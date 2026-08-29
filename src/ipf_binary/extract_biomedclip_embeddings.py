from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .ct_io import load_ct_slices


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract public BiomedCLIP CT slice embeddings")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/ct_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/embeddings/biomedclip_index"))
    parser.add_argument("--prompt-config", type=Path, default=Path("configs/medsiglip_prompts.json"))
    parser.add_argument("--model-id", default="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
    parser.add_argument("--cache-dir", type=Path, default=Path(".model_cache/huggingface"))
    parser.add_argument("--cohort", choices=("index", "all"), default="index")
    parser.add_argument("--slices", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-mode", choices=("lung", "tri"), default="lung")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": "string", "ct_id": "string"})
    if args.cohort == "index" and "is_index_ct" in manifest.columns:
        manifest = manifest.loc[manifest["is_index_ct"] == 1].copy()
    manifest = manifest.loc[manifest["series_path"].fillna("").ne("")].copy()
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    prompt_config = json.loads(args.prompt_config.read_text(encoding="utf-8"))
    positive_prompts = list(prompt_config["positive"])
    prompts = positive_prompts + list(prompt_config["negative"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_root = str(args.cache_dir.resolve())
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HF_HUB_CACHE", str((args.cache_dir / "hub").resolve()))
    import open_clip

    hub_name = f"hf-hub:{args.model_id}"
    model, preprocess = open_clip.create_model_from_pretrained(hub_name, cache_dir=str(args.cache_dir))
    tokenizer = open_clip.get_tokenizer(hub_name)
    model = model.to(device).eval()
    tokenized = tokenizer(prompts).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        text_features = model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "embedding_index.csv"
    existing = (
        pd.read_csv(index_path, dtype={"patient_id": "string", "ct_id": "string"})
        if index_path.exists()
        else pd.DataFrame()
    )
    existing_by_ct = set(existing["ct_id"].astype(str)) if not existing.empty else set()
    output_records = existing.to_dict("records") if not existing.empty else []

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="BiomedCLIP CT scans"):
        ct_id = str(row.ct_id)
        output_path = args.output_dir / f"{ct_id}.npz"
        if not args.overwrite and ct_id in existing_by_ct and output_path.exists():
            continue
        try:
            images, slice_indices = load_ct_slices(Path(row.series_path), args.slices, args.window_mode)
            slice_embeddings: list[np.ndarray] = []
            slice_scores: list[float] = []
            for start in range(0, len(images), args.batch_size):
                batch = torch.stack(
                    [preprocess(image) for image in images[start : start + args.batch_size]]
                ).to(device)
                with torch.inference_mode(), torch.autocast(
                    device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                ):
                    image_features = model.encode_image(batch)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                    logits = 100.0 * image_features @ text_features.T
                features_np = image_features.detach().float().cpu().numpy()
                logits_np = logits.detach().float().cpu().numpy()
                slice_embeddings.append(features_np)
                for values in logits_np:
                    positive = float(np.mean(values[: len(positive_prompts)]))
                    negative = float(np.mean(values[len(positive_prompts) :]))
                    slice_scores.append(sigmoid(positive - negative))
            embeddings = np.concatenate(slice_embeddings, axis=0)
            scores = np.asarray(slice_scores, dtype=np.float32)
            pooled = np.concatenate(
                [embeddings.mean(axis=0), embeddings.max(axis=0)], axis=0
            ).astype(np.float32)
            np.savez_compressed(
                output_path,
                pooled_embedding=pooled,
                slice_embeddings=embeddings.astype(np.float16),
                slice_indices=slice_indices.astype(np.int16),
                zero_shot_slice_scores=scores,
            )
            output_records = [
                record for record in output_records if str(record.get("ct_id")) != ct_id
            ]
            output_records.append(
                {
                    "patient_id": str(row.patient_id),
                    "ct_id": ct_id,
                    "label": int(row.label),
                    "split": row.split,
                    "scan_date": row.scan_date,
                    "embedding_path": str(output_path.resolve()),
                    "zero_shot_score": float(scores.mean()),
                    "slice_count": int(len(slice_indices)),
                    "window_mode": args.window_mode,
                    "model_id": args.model_id,
                    "status": "ok",
                }
            )
        except Exception as exc:
            output_records = [
                record for record in output_records if str(record.get("ct_id")) != ct_id
            ]
            output_records.append(
                {
                    "patient_id": str(row.patient_id),
                    "ct_id": ct_id,
                    "label": int(row.label),
                    "split": row.split,
                    "scan_date": row.scan_date,
                    "embedding_path": "",
                    "zero_shot_score": np.nan,
                    "slice_count": 0,
                    "window_mode": args.window_mode,
                    "model_id": args.model_id,
                    "status": f"error:{type(exc).__name__}:{exc}",
                }
            )
        pd.DataFrame.from_records(output_records).to_csv(index_path, index=False, encoding="utf-8-sig")

    print(
        json.dumps(
            {
                "device": str(device),
                "model_id": args.model_id,
                "requested_scans": int(len(manifest)),
                "successful_scans": int(
                    sum(record.get("status") == "ok" for record in output_records)
                ),
                "index_path": str(index_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
