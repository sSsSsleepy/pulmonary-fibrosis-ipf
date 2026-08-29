from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from .ct_io import load_ct_slices


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def extract_one(
    model: torch.nn.Module,
    processor: object,
    device: torch.device,
    images: list[Image.Image],
    prompts: list[str],
    positive_prompt_count: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    embeddings: list[np.ndarray] = []
    zero_shot_scores: list[float] = []
    for start in range(0, len(images), batch_size):
        batch_images = images[start : start + batch_size]
        inputs = processor(
            text=prompts,
            images=batch_images,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        autocast_enabled = device.type == "cuda"
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=autocast_enabled,
        ):
            outputs = model(**inputs)
        image_embeddings = outputs.image_embeds.detach().float().cpu().numpy()
        logits = outputs.logits_per_image.detach().float().cpu().numpy()
        embeddings.append(image_embeddings)
        for row in logits:
            positive = float(np.mean(row[:positive_prompt_count]))
            negative = float(np.mean(row[positive_prompt_count:]))
            zero_shot_scores.append(sigmoid(positive - negative))
    return np.concatenate(embeddings, axis=0), np.asarray(zero_shot_scores, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MedSigLIP CT slice embeddings")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/ct_manifest.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/embeddings/medsiglip_index"))
    parser.add_argument("--prompt-config", type=Path, default=Path("configs/medsiglip_prompts.json"))
    parser.add_argument("--model-id", default="google/medsiglip-448")
    parser.add_argument("--cache-dir", type=Path, default=Path(".model_cache/huggingface"))
    parser.add_argument("--cohort", choices=("index", "all"), default="index")
    parser.add_argument("--slices", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--window-mode", choices=("lung", "tri"), default="lung")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": "string", "ct_id": "string"})
    if args.cohort == "index":
        manifest = manifest.loc[manifest["is_index_ct"] == 1].copy()
    manifest = manifest.loc[manifest["series_path"].fillna("").ne("")].copy()
    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    prompt_config = json.loads(args.prompt_config.read_text(encoding="utf-8"))
    positive_prompts = list(prompt_config["positive"])
    negative_prompts = list(prompt_config["negative"])
    prompts = positive_prompts + negative_prompts

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        cache_dir=str(args.cache_dir),
    ).to(device)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id, cache_dir=str(args.cache_dir))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.output_dir / "embedding_index.csv"
    existing = pd.read_csv(index_path, dtype={"patient_id": "string", "ct_id": "string"}) if index_path.exists() else pd.DataFrame()
    existing_by_ct = set(existing["ct_id"].astype(str)) if not existing.empty else set()
    output_records = existing.to_dict("records") if not existing.empty else []

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="MedSigLIP CT scans"):
        ct_id = str(row.ct_id)
        output_path = args.output_dir / f"{ct_id}.npz"
        if not args.overwrite and ct_id in existing_by_ct and output_path.exists():
            continue
        try:
            images, slice_indices = load_ct_slices(Path(row.series_path), args.slices, args.window_mode)
            slice_embeddings, slice_scores = extract_one(
                model,
                processor,
                device,
                images,
                prompts,
                len(positive_prompts),
                args.batch_size,
            )
            pooled = np.concatenate(
                [slice_embeddings.mean(axis=0), slice_embeddings.max(axis=0)], axis=0
            ).astype(np.float32)
            np.savez_compressed(
                output_path,
                pooled_embedding=pooled,
                slice_embeddings=slice_embeddings.astype(np.float16),
                slice_indices=slice_indices.astype(np.int16),
                zero_shot_slice_scores=slice_scores,
            )
            output_records = [record for record in output_records if str(record.get("ct_id")) != ct_id]
            output_records.append(
                {
                    "patient_id": str(row.patient_id),
                    "ct_id": ct_id,
                    "label": int(row.label),
                    "split": row.split,
                    "scan_date": row.scan_date,
                    "embedding_path": str(output_path.resolve()),
                    "zero_shot_score": float(slice_scores.mean()),
                    "slice_count": int(len(slice_indices)),
                    "window_mode": args.window_mode,
                    "model_id": args.model_id,
                    "status": "ok",
                }
            )
        except Exception as exc:
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
                "successful_scans": int(sum(record.get("status") == "ok" for record in output_records)),
                "index_path": str(index_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
