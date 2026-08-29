from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a balanced end-to-end smoke-test cohort")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/manifests/index_ct_manifest.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifests/smoke_index_ct_manifest.csv"))
    parser.add_argument("--per-class-split", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    manifest = pd.read_csv(args.manifest, dtype={"patient_id": "string", "ct_id": "string"})
    parts: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            group = manifest.loc[(manifest["split"] == split) & (manifest["label"] == label)]
            selected = group.sort_values(["patient_id", "ct_id"]).head(args.per_class_split)
            if len(selected) < args.per_class_split:
                raise ValueError(f"Insufficient rows for split={split}, label={label}")
            parts.append(selected)
            counts[f"{split}_label_{label}"] = int(len(selected))
    output = pd.concat(parts, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": len(output), "counts": counts, "output": str(args.output.resolve())}, indent=2))


if __name__ == "__main__":
    main()
