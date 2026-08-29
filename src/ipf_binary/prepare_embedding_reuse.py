from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reuse embeddings when the selected source series is unchanged")
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    dtypes = {"patient_id": "string", "ct_id": "string"}
    source_manifest = pd.read_csv(args.source_manifest, dtype=dtypes)
    target_manifest = pd.read_csv(args.target_manifest, dtype=dtypes)
    source_index = pd.read_csv(args.source_index, dtype=dtypes)
    source_manifest_by_patient = source_manifest.set_index("patient_id", drop=False)
    source_index_by_ct = source_index.set_index("ct_id", drop=False)
    args.target_dir.mkdir(parents=True, exist_ok=True)
    reused_records: list[dict[str, object]] = []
    pending_ct_ids: list[str] = []
    for target in target_manifest.itertuples(index=False):
        patient_id = str(target.patient_id)
        ct_id = str(target.ct_id)
        if patient_id not in source_manifest_by_patient.index:
            pending_ct_ids.append(ct_id)
            continue
        source = source_manifest_by_patient.loc[patient_id]
        if isinstance(source, pd.DataFrame):
            source = source.iloc[0]
        if str(source["ct_id"]) != ct_id or str(source["series_path"]) != str(target.series_path):
            pending_ct_ids.append(ct_id)
            continue
        if ct_id not in source_index_by_ct.index:
            pending_ct_ids.append(ct_id)
            continue
        index_record = source_index_by_ct.loc[ct_id]
        if isinstance(index_record, pd.DataFrame):
            index_record = index_record.iloc[0]
        source_path = Path(str(index_record["embedding_path"]))
        if str(index_record["status"]) != "ok" or not source_path.is_file():
            pending_ct_ids.append(ct_id)
            continue
        target_path = args.target_dir / f"{ct_id}.npz"
        shutil.copy2(source_path, target_path)
        record = index_record.to_dict()
        record.update(
            {
                "patient_id": patient_id,
                "ct_id": ct_id,
                "label": int(target.label),
                "split": str(target.split),
                "scan_date": str(target.scan_date),
                "embedding_path": str(target_path.resolve()),
            }
        )
        reused_records.append(record)
    index_path = args.target_dir / "embedding_index.csv"
    pd.DataFrame.from_records(reused_records).to_csv(index_path, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "target_scans": int(len(target_manifest)),
                "reused_embeddings": len(reused_records),
                "pending_embeddings": len(pending_ct_ids),
                "pending_ct_ids": pending_ct_ids,
                "index_path": str(index_path.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
