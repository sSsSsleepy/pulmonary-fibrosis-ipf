from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from .leakage import audit_identifiers, embedding_similarity_summary, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit corrected IPF cohort inputs for leakage")
    parser.add_argument(
        "--cohort",
        type=Path,
        default=Path("artifacts/manifests_corrected/corrected_lung_index_ct_manifest.csv"),
    )
    parser.add_argument("--embedding-index", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/manifests_corrected"),
    )
    return parser.parse_args()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    cohort = pd.read_csv(args.cohort, dtype={"patient_id": "string", "ct_id": "string"})
    if not cohort["patient_id"].is_unique or not cohort["ct_id"].is_unique:
        raise AssertionError("corrected cohort must have unique patient and CT identifiers")
    identifier_audit = audit_identifiers(cohort)

    fingerprints: list[dict[str, object]] = []
    for row in tqdm(cohort.itertuples(index=False), total=len(cohort), desc="Hash CT inputs"):
        path = Path(row.series_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        fingerprints.append(
            {
                "patient_id": str(row.patient_id),
                "ct_id": str(row.ct_id),
                "series_sha256": sha256_file(path),
                "series_size_bytes": int(stat.st_size),
                "series_mtime_ns": int(stat.st_mtime_ns),
            }
        )
    fingerprint_frame = pd.DataFrame.from_records(fingerprints)
    if fingerprint_frame["series_sha256"].duplicated().any():
        raise AssertionError("exact duplicate CT input files detected across patients")

    embedding_audit: dict[str, object] | None = None
    if args.embedding_index is not None:
        index = pd.read_csv(
            args.embedding_index,
            dtype={"patient_id": "string", "ct_id": "string"},
        )
        index = cohort[["patient_id", "ct_id"]].merge(
            index.loc[index["status"].eq("ok")],
            on=["patient_id", "ct_id"],
            how="left",
            validate="one_to_one",
        )
        if index["embedding_path"].isna().any():
            raise AssertionError("embedding index does not cover the corrected cohort")
        vectors = []
        for path_text in index["embedding_path"]:
            with np.load(Path(path_text)) as data:
                vectors.append(data["pooled_embedding"].astype(np.float32))
        embedding_audit = embedding_similarity_summary(np.stack(vectors))
        if embedding_audit["exact_duplicate_vectors"] > 0:
            raise AssertionError("exact duplicate embedding vectors detected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_path = args.output_dir / "corrected_input_fingerprints.csv"
    fingerprint_frame.to_csv(fingerprint_path, index=False, encoding="utf-8-sig")
    audit = {
        "status": "passed",
        "patients": int(len(cohort)),
        "total_series_bytes": int(fingerprint_frame["series_size_bytes"].sum()),
        "exact_duplicate_series_files": 0,
        **identifier_audit,
        "embedding_similarity": embedding_audit,
        "fingerprint_file": str(fingerprint_path.resolve()),
    }
    audit_path = args.output_dir / "corrected_cohort_leakage_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
