"""Merge the primary table with the CHROMA-FIT rows for the cross-dataset experiments."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import add_config_args, load_paths

LABELS = ["CIE-L*", "CIE-a*", "CIE-b*"]


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    paths = load_paths(args.paths)

    if not paths.table_csv.is_file():
        sys.exit(f"Missing {paths.table_csv}, run src.data.build_table first.")
    if not paths.chroma_table.is_file():
        sys.exit(f"Missing {paths.chroma_table}, needed for the CHROMA-FIT rows.")

    dast = pd.read_csv(paths.table_csv, dtype={"subject_key": str})
    dast["dataset"] = "dast"

    other = pd.read_csv(paths.chroma_table)
    chroma = other[other.dataset == "chroma"].copy()
    chroma["subject_key"] = chroma.subject_key.astype(str)

    print(f"primary : {len(dast):5d} images / {dast.subject_key.nunique()} subjects")
    print(f"chroma  : {len(chroma):5d} images / {chroma.subject_key.nunique()} subjects")

    overlap = set(dast.subject_key) & set(chroma.subject_key)
    if overlap:
        sys.exit(f"ABORT: {len(overlap)} subject_key collision(s) across datasets")

    missing_files = [p for p in chroma.complete_path if not Path(p).is_file()]
    if missing_files:
        print(f"  [WARN] {len(missing_files)} CHROMA-FIT images not found on disk")

    merged = pd.concat([dast, chroma], axis=0, ignore_index=True, sort=False)

    for col in ["UnifiedQualityScore.scalar", "BackgroundUniformity.scalar"]:
        if col not in merged.columns:
            sys.exit(f"ABORT: '{col}' missing: the quality filter needs it.")
        n_na = merged[merged.dataset == "chroma"][col].isna().sum()
        if n_na:
            print(f"  [WARN] {n_na} CHROMA-FIT rows have no {col}")

    for lab in LABELS:
        if merged[lab].isna().any():
            n = merged[lab].isna().sum()
            sys.exit(f"ABORT: {n} rows have no {lab}, cannot train on them.")

    front = ["complete_path", "dataset", "mst_bin", "subject_key", "triple_key"] + LABELS
    rest = [c for c in merged.columns if c not in front]
    merged = merged[front + rest]

    merged.to_csv(paths.merged_csv, index=False)

    print()
    print(f"Wrote {paths.merged_csv}")
    print(f"  total rows     : {len(merged)}")
    print(f"  total subjects : {merged.subject_key.nunique()}")
    print(f"  columns        : {len(merged.columns)}")
    print()
    print("  Per-channel mean gap (primary minus CHROMA-FIT), the device offset")
    print("  the mean-shift correction estimates on the training split:")
    d = merged[merged.dataset == "dast"][LABELS].mean()
    c = merged[merged.dataset == "chroma"][LABELS].mean()
    for lab in LABELS:
        print(f"    {lab:8s} {d[lab] - c[lab]:+7.2f}")


if __name__ == "__main__":
    main()
