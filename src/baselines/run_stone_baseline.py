"""Pixel-based baseline: the `stone` CLI with a palette built from the CST centroids."""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import color as skcolor

from src.config import add_config_args, load_paths, load_params
from src.dataset import make_splits
from src.utils import CST_L, CST_A, CST_B

CST_LABELS = [f"C{chr(ord('A') + i)}" for i in range(10)]
LABEL_TO_BIN = {lab: i + 1 for i, lab in enumerate(CST_LABELS)}


def lab_to_hex(L, a, b):
    rgb = np.clip(skcolor.lab2rgb(np.array([[[L, a, b]]]))[0][0], 0, 1)
    return "#{:02X}{:02X}{:02X}".format(*(int(v * 255) for v in rgb))


def main():
    parser = add_config_args(argparse.ArgumentParser())
    parser.add_argument("--cv", action="store_true",
                        help="score the cross-validated image set instead of the "
                             "single test split")
    args = parser.parse_args()

    paths = load_paths(args.paths)
    P = load_params(args.params)
    csv_path = str(paths.table_csv)
    palette = [lab_to_hex(L, a, b) for L, a, b in zip(CST_L, CST_A, CST_B)]
    out_csv = paths.results / (
        "stone_cst_cv.csv" if args.cv else "stone_cst_test.csv")

    if args.cv:
        from src.cv_folds import fold_assignment
        _assign, train_only, test_df = fold_assignment(csv_path)
        test_df = test_df[~test_df.subject_key.isin(train_only)].reset_index(drop=True)
        scope = "CV-scored set"
    else:
        _, _, test_df = make_splits(csv_path, val_ratio=P["val_ratio"],
                                    test_ratio=P["test_ratio"], seed=P["seed"],
                                    verbose=False)
        scope = "single test split"
    test_df = test_df.copy()
    test_df["subject_key"] = test_df.subject_key.astype(str).str.zfill(3)
    print(f"{scope}: {len(test_df)} images / {test_df.subject_key.nunique()} subjects")

    folders = {}
    for r in test_df.itertuples(index=False):
        folders.setdefault(str(Path(r.complete_path).parent), (r.subject_key, r.mst_bin))

    work_dir = paths.stone_work
    work_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (folder, (sid, gt_bin)) in enumerate(sorted(folders.items()), 1):
        out_dir = work_dir / sid
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [{i}/{len(folders)}] bin {gt_bin}", flush=True)

        subprocess.run(
            ["stone", "-i", folder, "-t", "color",
             "-p", *palette, "-l", *CST_LABELS,
             "-o", str(out_dir)],
            capture_output=True,
        )

        res = out_dir / "result.csv"
        if not res.is_file():
            print("      no result.csv: stone failed on this folder")
            continue

        df = pd.read_csv(res)
        # stone can emit several rows per image; the most confident is kept.
        df = df.sort_values("accuracy(0-100)", ascending=False).drop_duplicates("file")
        df["subject_key"] = sid
        df["mst_bin_gt"] = gt_bin
        rows.append(df[["file", "subject_key", "mst_bin_gt", "tone label",
                        "accuracy(0-100)"]])

    if not rows:
        sys.exit("stone produced no output at all, check the CLI installation.")

    stone_df = pd.concat(rows, ignore_index=True)
    stone_df["predicted_bin"] = stone_df["tone label"].map(LABEL_TO_BIN)

    test_df["file"] = test_df.complete_path.map(lambda p: Path(p).name)
    merged = stone_df.merge(
        test_df[["subject_key", "file", "EL", "exposure_code", "camera", "session_id"]],
        on=["subject_key", "file"], how="left",
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)

    missing = merged.predicted_bin.isna().sum()
    print(f"\nWrote {out_csv}")
    print(f"  rows {len(merged)} of {len(test_df)} images"
          f" ({missing} with no stone prediction)")

    valid = merged.dropna(subset=["predicted_bin"])
    gt = valid.mst_bin_gt.values
    pred = valid.predicted_bin.values.astype(int)
    print(f"  OVERALL exact={(gt == pred).mean() * 100:.1f}%  "
          f"within1={(np.abs(gt - pred) <= 1).mean() * 100:.1f}%  "
          f"bias={(pred - gt).mean():+.2f}  (n={len(valid)})")
    print()
    print(f"  {'EL':<5} {'n':>4} {'exact':>8} {'within1':>9} {'bias':>7}")
    for el in ["EL1", "EL2", "EL3", "EL4", "EL5"]:
        s = valid[valid.EL == el]
        if s.empty:
            continue
        g = s.mst_bin_gt.values
        p = s.predicted_bin.values.astype(int)
        print(f"  {el:<5} {len(s):>4} {(g == p).mean() * 100:>7.1f}% "
              f"{(np.abs(g - p) <= 1).mean() * 100:>8.1f}% {(p - g).mean():>+7.2f}")


if __name__ == "__main__":
    main()
