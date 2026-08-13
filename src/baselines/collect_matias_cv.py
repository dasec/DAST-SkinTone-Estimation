"""Pool the fine-tuned baseline folds and put them beside the main model.

Both are pooled over the same images from the same subjects. The zero-shot row
comes from eval_matias.py --cv, which scores that same image set.

    python -m src.baselines.collect_matias_cv
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.config import add_config_args, load_paths
from src.utils import bin_accuracy, compute_metrics

LINES = []


def emit(s=""):
    print(s)
    LINES.append(s)


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    paths = load_paths(args.paths)
    ours_csv = paths.results / "cv_pooled_predictions.csv"
    matias_cv = paths.results / "cv_matias"
    matias_zs = paths.results / "matias_zeroshot_cv.csv"
    out = paths.results / "MATIAS_CV.txt"

    files = sorted(matias_cv.glob("fold*_test_predictions.csv"))
    if not files:
        sys.exit(f"no fine-tuned folds in {matias_cv} — run src.baselines.train_matias_cv")
    mf = pd.concat([pd.read_csv(f, dtype={"subject_key": str}) for f in files],
                   ignore_index=True)

    if not ours_csv.is_file():
        sys.exit(f"missing {ours_csv}")
    ours = pd.read_csv(ours_csv, dtype={"subject_key": str})

    emit("=" * 74)
    emit("EXTERNAL BASELINE UNDER THE SAME 5-FOLD CROSS-VALIDATION")
    emit("=" * 74)
    emit(f"folds pooled : {len(files)}")
    emit(f"ours         : {len(ours)} images / {ours.subject_key.nunique()} subjects")
    emit(f"baseline (ft): {len(mf)} images / {mf.subject_key.nunique()} subjects")

    same = set(ours.subject_key) == set(mf.subject_key) and len(ours) == len(mf)
    emit(f"same evaluation set: {'yes' if same else 'NO — comparison is invalid'}")

    rows = []

    zs = None
    if matias_zs.is_file():
        zs = pd.read_csv(matias_zs, dtype={"subject_key": str})
        if "EL" not in zs.columns:
            zs["EL"] = zs.exposure_code.map(
                {"ue2": "EL1", "ue1": "EL2", "optimal": "EL3",
                 "oe1": "EL4", "oe2": "EL5"})
        ex, w1 = bin_accuracy(zs.mst_gt, zs.mst_pred)
        rows.append(("Baseline, zero-shot", len(zs), ex * 100, w1 * 100,
                     np.nan, np.nan))
    else:
        emit(f"\n  WARNING: {matias_zs} missing — run src.baselines.eval_matias --cv")

    for label, d in (("Baseline, fine-tuned", mf), ("Ours", ours)):
        ex, w1 = bin_accuracy(d.mst_gt, d.bin_Lb)
        m = compute_metrics(d[["L_pred", "a_pred", "b_pred"]].values,
                            d[["L_gt", "a_gt", "b_gt"]].values)
        rows.append((label, len(d), ex * 100, w1 * 100, m["L*_R2"], d.dE76.mean()))

    emit()
    emit(f"  {'model':<28} {'n':>5} {'exact':>8} {'within 1':>10} "
         f"{'L* R2':>8} {'dE76':>7}")
    emit("  " + "-" * 70)
    for label, n, ex, w1, r2, de in rows:
        r2s = f"{r2:>8.3f}" if not np.isnan(r2) else f"{'n/a':>8}"
        des = f"{de:>7.2f}" if not np.isnan(de) else f"{'n/a':>7}"
        emit(f"  {label:<28} {n:>5} {ex:>7.1f}% {w1:>9.1f}% {r2s} {des}")
    emit("  " + "-" * 70)
    emit("  zero-shot predicts a class directly, so it has no CIELAB output")

    if zs is not None:
        emit()
        emit("  zero-shot by exposure level")
        emit(f"  {'EL':<5} {'n':>5} {'exact':>8} {'within 1':>10}")
        for el, g in zs.groupby("EL"):
            ex, w1 = bin_accuracy(g.mst_gt, g.mst_pred)
            emit(f"  {el:<5} {len(g):>5} {ex*100:>7.1f}% {w1*100:>9.1f}%")

    emit()
    emit("  fine-tuned, per fold")
    emit(f"  {'fold':>4} {'n':>5} {'exact':>8} {'within 1':>10}")
    for f, g in mf.groupby("fold"):
        ex, w1 = bin_accuracy(g.mst_gt, g.bin_Lb)
        emit(f"  {int(f):>4} {len(g):>5} {ex*100:>7.1f}% {w1*100:>9.1f}%")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(LINES) + "\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
