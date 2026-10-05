"""Pool the cross-validation folds into the headline numbers."""

import argparse
import sys

import numpy as np
import pandas as pd

from src.config import add_config_args, load_paths
from src.cv_folds import fold_assignment, MIN_SUBJECTS_TO_TEST
from src.eval.exposure_breakdown import exposure_table
from src.utils import assign_bin_Lb, bin_accuracy, compute_metrics, CHANNEL_NAMES

RULES = [("bin_L", "L* only (1D)"), ("bin_Lb", "L* + b* (2D)"),
         ("bin_Lab", "L*a*b* (3D)")]

LINES = []


def emit(s=""):
    print(s)
    LINES.append(s)


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    paths = load_paths(args.paths)
    cv_dir = paths.results / "cv"
    out_txt = paths.results / "CV_REPORT.txt"

    files = sorted(cv_dir.glob("fold*_test_predictions.csv"))
    if not files:
        sys.exit(f"no fold predictions in {cv_dir}, run src.train_cv first")

    folds = [pd.read_csv(f, dtype={"subject_key": str}) for f in files]
    d = pd.concat(folds, ignore_index=True)

    emit("=" * 74)
    emit("5-FOLD SUBJECT-WISE CROSS-VALIDATION")
    emit("=" * 74)
    emit(f"folds found : {len(folds)}")
    emit(f"images      : {len(d)}")
    emit(f"subjects    : {d.subject_key.nunique()}")

    dup = d.subject_key.value_counts()
    dup = dup[dup > 12]
    if len(dup):
        emit(f"  WARNING: {len(dup)} subject(s) appear in more than one fold")

    _assign, train_only, table = fold_assignment(str(paths.table_csv))
    if train_only:
        pinned_bins = sorted(table[table.subject_key.isin(train_only)].mst_bin.unique())
        emit(f"training-only : {len(train_only)} subjects in bin {pinned_bins} "
             f"(fewer than {MIN_SUBJECTS_TO_TEST} subjects); used for learning, "
             "never scored")
    missing = set(table.subject_key) - set(d.subject_key) - set(train_only)
    if missing:
        emit(f"  WARNING: {len(missing)} subject(s) never tested")

    p = d[["L_pred", "a_pred", "b_pred"]].values
    g = d[["L_gt", "a_gt", "b_gt"]].values

    emit()
    emit("=" * 74)
    emit("POOLED OVER ALL FOLDS")
    emit("=" * 74)
    m = compute_metrics(p, g)
    emit(f"  {'channel':<8} {'MAE':>8} {'RMSE':>8} {'R2':>8}")
    for ch in CHANNEL_NAMES:
        emit(f"  {ch:<8} {m[ch+'_MAE']:>8.3f} {m[ch+'_RMSE']:>8.3f} {m[ch+'_R2']:>8.3f}")
    emit(f"  {'avg':<8} {m['avg_MAE']:>8.3f} {m['avg_RMSE']:>8.3f} {m['avg_R2']:>8.3f}")
    emit(f"  dE76  mean={d.dE76.mean():.3f}  std={d.dE76.std():.3f}  "
         f"median={d.dE76.median():.3f}")
    emit()
    emit(f"  {'rule':<22} {'exact':>9} {'within 1':>10} {'errors >=2 bins':>16}")
    for col, lab in RULES:
        ex, w1 = bin_accuracy(d.mst_gt, d[col])
        big = int((np.abs(d[col] - d.mst_gt) >= 2).sum())
        emit(f"  {lab:<22} {ex*100:>8.1f}% {w1*100:>9.1f}% {big:>16d}")

    oracle = d.apply(lambda r: assign_bin_Lb(r.L_gt, r.b_gt), axis=1)
    ceiling = (oracle == d.mst_gt).mean() * 100
    achieved = bin_accuracy(d.mst_gt, d.bin_Lb)[0] * 100
    emit()
    emit(f"  2D-rule ceiling over the same images : {ceiling:.1f}%")
    emit(f"  achieved                             : {achieved:.1f}%  "
         f"({achieved / ceiling * 100:.1f}% of ceiling)")

    emit()
    emit("=" * 74)
    emit("PER-FOLD")
    emit("=" * 74)
    emit(f"  {'fold':>4} {'n':>4} {'subj':>5} {'L*MAE':>7} {'L*R2':>7} {'dE76':>7} "
         f"{'exact':>8} {'±1bin':>7}")
    rows = []
    for f in folds:
        fp = f[["L_pred", "a_pred", "b_pred"]].values
        fg = f[["L_gt", "a_gt", "b_gt"]].values
        fm = compute_metrics(fp, fg)
        ex, w1 = bin_accuracy(f.mst_gt, f.bin_Lb)
        rows.append((fm["L*_MAE"], fm["L*_R2"], f.dE76.mean(), ex * 100, w1 * 100))
        emit(f"  {int(f.fold.iloc[0]):>4} {len(f):>4} {f.subject_key.nunique():>5} "
             f"{fm['L*_MAE']:>7.2f} {fm['L*_R2']:>7.3f} {f.dE76.mean():>7.2f} "
             f"{ex*100:>7.1f}% {w1*100:>6.1f}%")
    arr = np.array(rows)
    emit("  " + "-" * 58)
    for stat, fn in (("mean", np.mean), ("std", np.std),
                     ("min", np.min), ("max", np.max)):
        emit(f"  {stat:>4} {'':>4} {'':>5} {fn(arr[:,0]):>7.2f} {fn(arr[:,1]):>7.3f} "
             f"{fn(arr[:,2]):>7.2f} {fn(arr[:,3]):>7.1f}% {fn(arr[:,4]):>6.1f}%")

    emit()
    emit("=" * 74)
    emit("POOLED BY EXPOSURE LEVEL  (2D rule)")
    emit("=" * 74)
    latex = exposure_table(d, emit)
    emit()
    emit("  LaTeX rows:")
    for line in latex:
        emit("  " + line)

    emit()
    emit("=" * 74)
    emit("POOLED BY GROUND-TRUTH BIN  (2D rule)")
    emit("=" * 74)
    emit(f"  {'bin':>4} {'subj':>5} {'n':>4} {'exact':>8} {'±1bin':>8} {'ceiling':>8} "
         f"{'L*MAE':>7}")
    d["_oracle"] = oracle
    for b, s in d.groupby("mst_gt"):
        ex, w1 = bin_accuracy(s.mst_gt, s.bin_Lb)
        emit(f"  {b:>4} {s.subject_key.nunique():>5} {len(s):>4} {ex*100:>7.1f}% "
             f"{w1*100:>7.1f}% {(s._oracle == b).mean()*100:>7.1f}% "
             f"{(s.L_pred - s.L_gt).abs().mean():>7.2f}")

    unreachable = [sid for sid, s in d.groupby("subject_key")
                   if not (s._oracle == s.mst_gt).all()]
    emit()
    emit(f"  structurally unreachable under the 2D rule: {len(unreachable)} of "
         f"{d.subject_key.nunique()} subjects")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(LINES) + "\n")
    pooled = paths.results / "cv_pooled_predictions.csv"
    d.drop(columns="_oracle").to_csv(pooled, index=False)
    print(f"\nsaved -> {out_txt}")
    print(f"saved -> {pooled}")


if __name__ == "__main__":
    main()
