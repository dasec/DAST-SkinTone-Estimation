"""Pool the cross-validated backbone ablation into one table."""

import argparse
import re

import numpy as np
import pandas as pd

from src.config import add_config_args, load_paths, load_params
from src.utils import bin_accuracy, compute_metrics

LINES = []


def emit(s=""):
    print(s)
    LINES.append(s)


def cell_name(stem):
    """cv3_efficientnet_b4_aligned_test_predictions -> (efficientnet_b4, True)."""
    m = re.match(r"cv(\d)_(.+?)(_aligned)?_test_predictions$", stem)
    if not m:
        return None
    return m.group(2), bool(m.group(3))


def load_cells(abl_dir, main_dir):
    cells = {}
    for f in sorted(abl_dir.glob("cv*_test_predictions.csv")):
        parsed = cell_name(f.stem)
        if parsed is None:
            continue
        cells.setdefault(parsed, []).append(f)
    main = sorted(main_dir.glob("fold*_test_predictions.csv"))
    if main:
        cells[("vit_small_patch16_224", True)] = main
    return cells


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    paths = load_paths(args.paths)
    n_folds = load_params(args.params)["n_folds"]
    abl_dir = paths.results / "cv_ablation"
    out = paths.results / "CV_ABLATION.txt"
    summary_csv = paths.results / "cv_ablation_summary.csv"

    cells = load_cells(abl_dir, paths.results / "cv")
    if not cells:
        raise SystemExit(f"no ablation folds found in {abl_dir}")

    emit("=" * 88)
    emit("BACKBONE ABLATION UNDER 5-FOLD SUBJECT-WISE CROSS-VALIDATION")
    emit("=" * 88)
    emit("Every cell uses the same folds as the main cross-validation.")
    emit()
    emit(f"  {'backbone':<34} {'align':<6} {'folds':>6} {'n':>6} {'L* R2':>8} "
         f"{'b* R2':>8} {'avgMAE':>8} {'dE76':>7} {'exact':>8} {'±1bin':>8}")
    emit("  " + "-" * 106)

    rows = []
    for (bb, aligned), files in sorted(cells.items()):
        d = pd.concat([pd.read_csv(f, dtype={"subject_key": str}) for f in files],
                      ignore_index=True)
        m = compute_metrics(d[["L_pred", "a_pred", "b_pred"]].values,
                            d[["L_gt", "a_gt", "b_gt"]].values)
        ex, w1 = bin_accuracy(d.mst_gt, d.bin_Lb)
        partial = "" if len(files) == n_folds else "  PARTIAL"
        emit(f"  {bb:<34} {'yes' if aligned else 'no':<6} {len(files):>6} "
             f"{len(d):>6} {m['L*_R2']:>8.3f} {m['b*_R2']:>8.3f} "
             f"{m['avg_MAE']:>8.2f} {d.dE76.mean():>7.2f} {ex*100:>7.1f}% "
             f"{w1*100:>7.1f}%{partial}")
        rows.append({
            "backbone": bb, "aligned": aligned, "folds": len(files), "n": len(d),
            "L_R2": m["L*_R2"], "a_R2": m["a*_R2"], "b_R2": m["b*_R2"],
            "avg_MAE": m["avg_MAE"], "dE76": d.dE76.mean(),
            "exact": ex * 100, "within1": w1 * 100,
        })
    emit("  " + "-" * 106)

    df = pd.DataFrame(rows)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_csv, index=False)

    complete = df[df.folds == n_folds]
    if len(complete):
        best = complete.sort_values("L_R2", ascending=False).iloc[0]
        emit()
        emit(f"  best by L* R2 among complete cells: {best.backbone} "
             f"{'aligned' if best.aligned else 'unaligned'} ({best.L_R2:.3f})")

    emit()
    emit("=" * 88)
    emit("EFFECT OF ALIGNMENT, per backbone (aligned minus unaligned)")
    emit("=" * 88)
    emit(f"  {'backbone':<34} {'d L* R2':>9} {'d b* R2':>9} {'d avgMAE':>10}")
    for bb in sorted(df.backbone.unique()):
        a = df[(df.backbone == bb) & df.aligned]
        u = df[(df.backbone == bb) & ~df.aligned]
        if a.empty or u.empty or a.folds.iloc[0] != n_folds or u.folds.iloc[0] != n_folds:
            emit(f"  {bb:<34} {'incomplete pair':>30}")
            continue
        emit(f"  {bb:<34} {a.L_R2.iloc[0]-u.L_R2.iloc[0]:>+9.3f} "
             f"{a.b_R2.iloc[0]-u.b_R2.iloc[0]:>+9.3f} "
             f"{a.avg_MAE.iloc[0]-u.avg_MAE.iloc[0]:>+10.2f}")
    emit("  (positive d R2 means alignment helps; positive d MAE means it hurts)")

    emit()
    emit("=" * 88)
    emit("PER-FOLD SPREAD, complete cells only")
    emit("=" * 88)
    emit(f"  {'backbone':<34} {'align':<6} {'L* R2 mean':>11} {'std':>7} "
         f"{'min':>7} {'max':>7}")
    for (bb, aligned), files in sorted(cells.items()):
        if len(files) != n_folds:
            continue
        per = []
        for f in files:
            d = pd.read_csv(f, dtype={"subject_key": str})
            per.append(compute_metrics(d[["L_pred", "a_pred", "b_pred"]].values,
                                       d[["L_gt", "a_gt", "b_gt"]].values)["L*_R2"])
        per = np.array(per)
        emit(f"  {bb:<34} {'yes' if aligned else 'no':<6} {per.mean():>11.3f} "
             f"{per.std():>7.3f} {per.min():>7.3f} {per.max():>7.3f}")

    missing = [(bb, al, len(f)) for (bb, al), f in cells.items() if len(f) != n_folds]
    if missing:
        emit()
        emit("INCOMPLETE CELLS")
        for bb, al, n in sorted(missing):
            emit(f"  {bb} {'aligned' if al else 'unaligned'}: {n}/{n_folds} folds")

    out.write_text("\n".join(LINES) + "\n")
    print(f"\nsaved -> {out}")
    print(f"saved -> {summary_csv}")


if __name__ == "__main__":
    main()
