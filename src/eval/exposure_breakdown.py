"""Bin accuracy per exposure level over the pooled cross-validation predictions.

Reports the robustness table of the paper, plus its rows in LaTeX form.

    python -m src.eval.exposure_breakdown
"""

import argparse
import sys

import pandas as pd

from src.config import add_config_args, load_paths
from src.utils import bin_accuracy

EL_ORDER = ["EL1", "EL2", "EL3", "EL4", "EL5"]
EL_NAME = {"EL1": "strongly underexposed", "EL2": "underexposed", "EL3": "normal",
           "EL4": "overexposed", "EL5": "strongly overexposed"}


def exposure_table(d, emit=print, rule="bin_Lb"):
    """Emit the per-exposure-level table; returns the LaTeX rows."""
    emit(f"  {'EL':<5} {'exposure':<24} {'n':>4} {'exact':>8} {'±1bin':>8} "
         f"{'bias':>7} {'L*MAE':>7} {'dE76':>7}")
    latex = []
    for el in EL_ORDER:
        s = d[d.EL == el]
        if s.empty:
            continue
        ex, w1 = bin_accuracy(s.mst_gt, s[rule])
        bias = (s[rule] - s.mst_gt).mean()
        lmae = (s.L_pred - s.L_gt).abs().mean()
        emit(f"  {el:<5} {EL_NAME[el]:<24} {len(s):>4} {ex*100:>7.1f}% "
             f"{w1*100:>7.1f}% {bias:>+7.2f} {lmae:>7.2f} {s.dE76.mean():>7.2f}")
        latex.append(f"    {el} & {EL_NAME[el].capitalize()} & {len(s)} & "
                     f"{ex*100:.1f}\\% & {w1*100:.1f}\\% & ${bias:+.2f}$ \\\\")
    ex, w1 = bin_accuracy(d.mst_gt, d[rule])
    emit(f"  {'All':<5} {'':<24} {len(d):>4} {ex*100:>7.1f}% {w1*100:>7.1f}% "
         f"{(d[rule] - d.mst_gt).mean():>+7.2f} "
         f"{(d.L_pred - d.L_gt).abs().mean():>7.2f} {d.dE76.mean():>7.2f}")
    return latex


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    paths = load_paths(args.paths)
    pooled = paths.results / "cv_pooled_predictions.csv"
    if not pooled.is_file():
        sys.exit(f"missing {pooled} — run src.eval.collect_cv first")

    d = pd.read_csv(pooled, dtype={"subject_key": str})
    print("=" * 74)
    print("BIN ACCURACY BY EXPOSURE LEVEL  (2D rule, pooled cross-validation)")
    print("=" * 74)
    latex = exposure_table(d)
    print("\n  LaTeX rows:")
    for line in latex:
        print("  " + line)


if __name__ == "__main__":
    main()
