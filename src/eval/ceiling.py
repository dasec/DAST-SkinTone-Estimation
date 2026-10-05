"""Exact-match ceiling of each bin rule."""

import argparse

import pandas as pd

from src.config import add_config_args, load_paths, load_params
from src.dataset import make_splits
from src.utils import assign_bin_L, assign_bin_Lb, assign_bin_Lab


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    paths = load_paths(args.paths)
    P = load_params(args.params)
    csv = str(paths.table_csv)

    d = pd.read_csv(csv, dtype={"subject_key": str})
    tr, va, te = make_splits(csv, val_ratio=P["val_ratio"],
                             test_ratio=P["test_ratio"], seed=P["seed"],
                             verbose=False)

    # Ground-truth bins use full CIELAB, so a rule on fewer channels cannot
    # reach 100% exact match even from the exact colorimeter values.
    rules = {
        "L* only (1D)": lambda r: assign_bin_L(r["CIE-L*"]),
        "L* + b* (2D)": lambda r: assign_bin_Lb(r["CIE-L*"], r["CIE-b*"]),
        "L*a*b* (3D)": lambda r: assign_bin_Lab(r["CIE-L*"], r["CIE-a*"], r["CIE-b*"]),
    }

    print("How well each rule reproduces the ground-truth bin from the exact")
    print(f"colorimeter values ({len(d)} images / {d.subject_key.nunique()} subjects):")
    print(f"  {'rule':<16} {'exact':>9} {'within 1':>10}")
    for name, fn in rules.items():
        p = d.apply(fn, axis=1)
        exact = (p == d.mst_bin).mean() * 100
        w1 = ((p - d.mst_bin).abs() <= 1).mean() * 100
        print(f"  {name:<16} {exact:>8.1f}% {w1:>9.1f}%")

    print("\nCeiling of the 2D rule per split:")
    print(f"  {'split':<7} {'images':>7} {'subjects':>9} {'ceiling':>9}")
    for name, s in (("train", tr), ("val", va), ("test", te)):
        oracle = s.apply(rules["L* + b* (2D)"], axis=1)
        print(f"  {name:<7} {len(s):>7} {s.subject_key.nunique():>9} "
              f"{(oracle == s.mst_bin).mean() * 100:>8.1f}%")

    print("\nReachability per ground-truth bin under the 2D rule:")
    oracle = d.apply(rules["L* + b* (2D)"], axis=1)
    print(f"  {'bin':>4} {'subjects':>9} {'reproduced':>11}")
    for b, g in d.groupby("mst_bin"):
        print(f"  {b:>4} {g.subject_key.nunique():>9} "
              f"{(oracle[g.index] == b).mean() * 100:>10.1f}%")


if __name__ == "__main__":
    main()
