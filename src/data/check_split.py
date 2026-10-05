"""Leakage gate on the single split: key format, subject disjointness, image hashes."""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from src.config import add_config_args, load_paths, load_params
from src.dataset import make_splits

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    paths = load_paths(args.paths)
    P = load_params(args.params)
    csv = str(paths.table_csv)

    d = pd.read_csv(csv, dtype={"subject_key": str})

    print("1. split key")
    check("subject_key carries no bin prefix",
          not d.subject_key.str.contains("_").any())
    check("subject_key is the bare 3-digit id",
          d.subject_key.str.fullmatch(r"\d{3}").all())
    check("each subject maps to one bin",
          (d.groupby("subject_key").mst_bin.nunique() == 1).all())

    tr, va, te = make_splits(csv, val_ratio=P["val_ratio"],
                             test_ratio=P["test_ratio"], seed=P["seed"],
                             verbose=False)

    print("\n2. split sizes")
    print(f"  {'split':<7} {'images':>7} {'subjects':>9}")
    for name, s in (("train", tr), ("val", va), ("test", te)):
        print(f"  {name:<7} {len(s):>7} {s.subject_key.nunique():>9}")
    check("every image assigned exactly once",
          len(tr) + len(va) + len(te) == len(d),
          f"{len(tr)}+{len(va)}+{len(te)} vs {len(d)}")

    print("\n3. subject disjointness")
    s_tr, s_va, s_te = (set(s.subject_key) for s in (tr, va, te))
    check("train/val disjoint", not (s_tr & s_va), f"{len(s_tr & s_va)} shared")
    check("train/test disjoint", not (s_tr & s_te), f"{len(s_tr & s_te)} shared")
    check("val/test disjoint", not (s_va & s_te), f"{len(s_va & s_te)} shared")

    print("\n4. no image content shared across splits")
    where = {}
    for name, s in (("train", tr), ("val", va), ("test", te)):
        for p in s.complete_path:
            h = hashlib.md5(Path(p).read_bytes()).hexdigest()
            where.setdefault(h, set()).add(name)
    shared = {h: v for h, v in where.items() if len(v) > 1}
    check("no hash in more than one split", not shared, f"{len(shared)} shared")
    check("all images distinct by content", len(where) == len(d),
          f"{len(where)} unique hashes for {len(d)} rows")

    print("\n5. exposure levels per split")
    els = sorted(d.EL.unique())
    print(f"  {'split':<7} " + " ".join(f"{e:>6}" for e in els) + f" {'total':>7}")
    for name, s in (("train", tr), ("val", va), ("test", te)):
        counts = s.EL.value_counts()
        print(f"  {name:<7} " + " ".join(f"{counts.get(e, 0):>6}" for e in els)
              + f" {len(s):>7}")

    print("\n6. bins per split (subjects)")
    print(f"  {'bin':>4} {'train':>6} {'val':>5} {'test':>5}")
    for b in sorted(d.mst_bin.unique()):
        row = [s[s.mst_bin == b].subject_key.nunique() for s in (tr, va, te)]
        print(f"  {b:>4} {row[0]:>6} {row[1]:>5} {row[2]:>5}")

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        sys.exit(1)
    print("Split is leakage-free.")


if __name__ == "__main__":
    main()
