"""Subject-wise 5-fold cross-validation folds, stratified by CST bin."""

import argparse

import numpy as np
import pandas as pd

from src.config import add_config_args, load_paths, load_params

_p = load_params()
N_FOLDS = _p["n_folds"]
SEED = _p["seed"]
MIN_SUBJECTS_TO_TEST = _p["min_subjects_to_test"]


def fold_assignment(csv_path, n_folds=N_FOLDS, seed=SEED,
                    min_subjects_to_test=MIN_SUBJECTS_TO_TEST):
    """Return (assignment, train_only_subjects, df).

    Subjects in bins with fewer than min_subjects_to_test members train in
    every fold and are never scored.
    """
    df = pd.read_csv(csv_path, dtype={"subject_key": str})

    spanning = df.groupby("subject_key").mst_bin.nunique()
    if (spanning > 1).any():
        raise ValueError(f"subjects spanning bins: {sorted(spanning[spanning>1].index)}")

    subjects = df.groupby("subject_key").mst_bin.first().reset_index()
    per_bin = subjects.groupby("mst_bin").size()
    small_bins = set(per_bin[per_bin < min_subjects_to_test].index)

    rng = np.random.default_rng(seed)
    assignment, train_only = {}, set()
    for bin_val, group in subjects.groupby("mst_bin"):
        ids = group.subject_key.tolist()
        rng.shuffle(ids)
        if bin_val in small_bins:
            train_only.update(ids)
            continue
        for k, sid in enumerate(ids):
            assignment[sid] = k % n_folds
    return assignment, train_only, df


def make_fold(csv_path, fold, n_folds=N_FOLDS, seed=SEED):
    """train/val/test for one fold: test = fold, val = fold+1 wrapping, train = rest."""
    if not 0 <= fold < n_folds:
        raise ValueError(f"fold must be in 0..{n_folds - 1}, got {fold}")

    assignment, train_only, df = fold_assignment(csv_path, n_folds, seed)
    df["_fold"] = df.subject_key.map(assignment)

    test_fold = fold
    val_fold = (fold + 1) % n_folds

    pinned = df.subject_key.isin(train_only)
    test_df = df[df._fold == test_fold].drop(columns="_fold").reset_index(drop=True)
    val_df = df[df._fold == val_fold].drop(columns="_fold").reset_index(drop=True)
    train_df = df[pinned | (~pinned & ~df._fold.isin([test_fold, val_fold]))] \
                 .drop(columns="_fold").reset_index(drop=True)

    s_tr, s_va, s_te = (set(x.subject_key) for x in (train_df, val_df, test_df))
    assert not (s_tr & s_va), "train/val subject overlap"
    assert not (s_tr & s_te), "train/test subject overlap"
    assert not (s_va & s_te), "val/test subject overlap"
    assert len(train_df) + len(val_df) + len(test_df) == len(df), "lost rows"

    return train_df, val_df, test_df


def describe(csv_path, n_folds=N_FOLDS, seed=SEED):
    assignment, train_only, df = fold_assignment(csv_path, n_folds, seed)
    df["_fold"] = df.subject_key.map(assignment)

    print(f"{df.subject_key.nunique()} subjects, {len(df)} images, "
          f"{n_folds} folds, seed {seed}")
    if train_only:
        pinned_bins = sorted(df[df.subject_key.isin(train_only)].mst_bin.unique())
        print(f"training-only (bin(s) {pinned_bins} have < {MIN_SUBJECTS_TO_TEST} "
              f"subjects): {len(train_only)} subjects")
        print(f"tested subjects: {len(assignment)}   "
              f"tested images: {len(df[~df.subject_key.isin(train_only)])}")
    print()

    print("subjects per bin per fold  (train-only bins show as 0)")
    print(f"  {'bin':>4} " + " ".join(f"{'f'+str(i):>4}" for i in range(n_folds))
          + f" {'tested':>7} {'pinned':>7}")
    for b, g in df.groupby("mst_bin"):
        counts = [g[g._fold == i].subject_key.nunique() for i in range(n_folds)]
        n_pin = g[g.subject_key.isin(train_only)].subject_key.nunique()
        print(f"  {b:>4} " + " ".join(f"{c:>4}" for c in counts)
              + f" {sum(counts):>7} {n_pin:>7}")
    totals = [df[df._fold == i].subject_key.nunique() for i in range(n_folds)]
    print(f"  {'all':>4} " + " ".join(f"{c:>4}" for c in totals)
          + f" {sum(totals):>7} {len(train_only):>7}")

    print("\nsplit sizes per fold")
    print(f"  {'fold':>4} {'train':>14} {'val':>14} {'test':>14}")
    for i in range(n_folds):
        tr, va, te = make_fold(csv_path, i, n_folds, seed)
        print(f"  {i:>4} "
              f"{f'{len(tr)}/{tr.subject_key.nunique()}s':>14} "
              f"{f'{len(va)}/{va.subject_key.nunique()}s':>14} "
              f"{f'{len(te)}/{te.subject_key.nunique()}s':>14}")

    covered, seen_counts = set(), {}
    for i in range(n_folds):
        tr, _, te = make_fold(csv_path, i, n_folds, seed)
        covered |= set(te.subject_key)
        for sid in train_only:
            seen_counts[sid] = seen_counts.get(sid, 0) + int(sid in set(tr.subject_key))
    print(f"\ntested exactly once: {len(covered)} of {len(assignment)} testable")
    if train_only:
        ok = all(v == n_folds for v in seen_counts.values())
        print(f"pinned subjects in training in all {n_folds} folds: "
              f"{'yes' if ok else 'NO'}")


if __name__ == "__main__":
    parser = add_config_args(argparse.ArgumentParser())
    args = parser.parse_args()
    describe(str(load_paths(args.paths).table_csv))
