"""Integrity gate on the dataset release; exits non-zero on the first real problem.

    python -m src.data.verify_dataset
"""

import argparse
import hashlib
import sys

import pandas as pd

from src.config import add_config_args, load_paths

# Subjects per CST class, from the release documentation.
EXPECTED_COMPOSITION = {1: 2, 2: 10, 3: 10, 4: 10, 5: 10, 6: 10, 7: 10, 8: 10, 9: 8}
EXPECTED_IMAGES = 960
EXPECTED_SUBJECTS = 80
IMAGES_PER_SUBJECT = 12

E_FROM_CODE = {"ue2": "E1", "ue1": "E2", "optimal": "E3", "oe1": "E4", "oe2": "E5"}

failures = []


def check(label, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    if not ok:
        failures.append(label)


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    paths = load_paths(args.paths)
    root = paths.dataset_root

    d = pd.read_csv(paths.metadata_csv)
    d["sid"] = d.subject_id.astype(str).str.zfill(3)
    print(f"metadata rows: {len(d)}\n")

    images = sorted(root.glob("subjects/*/*/Pictures/cropped/*.png"))
    subject_dirs = sorted(p for p in root.glob("subjects/*/*") if p.is_dir())

    print("1. counts")
    check(f"{EXPECTED_IMAGES} images on disk", len(images) == EXPECTED_IMAGES,
          f"found {len(images)}")
    check(f"{EXPECTED_SUBJECTS} subject folders",
          len(subject_dirs) == EXPECTED_SUBJECTS, f"found {len(subject_dirs)}")
    check(f"{EXPECTED_SUBJECTS} unique subject ids",
          d.sid.nunique() == EXPECTED_SUBJECTS, f"found {d.sid.nunique()}")
    per_subject = d.groupby("sid").size()
    off = per_subject[per_subject != IMAGES_PER_SUBJECT]
    check(f"every subject has {IMAGES_PER_SUBJECT} images", len(off) == 0,
          f"{len(off)} deviating" if len(off) else "")

    print("\n2. duplicate subjects across bins")
    ids = [p.name for p in subject_dirs]
    repeated = sorted({i for i in ids if ids.count(i) > 1})
    check("no subject id under two bin folders", not repeated,
          f"{len(repeated)} repeated")

    # Content hash rather than folder name: an earlier distribution shipped the
    # same subject under two bins, which a name-based check cannot see.
    seen, collisions = {}, []
    for p in images:
        h = hashlib.md5(p.read_bytes()).hexdigest()
        if h in seen:
            collisions.append((seen[h], str(p)))
        else:
            seen[h] = str(p)
    check("no two images byte-identical", not collisions, f"{len(collisions)} pairs")

    print("\n3. metadata and images line up")
    on_disk = {p.name for p in images}
    in_meta = set(d.image_filename)
    check("no image without a metadata row", not (on_disk - in_meta),
          f"{len(on_disk - in_meta)} unmatched")
    check("no metadata row without an image", not (in_meta - on_disk),
          f"{len(in_meta - on_disk)} unmatched")
    unresolved = [r for r in d.image_relpath if not (root.parent / r).is_file()]
    check("every image_relpath resolves", not unresolved, f"{len(unresolved)} missing")

    print("\n4. exposure encoding")
    d["E_from_name"] = d.image_filename.str.extract(r"-(E\d)-")[0]
    d["E_from_code"] = d.exposure_code.map(E_FROM_CODE)
    check("filename E-level agrees with exposure_code",
          (d.E_from_name == d.E_from_code).all(),
          f"{(d.E_from_name != d.E_from_code).sum()} mismatches")
    print(f"       {d.E_from_name.value_counts().sort_index().to_dict()}")

    print("\n5. ground truth")
    folder_bin = {p.name: int(p.parent.name) for p in subject_dirs}
    d["folder_bin"] = d.sid.map(folder_bin)
    check("folder bin equals the colorimeter CST class",
          (d.folder_bin == d["Colorimeter CST class (face)"]).all(),
          f"{(d.folder_bin != d['Colorimeter CST class (face)']).sum()} rows differ")
    lab_cols = ["avg L* (face)", "avg a* (face)", "avg b* (face)"]
    check("face Lab is constant within a subject",
          (d.groupby("sid")[lab_cols].nunique() == 1).all().all())

    print("\n6. composition")
    got = d.groupby("folder_bin").sid.nunique().to_dict()
    print(f"  {'bin':>4} {'subjects':>9} {'images':>7}   expected")
    for b in range(1, 11):
        n, e = int(got.get(b, 0)), EXPECTED_COMPOSITION.get(b, 0)
        flag = "" if n == e else "   <-- MISMATCH"
        print(f"  {b:>4} {n:>9} {len(d[d.folder_bin == b]):>7}   {e}{flag}")
    check("composition matches the release documentation",
          got == EXPECTED_COMPOSITION)

    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {failures}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
