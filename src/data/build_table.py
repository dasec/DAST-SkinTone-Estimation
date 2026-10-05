"""Build the training table from the dataset metadata csv."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.config import add_config_args, load_exposure_overrides, load_paths

# E1-E5 in the filenames are the levels EL1-EL5 of the paper.
EL_FROM_E = {"E1": "EL1", "E2": "EL2", "E3": "EL3", "E4": "EL4", "E5": "EL5"}
EL_NAME = {"EL1": "strongly underexposed", "EL2": "underexposed",
           "EL3": "normal", "EL4": "overexposed", "EL5": "strongly overexposed"}

# Coarser three-way grouping of the five levels.
EXPOSURE_3WAY = {"EL1": "under exposed", "EL2": "under exposed", "EL3": "normal",
                 "EL4": "over exposed", "EL5": "over exposed"}

LAB = {"avg L* (face)": "CIE-L*", "avg a* (face)": "CIE-a*", "avg b* (face)": "CIE-b*"}

# Quality-metric names differ between tool versions.
OFIQ_RENAME = {
    "NoCompressionArtifacts": "CompressionArtifacts",
    "LeftwardCropOfFace": "LeftwardCropOfTheFaceImage",
    "RightwardCropOfFace": "RightwardCropOfTheFaceImage",
    "MarginAboveTheFace": "MarginAboveOfTheFaceImage",
    "MarginBelowTheFace": "MarginBelowOfTheFaceImage",
    "HeadPoseYawFrontal": "HeadPoseYaw",
    "HeadPosePitchFrontal": "HeadPosePitch",
    "HeadPoseRollFrontal": "HeadPoseRoll",
}
OFIQ_FIRST = "UnifiedQualityScore"


def main():
    parser = add_config_args(argparse.ArgumentParser())
    parser.add_argument("--overrides", default=None,
                        help="path to an exposure_overrides.yaml")
    args = parser.parse_args()

    paths = load_paths(args.paths)
    exposure_override = load_exposure_overrides(args.overrides)

    if not paths.metadata_csv.is_file():
        sys.exit(f"missing {paths.metadata_csv}")

    d = pd.read_csv(paths.metadata_csv)
    d["subject_key"] = d.subject_id.astype(str).str.zfill(3)

    ofiq_cols = list(d.columns[list(d.columns).index(OFIQ_FIRST):])

    out = pd.DataFrame({
        "complete_path": d.image_relpath.map(
            lambda r: str(paths.dataset_root.parent / r)),
        "dataset": "dast",
        "mst_bin": d["Colorimeter CST class (face)"].astype(int),
        "subject_key": d.subject_key,
        "triple_key": d.subject_key + "_" + d.image_filename,
        "image_filename": d.image_filename,
        "session_id": d["Image identifier"],
        "camera": d.camera.str.capitalize(),
        "gender": d.gender,
        "stone_cst_class": d.stone_cst_class,
        "hand_cst_class": d["Colorimeter CST class (hands)"],
    })
    for src_col, dst in LAB.items():
        out[dst] = d[src_col]

    e_level = d.image_filename.str.extract(r"-(E\d)-")[0]
    exposure_code = d.exposure_code.copy()
    applied = []
    for fname, (e_new, code_new) in exposure_override.items():
        hit = d.image_filename == fname
        if not hit.any():
            sys.exit(f"ABORT: override target {fname} not found in the metadata")
        applied.append((fname, e_level[hit].iloc[0], e_new))
        e_level[hit] = e_new
        exposure_code[hit] = code_new

    out["EL"] = e_level.map(EL_FROM_E)
    out["exposure_label"] = out.EL.map(EL_NAME)
    out["exposure"] = out.EL.map(EXPOSURE_3WAY)
    out["exposure_code"] = exposure_code

    for c in ofiq_cols:
        out[f"{OFIQ_RENAME.get(c, c)}.scalar"] = d[c]

    missing = [p for p in out.complete_path if not Path(p).is_file()]
    if missing:
        sys.exit(f"ABORT: {len(missing)} image paths do not resolve")

    spanning = out.groupby("subject_key").mst_bin.nunique()
    if (spanning > 1).any():
        sys.exit(f"ABORT: {(spanning > 1).sum()} subject(s) span more than one bin")

    paths.table_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(paths.table_csv, index=False)

    print(f"wrote {paths.table_csv}")
    print(f"  exposure overrides applied: {len(applied)}")
    print(f"  images   {len(out)}")
    print(f"  subjects {out.subject_key.nunique()}")
    print(f"  cameras  {out.camera.value_counts().to_dict()}")
    print(f"  EL       {out.EL.value_counts().sort_index().to_dict()}")
    print(f"  OFIQ cols {len([c for c in out.columns if c.endswith('.scalar')])}")
    print()
    print(f"  {'bin':>4} {'subjects':>9} {'images':>7}")
    for b, g in out.groupby("mst_bin"):
        print(f"  {b:>4} {g.subject_key.nunique():>9} {len(g):>7}")


if __name__ == "__main__":
    main()
