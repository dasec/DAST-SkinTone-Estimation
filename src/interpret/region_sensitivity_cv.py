"""Occlusion sensitivity pooled over the cross-validation folds.

Each fold model is run on its own held-out subjects, so every subject is
measured exactly once by a model that never saw it.

    python -m src.interpret.region_sensitivity_cv
"""

import argparse
import sys

import pandas as pd
import torch

from src.config import add_config_args, load_paths, load_params
from src.cv_folds import make_fold
from src.dataset import get_val_transform
from src.interpret.occlusion import (
    SKIN_REGIONS, BG_REGIONS, region_sensitivity, make_aligner, load_aligned, IMG,
)
from src.model import SkinToneRegressor
from src.utils import assign_bin_Lb

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    paths = load_paths(args.paths)
    n_folds = load_params(args.params)["n_folds"]
    csv_path = str(paths.table_csv)
    ckpt_dir = paths.checkpoints / "cv"
    out_csv = paths.results / "cv_region_sensitivity.csv"

    app, fa = make_aligner()
    transform = get_val_transform(IMG)
    records = []

    for fold in range(n_folds):
        ckpt_path = ckpt_dir / f"cv{fold}_vit_small_patch16_224_aligned_best.pth"
        if not ckpt_path.is_file():
            sys.exit(f"ABORT: fold checkpoint missing: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = SkinToneRegressor(backbone_name=ckpt["backbone_name"],
                                  freeze_backbone=False).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        lm, ls = ckpt["label_mean"].to(DEVICE), ckpt["label_std"].to(DEVICE)

        _, _, test_df = make_fold(csv_path, fold)
        test_df = test_df.reset_index(drop=True)
        print(f"fold {fold}: {len(test_df)} images / "
              f"{test_df.subject_key.nunique()} subjects", flush=True)

        no_face = 0
        for i, row in test_df.iterrows():
            t, found = load_aligned(row["complete_path"], app, fa, transform)
            if not found:
                no_face += 1
            x = t.unsqueeze(0).to(DEVICE)
            deltas, _ = region_sensitivity(model, x, lm, ls)
            with torch.no_grad():
                pred = (model(x) * ls + lm)[0].cpu().numpy()
            records.append({
                "fold": fold,
                "subject_key": str(row["subject_key"]),
                "EL": row.get("EL", ""),
                "gt_bin": int(row["mst_bin"]),
                "pred_bin": assign_bin_Lb(float(pred[0]), float(pred[2])),
                "gt_L": float(row["CIE-L*"]),
                "pred_L": float(pred[0]),
                **{k: round(v, 4) for k, v in deltas.items()},
            })
            if (i + 1) % 60 == 0:
                print(f"  {i + 1}/{len(test_df)}", flush=True)
        if no_face:
            print(f"  {no_face} images without a detected face (used unaligned)")

    s = pd.DataFrame(records)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    s.to_csv(out_csv, index=False)

    s["background"] = s[BG_REGIONS].mean(axis=1)
    cheeks = s[["cheek_L", "cheek_R"]].mean(axis=1)
    skin = s[SKIN_REGIONS].mean(axis=1)

    print("\n" + "=" * 66)
    print(f"OCCLUSION SENSITIVITY, pooled over {n_folds} folds "
          f"(n={len(s)}, {s.subject_key.nunique()} subjects)")
    print("=" * 66)
    print(f"  {'region':<28} {'mean |dL*|':>11} {'std':>8}")
    for name, series in (("face oval (whole face)", s.face_oval),
                         ("outside oval", s.bg_oval),
                         ("eyes", s.eyes),
                         ("mouth / chin", s.mouth),
                         ("left cheek", s.cheek_L),
                         ("right cheek", s.cheek_R),
                         ("cheeks (average)", cheeks),
                         ("forehead", s.forehead),
                         ("background corners", s.background),
                         ("skin ROIs (cheeks+forehead)", skin)):
        print(f"  {name:<28} {series.mean():>11.3f} {series.std():>8.3f}")
    ratio = s.face_oval.mean() / max(s.bg_oval.mean(), 1e-9)
    print("  " + "-" * 50)
    print(f"  face oval / outside oval ratio : {ratio:.2f}x")

    print(f"\n  {'fold':>4} {'n':>5} {'face':>8} {'outside':>9} {'ratio':>7}")
    for f, g in s.groupby("fold"):
        print(f"  {int(f):>4} {len(g):>5} {g.face_oval.mean():>8.3f} "
              f"{g.bg_oval.mean():>9.3f} "
              f"{g.face_oval.mean()/max(g.bg_oval.mean(),1e-9):>7.2f}")

    print("\n  by exposure level (face oval)")
    for el, g in s.groupby("EL"):
        print(f"  {el:<5} n={len(g):>4}  face={g.face_oval.mean():>6.3f}  "
              f"outside={g.bg_oval.mean():>6.3f}")

    print(f"\nsaved -> {out_csv}")


if __name__ == "__main__":
    main()
