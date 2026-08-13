"""EigenCAM and occlusion maps from a cross-validation fold model.

Subjects are picked from the chosen fold's held-out test set, so the model
producing the maps never saw them, and they span the tone range present in that
fold.

    python -m src.interpret.eigencam_cv --fold 0 --subjects 3
"""

import argparse
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.config import add_config_args, load_paths
from src.cv_folds import make_fold
from src.dataset import get_val_transform
from src.interpret.eigencam import (
    IMG, occlusion_saliency, upsample, overlay, tensor_to_uint8, load_aligned,
    make_aligner, vit_reshape_transform,
)
from src.model import SkinToneRegressor
from src.utils import assign_bin_Lb

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Normal exposure plus one off-nominal level.
EXPOSURES = ["EL3", "EL5"]
EL_LABEL = {"EL1": "Strongly underexposed", "EL2": "Underexposed",
            "EL3": "Normal", "EL4": "Overexposed",
            "EL5": "Strongly overexposed"}


def pick_subjects(test_df, n):
    """Spread across the tone range present in this fold: lightest to darkest."""
    order = (test_df.groupby("subject_key")["CIE-L*"].first()
             .sort_values(ascending=False))
    if len(order) <= n:
        return list(order.index)
    idx = np.linspace(0, len(order) - 1, n).round().astype(int)
    return [order.index[i] for i in idx]


def main():
    parser = add_config_args(argparse.ArgumentParser())
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--subjects", type=int, default=3)
    args = parser.parse_args()

    paths = load_paths(args.paths)
    t0 = time.time()
    try:
        from pytorch_grad_cam import EigenCAM
    except ImportError:
        sys.exit("pytorch-grad-cam not installed: pip install grad-cam")

    ckpt_path = (paths.checkpoints / "cv" /
                 f"cv{args.fold}_vit_small_patch16_224_aligned_best.pth")
    if not ckpt_path.is_file():
        sys.exit(f"ABORT: fold checkpoint missing: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = SkinToneRegressor(backbone_name=ckpt["backbone_name"],
                              freeze_backbone=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    lm, ls = ckpt["label_mean"].to(DEVICE), ckpt["label_std"].to(DEVICE)

    run_dir = (paths.results /
               f"eigencam_cv{args.fold}_{time.strftime('%Y%m%d_%H%M%S')}")
    (run_dir / "individual").mkdir(parents=True, exist_ok=True)

    cam = EigenCAM(model=model,
                   target_layers=[model.backbone.blocks[-1].norm1],
                   reshape_transform=vit_reshape_transform)

    _, _, test_df = make_fold(str(paths.table_csv), args.fold)
    test_df = test_df.reset_index(drop=True)
    test_df["subject_key"] = test_df.subject_key.astype(str).str.zfill(3)
    subjects = pick_subjects(test_df, args.subjects)

    print(f"fold {args.fold}, checkpoint epoch {ckpt['epoch']}")
    print(f"{len(subjects)} subjects drawn from this fold's held-out test set")
    app, fa = make_aligner()
    transform = get_val_transform(IMG)

    rows = []
    for sid in subjects:
        for el in EXPOSURES:
            sub = test_df[(test_df.subject_key == sid) & (test_df.EL == el)]
            if sub.empty:
                print(f"  {el}: not present for one subject, skipped")
                continue
            row = sub.iloc[0]
            t, ok = load_aligned(row["complete_path"], app, fa, transform)
            x = t.unsqueeze(0).to(DEVICE)

            grayscale = cam(input_tensor=x, targets=None)[0]
            occ_map, _ = occlusion_saliency(model, x, lm, ls)
            with torch.no_grad():
                pred = (model(x) * ls + lm)[0].cpu().numpy()

            face = tensor_to_uint8(t)
            rows.append({
                "subject": sid, "EL": el, "face": face,
                "eigen": overlay(face, upsample(grayscale)),
                "occ": overlay(face, upsample(occ_map)),
                "gt_bin": int(row["mst_bin"]),
                "pred_bin": assign_bin_Lb(float(pred[0]), float(pred[2])),
                "gt_L": float(row["CIE-L*"]), "pred_L": float(pred[0]),
            })
            stem = f"{sid}_{el}"
            Image.fromarray(face).save(run_dir / "individual" / f"{stem}_orig.png")
            Image.fromarray(rows[-1]["eigen"]).save(
                run_dir / "individual" / f"{stem}_eigencam.png")
            Image.fromarray(rows[-1]["occ"]).save(
                run_dir / "individual" / f"{stem}_occlusion.png")
            print(f"  {el:<4} gt_L={row['CIE-L*']:.1f} pred_L={pred[0]:.1f} "
                  f"gt_bin={int(row['mst_bin'])} pred_bin={rows[-1]['pred_bin']}"
                  + ("" if ok else "   [no face detected]"))

    if not rows:
        sys.exit("nothing to plot")

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]
    for i, d in enumerate(rows):
        for j, (key, title) in enumerate((("face", "Aligned input"),
                                          ("eigen", "EigenCAM"),
                                          ("occ", "Occlusion |dL*|"))):
            axes[i, j].imshow(d[key])
            axes[i, j].axis("off")
            if i == 0:
                axes[i, j].set_title(title, fontsize=11)
        axes[i, 0].axis("on")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_ylabel(f"bin {d['gt_bin']}\n{EL_LABEL[d['EL']]}", fontsize=8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(run_dir / f"eigencam_grid.{ext}", dpi=300, bbox_inches="tight")

    prov = [
        f"fold          : {args.fold}",
        f"checkpoint    : {ckpt_path.name} (epoch {ckpt['epoch']})",
        f"subjects      : {len(subjects)} from this fold's held-out test set",
        f"exposures     : {' '.join(EXPOSURES)}",
        "selection     : spread over colorimeter L* within the fold",
    ]
    (run_dir / "PROVENANCE.txt").write_text("\n".join(prov) + "\n")

    print(f"\nsaved grid -> {run_dir}/eigencam_grid.pdf (+ .png)")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
