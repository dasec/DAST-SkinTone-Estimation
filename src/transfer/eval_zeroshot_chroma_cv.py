"""Zero-shot CHROMA-FIT transfer for each cross-validation fold model.

No training happens here. The quality filter is applied once before the loop, so
every fold sees an identical image set and the spread measures how stable the
transfer is with respect to the training partition.

    python -m src.transfer.eval_zeroshot_chroma_cv
"""

import argparse
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms as T

from src.config import add_config_args, load_paths, load_params
from src.dataset import IMAGENET_MEAN, IMAGENET_STD
from src.model import SkinToneRegressor
from src.utils import compute_metrics

LABELS = ["CIE-L*", "CIE-a*", "CIE-b*"]
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def make_aligner():
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    app = FaceAnalysis(name="buffalo_sc",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app, face_align


def filtered_chroma(csv_path, uqs_min, min_L):
    df = pd.read_csv(csv_path)
    chroma = df[df.dataset == "chroma"].reset_index(drop=True)
    n_all = len(chroma)

    chroma = chroma[chroma["CIE-L*"] >= min_L].reset_index(drop=True)
    n_after_L = len(chroma)

    bad = ((chroma["UnifiedQualityScore.scalar"] < uqs_min) |
           (chroma["BackgroundUniformity.scalar"] == -1))
    chroma = chroma[~bad].reset_index(drop=True)

    print("CHROMA-FIT filtering (applied once, shared by all folds)")
    print(f"  all images                      : {n_all}")
    print(f"  after dropping L*<{min_L}            : {n_after_L}  "
          f"({n_all - n_after_L} dropped)")
    print(f"  after quality filter            : {len(chroma)}  "
          f"({n_after_L - len(chroma)} dropped)")
    return chroma


def run_fold(ckpt_dir, fold, chroma, app, fa):
    ckpt_path = ckpt_dir / f"cv{fold}_vit_small_patch16_224_aligned_best.pth"
    if not ckpt_path.is_file():
        sys.exit(f"ABORT: fold checkpoint missing: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    model = SkinToneRegressor(backbone_name=ckpt["backbone_name"],
                              freeze_backbone=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    lm, ls = ckpt["label_mean"].to(DEVICE), ckpt["label_std"].to(DEVICE)

    preds, no_face = [], 0
    for _, row in chroma.iterrows():
        img = np.array(Image.open(row["complete_path"]).convert("RGB"))
        faces = app.get(img)
        if faces:
            img = fa.norm_crop(img, landmark=faces[0].kps, image_size=224)
        else:
            no_face += 1
        x = TRANSFORM(Image.fromarray(img)).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            preds.append((model(x) * ls + lm)[0].cpu().numpy())

    preds = np.array(preds)
    gts = chroma[LABELS].values.astype(np.float32)
    de = np.sqrt(((preds - gts) ** 2).sum(1))
    print(f"  fold {fold}: epoch {ckpt['epoch']}, {no_face} without a detected face")
    return preds, gts, de


def main():
    args = add_config_args(argparse.ArgumentParser()).parse_args()
    paths = load_paths(args.paths)
    P = load_params(args.params)
    n_folds = P["n_folds"]
    ckpt_dir = paths.checkpoints / "cv"
    out_csv = paths.results / "cv_zeroshot_chroma.csv"

    chroma = filtered_chroma(paths.merged_csv, P["chroma_uqs_min"], P["chroma_min_L"])
    app, fa = make_aligner()
    gts = chroma[LABELS].values.astype(np.float32)

    rows, all_preds = [], []
    print("\nrunning the fold models")
    for fold in range(n_folds):
        preds, _, de = run_fold(ckpt_dir, fold, chroma, app, fa)
        m = compute_metrics(preds, gts)
        rows.append({
            "fold": fold, "n": len(preds),
            "L_R2": m["L*_R2"], "a_R2": m["a*_R2"], "b_R2": m["b*_R2"],
            "L_MAE": m["L*_MAE"], "a_MAE": m["a*_MAE"], "b_MAE": m["b*_MAE"],
            "dE76": de.mean(),
            "delta_L": preds[:, 0].mean() - gts[:, 0].mean(),
            "delta_a": preds[:, 1].mean() - gts[:, 1].mean(),
            "delta_b": preds[:, 2].mean() - gts[:, 2].mean(),
        })
        all_preds.append(preds)

    per_fold = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    per_fold.to_csv(out_csv, index=False)

    pooled = np.mean(all_preds, axis=0)
    mp = compute_metrics(pooled, gts)
    dep = np.sqrt(((pooled - gts) ** 2).sum(1))

    print("\n" + "=" * 70)
    print(f"ZERO-SHOT CHROMA-FIT, per fold  (n={len(chroma)} images, same set each time)")
    print("=" * 70)
    print(f"  {'fold':>4} {'L* R2':>8} {'a* R2':>9} {'b* R2':>8} {'dE76':>7} "
          f"{'delta a*':>9}")
    for r in rows:
        print(f"  {r['fold']:>4} {r['L_R2']:>8.3f} {r['a_R2']:>9.3f} "
              f"{r['b_R2']:>8.3f} {r['dE76']:>7.2f} {r['delta_a']:>+9.2f}")
    print("  " + "-" * 52)
    for stat, fn in (("mean", np.mean), ("std", np.std),
                     ("min", np.min), ("max", np.max)):
        print(f"  {stat:>4} {fn(per_fold.L_R2):>8.3f} {fn(per_fold.a_R2):>9.3f} "
              f"{fn(per_fold.b_R2):>8.3f} {fn(per_fold.dE76):>7.2f} "
              f"{fn(per_fold.delta_a):>+9.2f}")
    print()
    print(f"  averaging the fold models' predictions: L* R2={mp['L*_R2']:.3f}  "
          f"a* R2={mp['a*_R2']:.3f}  b* R2={mp['b*_R2']:.3f}  dE76={dep.mean():.2f}")
    print(f"\nsaved -> {out_csv}")


if __name__ == "__main__":
    main()
