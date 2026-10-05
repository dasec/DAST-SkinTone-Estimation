"""Predict CIELAB skin tone and the nearest CST bin for a folder of face images."""

import argparse
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.dataset import align_face, get_val_transform, make_aligner
from src.model import SkinToneRegressor
from src.utils import assign_bin_Lab

# TF32 lowers convolution precision on recent NVIDIA GPUs, for larger batches
# only. Off, the result does not depend on batch size or device.
torch.backends.cudnn.allow_tf32 = False

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SETTINGS = ("backbone_name", "img_size", "align")


def expand_models(patterns):
    """Expand wildcards in --model; a shell on Windows passes them unexpanded."""
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            sys.exit(f"--model: no file matches '{pattern}'")
        paths.extend(m for m in matches if m not in paths)
    return paths


def load_model(path, device):
    """Return (model, label_mean, label_std, settings) for one released checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = SkinToneRegressor(ckpt["backbone_name"], pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    settings = {k: ckpt[k] for k in SETTINGS}
    return model, ckpt["label_mean"].to(device), ckpt["label_std"].to(device), settings


def load_models(paths, device):
    """Load every checkpoint and check that they share the same preprocessing."""
    models, settings = [], None
    for path in paths:
        model, mean, std, s = load_model(path, device)
        if settings is None:
            settings = s
        elif s != settings:
            sys.exit(f"{path} expects {s}, but {paths[0]} expects {settings}; "
                     f"models with different preprocessing cannot be averaged")
        models.append((model, mean, std))
    return models, settings


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


@torch.no_grad()
def predict(models, tensor, device):
    """Mean prediction over the models, each de-normalised with its own statistics."""
    x = tensor.unsqueeze(0).to(device)
    outs = [(model(x) * std + mean).squeeze(0).cpu().numpy()
            for model, mean, std in models]
    return np.mean(outs, axis=0)


def run(models, settings, images, device):
    """Return one result row per image, in the order given."""
    transform = get_val_transform(settings["img_size"])
    aligner = None
    if settings["align"]:
        aligner = make_aligner(ctx_id=0 if device.type == "cuda" else -1)

    rows = []
    for path in images:
        row = {"image": path.name, "L*": np.nan, "a*": np.nan, "b*": np.nan,
               "cst_bin": pd.NA, "face_detected": pd.NA, "status": "ok"}
        try:
            image = Image.open(path).convert("RGB")
        except Exception as exc:
            row["status"] = f"unreadable ({type(exc).__name__})"
            rows.append(row)
            continue

        if aligner is not None:
            image, found = align_face(aligner, image)
            row["face_detected"] = found
            if not found:
                row["status"] = "no face detected, whole image used"

        lab = predict(models, transform(image), device)
        row["L*"], row["a*"], row["b*"] = (round(float(v), 4) for v in lab)
        row["cst_bin"] = assign_bin_Lab(*lab)
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", nargs="+", required=True,
                        help="one or more released .pth checkpoints, wildcards allowed; "
                             "several are averaged")
    parser.add_argument("--input", required=True, help="folder of face images")
    parser.add_argument("--output", required=True, help="CSV file to write")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu; default: cuda when available")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    images = list_images(args.input)
    if not images:
        sys.exit(f"no images found in {args.input}")

    models, settings = load_models(expand_models(args.model), device)
    print(f"{len(models)} model(s), {len(images)} image(s), device {device}, "
          f"alignment {'on' if settings['align'] else 'off'}")

    out = pd.DataFrame(run(models, settings, images, device))
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} rows to {args.output}")

    unreadable = out[out.status.str.startswith("unreadable")]
    no_face = out[out.face_detected.eq(False)]
    if len(unreadable):
        print(f"\n{len(unreadable)} image(s) could not be read and have no prediction:")
        for name in unreadable.image:
            print(f"  {name}")
    if len(no_face):
        print(f"\nno face detected in {len(no_face)} image(s); the prediction uses "
              f"the whole image and is less reliable:")
        for name in no_face.image:
            print(f"  {name}")


if __name__ == "__main__":
    main()
