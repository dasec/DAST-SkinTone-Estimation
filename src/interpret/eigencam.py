"""EigenCAM and sliding-window occlusion maps for the ViT regressor."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.dataset import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from src.interpret.occlusion import IMG, TARGET_CH, make_aligner, load_aligned  # noqa: E402,F401

# EigenCAM, not raw attention: without register tokens, attention maps carry
# high-magnitude artefacts in low-information regions.
N_PATCH = IMG // 16
OCC_WINDOW, OCC_STRIDE, OCC_BATCH, OCC_FILL = 24, 8, 128, 0.0
MEAN = np.array(IMAGENET_MEAN)
STD = np.array(IMAGENET_STD)


def vit_reshape_transform(tensor, height=N_PATCH, width=N_PATCH):
    """Drop the CLS token and fold the patch tokens back into a HxW grid."""
    out = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    return out.permute(0, 3, 1, 2)


def tensor_to_uint8(t):
    """Undo normalisation and return an RGB uint8 image."""
    img = (t.cpu().permute(1, 2, 0).numpy() * STD + MEAN).clip(0, 1)
    return (img * 255).astype(np.uint8)


def occlusion_saliency(model, x, lm, ls):
    """Sliding-window occlusion map of |delta L*|; returns (map, baseline)."""
    model.eval()
    H, W = x.shape[-2:]
    with torch.no_grad():
        base = (model(x) * ls + lm)[0, TARGET_CH].item()
    positions = [(i * OCC_STRIDE, j * OCC_STRIDE)
                 for i in range((H - OCC_WINDOW) // OCC_STRIDE + 1)
                 for j in range((W - OCC_WINDOW) // OCC_STRIDE + 1)]
    deltas = []
    for k in range(0, len(positions), OCC_BATCH):
        bp = positions[k:k + OCC_BATCH]
        bx = x.repeat(len(bp), 1, 1, 1)
        for kk, (y0, x0) in enumerate(bp):
            bx[kk, :, y0:y0 + OCC_WINDOW, x0:x0 + OCC_WINDOW] = OCC_FILL
        with torch.no_grad():
            preds = (model(bx) * ls + lm)[:, TARGET_CH].cpu().numpy()
        deltas.extend(np.abs(preds - base))
    nh = (H - OCC_WINDOW) // OCC_STRIDE + 1
    nw = (W - OCC_WINDOW) // OCC_STRIDE + 1
    return np.array(deltas).reshape(nh, nw), base


def upsample(m, size=IMG):
    """Normalise a saliency map to [0,1] and resize it to the input resolution."""
    m = (m - m.min()) / (m.max() - m.min() + 1e-8)
    return np.array(Image.fromarray((m * 255).astype(np.uint8)).resize(
        (size, size), Image.BICUBIC), dtype=np.float32) / 255.0


def overlay(face, sal, alpha=0.55, cmap="inferno"):
    """Blend a colour-mapped saliency map over the face image."""
    heat = (plt.colormaps[cmap](sal)[:, :, :3] * 255).astype(np.uint8)
    return (face * (1 - alpha) + heat * alpha).clip(0, 255).astype(np.uint8)
