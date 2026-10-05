"""Occlusion sensitivity: replace a region with neutral grey, measure the shift in L*."""

import numpy as np
import torch
from PIL import Image

IMG = 224
OCC_FILL = 0.0      # ImageNet mean in normalised space
TARGET_CH = 0       # L*, the channel with the highest R2

# ROI geometry on the aligned 224x224 template.
EYE_L_XY = (76, 103)
EYE_R_XY = (147, 103)
MOUTH_ROW = 185
IED = EYE_R_XY[0] - EYE_L_XY[0]
EMD = MOUTH_ROW - EYE_L_XY[1]
# Cheek patches follow the ISO landmark convention.
CHEEK_S = int(0.3 * IED)
CHEEK_DY = int(0.5 * EMD)

FACE_CX, FACE_CY, FACE_RX, FACE_RY = 112, 118, 88, 102

REGIONS = {
    "cheek_L": (EYE_L_XY[0] - CHEEK_S, EYE_L_XY[1] + CHEEK_DY,
                EYE_L_XY[0], EYE_L_XY[1] + CHEEK_DY + CHEEK_S),
    "cheek_R": (EYE_R_XY[0], EYE_R_XY[1] + CHEEK_DY,
                EYE_R_XY[0] + CHEEK_S, EYE_R_XY[1] + CHEEK_DY + CHEEK_S),
    "forehead": (70, 35, 154, 85),
    "eyes": (55, 88, 169, 115),
    "mouth": (80, 150, 144, 180),
    "bg_TL": (0, 0, 30, 30),
    "bg_TR": (194, 0, 224, 30),
    "bg_BL": (0, 194, 30, 224),
    "bg_BR": (194, 194, 224, 224),
}
SKIN_REGIONS = ["cheek_L", "cheek_R", "forehead"]
BG_REGIONS = ["bg_TL", "bg_TR", "bg_BL", "bg_BR"]


def make_face_oval_mask(size=IMG):
    """Boolean mask of the face oval on the aligned template."""
    y, x = np.mgrid[0:size, 0:size]
    return ((x - FACE_CX) ** 2 / FACE_RX ** 2
            + (y - FACE_CY) ** 2 / FACE_RY ** 2) <= 1.0


_FACE_MASK = make_face_oval_mask()


def make_aligner():
    """Return (FaceAnalysis app, face_align module)."""
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    app = FaceAnalysis(name="buffalo_sc",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app, face_align


def load_aligned(path, app, fa, transform):
    """Load and align one image; returns (tensor, face_detected)."""
    img_np = np.array(Image.open(path).convert("RGB"))
    found = False
    if app is not None:
        faces = app.get(img_np)
        if faces:
            img_np = fa.norm_crop(img_np, landmark=faces[0].kps, image_size=IMG)
            found = True
    return transform(Image.fromarray(img_np)), found


def region_sensitivity(model, x, lm, ls):
    """Per-region |delta L*| for the ROIs plus face oval against everything outside it."""
    model.eval()
    with torch.no_grad():
        base = (model(x) * ls + lm)[0, TARGET_CH].item()

    out = {}
    for name, (x0, y0, x1, y1) in REGIONS.items():
        occ = x.clone()
        occ[..., y0:y1, x0:x1] = OCC_FILL
        with torch.no_grad():
            out[name] = abs((model(occ) * ls + lm)[0, TARGET_CH].item() - base)

    mask = torch.tensor(_FACE_MASK, device=x.device).unsqueeze(0).expand(3, -1, -1)
    occ_face = x.clone()
    occ_face[0][mask] = OCC_FILL
    occ_bg = x.clone()
    occ_bg[0][~mask] = OCC_FILL
    with torch.no_grad():
        out["face_oval"] = abs((model(occ_face) * ls + lm)[0, TARGET_CH].item() - base)
        out["bg_oval"] = abs((model(occ_bg) * ls + lm)[0, TARGET_CH].item() - base)
    return out, base
