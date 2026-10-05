"""External ViT-S/16 baseline (Matias et al.), evaluated zero-shot."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from src.config import add_config_args, load_paths, load_params
from src.dataset import IMAGENET_MEAN, IMAGENET_STD, make_splits

parser = add_config_args(argparse.ArgumentParser())
parser.add_argument("--all", action="store_true",
                    help="score every image instead of the single test split")
parser.add_argument("--cv", action="store_true",
                    help="score the cross-validated image set; the prototype model "
                         "does no training, so this is the CV-equivalent evaluation")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=None)
args = parser.parse_args()

PATHS = load_paths(args.paths)
P = load_params(args.params, num_workers=args.num_workers)
CSV_PATH = str(PATHS.table_csv)
NUM_WORKERS = P["num_workers"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def collate(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    cols = list(zip(*batch))
    return tuple(torch.stack(c) if isinstance(c[0], torch.Tensor) else torch.tensor(c)
                 for c in cols)


class _ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(384, 2048), nn.ReLU(), nn.Linear(2048, 128))

    def forward(self, x):
        return self.head(x)


class MatiasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_small_patch16_224",
                                          pretrained=False, num_classes=0)
        self.projection_head = _ProjectionHead()

    def forward(self, x):
        return self.projection_head(self.backbone(x))


class ProtoDataset(Dataset):
    """The baseline authors' per-class training images."""

    def __init__(self, proto_dir):
        self.records = []
        for cls in range(1, 11):
            d = os.path.join(proto_dir, str(cls))
            if not os.path.isdir(d):
                continue
            for f in sorted(os.listdir(d)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.records.append((os.path.join(d, f), cls))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        path, label = self.records[idx]
        try:
            return TRANSFORM(Image.open(path).convert("RGB")), label, idx
        except Exception:
            return None


class ImageDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            return TRANSFORM(Image.open(row["complete_path"]).convert("RGB")), idx
        except Exception:
            return None


def build_prototypes(model, proto_dir):
    # The released checkpoint has no classification head: prediction is the
    # nearest class prototype, the L2-normalised mean embedding per class.
    print("Building class prototypes from the baseline authors' training images...")
    ds = ProtoDataset(proto_dir)
    if len(ds) == 0:
        sys.exit(f"No prototype images under {proto_dir}")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate,
                        pin_memory=(DEVICE.type == "cuda"))
    sums, counts = torch.zeros(10, 128), torch.zeros(10)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if batch is None:
                continue
            imgs, labels, _ = batch
            emb = model(imgs.to(DEVICE)).cpu()
            for e, lbl in zip(emb, labels):
                sums[int(lbl) - 1] += e
                counts[int(lbl) - 1] += 1
            if (i + 1) % 30 == 0:
                print(f"  {(i + 1) * args.batch} / {len(ds)}", flush=True)
    if (counts == 0).any():
        empty = [i + 1 for i, c in enumerate(counts) if c == 0]
        print(f"  [WARN] no images for class(es) {empty}, those prototypes are undefined")
    protos = F.normalize(sums / counts.clamp(min=1).unsqueeze(1), dim=1)
    print(f"Done. images per class: {counts.int().tolist()}\n")
    return protos.to(DEVICE), counts


def main():
    ckpt_path = PATHS.matias_checkpoint
    if not ckpt_path.is_file():
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    if args.cv:
        from src.cv_folds import fold_assignment
        _assignment, train_only, df = fold_assignment(CSV_PATH)
        df = df[~df.subject_key.isin(train_only)].reset_index(drop=True)
        scope = "CV-scored set (every testable subject once)"
    elif args.all:
        df = pd.read_csv(CSV_PATH, dtype={"subject_key": str})
        scope = "all images"
    else:
        _, _, df = make_splits(CSV_PATH, val_ratio=P["val_ratio"],
                               test_ratio=P["test_ratio"], seed=P["seed"],
                               verbose=False)
        scope = "single test split"
    df = df.reset_index(drop=True)
    print(f"{scope}: {len(df)} images / {df.subject_key.nunique()} subjects\n")

    model = MatiasModel().to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded external baseline checkpoint (epoch {ckpt['epoch']})")

    protos, _counts = build_prototypes(model, str(PATHS.matias_prototype_dir))

    print("Running inference...")
    loader = DataLoader(ImageDataset(df), batch_size=args.batch, shuffle=False,
                        num_workers=NUM_WORKERS, collate_fn=collate,
                        pin_memory=(DEVICE.type == "cuda"))
    rows = []
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            imgs, idxs = batch
            emb = F.normalize(model(imgs.to(DEVICE)), dim=1)
            pred = (emb @ protos.T).argmax(dim=1) + 1
            for pr, idx in zip(pred.cpu().numpy(), idxs.numpy()):
                row = df.iloc[int(idx)]
                rows.append({
                    "subject_key": str(row["subject_key"]),
                    "mst_gt": int(row["mst_bin"]),
                    "mst_pred": int(pr),
                    "exposure": row.get("exposure", ""),
                    "exposure_code": row.get("exposure_code", ""),
                    "camera": row.get("camera", ""),
                })

    res = pd.DataFrame(rows)
    tag = "cv" if args.cv else ("all" if args.all else "test")
    out = PATHS.results / f"matias_zeroshot_{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    print(f"Saved {len(res)} predictions -> {out}\n")

    gt, pred = res.mst_gt.values, res.mst_pred.values
    print("=" * 66)
    print(f"External ViT-S/16 baseline, zero-shot on the {scope}")
    print(f"OVERALL  exact={(gt == pred).mean() * 100:.1f}%  "
          f"within1={(np.abs(gt - pred) <= 1).mean() * 100:.1f}%  "
          f"bias={(pred - gt).mean():+.2f}  (n={len(gt)})")
    print("=" * 66)

    print(f"\n{'GT':>3} {'n':>4} {'exact':>7} {'±1':>7}  predicted-as")
    print("-" * 66)
    for b in range(1, 11):
        m = gt == b
        if not m.any():
            continue
        counts_b = pd.Series(pred[m]).value_counts().sort_index()
        cs = "  ".join(f"bin{k}({v})" for k, v in counts_b.items())
        print(f"{b:>3} {m.sum():>4} {(pred[m] == b).mean() * 100:>6.1f}% "
              f"{(np.abs(pred[m] - b) <= 1).mean() * 100:>6.1f}%  {cs}")
    print("-" * 66)

    EL = {"ue2": "EL1", "ue1": "EL2", "optimal": "EL3", "oe1": "EL4", "oe2": "EL5"}
    if "exposure_code" in res.columns and res.exposure_code.notna().any():
        res["EL"] = res.exposure_code.map(EL)
        print(f"\n{'EL':<5} {'n':>4} {'exact':>8} {'±1 bin':>8} {'bias':>7}")
        print("-" * 36)
        for el in ["EL1", "EL2", "EL3", "EL4", "EL5"]:
            s = res[res.EL == el]
            if s.empty:
                continue
            g, p = s.mst_gt.values, s.mst_pred.values
            print(f"{el:<5} {len(s):>4} {(g == p).mean() * 100:>7.1f}% "
                  f"{(np.abs(g - p) <= 1).mean() * 100:>7.1f}% {(p - g).mean():>+7.2f}")
        print("-" * 36)


if __name__ == "__main__":
    main()
