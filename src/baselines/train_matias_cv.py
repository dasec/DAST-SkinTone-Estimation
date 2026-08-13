"""Fine-tune the external baseline backbone on one cross-validation fold.

Same folds, head, hyperparameters and schedule as src/train_cv.py, so the only
difference from the main model is how the backbone was pretrained.

    python -m src.baselines.train_matias_cv --fold 0 --align
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import add_config_args, load_paths, load_params
from src.cv_folds import make_fold, N_FOLDS
from src.dataset import (
    SkinToneDataset, get_train_transform, get_val_transform, collate_skip_none,
    collate_skip_none_indexed,
)
from src.utils import (
    compute_metrics, print_metrics, assign_bin_L, assign_bin_Lb, assign_bin_Lab,
    bin_accuracy,
)

parser = add_config_args(argparse.ArgumentParser())
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--align", action="store_true", default=False)
parser.add_argument("--batch", type=int, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--backbone_lr", type=float, default=None)
parser.add_argument("--ckpt_dir", default=None)
parser.add_argument("--num_workers", type=int, default=None)
args = parser.parse_args()

PATHS = load_paths(args.paths)
P = load_params(args.params, batch_size=args.batch, epochs=args.epochs,
                lr_backbone=args.backbone_lr, num_workers=args.num_workers)

CSV_PATH = str(PATHS.table_csv)
ALIGN = args.align
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG_SIZE = P["img_size"]
TAG = f"cvmatias{args.fold}" + ("_aligned" if ALIGN else "")
CKPT_DIR = Path(args.ckpt_dir) if args.ckpt_dir else PATHS.checkpoints / "cv_matias"
CHECKPOINT_SAVE = str(CKPT_DIR / f"{TAG}_best.pth")

BATCH_SIZE = P["batch_size"]
EPOCHS = P["epochs"]
WARMUP_EPOCHS = P["warmup_epochs"]
LR_HEAD = P["lr_head"]
LR_HEAD_PH2 = P["lr_head_phase2"]
LR_BACKBONE = P["lr_backbone"]
WEIGHT_DECAY = P["weight_decay"]
LR_PATIENCE = P["lr_patience"]
LR_FACTOR = P["lr_factor"]
EARLY_STOP_PAT = P["early_stop_patience"]
CHANNEL_WEIGHTS = P["channel_weights"]
NUM_WORKERS = P["num_workers"]


class MatiasFinetune(nn.Module):
    """External baseline backbone with our regression head; projection head dropped."""

    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("vit_small_patch16_224",
                                          pretrained=False, num_classes=0)
        self.regression_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("Backbone unfrozen.")

    def forward(self, x):
        return self.regression_head(self.backbone(x))


def load_matias_backbone(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state = {k[len("backbone."):]: v for k, v in ckpt["model_state_dict"].items()
             if k.startswith("backbone.")}
    if not state:
        raise RuntimeError("no 'backbone.' keys in the baseline checkpoint")
    missing, _ = model.backbone.load_state_dict(state, strict=True)
    assert not missing, f"missing backbone keys: {missing}"
    print(f"loaded baseline backbone (epoch {ckpt['epoch']}, projection head dropped)")


def make_loader(df, train=False, indexed=False):
    tf = get_train_transform(IMG_SIZE) if train else get_val_transform(IMG_SIZE)
    ds = SkinToneDataset(df, transform=tf, align=ALIGN, index=indexed)
    return DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=train, num_workers=NUM_WORKERS,
        collate_fn=collate_skip_none_indexed if indexed else collate_skip_none,
        pin_memory=(DEVICE.type == "cuda"),
    )


def label_stats(df):
    v = df[["CIE-L*", "CIE-a*", "CIE-b*"]].values.astype(np.float32)
    return (torch.tensor(v.mean(0), dtype=torch.float32, device=DEVICE),
            torch.tensor(v.std(0), dtype=torch.float32, device=DEVICE))


def run_epoch(model, loader, criterion, optimizer=None, lm=None, ls=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total, n, preds, tgts = 0.0, 0, [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for images, labels in loader:
            if images is None:
                continue
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            out = model(images)
            loss = criterion(out, (labels - lm) / ls)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item()
            n += 1
            preds.append((out.detach() * ls + lm).cpu())
            tgts.append(labels.cpu())
    return total / max(n, 1), torch.cat(preds).numpy(), torch.cat(tgts).numpy()


def predict_indexed(model, df, lm, ls):
    """Inference tracking row indices, so a dropped image cannot misalign results."""
    loader = make_loader(df, train=False, indexed=True)
    got = {}
    with torch.no_grad():
        for images, _labels, idxs in loader:
            if images is None:
                continue
            out = (model(images.to(DEVICE)) * ls + lm).cpu().numpy()
            for i, ri in enumerate(idxs.numpy()):
                got[int(ri)] = out[i]
    if len(got) != len(df):
        raise RuntimeError(f"{len(df) - len(got)} image(s) failed to load")
    return np.array([got[i] for i in range(len(df))])


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (PATHS.results / "cv_matias").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Device: {DEVICE}")
    print(f"fold {args.fold} of {N_FOLDS}   align={ALIGN}\n")

    train_df, val_df, test_df = make_fold(CSV_PATH, args.fold)
    print(f"train {len(train_df)}/{train_df.subject_key.nunique()}s   "
          f"val {len(val_df)}/{val_df.subject_key.nunique()}s   "
          f"test {len(test_df)}/{test_df.subject_key.nunique()}s")

    lm, ls = label_stats(train_df)
    model = MatiasFinetune().to(DEVICE)
    load_matias_backbone(model, PATHS.matias_checkpoint)
    model.freeze_backbone()

    train_loader = make_loader(train_df, True)
    val_loader = make_loader(val_df, False)

    w = torch.tensor(CHANNEL_WEIGHTS, dtype=torch.float32, device=DEVICE)
    criterion = lambda p, t: (
        nn.functional.huber_loss(p, t, reduction="none", delta=1.0) * w).mean()

    optimizer = optim.AdamW(model.regression_head.parameters(),
                            lr=LR_HEAD, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE)

    best, no_improve, phase = float("inf"), 0, 1
    for epoch in range(1, EPOCHS + 1):
        if epoch == WARMUP_EPOCHS + 1 and phase == 1:
            phase = 2
            model.unfreeze_backbone()
            optimizer = optim.AdamW([
                {"params": model.backbone.parameters(), "lr": LR_BACKBONE},
                {"params": model.regression_head.parameters(), "lr": LR_HEAD_PH2},
            ], weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE)
            no_improve = 0

        tl, tp, tt = run_epoch(model, train_loader, criterion, optimizer, lm, ls)
        _, vp, vt = run_epoch(model, val_loader, criterion, lm=lm, ls=ls)
        vm = compute_metrics(vp, vt)
        scheduler.step(vm["avg_MAE"])
        print(f"Ep {epoch:3d}/{EPOCHS}  Ph{phase}  train_loss={tl:.4f}  "
              f"val_MAE={vm['avg_MAE']:.3f}  val_R2={vm['avg_R2']:.3f}")

        if vm["avg_MAE"] < best:
            best, no_improve = vm["avg_MAE"], 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_metrics": vm, "label_mean": lm.cpu(),
                        "label_std": ls.cpu(), "fold": args.fold}, CHECKPOINT_SAVE)
            print(f"  -> new best val MAE={best:.3f}")
        else:
            no_improve += 1
        if no_improve >= EARLY_STOP_PAT:
            print(f"\nEarly stopping after {epoch} epochs.")
            break

    ckpt = torch.load(CHECKPOINT_SAVE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    lm, ls = ckpt["label_mean"].to(DEVICE), ckpt["label_std"].to(DEVICE)

    preds = predict_indexed(model, test_df, lm, ls)
    gts = test_df[["CIE-L*", "CIE-a*", "CIE-b*"]].values.astype(np.float32)
    de = np.sqrt(((preds - gts) ** 2).sum(1))

    print(f"\nFOLD {args.fold} TEST (best epoch {ckpt['epoch']}, n={len(test_df)})")
    print_metrics(compute_metrics(preds, gts))
    print(f"  dE76 mean={de.mean():.3f}")

    out = pd.DataFrame({
        "fold": args.fold,
        "subject_key": test_df.subject_key.values,
        "EL": test_df.EL.values,
        "mst_gt": test_df.mst_bin.values,
        "L_gt": gts[:, 0], "a_gt": gts[:, 1], "b_gt": gts[:, 2],
        "L_pred": preds[:, 0].round(4), "a_pred": preds[:, 1].round(4),
        "b_pred": preds[:, 2].round(4), "dE76": de.round(4),
    })
    out["bin_L"] = [assign_bin_L(l) for l in preds[:, 0]]
    out["bin_Lb"] = [assign_bin_Lb(l, b) for l, b in preds[:, [0, 2]]]
    out["bin_Lab"] = [assign_bin_Lab(*p) for p in preds]
    csv_out = PATHS.results / "cv_matias" / f"fold{args.fold}_test_predictions.csv"
    out.to_csv(csv_out, index=False)

    for col, lab in (("bin_L", "L* only"), ("bin_Lb", "L* + b*"),
                     ("bin_Lab", "L*a*b*")):
        ex, w1 = bin_accuracy(out.mst_gt, out[col])
        print(f"  {lab:<12} exact={ex*100:5.1f}%  within1={w1*100:5.1f}%")
    print(f"\nsaved -> {csv_out}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
