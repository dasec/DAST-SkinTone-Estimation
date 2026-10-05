"""Combined training on both datasets with the mean-shift correction, one fold."""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import add_config_args, load_paths, load_params
from src.cv_folds import make_fold
from src.dataset import (
    SkinToneDataset, get_train_transform, get_val_transform, collate_skip_none,
)
from src.model import SkinToneRegressor
from src.utils import compute_metrics, print_metrics

LABELS = ["CIE-L*", "CIE-a*", "CIE-b*"]

parser = add_config_args(argparse.ArgumentParser())
parser.add_argument("--fold", type=int, required=True)
parser.add_argument("--backbone", default=None)
parser.add_argument("--batch", type=int, default=None)
parser.add_argument("--epochs", type=int, default=None)
parser.add_argument("--ckpt_dir", default=None)
parser.add_argument("--num_workers", type=int, default=None)
parser.add_argument("--no-shift", dest="shift", action="store_false", default=True)
args = parser.parse_args()

PATHS = load_paths(args.paths)
P = load_params(args.params, backbone=args.backbone, batch_size=args.batch,
                epochs=args.epochs, num_workers=args.num_workers)

DAST_CSV = str(PATHS.table_csv)
MERGED_CSV = str(PATHS.merged_csv)
N_FOLDS = P["n_folds"]
SEED = P["seed"]
UQS_MIN = P["chroma_uqs_min"]
MIN_L = P["chroma_min_L"]
ALIGN = P["align"]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TAG = f"cvcross{args.fold}" + ("" if args.shift else "_noshift")
CKPT_DIR = Path(args.ckpt_dir) if args.ckpt_dir else PATHS.checkpoints / "cv_cross"
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


def chroma_folds():
    """Split the filtered CHROMA-FIT subjects into the same number of folds."""
    df = pd.read_csv(MERGED_CSV, dtype={"subject_key": str})
    c = df[df.dataset == "chroma"].reset_index(drop=True)
    c = c[c["CIE-L*"] >= MIN_L].reset_index(drop=True)
    bad = ((c["UnifiedQualityScore.scalar"] < UQS_MIN) |
           (c["BackgroundUniformity.scalar"] == -1))
    c = c[~bad].reset_index(drop=True)

    subjects = c.groupby("subject_key").mst_bin.first().reset_index()
    rng = np.random.default_rng(SEED)
    assignment = {}
    for _b, g in subjects.groupby("mst_bin"):
        ids = g.subject_key.tolist()
        rng.shuffle(ids)
        for k, sid in enumerate(ids):
            assignment[sid] = k % N_FOLDS
    c["_fold"] = c.subject_key.map(assignment)
    return c


def build_split(fold):
    d_tr, d_va, d_te = make_fold(DAST_CSV, fold)
    c = chroma_folds()
    test_f, val_f = fold, (fold + 1) % N_FOLDS
    c_te = c[c._fold == test_f].drop(columns="_fold")
    c_va = c[c._fold == val_f].drop(columns="_fold")
    c_tr = c[~c._fold.isin([test_f, val_f])].drop(columns="_fold")

    tr = pd.concat([d_tr, c_tr], ignore_index=True, sort=False)
    va = pd.concat([d_va, c_va], ignore_index=True, sort=False)
    te = pd.concat([d_te, c_te], ignore_index=True, sort=False)
    return tr, va, te


def compute_shift(train_df):
    """Per-channel device offset, estimated on the training split only."""
    # Estimated over the whole pool, it would leak test labels into training.
    d = train_df[train_df.dataset == "dast"][LABELS].mean()
    c = train_df[train_df.dataset == "chroma"][LABELS].mean()
    shift = d - c
    print(f"mean-shift estimated on TRAIN only: L*={shift['CIE-L*']:+.2f}  "
          f"a*={shift['CIE-a*']:+.2f}  b*={shift['CIE-b*']:+.2f}")
    return shift


def apply_shift(df, shift):
    df = df.copy()
    m = df.dataset == "chroma"
    for c in LABELS:
        df.loc[m, c] += shift[c]
    return df


def label_stats(df):
    v = df[LABELS].values.astype(np.float32)
    return (torch.tensor(v.mean(0), dtype=torch.float32, device=DEVICE),
            torch.tensor(v.std(0), dtype=torch.float32, device=DEVICE))


def make_loader(df, train=False, img_size=224):
    tf = get_train_transform(img_size) if train else get_val_transform(img_size)
    return DataLoader(SkinToneDataset(df, transform=tf, align=ALIGN),
                      batch_size=BATCH_SIZE, shuffle=train,
                      num_workers=NUM_WORKERS, collate_fn=collate_skip_none,
                      pin_memory=(DEVICE.type == "cuda"))


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


def main():
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    (PATHS.results / "cv_cross").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"Device: {DEVICE}   fold {args.fold} of {N_FOLDS}   shift={args.shift}\n")
    tr, va, te = build_split(args.fold)
    for name, d in (("train", tr), ("val", va), ("test", te)):
        print(f"  {name:5s}  primary {(d.dataset=='dast').sum():4d}  "
              f"chroma {(d.dataset=='chroma').sum():4d}")

    shift = None
    if args.shift:
        shift = compute_shift(tr)
        tr, va, te = (apply_shift(x, shift) for x in (tr, va, te))
    else:
        print("mean-shift disabled (control run)")

    lm, ls = label_stats(tr)
    model = SkinToneRegressor(backbone_name=P["backbone"],
                              freeze_backbone=True).to(DEVICE)
    from timm.data import resolve_data_config
    img_size = resolve_data_config(model.backbone.pretrained_cfg)["input_size"][1]

    train_loader = make_loader(tr, True, img_size)
    val_loader = make_loader(va, False, img_size)

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

        tl, _, _ = run_epoch(model, train_loader, criterion, optimizer, lm, ls)
        _, vp, vt = run_epoch(model, val_loader, criterion, lm=lm, ls=ls)
        vm = compute_metrics(vp, vt)
        scheduler.step(vm["avg_MAE"])
        print(f"Ep {epoch:3d}/{EPOCHS}  Ph{phase}  train_loss={tl:.4f}  "
              f"val_MAE={vm['avg_MAE']:.3f}")

        if vm["avg_MAE"] < best:
            best, no_improve = vm["avg_MAE"], 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_metrics": vm, "label_mean": lm.cpu(),
                        "label_std": ls.cpu(), "fold": args.fold,
                        "backbone_name": P["backbone"], "align": ALIGN,
                        "mean_shift": None if shift is None else dict(shift)},
                       CHECKPOINT_SAVE)
            print(f"  -> new best val MAE={best:.3f}")
        else:
            no_improve += 1
        if no_improve >= EARLY_STOP_PAT:
            print(f"\nEarly stopping after {epoch} epochs.")
            break

    ckpt = torch.load(CHECKPOINT_SAVE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    lm, ls = ckpt["label_mean"].to(DEVICE), ckpt["label_std"].to(DEVICE)

    out_rows = []
    print(f"\nFOLD {args.fold} TEST (best epoch {ckpt['epoch']})")
    for label, sub in (("combined", te), ("dast", te[te.dataset == "dast"]),
                       ("chroma", te[te.dataset == "chroma"])):
        if sub.empty:
            continue
        _, p, t = run_epoch(model, make_loader(sub, False, img_size),
                            criterion, lm=lm, ls=ls)
        m = compute_metrics(p, t)
        de = np.sqrt(((p - t) ** 2).sum(1))
        print(f"  [{label}]  n={len(sub)}")
        print_metrics(m, indent="    ")
        print(f"    dE76 mean={de.mean():.3f}")
        out_rows.append({
            "fold": args.fold, "subset": label, "n": len(sub),
            "L_R2": m["L*_R2"], "a_R2": m["a*_R2"], "b_R2": m["b*_R2"],
            "L_MAE": m["L*_MAE"], "a_MAE": m["a*_MAE"], "b_MAE": m["b*_MAE"],
            "dE76": de.mean(),
            "shift_L": None if shift is None else shift["CIE-L*"],
            "shift_a": None if shift is None else shift["CIE-a*"],
            "shift_b": None if shift is None else shift["CIE-b*"],
        })

    suffix = "" if args.shift else "_noshift"
    pd.DataFrame(out_rows).to_csv(
        PATHS.results / "cv_cross" / f"fold{args.fold}{suffix}.csv", index=False)
    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
