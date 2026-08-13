"""Dataset, transforms and the subject-wise split."""

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

try:
    from insightface.app import FaceAnalysis
    from insightface.utils import face_align
    _INSIGHTFACE_AVAILABLE = True
except ImportError:
    _INSIGHTFACE_AVAILABLE = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

LABEL_COLS = ["CIE-L*", "CIE-a*", "CIE-b*"]

# Rows below this lightness in the second dataset are measurement failures and
# are always dropped; see the cross-dataset section of the paper.
CHROMA_MIN_L = 20


def get_train_transform(img_size=224):
    """Geometry and optical augmentation only."""
    # No colour jitter: the labels are the image colours, so perturbing colour
    # would break the input-label relationship.
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=5),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(img_size=224):
    """Resize and normalise, no augmentation."""
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def make_splits(csv_path, val_ratio=0.15, test_ratio=0.15, seed=42, verbose=True,
                dataset_filter=None, uqs_min=None):
    """Subject-level stratified split; returns (train_df, val_df, test_df).

    dataset_filter keeps only rows of one dataset; uqs_min drops second-dataset
    rows below that quality score or with an invalid background-uniformity
    value. Both default to off.
    """
    df = pd.read_csv(csv_path, dtype={"subject_key": str})

    if dataset_filter is not None:
        df = df[df["dataset"] == dataset_filter].reset_index(drop=True)

    if "dataset" in df.columns and "CIE-L*" in df.columns:
        bad = (df["dataset"] == "chroma") & (df["CIE-L*"] < CHROMA_MIN_L)
        if bad.sum():
            if verbose:
                print(f"Dropping {bad.sum()} rows with L*<{CHROMA_MIN_L} "
                      "(measurement errors)")
            df = df[~bad].reset_index(drop=True)

    if uqs_min is not None and "UnifiedQualityScore.scalar" in df.columns:
        before = len(df)
        is_chroma = df["dataset"] == "chroma"
        low_uqs = df["UnifiedQualityScore.scalar"] < uqs_min
        bad_bg = df["BackgroundUniformity.scalar"] == -1
        df = df[~(is_chroma & (low_uqs | bad_bg))].reset_index(drop=True)
        if verbose:
            print(f"Quality filter (UQS min={uqs_min}): dropped {before - len(df)} "
                  f"rows, {len(df)} remain")

    # A subject spanning two bins means the table is defective and any split
    # drawn from it can leak.
    spanning = df.groupby("subject_key")["mst_bin"].nunique()
    bad = spanning[spanning > 1]
    if len(bad):
        raise ValueError(f"subject_key(s) mapped to multiple bins: {sorted(bad.index)}")

    subjects = df.groupby("subject_key")["mst_bin"].first().reset_index()

    rng = np.random.default_rng(seed)
    train_keys, val_keys, test_keys = [], [], []

    for _bin_val, group in subjects.groupby("mst_bin"):
        subj_list = group["subject_key"].tolist()
        rng.shuffle(subj_list)
        n = len(subj_list)

        n_test = max(0, round(n * test_ratio))
        n_val = max(0, round(n * val_ratio))
        n_train = n - n_val - n_test

        if n_train < 1:
            n_train = min(1, n)
            leftover = n - n_train
            n_val = leftover // 2
            n_test = leftover - n_val

        train_keys.extend(subj_list[:n_train])
        val_keys.extend(subj_list[n_train:n_train + n_val])
        test_keys.extend(subj_list[n_train + n_val:])

    train_df = df[df.subject_key.isin(train_keys)].reset_index(drop=True)
    val_df = df[df.subject_key.isin(val_keys)].reset_index(drop=True)
    test_df = df[df.subject_key.isin(test_keys)].reset_index(drop=True)

    assert not (set(train_keys) & set(val_keys)), "train/val subject overlap"
    assert not (set(train_keys) & set(test_keys)), "train/test subject overlap"
    assert not (set(val_keys) & set(test_keys)), "val/test subject overlap"

    if verbose:
        print(
            f"Split — train: {len(train_df)} imgs / {len(train_keys)} subjects"
            f" | val: {len(val_df)} imgs / {len(val_keys)} subjects"
            f" | test: {len(test_df)} imgs / {len(test_keys)} subjects"
        )

    return train_df, val_df, test_df


class SkinToneDataset(Dataset):
    """Returns (image, label), or (image, label, row_index) when index=True."""

    def __init__(self, df, transform=None, align=False, index=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform if transform is not None else get_val_transform()
        self.index = index
        self.aligner = None
        self._fa = None

        if align:
            if not _INSIGHTFACE_AVAILABLE:
                raise ImportError(
                    "insightface required for alignment: pip install insightface"
                )
            app = FaceAnalysis(
                name="buffalo_sc",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
            self.aligner = app
            self._fa = face_align

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            image = Image.open(row["complete_path"]).convert("RGB")
        except Exception as exc:
            print(f"[WARN] cannot open {row['complete_path']}: {exc}")
            return None

        if self.aligner is not None:
            img_np = np.array(image)
            faces = self.aligner.get(img_np)
            if faces:
                aligned = self._fa.norm_crop(
                    img_np, landmark=faces[0].kps, image_size=224
                )
                image = Image.fromarray(aligned)
            # No face detected: keep the original crop.

        image = self.transform(image)
        label = torch.tensor(
            [row[c] for c in LABEL_COLS], dtype=torch.float32
        )
        if self.index:
            return image, label, idx
        return image, label


def collate_skip_none(batch):
    """Drop unreadable images from a batch."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)


def collate_skip_none_indexed(batch):
    """As collate_skip_none, keeping row indices so predictions stay aligned."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None, None, None
    images, labels, idxs = zip(*batch)
    return torch.stack(images), torch.stack(labels), torch.tensor(idxs)
