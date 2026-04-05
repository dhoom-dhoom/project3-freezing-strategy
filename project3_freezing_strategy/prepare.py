"""
Fixed infrastructure for Project 3 on IndiWASTE.

This file defines:
- dataset loading
- fixed train/val/test split handling
- fixed transforms
- fixed metrics
- fixed 300-second training budget

Do not modify during the autoresearch loop unless explicitly requested by the human.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.models import ResNet18_Weights

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify in the loop)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300
IMAGE_SIZE = 224
NUM_CLASSES = 10
DEFAULT_BATCH_SIZE = 16
DEFAULT_EVAL_BATCH_SIZE = 32
NUM_WORKERS = 0
RANDOM_SEED = 42

CLASS_NAMES = [
    "battery",
    "biological",
    "cardboard",
    "clothes",
    "glass",
    "metal",
    "paper",
    "plastic",
    "shoes",
    "trash",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}


def _default_dataset_root() -> Path:
    env_root = os.environ.get("INDIWASTE_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / "IndiWASTE").resolve()


DATASET_ROOT = _default_dataset_root()
SPLITS_DIR = DATASET_ROOT / "splits"
IMAGES_DIR = DATASET_ROOT / "images"

_WEIGHTS = ResNet18_Weights.DEFAULT
_MEAN = _WEIGHTS.transforms().mean
_STD = _WEIGHTS.transforms().std


@dataclass(frozen=True)
class EvalResult:
    val_macro_f1: float
    val_accuracy: float
    val_error: float
    val_loss: float
    confusion: list[list[int]]
    per_class_f1: dict[str, float]


class IndiWasteSplitDataset(Dataset):
    def __init__(self, root: Path, split: str, transform):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        split_path = self.root / "splits" / f"{split}.csv"
        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing split file: {split_path}. Set INDIWASTE_ROOT or place the dataset at ./IndiWASTE."
            )
        frame = pd.read_csv(split_path)
        required_cols = {"image_id", "filename", "label"}
        if not required_cols.issubset(frame.columns):
            raise RuntimeError(f"Unexpected split schema in {split_path}: {frame.columns.tolist()}")
        self.rows = frame.to_dict("records")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        label_name = row["label"]
        image_path = self.root / "images" / label_name / row["filename"]
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image referenced by split file: {image_path}")
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        label_idx = CLASS_TO_IDX[label_name]
        return image, label_idx


def _build_train_transform():
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )


def _build_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=_MEAN, std=_STD),
        ]
    )


def _maybe_subset(dataset, limit: int | None):
    if limit is None or limit >= len(dataset):
        return dataset
    return Subset(dataset, range(limit))


def make_dataloaders(
    dataset_root: Path | str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    smoke_test: bool = False,
):
    root = Path(dataset_root).resolve() if dataset_root is not None else DATASET_ROOT
    train_dataset = IndiWasteSplitDataset(root, "train", _build_train_transform())
    val_dataset = IndiWasteSplitDataset(root, "val", _build_eval_transform())
    test_dataset = IndiWasteSplitDataset(root, "test", _build_eval_transform())

    if smoke_test:
        train_dataset = _maybe_subset(train_dataset, 64)
        val_dataset = _maybe_subset(val_dataset, 64)
        test_dataset = _maybe_subset(test_dataset, 64)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate_classifier(model, loader, device, criterion):
    model.eval()
    losses = []
    all_targets = []
    all_preds = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)

        losses.append(loss.item())
        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    val_macro_f1 = f1_score(all_targets, all_preds, average="macro")
    val_accuracy = accuracy_score(all_targets, all_preds)
    val_error = 1.0 - val_macro_f1
    val_loss = float(np.mean(losses)) if losses else math.nan
    confusion = confusion_matrix(all_targets, all_preds, labels=list(range(NUM_CLASSES)))
    per_class = f1_score(all_targets, all_preds, labels=list(range(NUM_CLASSES)), average=None)
    per_class_f1 = {IDX_TO_CLASS[idx]: float(score) for idx, score in enumerate(per_class)}

    return EvalResult(
        val_macro_f1=float(val_macro_f1),
        val_accuracy=float(val_accuracy),
        val_error=float(val_error),
        val_loss=float(val_loss),
        confusion=confusion.tolist(),
        per_class_f1=per_class_f1,
    )


def dataset_summary(dataset_root: Path | str | None = None):
    root = Path(dataset_root).resolve() if dataset_root is not None else DATASET_ROOT
    summary = {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(root / "splits" / f"{split}.csv")
        summary[split] = {
            "num_images": int(len(frame)),
            "label_counts": dict(Counter(frame["label"].tolist())),
        }
    return summary


def save_eval_artifacts(result: EvalResult, output_path: str | os.PathLike = "last_eval.json"):
    payload = {
        "val_macro_f1": result.val_macro_f1,
        "val_accuracy": result.val_accuracy,
        "val_error": result.val_error,
        "val_loss": result.val_loss,
        "confusion": result.confusion,
        "per_class_f1": result.per_class_f1,
        "saved_unix": int(time.time()),
    }
    Path(output_path).write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    print(f"Dataset root: {DATASET_ROOT}")
    print(json.dumps(dataset_summary(), indent=2))
