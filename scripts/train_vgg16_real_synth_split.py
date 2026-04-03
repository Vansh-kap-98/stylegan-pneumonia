#!/usr/bin/env python3
"""Train VGG16 classifier with 50/50 real-synthetic split at 256x256 source resolution.

This script keeps evaluation on held-out real images to avoid synthetic leakage.

Usage example:
python scripts/train_vgg16_real_synth_split.py \
  --real-root data/processed/mendeley_256 \
  --synth-root outputs/snapshot200/for_classifier \
  --epochs 10 --batch-size 32
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class PathLabelDataset(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]], transform):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGG16 real+synthetic 50/50 training")
    parser.add_argument("--real-root", required=True, help="Root with class folders 0_normal and 1_pneumonia")
    parser.add_argument("--synth-root", required=True, help="Root with synthetic class folders 0_normal and 1_pneumonia")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collect_class_images(root: Path) -> List[Tuple[Path, int]]:
    out: List[Tuple[Path, int]] = []
    class_map = {
        "0_normal": 0,
        "1_pneumonia": 1,
    }

    for folder_name, label in class_map.items():
        cls_dir = root / folder_name
        if not cls_dir.exists():
            raise ValueError(f"Missing class directory: {cls_dir}")
        for p in cls_dir.glob("**/*"):
            if p.suffix.lower() in ALLOWED_EXT:
                out.append((p, label))
    if not out:
        raise ValueError(f"No images found in {root}")
    return out


def stratified_split(samples: List[Tuple[Path, int]], seed: int):
    labels = [label for _, label in samples]
    train_val, test = train_test_split(samples, test_size=0.2, stratify=labels, random_state=seed)

    train_val_labels = [label for _, label in train_val]
    train, val = train_test_split(train_val, test_size=0.1, stratify=train_val_labels, random_state=seed)
    return train, val, test


def make_50_50_train(real_train: List[Tuple[Path, int]], synth_all: List[Tuple[Path, int]], seed: int):
    rng = random.Random(seed)
    synth_by_class = defaultdict(list)
    for p, label in synth_all:
        synth_by_class[label].append((p, label))

    real_by_class = defaultdict(list)
    for p, label in real_train:
        real_by_class[label].append((p, label))

    balanced_synth = []
    for label in (0, 1):
        n_real = len(real_by_class[label])
        n_synth = len(synth_by_class[label])
        n = min(n_real, n_synth)
        if n == 0:
            raise ValueError(
                f"Not enough synthetic images for class {label}: "
                f"real={n_real}, synthetic={n_synth}"
            )

        # Use the maximum matched count available so the train set stays 50/50
        # without requiring more synthetic samples than were generated.
        real_by_class[label] = rng.sample(real_by_class[label], n)
        balanced_synth.extend(rng.sample(synth_by_class[label], n))

    mixed = list(real_train) + balanced_synth
    rng.shuffle(mixed)
    return mixed


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(y.numpy().tolist())
    acc = accuracy_score(y_true, y_pred)
    return acc, y_true, y_pred


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    real_samples = collect_class_images(Path(args.real_root))
    synth_samples = collect_class_images(Path(args.synth_root))

    real_train, real_val, real_test = stratified_split(real_samples, args.seed)
    mixed_train = make_50_50_train(real_train, synth_samples, args.seed)

    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.0),
        transforms.RandomRotation(degrees=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = PathLabelDataset(mixed_train, train_tf)
    val_ds = PathLabelDataset(real_val, eval_tf)
    test_ds = PathLabelDataset(real_test, eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hasattr(models, "VGG16_Weights"):
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
    else:
        model = models.vgg16(pretrained=True)
    for p in model.features.parameters():
        p.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)

    best_state = None
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_ds)
        val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d}/{args.epochs} - train_loss={train_loss:.4f} - val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print("\nFinal held-out real test results")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"], digits=4))


if __name__ == "__main__":
    main()
