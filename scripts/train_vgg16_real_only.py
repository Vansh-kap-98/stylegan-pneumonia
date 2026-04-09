#!/usr/bin/env python3
"""Train VGG16 on real-only data and evaluate on held-out real test split."""

import argparse
import sys
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

# Make sibling imports work when running `python scripts/<file>.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_vgg16_real_synth_split import (
    PathLabelDataset,
    collect_class_images,
    evaluate,
    set_seed,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGG16 real-only baseline training")
    parser.add_argument("--real-root", required=True, help="Root with class folders 0_normal and 1_pneumonia")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    real_samples = collect_class_images(Path(args.real_root))
    real_train, real_val, real_test = stratified_split(real_samples, args.seed)

    train_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(p=0.0),
            transforms.RandomRotation(degrees=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = PathLabelDataset(real_train, train_tf)
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
        print(f"Epoch {epoch:02d}/{args.epochs} - train_loss={train_loss:.4f} - val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    test_acc, y_true, y_pred = evaluate(model, test_loader, device)
    print("\nFinal held-out real test results (VGG16 baseline real-only)")
    print(f"Test accuracy: {test_acc * 100:.2f}%")
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("Classification report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"], digits=4))


if __name__ == "__main__":
    main()
