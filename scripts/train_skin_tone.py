#!/usr/bin/env python3
import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/skin_tone_dataset",
                    help="Folder with subfolders: Black/, Brown/, White/")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_split", type=float, default=0.15)
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--patience", type=int, default=3, help="early stopping patience")
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir.resolve()}")

    # Tone classification depends on color; keep geometry changes mild.
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(args.img_size, scale=(0.90, 1.00), ratio=(0.90, 1.10)),
        transforms.RandomHorizontalFlip(p=0.5),

        # Lighting robustness (main challenge)
        transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.30, hue=0.02),
        transforms.RandomGrayscale(p=0.02),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Load full dataset
    full_ds = datasets.ImageFolder(root=str(data_dir), transform=train_tf)
    classes = full_ds.classes  # alphabetical by folder
    num_classes = len(classes)
    print("Classes found:", classes)

    # Split train/val
    n_total = len(full_ds)
    n_val = int(round(n_total * args.val_split))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Ensure val uses eval transforms
    val_ds.dataset = datasets.ImageFolder(root=str(data_dir), transform=eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model: ResNet34 often improves slightly over ResNet18
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if args.freeze_backbone:
        for name, p in model.named_parameters():
            if not name.startswith("fc."):
                p.requires_grad = False

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.out_dir) / "skin_tone_resnet34.pt"

    best_val_acc = -1.0
    bad = 0

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * x.size(0)
            tr_correct += (logits.argmax(dim=1) == y).sum().item()
            tr_total += x.size(0)

        train_loss = tr_loss / max(1, tr_total)
        train_acc = tr_correct / max(1, tr_total)

        # val
        model.eval()
        va_loss = 0.0
        va_correct = 0
        va_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = criterion(logits, y)

                va_loss += loss.item() * x.size(0)
                va_correct += (logits.argmax(dim=1) == y).sum().item()
                va_total += x.size(0)

        val_loss = va_loss / max(1, va_total)
        val_acc = va_correct / max(1, va_total)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        # save best + early stopping
        if val_acc > best_val_acc + 1e-4:
            best_val_acc = val_acc
            bad = 0
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "backbone": "resnet34",
            }, ckpt_path)
            print(f"  ✓ Saved best to {ckpt_path} (val acc={best_val_acc:.3f})")
        else:
            bad += 1
            if bad >= args.patience:
                print("Early stopping.")
                break

    print("Done. Best val acc:", best_val_acc)
    print("Checkpoint:", ckpt_path.resolve())


if __name__ == "__main__":
    main()