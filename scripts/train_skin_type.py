#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Root with train/validate/test folders")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--out", default="models/skin_type_resnet18.pt")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "validate"
    if not val_dir.exists():
        # Some datasets use "val"
        val_dir = data_dir / "val"

    test_dir = data_dir / "test"

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    train_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1, 0.1, 0.1, 0.02),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = ImageFolder(train_dir, transform=train_tf)
    val_ds = ImageFolder(val_dir, transform=eval_tf) if val_dir.exists() else None
    test_ds = ImageFolder(test_dir, transform=eval_tf) if test_dir.exists() else None

    print("Classes:", train_ds.classes)  # important
    num_classes = len(train_ds.classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4) if val_ds else None
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4) if test_ds else None

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_loss = loss_sum / total
        train_acc = correct / total

        val_acc = None
        if val_loader:
            model.eval()
            v_total, v_correct = 0, 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    logits = model(x)
                    preds = logits.argmax(dim=1)
                    v_correct += (preds == y).sum().item()
                    v_total += x.size(0)
            val_acc = v_correct / max(1, v_total)

            # save best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "model": model.state_dict(),
                    "classes": train_ds.classes
                }, args.out)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc}")
    
    # If no validation set, save final model anyway
    if not val_loader:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "classes": train_ds.classes
        }, args.out)
        print(f"✓ Saved final model to {args.out}")


    # optional test
    if test_loader:
        # load best checkpoint for test
        ckpt = torch.load(args.out, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        t_total, t_correct = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=1)
                t_correct += (preds == y).sum().item()
                t_total += x.size(0)
        print("TEST ACC:", t_correct / max(1, t_total))


if __name__ == "__main__":
    main()
