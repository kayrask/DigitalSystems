#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import models, transforms

from dataset_skin import SkinDefectsDataset

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_class_weights(df, label_to_idx):
    # weights = 1 / freq
    counts = df[df["split"]=="train"]["label"].value_counts().to_dict()
    n_classes = len(label_to_idx)
    weights = torch.ones(n_classes, dtype=torch.float32)
    for lbl, idx in label_to_idx.items():
        freq = counts.get(lbl, 1)
        weights[idx] = 1.0 / float(freq)
    # normalize a bit
    weights = weights * (n_classes / weights.sum())
    return weights

def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

def build_model(num_classes, freeze_backbone=False):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    if freeze_backbone:
        for n, p in m.named_parameters():
            if not n.startswith("fc."):
                p.requires_grad = False
    return m

def accuracy(logits, y):
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/processed/b.csv")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--use_weighted_sampler", action="store_true",
                    help="Use WeightedRandomSampler on train set to handle imbalance")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(args.csv)
    # Build a consistent label map from all splits
    classes = sorted(df["label"].unique())
    label_to_idx = {c:i for i,c in enumerate(classes)}
    num_classes = len(classes)

    train_tf, eval_tf = get_transforms(args.img_size)

    train_ds = SkinDefectsDataset(args.csv, split="train", transform=train_tf, label_to_idx=label_to_idx)
    val_ds   = SkinDefectsDataset(args.csv, split="val",   transform=eval_tf, label_to_idx=label_to_idx)
    test_ds  = SkinDefectsDataset(args.csv, split="test",  transform=eval_tf, label_to_idx=label_to_idx)

    if args.use_weighted_sampler:
        # Build per-sample weights based on class frequency
        counts = train_ds.df["label"].value_counts().to_dict()
        sample_weights = [1.0 / counts[row["label"]] for _, row in train_ds.df.iterrows()]
        sampler = WeightedRandomSampler(weights=torch.DoubleTensor(sample_weights),
                                        num_samples=len(sample_weights),
                                        replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=4, pin_memory=True)

    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)

    model = build_model(num_classes, freeze_backbone=args.freeze_backbone).to(device)
    class_weights = compute_class_weights(df, label_to_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = -1
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.outdir) / "skin_resnet18.pt"

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss = tr_acc = n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_acc += accuracy(logits, y)
            n_batches += 1

        model.eval()
        va_loss = va_acc = v_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                va_loss += loss.item()
                va_acc += accuracy(logits, y)
                v_batches += 1

        tr_loss /= max(1, n_batches)
        tr_acc  /= max(1, n_batches)
        va_loss /= max(1, v_batches)
        va_acc  /= max(1, v_batches)
        scheduler.step()

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            torch.save({
                "state_dict": model.state_dict(),
                "label_to_idx": label_to_idx,
                "classes": classes,
                "img_size": args.img_size,
            }, ckpt_path)
            print(f"  ✓ Saved best to {ckpt_path} (val_acc={best_val:.3f})")

    print("Training complete. Best val acc:", best_val)

    # quick test pass with best checkpoint
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        te_acc = te_loss = t_batches = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                te_loss += loss.item()
                te_acc += accuracy(logits, y)
                t_batches += 1
        te_loss /= max(1, t_batches)
        te_acc  /= max(1, t_batches)
        print(f"[TEST] loss {te_loss:.4f} acc {te_acc:.3f}")
    else:
        print("No checkpoint found for test evaluation.")
        
if __name__ == "__main__":
    main()
