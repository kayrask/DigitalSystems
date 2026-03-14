#!/usr/bin/env python3
# Multilabel trainer for acne/bags/blackheads/hyperpigmentation/redness
import argparse, os, random, json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import f1_score

LABELS = ["acne","bags","blackheads","hyperpigmentation","redness"]

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class MultiLabelCSVDataset(Dataset):
    def __init__(self, csv_path, split, transform=None, img_col="image_path"):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"]==split].reset_index(drop=True)
        self.transform = transform
        self.img_col = img_col
        # Ensure label columns exist
        missing = [l for l in LABELS if l not in self.df.columns]
        if missing:
            raise ValueError(f"Missing label columns in CSV: {missing}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[self.img_col]).convert("RGB")
        if self.transform: img = self.transform(img)
        y = torch.tensor(row[LABELS].values.astype(np.float32))  # multi-hot
        return img, y

def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        # Strongest anti-background shortcut + scale robustness
        transforms.RandomResizedCrop(
            size=img_size,
            scale=(0.78, 1.00),
            ratio=(0.85, 1.15)
        ),

        # your dataset has multi-angle shots; keep flip because left/right symmetry is fine
        transforms.RandomHorizontalFlip(p=0.5),

        # Lighting & camera pipeline variation
        transforms.ColorJitter(
            brightness=0.22,
            contrast=0.22,
            saturation=0.12,
            hue=0.03
        ),
        transforms.RandomGrayscale(p=0.08),

        # Pose/angle robustness (helps different viewpoints)
        transforms.RandomPerspective(distortion_scale=0.12, p=0.25),

        # Focus issues (keep it subtle)
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),

        transforms.ToTensor(),

        # Occlusion robustness (hair, hands, shadows)
        transforms.RandomErasing(
            p=0.35,
            scale=(0.02, 0.12),
            ratio=(0.3, 3.3),
            value="random"
        ),

        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    return train_tf, eval_tf


def build_model(num_classes, arch="resnet18", freeze_backbone=False):
    if arch == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        head_prefix = "classifier"
    else:  # default: resnet18
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head_prefix = "fc"
    if freeze_backbone:
        for n, p in m.named_parameters():
            if not n.startswith(head_prefix):
                p.requires_grad = False
    return m

def micro_f1_from_logits(logits, y_true, thresh=0.5):
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    y = y_true.detach().cpu().numpy()
    y_hat = (probs >= thresh).astype(np.int32)
    return f1_score(y.reshape(-1), y_hat.reshape(-1), zero_division=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="CSV with image_path, split, and one-hot label cols")
    ap.add_argument("--images_root", default=".", help="root folder for relative image paths (optional)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--freeze_backbone", action="store_true")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "efficientnet_b3"],
                    help="Model architecture (default: resnet18)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Fix relative image paths if needed
    df = pd.read_csv(args.metadata)
    if "image_path" not in df.columns:
        # try 'path' fallback
        if "path" in df.columns:
            df.rename(columns={"path":"image_path"}, inplace=True)
            df.to_csv(args.metadata, index=False)
        else:
            raise ValueError("CSV must have 'image_path' (or 'path') column.")

    if args.images_root not in ["", ".", "./"]:
        df["image_path"] = df["image_path"].apply(lambda p: os.path.join(args.images_root, p))
        df.to_csv(args.metadata, index=False)

    # Datasets / loaders
    train_tf, eval_tf = get_transforms(args.img_size)
    train_ds = MultiLabelCSVDataset(args.metadata, split="train", transform=train_tf)
    val_ds   = MultiLabelCSVDataset(args.metadata, split="val",   transform=eval_tf)
    test_ds  = MultiLabelCSVDataset(args.metadata, split="test",  transform=eval_tf)

    # pos_weight from train prevalence
    pos = train_ds.df[LABELS].sum().values.astype(float)
    neg = len(train_ds) - pos
    pos_weight = torch.tensor(neg / np.maximum(pos, 1.0), device=device, dtype=torch.float32)
    print("pos_weight:", pos_weight.tolist())

    # label-aware sampler (oversample rare-label images)
    freq = pos / len(train_ds)
    inv = 1.0 / np.maximum(freq, 1e-6)
    sample_mat = train_ds.df[LABELS].values
    weights = (sample_mat * inv).sum(axis=1)
    weights = weights / weights.mean()
    sampler = WeightedRandomSampler(weights=torch.as_tensor(weights, dtype=torch.double),
                                    num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model / loss / opt
    model = build_model(num_classes=len(LABELS), arch=args.arch, freeze_backbone=args.freeze_backbone).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    # Train
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.out_dir) / "best_multilabel.pt"
    best_f1, patience, bad = -1.0, 4, 0

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_loss = tr_f1 = n_batches = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).float()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
            tr_f1 += micro_f1_from_logits(logits, y, 0.5)
            n_batches += 1

        model.eval()
        va_loss = va_f1 = v_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).float()
                logits = model(x)
                loss = criterion(logits, y)
                va_loss += loss.item()
                va_f1 += micro_f1_from_logits(logits, y, 0.5)
                v_batches += 1

        tr_loss /= max(1, n_batches); tr_f1 /= max(1, n_batches)
        va_loss /= max(1, v_batches); va_f1 /= max(1, v_batches)
        scheduler.step(va_loss)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} microF1 {tr_f1:.3f} | "
              f"val loss {va_loss:.4f} microF1 {va_f1:.3f}")

        if va_f1 > best_f1 + 1e-4:
            best_f1 = va_f1; bad = 0
            torch.save({
                "state_dict": model.state_dict(),
                "labels": LABELS,
                "img_size": args.img_size,
            }, ckpt_path)
            print(f"  ✓ Saved best to {ckpt_path} (val microF1={best_f1:.3f})")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping.")
                break

    print("Training done. Best val microF1:", best_f1)

    # quick test pass
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"]); model.eval()
        te_loss = te_f1 = t_batches = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device).float()
                logits = model(x)
                loss = criterion(logits, y)
                te_loss += loss.item()
                te_f1 += micro_f1_from_logits(logits, y, 0.5)
                t_batches += 1
        te_loss /= max(1, t_batches); te_f1 /= max(1, t_batches)
        print(f"[TEST] loss {te_loss:.4f} microF1 {te_f1:.3f}")
    else:
        print("No checkpoint found for test evaluation.")

if __name__ == "__main__":
    main()
