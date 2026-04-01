#!/usr/bin/env python3
"""
Multilabel trainer for acne/bags/blackheads/hyperpigmentation/redness.

Supports multiple architectures, loss functions, augmentation strategies,
and LR schedules — all behind CLI flags for experiment comparison.

Usage:
  # Baseline
  python scripts/train_multilabel.py --metadata data/processed/all_multilabel_onehot.csv

  # Full experiment
  python scripts/train_multilabel.py \
    --metadata data/processed/all_multilabel_onehot.csv \
    --arch efficientnet_b3 --epochs 40 --lr 1e-4 \
    --loss focal --label_smoothing 0.05 --mixup_alpha 0.4 \
    --scheduler cosine --warmup_epochs 3 --accum_steps 2 \
    --run_name exp5_effb3_full --out_dir models
"""
import argparse, os, random, json, time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from sklearn.metrics import f1_score, roc_auc_score

LABELS = ["acne", "bags", "blackheads", "hyperpigmentation", "redness"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
class FocalLoss(nn.Module):
    """Binary focal loss for multilabel classification.

    Reduces the relative loss for well-classified examples, putting more focus
    on hard, misclassified examples.  When gamma=0 this is equivalent to BCE.
    """

    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha          # tensor (num_classes,), same role as pos_weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma

        if self.alpha is not None:
            # alpha weighting: use alpha for positives, 1.0 for negatives
            alpha_t = targets * self.alpha.unsqueeze(0) + (1 - targets) * 1.0
            focal_weight = focal_weight * alpha_t

        loss = focal_weight * bce
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ---------------------------------------------------------------------------
# MixUp
# ---------------------------------------------------------------------------
def mixup_data(x, y, alpha=0.4):
    """MixUp: interpolate pairs of images and labels.

    Especially useful when the dataset is single-label — creates synthetic
    multi-label examples (e.g. mixing an acne image with a blackheads image).
    """
    if alpha <= 0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # keep lam >= 0.5 so the primary image dominates
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class MultiLabelCSVDataset(Dataset):
    def __init__(self, csv_path, split, transform=None, img_col="image_path",
                 label_smoothing=0.0):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.img_col = img_col
        self.label_smoothing = label_smoothing
        missing = [l for l in LABELS if l not in self.df.columns]
        if missing:
            raise ValueError(f"Missing label columns in CSV: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row[self.img_col]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(row[LABELS].values.astype(np.float32))
        if self.label_smoothing > 0:
            # Smooth: 1 -> 1-eps, 0 -> eps/(num_classes-1)
            eps = self.label_smoothing
            y = y * (1 - eps) + (1 - y) * (eps / max(len(LABELS) - 1, 1))
        return img, y


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
def get_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            size=img_size, scale=(0.78, 1.00), ratio=(0.85, 1.15)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.22, contrast=0.22, saturation=0.18, hue=0.03
        ),
        transforms.RandomGrayscale(p=0.08),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2)),
        transforms.ToTensor(),
        transforms.RandomErasing(
            p=0.35, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"
        ),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_model(num_classes, arch="resnet18", freeze_backbone=False):
    if arch == "efficientnet_b3":
        m = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        head_prefix = "classifier"
    else:
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        head_prefix = "fc"
    if freeze_backbone:
        for n, p in m.named_parameters():
            if not n.startswith(head_prefix):
                p.requires_grad = False
    return m


# ---------------------------------------------------------------------------
# Metrics (epoch-level, not batch-averaged)
# ---------------------------------------------------------------------------
def compute_metrics(all_logits, all_targets, thresh=0.5):
    """Compute comprehensive metrics from accumulated epoch logits/targets."""
    probs = torch.sigmoid(all_logits).numpy()
    y = all_targets.numpy()
    # Binarize targets (needed when label smoothing makes them continuous)
    y = (y >= 0.5).astype(np.float32)
    y_hat = (probs >= thresh).astype(np.int32)

    # Micro & macro F1
    micro_f1 = f1_score(y.ravel(), y_hat.ravel(), zero_division=0)

    per_class_f1 = {}
    per_class_auroc = {}
    for i, label in enumerate(LABELS):
        per_class_f1[label] = float(f1_score(y[:, i], y_hat[:, i], zero_division=0))
        try:
            per_class_auroc[label] = float(roc_auc_score(y[:, i], probs[:, i]))
        except ValueError:
            per_class_auroc[label] = 0.0

    macro_f1 = float(np.mean(list(per_class_f1.values())))
    macro_auroc = float(np.mean(list(per_class_auroc.values())))

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": macro_f1,
        "macro_auroc": macro_auroc,
        "per_class_f1": per_class_f1,
        "per_class_auroc": per_class_auroc,
    }


def print_metrics(tag, metrics):
    """Pretty-print metrics for an epoch."""
    print(f"  [{tag}] micro_f1={metrics['micro_f1']:.3f}  "
          f"macro_f1={metrics['macro_f1']:.3f}  "
          f"macro_auroc={metrics['macro_auroc']:.3f}")
    for label in LABELS:
        f1 = metrics["per_class_f1"][label]
        auc = metrics["per_class_auroc"][label]
        marker = " *" if f1 < 0.3 else ""
        print(f"    {label:20s}  F1={f1:.3f}  AUROC={auc:.3f}{marker}")


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------
def log_experiment(log_path, run_name, args_dict, best_val_metrics, test_metrics,
                   train_time_s):
    """Append experiment results to a JSON log file."""
    entry = {
        "run_name": run_name,
        "timestamp": datetime.now().isoformat(),
        "train_time_s": round(train_time_s, 1),
        "args": {k: v for k, v in args_dict.items()
                 if k not in ("metadata", "images_root", "out_dir")},
        "best_val": best_val_metrics,
        "test": test_metrics,
    }

    log = []
    if log_path.exists():
        try:
            log = json.loads(log_path.read_text())
        except (json.JSONDecodeError, ValueError):
            log = []
    log.append(entry)
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\n  Experiment logged to {log_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Multilabel skin condition trainer")

    # Data
    ap.add_argument("--metadata", required=True,
                    help="CSV with image_path, split, and one-hot label cols")
    ap.add_argument("--images_root", default=".",
                    help="Root folder for relative image paths")

    # Architecture
    ap.add_argument("--arch", default="resnet18",
                    choices=["resnet18", "efficientnet_b3"])
    ap.add_argument("--freeze_backbone", action="store_true")

    # Training
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=6,
                    help="Early stopping patience (epochs without improvement)")

    # Loss
    ap.add_argument("--loss", default="bce", choices=["bce", "focal"],
                    help="Loss function (default: bce)")
    ap.add_argument("--focal_gamma", type=float, default=2.0,
                    help="Focal loss gamma (default: 2.0)")

    # Regularisation
    ap.add_argument("--label_smoothing", type=float, default=0.0,
                    help="Label smoothing epsilon (0=off, 0.05 recommended)")
    ap.add_argument("--mixup_alpha", type=float, default=0.0,
                    help="MixUp alpha (0=off, 0.4 recommended)")

    # LR schedule
    ap.add_argument("--scheduler", default="plateau",
                    choices=["plateau", "cosine"],
                    help="LR scheduler (default: plateau)")
    ap.add_argument("--warmup_epochs", type=int, default=0,
                    help="Linear LR warmup epochs (0=off, 3 recommended for effnet)")

    # Gradient accumulation
    ap.add_argument("--accum_steps", type=int, default=1,
                    help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")

    # Output
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--run_name", default=None,
                    help="Experiment name (auto-generated if not set)")

    args = ap.parse_args()

    if args.run_name is None:
        args.run_name = f"{args.arch}_{args.loss}_ls{args.label_smoothing}_mx{args.mixup_alpha}"

    set_seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}")
    print(f"Run: {args.run_name}")
    print(f"Config: arch={args.arch} loss={args.loss} lr={args.lr} "
          f"label_smooth={args.label_smoothing} mixup={args.mixup_alpha} "
          f"scheduler={args.scheduler} warmup={args.warmup_epochs} "
          f"accum={args.accum_steps}")

    # ------------------------------------------------------------------
    # Fix relative image paths if needed
    # ------------------------------------------------------------------
    df = pd.read_csv(args.metadata)
    if "image_path" not in df.columns:
        if "path" in df.columns:
            df.rename(columns={"path": "image_path"}, inplace=True)
            df.to_csv(args.metadata, index=False)
        else:
            raise ValueError("CSV must have 'image_path' (or 'path') column.")

    if args.images_root not in ["", ".", "./"]:
        df["image_path"] = df["image_path"].apply(
            lambda p: os.path.join(args.images_root, p)
        )
        df.to_csv(args.metadata, index=False)

    # ------------------------------------------------------------------
    # Datasets / loaders
    # ------------------------------------------------------------------
    train_tf, eval_tf = get_transforms(args.img_size)
    train_ds = MultiLabelCSVDataset(
        args.metadata, split="train", transform=train_tf,
        label_smoothing=args.label_smoothing
    )
    val_ds = MultiLabelCSVDataset(args.metadata, split="val", transform=eval_tf)
    test_ds = MultiLabelCSVDataset(args.metadata, split="test", transform=eval_tf)

    print(f"\nSplit sizes: train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    # pos_weight from train prevalence
    pos = train_ds.df[LABELS].sum().values.astype(float)
    neg = len(train_ds) - pos
    pos_weight = torch.tensor(
        neg / np.maximum(pos, 1.0), device=device, dtype=torch.float32
    )
    print(f"pos_weight: {dict(zip(LABELS, pos_weight.tolist()))}")

    # Label-aware sampler (oversample rare-label images)
    freq = pos / len(train_ds)
    inv = 1.0 / np.maximum(freq, 1e-6)
    sample_mat = train_ds.df[LABELS].values
    weights = (sample_mat * inv).sum(axis=1)
    weights = weights / weights.mean()
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights), replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True
    )

    # ------------------------------------------------------------------
    # Model / loss / optimizer / scheduler
    # ------------------------------------------------------------------
    model = build_model(
        num_classes=len(LABELS), arch=args.arch,
        freeze_backbone=args.freeze_backbone
    ).to(device)

    if args.loss == "focal":
        criterion = FocalLoss(alpha=pos_weight, gamma=args.focal_gamma)
        print(f"Loss: Focal (gamma={args.focal_gamma})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("Loss: BCEWithLogitsLoss")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if args.scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        print("Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2)")
    else:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2
        )
        print("Scheduler: ReduceLROnPlateau (factor=0.5, patience=2)")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(args.out_dir) / "best_multilabel.pt"
    best_macro_f1 = -1.0
    best_val_metrics = {}
    bad = 0

    start_time = time.time()
    print(f"\n{'='*70}")
    print(f"Starting training: {args.epochs} epochs, effective batch={args.batch_size * args.accum_steps}")
    print(f"{'='*70}\n")

    for epoch in range(1, args.epochs + 1):

        # --- Linear warmup ---
        if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
            warmup_lr = args.lr * (epoch / args.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        current_lr = optimizer.param_groups[0]["lr"]

        # --- Train ---
        model.train()
        epoch_loss = 0.0
        all_logits_train = []
        all_targets_train = []
        optimizer.zero_grad()

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device).float()

            # MixUp (applied after moving to device)
            if args.mixup_alpha > 0:
                x, y = mixup_data(x, y, alpha=args.mixup_alpha)

            logits = model(x)
            loss = criterion(logits, y)
            loss = loss / args.accum_steps
            loss.backward()

            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * args.accum_steps
            # Store un-mixed logits/targets for metrics (only when no mixup
            # for cleaner metric computation — still accumulate for loss tracking)
            if args.mixup_alpha <= 0:
                all_logits_train.append(logits.detach().cpu())
                all_targets_train.append(y.detach().cpu())

        epoch_loss /= max(1, len(train_loader))

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_logits_val = []
        all_targets_val = []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).float()
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item()
                all_logits_val.append(logits.detach().cpu())
                all_targets_val.append(y.detach().cpu())

        val_loss /= max(1, len(val_loader))

        # --- Compute epoch-level metrics ---
        val_logits = torch.cat(all_logits_val)
        val_targets = torch.cat(all_targets_val)
        val_metrics = compute_metrics(val_logits, val_targets)

        train_metrics_str = ""
        if all_logits_train:
            train_logits = torch.cat(all_logits_train)
            train_targets = torch.cat(all_targets_train)
            train_metrics = compute_metrics(train_logits, train_targets)
            train_metrics_str = f"train_macroF1={train_metrics['macro_f1']:.3f}  "

        # --- Scheduler step ---
        if args.scheduler == "cosine":
            if epoch > args.warmup_epochs:
                scheduler.step(epoch)
        else:
            scheduler.step(val_metrics["macro_f1"])

        # --- Print ---
        print(f"Epoch {epoch:02d}/{args.epochs}  lr={current_lr:.2e}  "
              f"loss={epoch_loss:.4f}/{val_loss:.4f}  "
              f"{train_metrics_str}"
              f"val_macroF1={val_metrics['macro_f1']:.3f}  "
              f"val_macroAUROC={val_metrics['macro_auroc']:.3f}")
        print_metrics("val", val_metrics)

        # --- Checkpoint (best macro F1) ---
        if val_metrics["macro_f1"] > best_macro_f1 + 1e-4:
            best_macro_f1 = val_metrics["macro_f1"]
            best_val_metrics = val_metrics
            bad = 0
            torch.save({
                "state_dict": model.state_dict(),
                "labels": LABELS,
                "img_size": args.img_size,
                "arch": args.arch,
                "run_name": args.run_name,
            }, ckpt_path)
            print(f"  >>> Saved best to {ckpt_path} (macro_f1={best_macro_f1:.3f})")
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  Early stopping after {args.patience} epochs without improvement.")
                break

        print()

    train_time = time.time() - start_time
    print(f"\nTraining done in {train_time:.0f}s. Best val macro_f1: {best_macro_f1:.3f}")

    # ------------------------------------------------------------------
    # Test evaluation
    # ------------------------------------------------------------------
    test_metrics = {}
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        all_logits_test = []
        all_targets_test = []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device).float()
                logits = model(x)
                all_logits_test.append(logits.detach().cpu())
                all_targets_test.append(y.detach().cpu())

        test_logits = torch.cat(all_logits_test)
        test_targets = torch.cat(all_targets_test)
        test_metrics = compute_metrics(test_logits, test_targets)

        print(f"\n{'='*70}")
        print("TEST RESULTS")
        print(f"{'='*70}")
        print_metrics("test", test_metrics)
    else:
        print("No checkpoint found for test evaluation.")

    # ------------------------------------------------------------------
    # Log experiment
    # ------------------------------------------------------------------
    log_path = Path(args.out_dir) / "experiment_log.json"
    log_experiment(log_path, args.run_name, vars(args),
                   best_val_metrics, test_metrics, train_time)


if __name__ == "__main__":
    main()
