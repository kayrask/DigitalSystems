#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import argparse

import torch
import pandas as pd
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
)
from tqdm import tqdm

# IMPORTANT: must match the label order used in train_multilabel.py
LABELS = ["acne", "bags", "blackheads", "hyperpigmentation", "redness"]


# -------- Dataset --------

class MultiLabelCSVDataset(Dataset):
    def __init__(self, csv_path, images_root, split, img_col=None, transform=None):
        df = pd.read_csv(csv_path)
        if "split" not in df.columns:
            raise ValueError(f"'split' column not found in {csv_path}")

        df = df[df["split"] == split].reset_index(drop=True)
        self.df = df
        self.images_root = images_root
        self.transform = transform

        # decide which column holds image path
        cols = df.columns
        if img_col is not None:
            self.img_col = img_col
        elif "image" in cols:
            self.img_col = "image"
        elif "image_path" in cols:
            self.img_col = "image_path"
        else:
            raise ValueError(
                f"No obvious image column found. Columns are: {list(cols)}. "
                f"Expected 'image' or 'image_path'."
            )

        # sanity check label columns
        missing = [l for l in LABELS if l not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns in {csv_path}: {missing}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_root, str(row[self.img_col]))
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        # multi-hot labels
        y = torch.tensor(row[LABELS].values.astype(np.float32))
        return img, y


# -------- Model builder --------

def build_resnet18_multilabel(num_classes: int):
    """
    Same architecture as train_multilabel.py:
    ResNet18 + final Linear(num_classes).
    """
    m = models.resnet18(weights=None)  # weights overwritten by checkpoint
    in_features = m.fc.in_features
    m.fc = torch.nn.Linear(in_features, num_classes)
    return m


def load_model(checkpoint_path, num_classes, device):
    """
    Load model from checkpoint.
    Supports:
      - {"model": state_dict}
      - {"state_dict": state_dict}
    """
    state = torch.load(checkpoint_path, map_location=device)

    if "model" in state:
        sd = state["model"]
    elif "state_dict" in state:
        sd = state["state_dict"]
    else:
        raise KeyError(
            f"Checkpoint missing 'model' or 'state_dict' keys. Got: {state.keys()}"
        )

    model = build_resnet18_multilabel(num_classes=num_classes)
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()
    return model


# -------- Main eval script --------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metadata",
        required=True,
        help="CSV with image paths, split column, and one-hot label columns.",
    )
    ap.add_argument(
        "--images_root",
        required=True,
        help="Root folder that image paths are relative to.",
    )
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained multilabel checkpoint (.pt).",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Global decision threshold applied to all classes.",
    )
    args = ap.parse_args()

    # Device: prefer MPS (Apple GPU), then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    # Ensure 'image' column exists for compatibility if we only have 'image_path'
    df_meta = pd.read_csv(args.metadata)
    cols_before = df_meta.columns.tolist()
    if "image" not in df_meta.columns and "image_path" in df_meta.columns:
        df_meta["image"] = df_meta["image_path"]
        df_meta.to_csv(args.metadata, index=False)
        print("Added 'image' column as a copy of 'image_path'.")
    else:
        print("Metadata columns:", cols_before)

    # Transforms (match eval transforms in train_multilabel)
    img_size = 224  # if you changed img_size in train_multilabel, update this too
    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Datasets / loaders
    ds_test = MultiLabelCSVDataset(
        csv_path=args.metadata,
        images_root=args.images_root,
        split="test",
        transform=tf,
    )
    classes = LABELS  # label names in correct order

    dl_test = DataLoader(
        ds_test,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=False,  # avoids MPS warning
    )

    # Model
    model = load_model(
        checkpoint_path=args.checkpoint,
        num_classes=len(classes),
        device=device,
    )

    # Collect predictions
    ys_true, ys_prob = [], []
    with torch.no_grad():
        for x, y in tqdm(dl_test, desc="Testing"):
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu()
            ys_prob.append(prob)
            ys_true.append(y)

    y_true = torch.cat(ys_true, dim=0).numpy()  # [N, C]
    y_prob = torch.cat(ys_prob, dim=0).numpy()  # [N, C]
    # try to load per-class thresholds if available
    per_class_thresh = {lab: args.threshold for lab in LABELS}
    json_path = "models/per_class_thresholds.json"
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            loaded = json.load(f)
        for lab in LABELS:
            if lab in loaded:
                per_class_thresh[lab] = float(loaded[lab])
        print("Using per-class thresholds:", per_class_thresh)
    else:
        print("Using global threshold:", args.threshold)

# apply thresholds per class
    y_pred = np.zeros_like(y_prob, dtype="int32")
    for j, lab in enumerate(LABELS):
        t = per_class_thresh[lab]
        y_pred[:, j] = (y_prob[:, j] >= t).astype("int32")


    # Metrics
    if y_true.sum() > 0:
        macro_auroc = float(roc_auc_score(y_true, y_prob, average="macro"))
        micro_auroc = float(roc_auc_score(y_true, y_prob, average="micro"))
    else:
        macro_auroc = micro_auroc = float("nan")

    report = {
        "macro_auroc": macro_auroc,
        "micro_auroc": micro_auroc,
        "macro_ap": float(average_precision_score(y_true, y_prob, average="macro")),
        "micro_ap": float(average_precision_score(y_true, y_prob, average="micro")),
        "macro_f1@{:.2f}".format(args.threshold): float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "micro_f1@{:.2f}".format(args.threshold): float(
            f1_score(y_true, y_pred, average="micro", zero_division=0)
        ),
        "per_class": {},
    }

    cr = classification_report(
        y_true,
        y_pred,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )
    for cls in classes:
        d = cr.get(cls, {})
        report["per_class"][cls] = {
            "precision": float(d.get("precision", 0.0)),
            "recall": float(d.get("recall", 0.0)),
            "f1": float(d.get("f1-score", 0.0)),
        }

    os.makedirs("experiments/checkpoints", exist_ok=True)
    out_path = "experiments/checkpoints/test_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

