#!/usr/bin/env python3
import os, json, argparse

import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.metrics import f1_score
from tqdm import tqdm

LABELS = ["acne", "bags", "blackheads", "hyperpigmentation", "redness"]

class MultiLabelValDataset(Dataset):
    def __init__(self, csv_path, images_root, split="val", img_col=None, transform=None):
        df = pd.read_csv(csv_path)
        df = df[df["split"] == split].reset_index(drop=True)
        self.df = df
        self.images_root = images_root
        self.transform = transform

        cols = df.columns
        if img_col is not None:
            self.img_col = img_col
        elif "image" in cols:
            self.img_col = "image"
        elif "image_path" in cols:
            self.img_col = "image_path"
        else:
            raise ValueError(f"No image column found. Columns: {list(cols)}")

        missing = [l for l in LABELS if l not in df.columns]
        if missing:
            raise ValueError(f"Missing label columns: {missing}")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = os.path.join(self.images_root, str(row[self.img_col]))
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        y = torch.tensor(row[LABELS].values.astype(np.float32))
        return img, y

def build_resnet18_multilabel(num_classes):
    m = models.resnet18(weights=None)
    in_features = m.fc.in_features
    m.fc = torch.nn.Linear(in_features, num_classes)
    return m

def load_model(ckpt_path, num_classes, device):
    state = torch.load(ckpt_path, map_location=device)
    if "model" in state:
        sd = state["model"]
    elif "state_dict" in state:
        sd = state["state_dict"]
    else:
        raise KeyError(f"Checkpoint missing 'model' or 'state_dict' keys. Got: {state.keys()}")
    m = build_resnet18_multilabel(num_classes)
    m.load_state_dict(sd, strict=True)
    m.to(device).eval()
    return m

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="models/per_class_thresholds.json")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print("Device:", device)

    # make sure 'image' exists if only 'image_path' is there
    df = pd.read_csv(args.metadata)
    if "image" not in df.columns and "image_path" in df.columns:
        df["image"] = df["image_path"]
        df.to_csv(args.metadata, index=False)

    img_size = 224
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225]),
    ])

    ds_val = MultiLabelValDataset(args.metadata, args.images_root, split="val", transform=tf)
    dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=2, pin_memory=False)

    model = load_model(args.checkpoint, num_classes=len(LABELS), device=device)

    ys_true, ys_prob = [], []
    with torch.no_grad():
        for x, y in tqdm(dl_val, desc="Val"):
            x = x.to(device)
            logits = model(x)
            prob = torch.sigmoid(logits).cpu()
            ys_prob.append(prob)
            ys_true.append(y)

    y_true = torch.cat(ys_true, dim=0).numpy()
    y_prob = torch.cat(ys_prob, dim=0).numpy()

    thresholds = {}
    for j, lab in enumerate(LABELS):
        ys = y_true[:, j]
        ps = y_prob[:, j]
        best_f1, best_t = 0.0, 0.5
        for t in np.linspace(0.05, 0.95, 19):
            pred = (ps >= t).astype(int)
            f1 = f1_score(ys, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[lab] = float(best_t)
        print(f"{lab}: best F1={best_f1:.3f} at t={best_t:.2f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(thresholds, f, indent=2)

    print("Saved per-class thresholds to", args.out)
