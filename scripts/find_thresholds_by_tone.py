#!/usr/bin/env python3
"""
Tone-aware threshold calibration with dataset-aware label filtering.

Why:
- a_fullface: labels acne, bags, redness
- b_fullface: labels acne only
- patches: labels acne, blackheads, hyperpigmentation

So we ONLY calibrate thresholds for each label on datasets that truly annotate that label.
This prevents insane thresholds caused by "unknown" labels treated as 0.

Outputs JSON:
{
  "tone_conf_min": 0.8,
  "labels": [...],
  "tone_classes": ["Black","Brown","White"],
  "thresholds": {
     "default": {...},
     "Black": {...},
     "Brown": {...},
     "White": {...}
  },
  "counts": {...}   # helpful debugging
}
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


LABELS = ["acne", "bags", "blackheads", "hyperpigmentation", "redness"]

# ✅ Dataset coverage map (based on your dataset structure)
LABEL_DATASETS = {
    "acne": {"a_fullface", "b_fullface", "patches"},
    "bags": {"a_fullface"},
    "redness": {"a_fullface"},
    "blackheads": {"patches"},
    "hyperpigmentation": {"patches"},
}


def load_multilabel_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)

    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(LABELS))

    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt["model"]
    m.load_state_dict(state)
    m.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return m, tf


def load_skin_tone_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device)
    classes = ckpt.get("classes", ["Black", "Brown", "White"])

    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, len(classes))
    m.load_state_dict(ckpt["model"])
    m.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return m, classes, tf


@torch.no_grad()
def predict_multilabel_probs(model, tf, device, pil_img: Image.Image) -> np.ndarray:
    x = tf(pil_img).unsqueeze(0).to(device)
    logits = model(x)              # (1,5)
    probs = torch.sigmoid(logits)[0].detach().cpu().numpy()
    return probs.astype(np.float32)


@torch.no_grad()
def predict_tone(model, tf, device, pil_img: Image.Image):
    x = tf(pil_img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
    idx = int(probs.argmax())
    return idx, float(probs[idx]), probs.astype(np.float32)


def f1_for_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    y_pred = (y_prob >= thr).astype(np.int32)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    if tp == 0:
        return 0.0
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray) -> float:
    best_thr = 0.5
    best_f1 = -1.0
    for thr in grid:
        f1 = f1_for_threshold(y_true, y_prob, float(thr))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--img_col", default="image_path")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--multilabel_ckpt", default="models/best_multilabel.pt")
    ap.add_argument("--tone_ckpt", default="models/skin_tone_resnet18.pt")
    ap.add_argument("--out_json", default="models/per_class_thresholds_by_tone.json")
    ap.add_argument("--tone_conf_min", type=float, default=0.80)
    ap.add_argument("--grid_steps", type=int, default=101)

    # ✅ guardrails
    ap.add_argument("--min_group_n", type=int, default=40)
    ap.add_argument("--min_pos", type=int, default=10)
    ap.add_argument("--thr_min", type=float, default=0.10)
    ap.add_argument("--thr_max", type=float, default=0.90)

    args = ap.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    df = pd.read_csv(args.metadata)
    df = df[df["split"] == args.split].reset_index(drop=True)
    if "dataset" not in df.columns:
        raise ValueError("CSV must contain a 'dataset' column (a_fullface/b_fullface/patches).")

    ml_model, ml_tf = load_multilabel_model(args.multilabel_ckpt, device)
    tone_model, tone_classes, tone_tf = load_skin_tone_model(args.tone_ckpt, device)
    print("Tone classes:", tone_classes)

    groups = ["default"] + tone_classes

    probs_by_group = {g: {lbl: [] for lbl in LABELS} for g in groups}
    true_by_group  = {g: {lbl: [] for lbl in LABELS} for g in groups}

    # Counts for debugging
    group_counts = {g: 0 for g in groups}
    pos_counts = {g: {lbl: 0 for lbl in LABELS} for g in groups}

    for i, row in df.iterrows():
        path = row[args.img_col]
        ds = str(row["dataset"])

        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print("[WARN] unreadable image:", path, e)
            continue

        p = predict_multilabel_probs(ml_model, ml_tf, device, img)

        tone_idx, tone_conf, _ = predict_tone(tone_model, tone_tf, device, img)
        tone_group = tone_classes[tone_idx] if tone_conf >= args.tone_conf_min else None

        # Track group counts (for overall sample volume)
        group_counts["default"] += 1
        if tone_group is not None:
            group_counts[tone_group] += 1

        # ✅ Dataset-aware label filtering
        for j, lbl in enumerate(LABELS):
            if ds not in LABEL_DATASETS[lbl]:
                continue  # label not reliably annotated in this dataset

            y = int(row[lbl])

            probs_by_group["default"][lbl].append(float(p[j]))
            true_by_group["default"][lbl].append(y)
            pos_counts["default"][lbl] += y

            if tone_group is not None:
                probs_by_group[tone_group][lbl].append(float(p[j]))
                true_by_group[tone_group][lbl].append(y)
                pos_counts[tone_group][lbl] += y

        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{len(df)}")

    grid = np.linspace(0.0, 1.0, args.grid_steps)

    thresholds = {}

    # Compute default thresholds first (no tone split)
    thresholds["default"] = {}
    for lbl in LABELS:
        y_true = np.array(true_by_group["default"][lbl], dtype=np.int32)
        y_prob = np.array(probs_by_group["default"][lbl], dtype=np.float32)
        if len(y_true) == 0 or y_true.sum() == 0:
            thresholds["default"][lbl] = 0.5
            continue
        thr = best_threshold(y_true, y_prob, grid)
        thr = max(args.thr_min, min(args.thr_max, thr))
        thresholds["default"][lbl] = float(round(thr, 4))

    # Compute per-tone thresholds with guardrails
    for g in tone_classes:
        thresholds[g] = {}
        for lbl in LABELS:
            y_true = np.array(true_by_group[g][lbl], dtype=np.int32)
            y_prob = np.array(probs_by_group[g][lbl], dtype=np.float32)

            # Guardrail: enough samples + positives?
            if len(y_true) < args.min_group_n or y_true.sum() < args.min_pos:
                thresholds[g][lbl] = thresholds["default"][lbl]
                continue

            thr = best_threshold(y_true, y_prob, grid)
            thr = max(args.thr_min, min(args.thr_max, thr))
            thresholds[g][lbl] = float(round(thr, 4))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "tone_conf_min": args.tone_conf_min,
        "labels": LABELS,
        "tone_classes": tone_classes,
        "label_datasets": {k: sorted(list(v)) for k, v in LABEL_DATASETS.items()},
        "thresholds": thresholds,
        "counts": {
            "group_counts": group_counts,
            "pos_counts": pos_counts,
            "n_per_label_default": {lbl: len(true_by_group["default"][lbl]) for lbl in LABELS},
            "pos_per_label_default": {lbl: int(np.sum(true_by_group["default"][lbl])) for lbl in LABELS},
        },
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    print("Wrote:", out_path.resolve())
    print("Group counts (confident tone only counts in tone groups):", group_counts)
    print("Default thresholds:", thresholds["default"])
    for g in tone_classes:
        print(g, "thresholds:", thresholds[g])


if __name__ == "__main__":
    main()