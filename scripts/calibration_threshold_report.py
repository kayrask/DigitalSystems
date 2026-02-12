#!/usr/bin/env python3
"""
Generate calibration + threshold tuning report by class and skin tone group.

Input CSV is expected to have:
  - split column
  - image_path (or path)
  - label columns: acne,bags,blackheads,hyperpigmentation,redness
"""
import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.inference import SkinModel, SkinToneModel, LABELS


def _best_threshold(y_true: np.ndarray, y_prob: np.ndarray, grid: np.ndarray) -> float:
    best_t = 0.5
    best_f1 = -1.0
    for t in grid:
        y_pred = (y_prob >= t).astype(np.int32)
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t


def _metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> dict:
    y_pred = (y_prob >= thr).astype(np.int32)
    p, r, f1, s = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "support_pos": int(np.sum(y_true)),
        "support_total": int(len(y_true)),
        "threshold": float(thr),
    }


def _predict_probs(skin_model: SkinModel, img: Image.Image) -> dict:
    x = skin_model.transform(img).unsqueeze(0).to(skin_model.device)
    with torch.no_grad():
        logits = skin_model.model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    return {label: float(probs[i]) for i, label in enumerate(LABELS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="CSV with labels and split")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--img_col", default="", help="image column override")
    ap.add_argument("--model_ckpt", default="models/best_multilabel.pt")
    ap.add_argument("--thresholds_json", default="models/per_class_thresholds.json")
    ap.add_argument("--tone_ckpt", default="models/skin_tone_resnet18.pt")
    ap.add_argument("--tone_conf_min", type=float, default=0.80)
    ap.add_argument("--min_samples", type=int, default=30)
    ap.add_argument("--min_pos", type=int, default=8)
    ap.add_argument("--grid_steps", type=int, default=101)
    ap.add_argument(
        "--out_report", default="models/calibration_threshold_report_by_tone.json"
    )
    ap.add_argument(
        "--out_thresholds",
        default="models/per_class_thresholds_by_tone_tuned.json",
        help="Threshold JSON generated from this calibration run",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.metadata)
    if args.img_col:
        img_col = args.img_col
    elif "image_path" in df.columns:
        img_col = "image_path"
    elif "path" in df.columns:
        img_col = "path"
    else:
        raise ValueError("No image column found. Use --img_col.")

    missing = [l for l in LABELS if l not in df.columns]
    if missing:
        raise ValueError(f"Missing label columns: {missing}")

    if "split" not in df.columns:
        raise ValueError("metadata CSV must contain a 'split' column")

    df = df[df["split"] == args.split].reset_index(drop=True)

    skin_model = SkinModel(args.model_ckpt, args.thresholds_json)
    tone_model = SkinToneModel(args.tone_ckpt)

    groups = ["default"] + tone_model.classes
    probs_by_group = {g: {l: [] for l in LABELS} for g in groups}
    true_by_group = {g: {l: [] for l in LABELS} for g in groups}

    for i, row in df.iterrows():
        try:
            img = Image.open(str(row[img_col])).convert("RGB")
        except Exception:
            continue

        probs = _predict_probs(skin_model, img)
        tone_out = tone_model.predict(img)
        tone_group = (
            tone_out["group"] if tone_out["confidence"] >= args.tone_conf_min else None
        )

        for l in LABELS:
            y = int(row[l])
            probs_by_group["default"][l].append(float(probs[l]))
            true_by_group["default"][l].append(y)
            if tone_group is not None:
                probs_by_group[tone_group][l].append(float(probs[l]))
                true_by_group[tone_group][l].append(y)

        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{len(df)}")

    # Current threshold map (default + tone fallback)
    current_thresholds = {}
    current_thresholds["default"] = {
        l: float(skin_model.thresholds.get(l, 0.5)) for l in LABELS
    }
    if skin_model.thresholds_by_tone:
        for g in tone_model.classes:
            if g in skin_model.thresholds_by_tone:
                current_thresholds[g] = {
                    l: float(skin_model.thresholds_by_tone[g].get(l, current_thresholds["default"][l]))
                    for l in LABELS
                }
            else:
                current_thresholds[g] = dict(current_thresholds["default"])
    else:
        for g in tone_model.classes:
            current_thresholds[g] = dict(current_thresholds["default"])

    grid = np.linspace(0.05, 0.95, args.grid_steps)
    tuned_thresholds = {g: {} for g in groups}
    metrics_current = {g: {} for g in groups}
    metrics_tuned = {g: {} for g in groups}

    for g in groups:
        for l in LABELS:
            y_true = np.array(true_by_group[g][l], dtype=np.int32)
            y_prob = np.array(probs_by_group[g][l], dtype=np.float32)

            if len(y_true) == 0:
                metrics_current[g][l] = None
                metrics_tuned[g][l] = None
                tuned_thresholds[g][l] = float(current_thresholds[g][l])
                continue

            thr_cur = float(current_thresholds[g][l])
            metrics_current[g][l] = _metrics_at_threshold(y_true, y_prob, thr_cur)

            if len(y_true) < args.min_samples or int(np.sum(y_true)) < args.min_pos:
                thr_tuned = float(current_thresholds["default"][l])
            else:
                thr_tuned = _best_threshold(y_true, y_prob, grid)
            tuned_thresholds[g][l] = float(round(thr_tuned, 4))
            metrics_tuned[g][l] = _metrics_at_threshold(y_true, y_prob, thr_tuned)

    report = {
        "generated_at_utc": datetime.utcnow().isoformat(),
        "split": args.split,
        "tone_conf_min": float(args.tone_conf_min),
        "labels": LABELS,
        "tone_classes": tone_model.classes,
        "guards": {"min_samples": args.min_samples, "min_pos": args.min_pos},
        "thresholds_current": current_thresholds,
        "thresholds_tuned": tuned_thresholds,
        "metrics_current": metrics_current,
        "metrics_tuned": metrics_tuned,
    }

    out_report = Path(args.out_report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(out_report, "w") as f:
        json.dump(report, f, indent=2)

    out_thr = Path(args.out_thresholds)
    out_thr.parent.mkdir(parents=True, exist_ok=True)
    with open(out_thr, "w") as f:
        json.dump(
            {
                "tone_conf_min": float(args.tone_conf_min),
                "labels": LABELS,
                "tone_classes": tone_model.classes,
                "thresholds": tuned_thresholds,
            },
            f,
            indent=2,
        )

    print("Wrote report:", out_report.resolve())
    print("Wrote tuned thresholds:", out_thr.resolve())


if __name__ == "__main__":
    main()
