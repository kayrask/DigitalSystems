#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn


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
def predict_tone_probs(model, tf, device, pil_img):
    x = tf(pil_img).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (3,)
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", required=True, help="CSV used for multilabel training (image_path, split, labels...)")
    ap.add_argument("--tone_ckpt", default="models/skin_tone_resnet18.pt", help="Tone model checkpoint")
    ap.add_argument("--out_csv", required=True, help="Output CSV with added tone columns")
    ap.add_argument("--img_col", default="image_path")
    ap.add_argument("--max_rows", type=int, default=0, help="Optional limit for debugging (0 = all)")
    args = ap.parse_args()

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)

    df = pd.read_csv(args.metadata)
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    tone_model, classes, tf = load_skin_tone_model(args.tone_ckpt, device)
    print("Tone classes:", classes)

    # Add columns (match exact class names)
    for c in classes:
        col = f"tone_{c}"
        if col not in df.columns:
            df[col] = 0.0

    for i, row in df.iterrows():
        path = row[args.img_col]
        try:
            img = Image.open(path).convert("RGB")
            probs = predict_tone_probs(tone_model, tf, device, img)

            for j, c in enumerate(classes):
                df.at[i, f"tone_{c}"] = float(probs[j])

        except Exception as e:
            print(f"[WARN] Failed on {path}: {e}")
            # leave zeros

        if (i + 1) % 200 == 0:
            print(f"Processed {i+1}/{len(df)}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Wrote:", out_path.resolve())


if __name__ == "__main__":
    main()