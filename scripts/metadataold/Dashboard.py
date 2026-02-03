#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import streamlit as st
import numpy as np
import pandas as pd

# Must match your training / eval
LABELS = ["acne", "bags", "blackheads", "hyperpigmentation", "redness"]
CKPT_PATH = "models/best_multilabel.pt"
THRESH_PATH = "models/per_class_thresholds.json"
IMG_SIZE = 224


def build_resnet18_multilabel(num_classes: int):
    """Same architecture as the training script."""
    m = models.resnet18(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


@st.cache_resource
def load_model_and_thresholds():
    # Device selection: MPS > CUDA > CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found at {CKPT_PATH}")

    state = torch.load(CKPT_PATH, map_location=device)
    # support {"model": ...} or {"state_dict": ...}
    if "model" in state:
        sd = state["model"]
    elif "state_dict" in state:
        sd = state["state_dict"]
    else:
        raise KeyError(f"Checkpoint missing 'model' or 'state_dict' keys. Got: {state.keys()}")

    model = build_resnet18_multilabel(num_classes=len(LABELS))
    model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    # thresholds: default 0.5 everywhere, override if JSON exists
    thresholds = {lab: 0.5 for lab in LABELS}
    if os.path.exists(THRESH_PATH):
        try:
            with open(THRESH_PATH, "r") as f:
                loaded = json.load(f)
            for lab in LABELS:
                if lab in loaded:
                    thresholds[lab] = float(loaded[lab])
        except Exception as e:
            print("Could not load per-class thresholds:", e)

    # eval transforms (match eval_skin / train_multilabel)
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return model, device, tf, thresholds


def predict_image(pil_img: Image.Image, model, device, tf, thresholds):
    """Run model on a single PIL image and return dict with probabilities & flags."""
    model.eval()
    x = tf(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # shape [C]

    results = []
    for i, lab in enumerate(LABELS):
        p = float(probs[i])
        thr = thresholds.get(lab, 0.5)
        present = p >= thr
        results.append({
            "condition": lab,
            "probability": p,
            "threshold": thr,
            "present": bool(present),
        })
    return results


# --------------------- STREAMLIT UI --------------------- #

st.set_page_config(page_title="Facial Skin Analysis Demo", layout="centered")

st.title("🧴 Facial Skin Condition Demo Dashboard")
st.write(
    "Upload a face image or use your webcam, and this demo model will predict "
    "the likelihood of the following conditions:\n\n"
    "- **Acne**\n- **Under-eye bags**\n- **Blackheads**\n- **Hyperpigmentation**\n- **Redness**"
)

# Load model + preprocessing
with st.spinner("Loading model..."):
    model, device, tf, thresholds = load_model_and_thresholds()
st.success(f"Model loaded on `{device}`")

st.sidebar.header("Input mode")
mode = st.sidebar.radio(
    "Choose how to provide an image:",
    ("Upload image", "Use webcam"),
)

uploaded_image = None

if mode == "Upload image":
    file = st.file_uploader(
        "Upload a facial image (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )
    if file is not None:
        uploaded_image = Image.open(file).convert("RGB")
elif mode == "Use webcam":
    cam = st.camera_input("Take a photo")
    if cam is not None:
        uploaded_image = Image.open(cam).convert("RGB")

if uploaded_image is not None:
    st.subheader("Input image")
    st.image(uploaded_image, caption="Image used for prediction", use_column_width=True)

    if st.button("🔍 Analyze skin"):
        with st.spinner("Running prediction..."):
            results = predict_image(uploaded_image, model, device, tf, thresholds)

        # Create a nice table
        df = pd.DataFrame(results)
        df["probability (%)"] = (df["probability"] * 100).round(1)
        df["threshold (%)"] = (df["threshold"] * 100).round(1)
        df["present?"] = df["present"].map({True: "✅ yes", False: "❌ no"})

        st.subheader("Results")
        st.dataframe(
            df[["condition", "probability (%)", "threshold (%)", "present?"]],
            use_container_width=True,
        )

        # Optional: simple bar chart of probs
        st.subheader("Probabilities")
        chart_df = df[["condition", "probability"]].set_index("condition")
        st.bar_chart(chart_df)

else:
    st.info("Please upload an image or use the webcam to start.")
