# api/explainability.py
from __future__ import annotations

from typing import Any, Dict

import base64
import io

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F


LABELS = ["acne", "bags", "blackheads", "hyperpigmentation", "redness"]


def _make_red_overlay_rgba(cam: np.ndarray, strength: float = 0.85) -> Image.Image:
    """Return a transparent (RGBA) red overlay.

    - Red channel is 255 everywhere.
    - Alpha channel is proportional to CAM intensity.
    - No blending with the input image.

    The returned PNG should be alpha-composited on top of a base image.
    """

    cam = np.asarray(cam, dtype=np.float32)
    cam = np.clip(cam, 0.0, 1.0)

    # --- Make CAM visible (percentile rescale + gamma) ---
    lo = float(np.percentile(cam, 70))
    hi = float(np.percentile(cam, 99))
    cam = np.clip((cam - lo) / (hi - lo + 1e-8), 0.0, 1.0)
    cam = cam ** 0.6

    h, w = cam.shape

    # optional soft ellipse mask (keeps focus toward face region)
    yy, xx = np.mgrid[0:h, 0:w]
    cx, cy = w / 2.0, h / 2.0
    rx, ry = w * 0.34, h * 0.44
    ellipse = ((xx - cx) ** 2) / (rx ** 2) + ((yy - cy) ** 2) / (ry ** 2)
    mask = (np.clip(1.0 - ellipse, 0.0, 1.0) ** 0.8).astype(np.float32)

    # additional visibility boost
    cam = cam ** 0.5

    alpha = (cam * mask * float(strength) * 255.0).astype(np.uint8)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255  # red
    rgba[..., 3] = alpha

    return Image.fromarray(rgba, mode="RGBA")


def gradcam_overlay_base64(
    skin_model,
    image: Image.Image,
    target_label: str,
) -> Dict[str, Any]:
    """Generate a Grad-CAM overlay and return it as base64 PNG (RGBA overlay)."""

    if target_label not in LABELS:
        raise ValueError(f"target_label must be one of {LABELS}")

    device = skin_model.device
    model = skin_model.model
    model.eval()

    target_layer = model.layer4  # last conv block for resnet18

    activations = []
    gradients = []

    def fwd_hook(_, __, out):
        activations.append(out)

    def bwd_hook(_, grad_in, grad_out):
        gradients.append(grad_out[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        x = skin_model.transform(image).unsqueeze(0).to(device)
        x.requires_grad_(True)

        logits = model(x)
        idx = LABELS.index(target_label)
        score = logits[0, idx]

        model.zero_grad(set_to_none=True)
        score.backward()

        act = activations[-1]
        grad = gradients[-1]

        w = grad.mean(dim=(2, 3), keepdim=True)
        cam = (w * act).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        H, W = image.size[1], image.size[0]
        cam_up = F.interpolate(
            cam[None, None, ...],
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam_np = cam_up.detach().cpu().numpy()

        overlay = _make_red_overlay_rgba(cam_np, strength=0.85)

        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {
            "target": target_label,
            "overlay_png_base64": b64,
            "cam_minmax": [float(cam_np.min()), float(cam_np.max())],
        }

    finally:
        h1.remove()
        h2.remove()
