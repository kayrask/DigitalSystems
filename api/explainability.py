# api/explainability.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import base64
import io

import numpy as np
from PIL import Image
from PIL import ImageFilter

import torch
import torch.nn.functional as F


LABELS = ["acne", "bags", "blackheads", "hyperpigmentation", "redness"]


def _ensure_mask01(mask: np.ndarray, size_hw: Optional[Tuple[int, int]] = None) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("mask must be HxW")
    if size_hw is not None and (arr.shape[0] != size_hw[0] or arr.shape[1] != size_hw[1]):
        pil = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")
        pil = pil.resize((size_hw[1], size_hw[0]), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _morph_mask(mask01: np.ndarray, op: str, k: int = 5, n: int = 1) -> np.ndarray:
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    out = np.clip(mask01 * 255.0, 0, 255).astype(np.uint8)
    for _ in range(max(1, int(n))):
        img = Image.fromarray(out, mode="L")
        if op == "dilate":
            img = img.filter(ImageFilter.MaxFilter(k))
        elif op == "erode":
            img = img.filter(ImageFilter.MinFilter(k))
        else:
            raise ValueError("op must be dilate|erode")
        out = np.asarray(img, dtype=np.uint8)
    return out.astype(np.float32) / 255.0


def refine_gradcam_mask(
    cam: np.ndarray,
    exclude_mask: Optional[np.ndarray] = None,
    percentile: float = 86.0,
    blur_radius: float = 2.0,
) -> np.ndarray:
    """
    Convert raw Grad-CAM heatmap into a cleaner binary-ish mask in [0,1].
    """
    cam01 = _ensure_mask01(cam)
    hi = float(np.percentile(cam01, np.clip(percentile, 60.0, 99.5)))
    lo = float(np.percentile(cam01, 55.0))
    cam01 = np.clip((cam01 - lo) / (max(hi - lo, 1e-6)), 0.0, 1.0)
    binary = (cam01 >= 0.45).astype(np.float32)

    # close then open to reduce speckle and fill small holes
    binary = _morph_mask(binary, "dilate", k=7, n=1)
    binary = _morph_mask(binary, "erode", k=7, n=1)
    binary = _morph_mask(binary, "erode", k=5, n=1)
    binary = _morph_mask(binary, "dilate", k=5, n=1)

    if exclude_mask is not None:
        ex = _ensure_mask01(exclude_mask, size_hw=binary.shape)
        binary[ex >= 0.5] = 0.0

    # soften edges to avoid blocky overlays
    soft = Image.fromarray(np.clip(binary * 255.0, 0, 255).astype(np.uint8), mode="L")
    soft = soft.filter(ImageFilter.GaussianBlur(radius=max(0.0, float(blur_radius))))
    soft01 = np.asarray(soft, dtype=np.float32) / 255.0
    return np.clip(soft01, 0.0, 1.0)


def make_red_overlay_from_mask(mask01: np.ndarray, strength: float = 0.82) -> Image.Image:
    m = _ensure_mask01(mask01)
    alpha = np.clip(m * float(strength) * 255.0, 0, 255).astype(np.uint8)
    h, w = m.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = alpha
    return Image.fromarray(rgba, mode="RGBA")


def detector_pixel_mask(
    face_size_wh: Tuple[int, int],
    boxes: List[Dict[str, Any]],
    expansion: float = 0.22,
    blur_radius: float = 2.6,
) -> np.ndarray:
    """
    Build soft mask from YOLO detections so only detected bits are highlighted.
    """
    fw, fh = int(face_size_wh[0]), int(face_size_wh[1])
    m = np.zeros((fh, fw), dtype=np.float32)
    for b in boxes:
        x1, y1, x2, y2 = [float(v) for v in b["box"]]
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        pad_x = bw * float(expansion)
        pad_y = bh * float(expansion)
        xa = max(0, int(round(x1 - pad_x)))
        ya = max(0, int(round(y1 - pad_y)))
        xb = min(fw, int(round(x2 + pad_x)))
        yb = min(fh, int(round(y2 + pad_y)))
        if xb > xa and yb > ya:
            m[ya:yb, xa:xb] = 1.0

    if np.max(m) <= 0:
        return m
    soft = Image.fromarray((m * 255).astype(np.uint8), mode="L")
    soft = soft.filter(ImageFilter.GaussianBlur(radius=max(0.0, float(blur_radius))))
    return np.asarray(soft, dtype=np.float32) / 255.0


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


def gradcam_cam_array(
    skin_model,
    image: Image.Image,
    target_label: str,
) -> np.ndarray:
    """
    Return raw upsampled Grad-CAM array in [0,1], shape HxW.
    """
    if target_label not in LABELS:
        raise ValueError(f"target_label must be one of {LABELS}")

    device = skin_model.device
    model = skin_model.model
    model.eval()

    target_layer = model.layer4
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

        h, w_img = image.size[1], image.size[0]
        cam_up = F.interpolate(
            cam[None, None, ...],
            size=(h, w_img),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return np.clip(cam_up.detach().cpu().numpy().astype(np.float32), 0.0, 1.0)
    finally:
        h1.remove()
        h2.remove()


def gradcam_overlay_base64(
    skin_model,
    image: Image.Image,
    target_label: str,
) -> Dict[str, Any]:
    """Generate a Grad-CAM overlay and return it as base64 PNG (RGBA overlay)."""

    if target_label not in LABELS:
        raise ValueError(f"target_label must be one of {LABELS}")

    cam_np = gradcam_cam_array(skin_model, image, target_label)
    overlay = _make_red_overlay_rgba(cam_np, strength=0.85)
    buf = io.BytesIO()
    overlay.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    return {
        "target": target_label,
        "overlay_png_base64": b64,
        "cam_minmax": [float(cam_np.min()), float(cam_np.max())],
    }
