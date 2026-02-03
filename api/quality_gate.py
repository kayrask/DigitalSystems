# api/quality_gate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from PIL import Image
import torch
import numpy as np


@dataclass
class QualityConfig:
    min_short_side: int = 320

    # grayscale mean brightness in [0..1]
    brightness_min: float = 0.20
    brightness_max: float = 0.85

    # grayscale std (contrast) in [0..1]
    contrast_min: float = 0.05

    # focus measure threshold (variance of Laplacian response)
    blur_var_min: float = 0.0015


def _pil_to_gray_tensor(img: Image.Image) -> torch.Tensor:
    """
    Return grayscale tensor [1,1,H,W] float in [0..1].
    """
    rgb = img.convert("RGB")
    arr = np.asarray(rgb).astype("float32") / 255.0  # [H,W,3]
    # grayscale
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    t = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    return t.clamp(0.0, 1.0)


def _variance_of_laplacian(gray: torch.Tensor) -> float:
    """
    Simple blur metric: var(Laplacian(gray)).
    """
    k = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        dtype=gray.dtype,
    ).view(1, 1, 3, 3)

    lap = torch.nn.functional.conv2d(gray, k, padding=1)
    return float(lap.var().item())


def assess_image_quality(img: Image.Image, cfg: QualityConfig | None = None) -> Dict[str, Any]:
    cfg = cfg or QualityConfig()
    reasons: List[str] = []

    w, h = img.size
    short_side = min(w, h)
    if short_side < cfg.min_short_side:
        reasons.append(f"low_resolution (min_short_side={cfg.min_short_side}, got={short_side})")

    gray = _pil_to_gray_tensor(img)
    brightness = float(gray.mean().item())
    contrast = float(gray.std().item())
    blur_var = _variance_of_laplacian(gray)

    if brightness < cfg.brightness_min:
        reasons.append(f"too_dark (brightness={brightness:.3f})")
    if brightness > cfg.brightness_max:
        reasons.append(f"too_bright (brightness={brightness:.3f})")
    if contrast < cfg.contrast_min:
        reasons.append(f"low_contrast (contrast={contrast:.3f})")
    if blur_var < cfg.blur_var_min:
        reasons.append(f"blurry (blur_var={blur_var:.6f})")

    passed = len(reasons) == 0

    return {
        "passed": passed,
        "reasons": reasons,
        "metrics": {
            "width": w,
            "height": h,
            "short_side": short_side,
            "brightness": brightness,
            "contrast": contrast,
            "blur_var": blur_var,
        },
        "thresholds": {
            "min_short_side": cfg.min_short_side,
            "brightness_min": cfg.brightness_min,
            "brightness_max": cfg.brightness_max,
            "contrast_min": cfg.contrast_min,
            "blur_var_min": cfg.blur_var_min,
        }
    }
