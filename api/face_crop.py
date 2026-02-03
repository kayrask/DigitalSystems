# api/face_crop.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from PIL import Image, ImageFilter

import numpy as np
from PIL import Image
import mediapipe as mp

_mp_face = mp.solutions.face_detection


def crop_face(
    image: Image.Image,
    margin: float = 0.20,          # expand bbox to include cheeks/forehead/chin
    min_confidence: float = 0.6,   # detection confidence
) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
    """
    Returns (cropped_face_image or None, meta)
    meta includes bbox in original image coords.
    """
    img = image.convert("RGB")
    arr = np.asarray(img)  # H,W,3 uint8
    h, w = arr.shape[:2]

    meta: Dict[str, Any] = {
        "face_found": False,
        "score": None,
        "bbox": None,  # {x,y,w,h} in pixels (expanded)
    }

    with _mp_face.FaceDetection(model_selection=0, min_detection_confidence=min_confidence) as fd:
        results = fd.process(arr)

    if not results.detections:
        return None, meta

    best = max(results.detections, key=lambda d: float(d.score[0]) if d.score else 0.0)
    score = float(best.score[0]) if best.score else 0.0

    bbox = best.location_data.relative_bounding_box
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    # Expand bbox
    mx = int(bw * margin)
    my = int(bh * margin)
    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(w, x + bw + mx)
    y2 = min(h, y + bh + my)

    if x2 <= x1 or y2 <= y1:
        return None, meta

    crop = img.crop((x1, y1, x2, y2))

    meta["face_found"] = True
    meta["score"] = score
    meta["bbox"] = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
    return crop, meta

def apply_oval_mask(pil_img: Image.Image, feather: int = 18) -> Image.Image:
    """
    Keeps an oval region, soft-feathers edges, neutralizes outside region.
    """
    img = pil_img.convert("RGB")
    w, h = img.size

    # Create oval mask (0 outside, 255 inside)
    mask = Image.new("L", (w, h), 0)
    from PIL import ImageDraw
    draw = ImageDraw.Draw(mask)

    # oval fits nicely inside crop; tweak padding if needed
    pad_x = int(w * 0.10)
    pad_y = int(h * 0.06)
    draw.ellipse((pad_x, pad_y, w - pad_x, h - pad_y), fill=255)

    # feather edges
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))

    # background = slightly blurred version (looks natural)
    bg = img.filter(ImageFilter.GaussianBlur(radius=10))

    # composite: oval from original, outside from bg
    out = Image.composite(img, bg, mask)
    return out