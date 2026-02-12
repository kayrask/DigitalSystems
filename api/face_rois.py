# api/face_rois.py
from __future__ import annotations
from typing import Dict, Tuple, Optional
import numpy as np
from PIL import Image
from PIL import ImageDraw
import mediapipe as mp

_mp_mesh = mp.solutions.face_mesh

# ----------------------------
# Landmark groups (approximate)
# ----------------------------
# Nose / center-face
NOSE_LM = [1, 2, 98, 327, 168, 197, 195, 5]

# Eye contours (we derive under-eye by shifting bbox down)
LEFT_EYE_LM  = [33, 133, 160, 159, 158, 157, 173, 246]
RIGHT_EYE_LM = [263, 362, 387, 386, 385, 384, 398, 466]

# Cheek reference points (mid-cheek area)
LEFT_CHEEK_LM  = [50, 101, 205, 206]
RIGHT_CHEEK_LM = [280, 330, 425, 426]

# Lips / mouth landmarks
LIPS_LM = [61, 291, 0, 17, 13, 14, 78, 308]
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Forehead-ish references (top of face)
FOREHEAD_LM = [10, 109, 338, 67, 297]  # approximate spread across upper face

# Chin / jaw references
CHIN_LM = [152, 377, 148, 176, 400]
JAW_LM = [234, 93, 132, 58, 172, 136, 150, 149, 378, 379, 365, 397, 288]


def _pts(lms, idxs) -> np.ndarray:
    return np.array([[lms[i].x, lms[i].y] for i in idxs], dtype=np.float32)


def _lm_to_px(lm, w: int, h: int) -> tuple[int, int]:
    x = int(np.clip(lm.x, 0.0, 1.0) * (w - 1))
    y = int(np.clip(lm.y, 0.0, 1.0) * (h - 1))
    return x, y


def build_mouth_moustache_exclusion_mask(
    img_bgr_or_rgb: np.ndarray,
    landmarks,
    moustache_band: float = 0.35,
    lip_expand: float = 0.08,
) -> np.ndarray:
    """
    Returns uint8 mask (H,W) where 1 means exclude (mouth + moustache band).
    """
    h, w = img_bgr_or_rgb.shape[:2]
    mask_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask_img)

    outer = [_lm_to_px(landmarks[i], w, h) for i in LIPS_OUTER]
    inner = [_lm_to_px(landmarks[i], w, h) for i in LIPS_INNER]

    xs = [p[0] for p in outer]
    ys = [p[1] for p in outer]
    x0, y0 = min(xs), min(ys)
    x1, y1 = max(xs), max(ys)
    pad_x = int(lip_expand * max(1, (x1 - x0 + 1)))
    pad_y = int(lip_expand * max(1, (y1 - y0 + 1)))
    x0e, y0e = max(0, x0 - pad_x), max(0, y0 - pad_y)
    x1e, y1e = min(w - 1, x1 + pad_x), min(h - 1, y1 + pad_y)

    # Mouth polygon with inner part punched out.
    draw.polygon(outer, fill=1)
    draw.polygon(inner, fill=0)

    arr = np.array(mask_img, dtype=np.uint8)
    arr[y0e : y1e + 1, x0e : x1e + 1] = np.maximum(arr[y0e : y1e + 1, x0e : x1e + 1], 1)

    # Moustache band above upper lip bbox.
    mouth_h = max(1, (y1 - y0 + 1))
    band_h = int(moustache_band * mouth_h)
    band_y0 = max(0, y0 - band_h)
    band_y1 = y0
    arr[band_y0 : band_y1 + 1, x0e : x1e + 1] = 1
    return arr


def apply_exclusion_mask_pil(
    img: Image.Image, exclude_mask_hw: np.ndarray, fill_rgb=(127, 127, 127)
) -> Image.Image:
    arr = np.array(img).copy()
    m = exclude_mask_hw.astype(bool)
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr[m] = np.array(fill_rgb, dtype=arr.dtype)
    return Image.fromarray(arr)


def _bbox_from_points(pts_xy: np.ndarray, w: int, h: int, pad: float = 0.25) -> Tuple[int, int, int, int]:
    xs = pts_xy[:, 0] * w
    ys = pts_xy[:, 1] * h

    x1, x2 = float(xs.min()), float(xs.max())
    y1, y2 = float(ys.min()), float(ys.max())

    dx = (x2 - x1) * pad
    dy = (y2 - y1) * pad

    x1 = int(max(0, x1 - dx))
    y1 = int(max(0, y1 - dy))
    x2 = int(min(w, x2 + dx))
    y2 = int(min(h, y2 + dy))

    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)

    return x1, y1, x2, y2


def _crop(img: Image.Image, box: Tuple[int, int, int, int]) -> Image.Image:
    x1, y1, x2, y2 = box
    return img.crop((x1, y1, x2, y2))


def extract_rois_with_boxes(face_img: Image.Image) -> Dict[str, Dict]:
    """
    Extract multiple ROIs inside the FACE CROP using MediaPipe FaceMesh.

    Returns:
      {
        "nose": {"box": (x1,y1,x2,y2), "img": PIL.Image},
        ...
      }

    All boxes are in the coordinate space of `face_img` (not the full original image).
    If no face mesh is found, returns {}.
    """
    img = face_img.convert("RGB")
    arr = np.asarray(img)
    h, w = arr.shape[:2]

    with _mp_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as fm:
        out = fm.process(arr)

    if not out.multi_face_landmarks:
        return {}

    lms = out.multi_face_landmarks[0].landmark
    rois: Dict[str, Dict] = {}
    exclude_mouth = build_mouth_moustache_exclusion_mask(arr, lms)
    rois["exclude_mouth_moustache_mask"] = exclude_mouth

    # -----------------
    # Nose + T-zone ROIs
    # -----------------
    nose_box = _bbox_from_points(_pts(lms, NOSE_LM), w, h, pad=0.35)
    rois["nose"] = {"box": nose_box, "img": _crop(img, nose_box)}

    # T-zone: expand nose up/down and sideways (captures nose + central forehead + upper lip area)
    x1, y1, x2, y2 = nose_box
    tz_pad_x = int((x2 - x1) * 0.55)
    tz_up = int((y2 - y1) * 1.10)
    tz_down = int((y2 - y1) * 0.90)
    tzone_box = (
        max(0, x1 - tz_pad_x),
        max(0, y1 - tz_up),
        min(w, x2 + tz_pad_x),
        min(h, y2 + tz_down),
    )
    rois["t_zone"] = {"box": tzone_box, "img": _crop(img, tzone_box)}

    # Lips / mouth ROI (used to block moustache/mouth confounders)
    lips_box = _bbox_from_points(_pts(lms, LIPS_LM), w, h, pad=0.35)
    rois["lips"] = {"box": lips_box, "img": _crop(img, lips_box)}

    # T-zone upper ROI: T-zone but clipped to stop above lips (keeps nose/forehead, removes mouth/moustache)
    tx1, ty1, tx2, ty2 = rois["t_zone"]["box"]
    lx1, ly1, lx2, ly2 = lips_box
    tzu_box = (tx1, ty1, tx2, min(ty2, ly1))  # end at top of lips
    if tzu_box[3] > tzu_box[1] + 10:
        rois["t_zone_upper"] = {"box": tzu_box, "img": _crop(img, tzu_box)}

    # -----------------
    # Under-eye ROIs
    # -----------------
    def under_eye_box(eye_idxs, pad=0.35, shift_down=0.30, extra_down=0.55):
        ex1, ey1, ex2, ey2 = _bbox_from_points(_pts(lms, eye_idxs), w, h, pad=pad)
        height = (ey2 - ey1)
        y1n = int(min(h - 1, ey1 + height * shift_down))
        y2n = int(min(h, ey2 + height * (shift_down + extra_down)))
        return (ex1, y1n, ex2, y2n)

    under_l = under_eye_box(LEFT_EYE_LM)
    under_r = under_eye_box(RIGHT_EYE_LM)
    rois["under_eye_left"] = {"box": under_l, "img": _crop(img, under_l)}
    rois["under_eye_right"] = {"box": under_r, "img": _crop(img, under_r)}

    # Combined under-eye region
    ux1 = min(under_l[0], under_r[0])
    uy1 = min(under_l[1], under_r[1])
    ux2 = max(under_l[2], under_r[2])
    uy2 = max(under_l[3], under_r[3])
    under_box = (ux1, uy1, ux2, uy2)
    rois["under_eye"] = {"box": under_box, "img": _crop(img, under_box)}

    # -----------------
    # Cheeks (inner + mid)
    # -----------------
    cheek_pts = np.vstack([_pts(lms, LEFT_CHEEK_LM), _pts(lms, RIGHT_CHEEK_LM)])
    cheeks_box = _bbox_from_points(cheek_pts, w, h, pad=0.55)

    # Keep cheeks below under-eye so bags don't contaminate cheeks ROI
    cx1, cy1, cx2, cy2 = cheeks_box
    cy1 = max(cy1, under_box[3])
    cheeks_box = (cx1, cy1, cx2, cy2)
    rois["mid_cheeks"] = {"box": cheeks_box, "img": _crop(img, cheeks_box)}

    # Inner cheeks: narrower than mid_cheeks, closer to nose (useful for blackheads too)
    # derive from t_zone width, anchored around nose center
    nose_cx = int((nose_box[0] + nose_box[2]) / 2)
    inner_w = int((tzone_box[2] - tzone_box[0]) * 0.55)
    ix1 = max(0, nose_cx - inner_w // 2)
    ix2 = min(w, nose_cx + inner_w // 2)
    iy1 = max(0, cheeks_box[1])
    iy2 = min(h, cheeks_box[1] + int((cheeks_box[3] - cheeks_box[1]) * 0.65))
    inner_cheeks_box = (ix1, iy1, ix2, iy2)
    rois["inner_cheeks"] = {"box": inner_cheeks_box, "img": _crop(img, inner_cheeks_box)}

    # -----------------
    # Forehead ROI
    # -----------------
    forehead_box = _bbox_from_points(_pts(lms, FOREHEAD_LM), w, h, pad=0.60)

    # Keep forehead above eyes (avoid mixing with under-eye)
    fx1, fy1, fx2, fy2 = forehead_box
    fy2 = min(fy2, under_box[1])  # end at top of under-eye region
    if fy2 > fy1 + 5:
        forehead_box = (fx1, fy1, fx2, fy2)
        rois["forehead"] = {"box": forehead_box, "img": _crop(img, forehead_box)}

    # -----------------
    # Chin + Jaw ROI
    # -----------------
    chin_box = _bbox_from_points(_pts(lms, CHIN_LM), w, h, pad=0.50)
    rois["chin"] = {"box": chin_box, "img": _crop(img, chin_box)}

    jaw_box = _bbox_from_points(_pts(lms, JAW_LM), w, h, pad=0.35)
    rois["jaw"] = {"box": jaw_box, "img": _crop(img, jaw_box)}

    return rois


def extract_rois(face_img: Image.Image) -> Dict[str, Optional[Image.Image]]:
    """
    Convenience wrapper that returns only images:
      {"t_zone": PIL.Image, ...}
    """
    rois = extract_rois_with_boxes(face_img)
    return {k: v["img"] for k, v in rois.items() if isinstance(v, dict) and "img" in v}
