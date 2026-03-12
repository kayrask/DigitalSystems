from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from api.inference import SkinModel, SkinTypeModel, SkinToneModel, LABELS
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from pydantic import BaseModel
from api.recommender import recommend_routine
from typing import Optional, List
from api.quality_gate import assess_image_quality, QualityConfig
from api.safety_filter import filter_routine_for_user
from api.outcome_risk import compute_outcome_and_risk
from api.explainability import (
    gradcam_overlay_base64,
    gradcam_cam_array,
    refine_gradcam_mask,
    make_red_overlay_from_mask,
    detector_pixel_mask,
)
from api.face_crop import crop_face, apply_oval_mask
from api.face_rois import extract_rois_with_boxes, apply_exclusion_mask_pil
from api.face_parse import FaceParser, apply_skin_mask
from api.yolo_detector import YoloSkinDetector
from datetime import datetime


import numpy as np
import os
import io
import hashlib
import json
import base64

import mysql.connector
from mysql.connector import Error, IntegrityError

app = FastAPI()

# --- Coordinate conversion helpers ---
def clamp_box(box, w, h):
    """Clamp bounding box to image bounds."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(0, min(w, int(round(x2))))
    y2 = max(0, min(h, int(round(y2))))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return [x1, y1, x2, y2]

def face_box_to_full(box_face, face_size, full_face_bbox):
    """
    Convert bounding box from face crop coordinates to full image coordinates.
    
    Args:
        box_face: [x1, y1, x2, y2] in face crop pixels
        face_size: (face_width, face_height)
        full_face_bbox: {"x": int, "y": int, "w": int, "h": int} of face crop in full image
        
    Returns:
        [X1, Y1, X2, Y2] in full image pixels
    """
    fw, fh = face_size
    bx = int(full_face_bbox["x"])
    by = int(full_face_bbox["y"])
    bw = int(full_face_bbox["w"])
    bh = int(full_face_bbox["h"])

    sx = bw / max(1, fw)
    sy = bh / max(1, fh)

    x1, y1, x2, y2 = box_face
    X1 = bx + x1 * sx
    Y1 = by + y1 * sy
    X2 = bx + x2 * sx
    Y2 = by + y2 * sy
    return [float(X1), float(Y1), float(X2), float(Y2)]

def draw_boxes(pil_img: Image.Image, boxes: List[dict], line_width: int = 3, label_color: str = "red", box_color: str = "red") -> Image.Image:
    """
    Draw bounding boxes with labels on a PIL image.
    
    Args:
        pil_img: PIL Image to draw on
        boxes: List of dicts with keys: "box" ([x1, y1, x2, y2]), "label" (str), "conf" (float)
        line_width: Thickness of box outline
        label_color: Color of text and box outline
        box_color: Color of box outline
        
    Returns:
        PIL Image (RGB) with boxes drawn
    """
    out = pil_img.convert("RGB").copy()
    d = ImageDraw.Draw(out)
    
    for b in boxes:
        x1, y1, x2, y2 = [int(v) for v in b["box"]]
        label = b.get("label", "unknown")
        conf = b.get("conf", 0.0)
        text_label = f"{label} {conf:.0%}"
        
        # Draw rectangle
        d.rectangle([x1, y1, x2, y2], outline=box_color, width=line_width)
        
        # Draw text with background
        text_y = max(0, y1 - 16)
        d.text((x1, text_y), text_label, fill=label_color)
    
    return out

# --- ROI map for region-aware inference/explainability ---
LABEL_TO_ROIS = {
    "blackheads": ["nose", "t_zone"],
    "bags": ["under_eye_left", "under_eye_right"],
}

DIFFUSE = {"acne", "redness", "hyperpigmentation"}
LOCAL = {"blackheads", "bags"}
SENSITIVE_TARGETS = {"acne", "redness", "hyperpigmentation"}
NO_EXCLUDE_ROIS = {"nose"}

# Per-label Grad-CAM refinement tuning (Week 3 calibration pass).
EXPLAIN_REFINE_CFG = {
    "acne": {"percentile": 84.0, "blur_radius": 1.8},
    "bags": {"percentile": 88.0, "blur_radius": 1.6},
    "blackheads": {"percentile": 89.0, "blur_radius": 1.6},
    "hyperpigmentation": {"percentile": 83.0, "blur_radius": 2.4},
    "redness": {"percentile": 82.0, "blur_radius": 2.6},
}

REGION_CONSISTENCY_CFG = {
    "roi_support_margin": 0.03,
    "roi_prob_cap_for_penalty": 0.72,
    "ignore_overlap_threshold": 0.55,
    "no_detector_prob_floor": 0.62,
    "penalty_low_roi_support": 0.82,
    "penalty_ignore_overlap": 0.75,
    "penalty_no_detector_support": 0.80,
}


def _load_week3_config() -> None:
    """
    Optional runtime override for Week 3 knobs.
    File: models/week3_consistency_config.json
    {
      "explain_refine_cfg": {...},
      "region_consistency_cfg": {...}
    }
    """
    cfg_path = os.path.join("models", "week3_consistency_config.json")
    if not os.path.exists(cfg_path):
        return
    try:
        with open(cfg_path, "r") as f:
            j = json.load(f)
        for k, v in (j.get("explain_refine_cfg") or {}).items():
            if isinstance(v, dict):
                EXPLAIN_REFINE_CFG[k] = {
                    "percentile": float(v.get("percentile", EXPLAIN_REFINE_CFG.get(k, {}).get("percentile", 86.0))),
                    "blur_radius": float(v.get("blur_radius", EXPLAIN_REFINE_CFG.get(k, {}).get("blur_radius", 2.0))),
                }
        for k, v in (j.get("region_consistency_cfg") or {}).items():
            if k in REGION_CONSISTENCY_CFG:
                REGION_CONSISTENCY_CFG[k] = float(v)
        print(f"[Week3] Loaded consistency config from {cfg_path}")
    except Exception as e:
        print(f"[Week3] Failed to load {cfg_path}: {e}")

# YOLO class id mapping (from your data.yaml)
YOLO_CLASS_IDS = {
    "acne": {0},
    "blackheads": {1},
    "hyperpigmentation": {2},  # Dark-Spots
    "bags": {5},               # Eyebags
    "redness": {7},            # Skin-Redness
}

def compute_shadow_score(face_img: Image.Image) -> float:
    """
    Simple left-right illumination imbalance score.
    Higher = stronger shadow/uneven lighting.
    """
    gray = np.asarray(face_img.convert("L")).astype(np.float32)
    h, w = gray.shape[:2]
    if w < 2:
        return 0.0
    left = gray[:, : w // 2].mean()
    right = gray[:, w // 2 :].mean()
    denom = max((left + right) / 2.0, 1.0)
    return float(abs(left - right) / denom)


def _add_face_pose_quality_checks(quality: dict, face_meta: dict, full_img: Image.Image) -> dict:
    """
    Adds simple distance/centering checks based on detected face bbox.
    """
    out = dict(quality)
    out.setdefault("reasons", [])
    out.setdefault("metrics", {})
    out.setdefault("thresholds", {})

    bbox = (face_meta or {}).get("bbox") or {}
    fx = float(bbox.get("x", 0))
    fy = float(bbox.get("y", 0))
    fw = float(bbox.get("w", 0))
    fh = float(bbox.get("h", 0))
    W, H = full_img.size

    if fw > 0 and fh > 0 and W > 0 and H > 0:
        face_area_ratio = (fw * fh) / float(max(1, W * H))
        face_cx = fx + fw / 2.0
        face_cy = fy + fh / 2.0
        img_cx = W / 2.0
        img_cy = H / 2.0
        center_offset = float(
            np.sqrt((face_cx - img_cx) ** 2 + (face_cy - img_cy) ** 2)
            / max(1.0, np.sqrt(img_cx**2 + img_cy**2))
        )

        out["metrics"]["face_area_ratio"] = face_area_ratio
        out["metrics"]["face_center_offset"] = center_offset
        out["thresholds"]["face_area_ratio_min"] = 0.10
        out["thresholds"]["face_center_offset_max"] = 0.22

        if face_area_ratio < 0.10:
            out["reasons"].append(f"face_too_small (face_area_ratio={face_area_ratio:.3f})")
        if center_offset > 0.22:
            out["reasons"].append(f"face_off_center (offset={center_offset:.3f})")

    out["passed"] = len(out["reasons"]) == 0
    return out


def _retake_guidance_from_quality(quality: dict) -> list[str]:
    reasons = quality.get("reasons", [])
    tips = []
    for r in reasons:
        if r.startswith("too_dark"):
            tips.append("Increase front lighting and avoid backlight.")
        elif r.startswith("too_bright"):
            tips.append("Avoid direct flash/sun glare; use softer, even light.")
        elif r.startswith("blurry"):
            tips.append("Hold phone steady and clean the camera lens.")
        elif r.startswith("low_contrast"):
            tips.append("Use neutral, even lighting to improve skin detail.")
        elif r.startswith("low_resolution"):
            tips.append("Move closer so your face fills most of the frame.")
        elif r.startswith("face_too_small"):
            tips.append("Move closer; keep your full face large and centered.")
        elif r.startswith("face_off_center"):
            tips.append("Center your face in the guide oval before scanning.")
        elif r.startswith("uneven_lighting"):
            tips.append("Use even front lighting and avoid strong side shadows.")
        elif r.startswith("landmarks_unstable"):
            tips.append("Keep your head straight and face the camera directly.")
        elif r.startswith("eye_region_occluded"):
            tips.append("Make sure both eyes and under-eye regions are clearly visible.")
        elif r.startswith("face_partially_out_of_frame"):
            tips.append("Keep your whole face inside the frame; avoid edge clipping.")
        elif r.startswith("skin_visible_too_low"):
            tips.append("Remove occlusions like hair, hand, or mask from facial skin area.")

    # dedupe while preserving order
    seen = set()
    out = []
    for t in tips:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _add_face_surface_quality_checks(
    quality: dict,
    face_img: Image.Image,
    skin_mask: np.ndarray | None,
    rois: dict | None,
) -> dict:
    """
    Adds occlusion/lighting/landmark stability checks on the raw face crop.
    """
    out = dict(quality)
    out.setdefault("reasons", [])
    out.setdefault("metrics", {})
    out.setdefault("thresholds", {})

    arr = np.asarray(face_img.convert("RGB")).astype(np.float32) / 255.0
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    h, w = gray.shape

    # Side-shadow / uneven illumination proxy.
    left = float(gray[:, : max(1, w // 2)].mean())
    right = float(gray[:, max(1, w // 2):].mean())
    denom = max(1e-4, (left + right) * 0.5)
    imbalance = abs(left - right) / denom
    out["metrics"]["lr_light_imbalance"] = float(imbalance)
    out["thresholds"]["lr_light_imbalance_max"] = 0.28
    if imbalance > 0.28:
        out["reasons"].append(f"uneven_lighting (lr_imbalance={imbalance:.3f})")

    # Skin visibility proxy for occlusion.
    if skin_mask is not None:
        sm = np.asarray(skin_mask).astype(np.float32)
        if sm.max() > 1.0:
            sm = sm / 255.0
        skin_visible_ratio = float(np.mean(sm > 0.5))
        out["metrics"]["skin_visible_ratio"] = skin_visible_ratio
        out["thresholds"]["skin_visible_ratio_min"] = 0.28
        if skin_visible_ratio < 0.28:
            out["reasons"].append(f"skin_visible_too_low (ratio={skin_visible_ratio:.3f})")

    rois = rois or {}
    left_eye = rois.get("under_eye_left")
    right_eye = rois.get("under_eye_right")
    if not left_eye or not right_eye:
        out["reasons"].append("landmarks_unstable")
    else:
        for side, roi in (("left", left_eye), ("right", right_eye)):
            x1, y1, x2, y2 = [int(v) for v in roi["box"]]
            if x1 <= 2 or y1 <= 2 or x2 >= (w - 2) or y2 >= (h - 2):
                out["reasons"].append(f"face_partially_out_of_frame ({side})")
                continue
            if skin_mask is not None:
                sm = np.asarray(skin_mask).astype(np.float32)
                if sm.max() > 1.0:
                    sm = sm / 255.0
                roi_sm = sm[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                if roi_sm.size > 0:
                    vis = float(np.mean(roi_sm > 0.5))
                    out["metrics"][f"under_eye_{side}_skin_ratio"] = vis
                    if vis < 0.20:
                        out["reasons"].append(f"eye_region_occluded ({side}, ratio={vis:.3f})")

    out["passed"] = len(out["reasons"]) == 0
    return out


def _box_ignore_overlap(box_xyxy: list[float], ignore_mask: np.ndarray) -> float:
    """
    Fraction of box pixels that overlap ignore mask.
    """
    if ignore_mask is None:
        return 0.0
    h, w = ignore_mask.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box_xyxy]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    area = float((x2 - x1) * (y2 - y1))
    if area <= 0:
        return 0.0
    m = ignore_mask[y1:y2, x1:x2]
    if m.size == 0:
        return 0.0
    mm = m.astype(np.float32)
    if mm.max() > 1.0:
        mm = mm / 255.0
    return float(np.mean(mm > 0.5))


def _apply_regional_consistency(
    results: dict,
    thr_map: dict,
    roi_evidence: dict,
    det_count: dict,
    yolo_boxes: list[dict],
    ignore_mask: np.ndarray | None,
    cfg: dict | None = None,
) -> tuple[dict, dict]:
    """
    Week 3: penalize inconsistent positives using ROI support and ignore-zone overlap.
    """
    cfg = cfg or REGION_CONSISTENCY_CFG
    out = {k: {"probability": float(v["probability"]), "prediction": int(v["prediction"])} for k, v in results.items()}
    diag = {}

    for label in ["blackheads", "bags"]:
        if label not in out:
            continue

        prob = float(out[label]["probability"])
        thr = float(thr_map.get(label, 0.5))
        pred = int(out[label]["prediction"])
        reasons = []
        roi_prob = roi_evidence.get(label)

        if pred == 1 and roi_prob is not None and roi_prob < (thr + cfg["roi_support_margin"]) and prob < cfg["roi_prob_cap_for_penalty"]:
            prob *= cfg["penalty_low_roi_support"]
            reasons.append("low_roi_support")

        # If detector boxes mostly live in ignored zones (brow/lip/beard), downweight.
        ids = YOLO_CLASS_IDS.get(label, set())
        overlaps = []
        for b in yolo_boxes or []:
            if int(b.get("cls", -1)) in ids:
                overlaps.append(_box_ignore_overlap(b.get("box", [0, 0, 0, 0]), ignore_mask))
        max_ov = max(overlaps) if overlaps else 0.0
        if pred == 1 and max_ov >= cfg["ignore_overlap_threshold"] and prob < cfg["roi_prob_cap_for_penalty"]:
            prob *= cfg["penalty_ignore_overlap"]
            reasons.append("high_ignore_zone_overlap")

        if pred == 1 and det_count.get(label, 0) == 0 and prob < max(cfg["no_detector_prob_floor"], thr + 0.08):
            prob *= cfg["penalty_no_detector_support"]
            reasons.append("no_detector_support")

        prob = float(np.clip(prob, 0.0, 1.0))
        out[label]["probability"] = prob
        out[label]["prediction"] = int(prob >= thr)
        diag[label] = {
            "roi_prob": None if roi_prob is None else float(roi_prob),
            "det_count": int(det_count.get(label, 0)),
            "max_ignore_overlap": float(max_ov),
            "reasons": reasons,
        }

    return out, diag


def _apply_uncertainty_gating(results: dict, thr_map: dict, det_count: dict) -> tuple[dict, dict]:
    """
    Suppress weak positives and expose uncertainty diagnostics.
    """
    gated = {}
    uncertainty = {}
    # Conservative floors for "present" decisions to avoid low-threshold false positives.
    min_positive_prob = {
        "acne": 0.55,
        "bags": 0.50,
        "blackheads": 0.45,
        "hyperpigmentation": 0.55,
        "redness": 0.55,
    }

    acne_recall_floor = 0.58

    for label, v in results.items():
        prob = float(v["probability"])
        thr = float(thr_map.get(label, 0.5))
        pred = int(v["prediction"])

        # Rescue high-confidence acne misses caused by aggressive tone thresholds.
        if label == "acne" and pred == 0 and prob >= acne_recall_floor:
            pred = 1

        margin = abs(prob - thr)
        near_threshold = margin < 0.06
        below_floor = pred == 1 and prob < float(min_positive_prob.get(label, 0.5))
        # keep this rule only for localized labels; diffuse labels should not be over-suppressed here
        weak_positive = label in LOCAL and pred == 1 and prob < max(0.45, thr + 0.04)
        yolo_disagree = label in LOCAL and pred == 1 and det_count.get(label, 0) == 0 and prob < 0.65

        suppressed = False
        reasons = []
        if near_threshold:
            reasons.append("near_threshold")
        if below_floor:
            reasons.append("below_label_floor")
        if weak_positive:
            reasons.append("weak_positive")
        if yolo_disagree:
            reasons.append("yolo_disagree")

        suppress_reasons = {"below_label_floor", "weak_positive", "yolo_disagree"}
        if any(r in suppress_reasons for r in reasons):
            pred = 0
            suppressed = True

        gated[label] = {
            "probability": prob,
            "prediction": pred,
        }
        uncertainty[label] = {
            "threshold": thr,
            "floor": float(min_positive_prob.get(label, 0.5)),
            "margin": margin,
            "suppressed": suppressed,
            "reasons": reasons,
            "confidence_band": (
                "high" if margin >= 0.15 else "medium" if margin >= 0.08 else "low"
            ),
        }
    return gated, uncertainty

# === MySQL CONFIG ===
MYSQL_HOST = "localhost"
MYSQL_PORT = 3306
MYSQL_USER = "root"              # <-- change if needed
MYSQL_PASSWORD = "kayra0505"     # <-- put your MySQL password here
MYSQL_DB = "aurai"               # the DB you created in Workbench




def get_connection():
    """Create a new MySQL connection."""
    return mysql.connector.connect(
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
    )


def init_db():
    """Create tables if they do not exist."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Users table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL UNIQUE,
                password_hash VARCHAR(64) NOT NULL,
                role VARCHAR(20) NOT NULL DEFAULT 'user',
                phone VARCHAR(50) NULL,
                age INT NULL,
                address TEXT NULL,
                allergies TEXT NULL
            )
            """
        )
        # Backwards compatibility for existing DBs created before role support.
        try:
            cur.execute("ALTER TABLE users ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'user'")
        except Exception:
            pass

        # Scans table
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scans (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id INT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                results_json TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            )
            """
        )

        # Admin annotation table (latest payload per scan)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_annotations (
                id INT AUTO_INCREMENT PRIMARY KEY,
                scan_id INT NOT NULL UNIQUE,
                user_id INT NULL,
                image_kind VARCHAR(32) NOT NULL DEFAULT 'face_raw',
                annotations_json LONGTEXT NOT NULL,
                notes TEXT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
            )
            """
        )

        conn.commit()
        cur.close()
    except Error as e:
        print("Error while initializing DB:", e)
    finally:
        if conn is not None and conn.is_connected():
            conn.close()


def hash_password(password: str) -> str:
    # For a uni project this is fine; in production use bcrypt/argon2
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def is_admin_user(user_id: Optional[int]) -> bool:
    if user_id is None:
        return False
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT role FROM users WHERE id=%s LIMIT 1", (user_id,))
        row = cur.fetchone()
        cur.close()
        if conn.is_connected():
            conn.close()
        return bool(row and str(row[0]).lower() == "admin")
    except Exception:
        if conn is not None and getattr(conn, "is_connected", lambda: False)():
            conn.close()
        return False


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    address: Optional[str] = None
    allergies: Optional[str] = None



class LoginRequest(BaseModel):
    email: str
    password: str


class ScanCreate(BaseModel):
    user_id: int
    results: dict

class AnnotationSave(BaseModel):
    scan_id: int
    user_id: Optional[int] = None
    image_kind: Optional[str] = "face_raw"
    boxes: List[dict]
    strokes: Optional[List[dict]] = None
    notes: Optional[str] = None

class AdminPromoteRequest(BaseModel):
    email: str
    code: str

class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    address: Optional[str] = None
    allergies: Optional[str] = None

UPLOAD_DIR = "uploads"

def save_upload_image(contents: bytes, suffix: str = "") -> str:
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    h = hashlib.sha256(contents).hexdigest()[:16]
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"scan_{ts}_{h}{suffix}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(contents)
    return path

def base64png_to_pil_rgb(b64: str) -> Image.Image:
    """Decode base64 PNG/JPEG into a PIL RGB image."""
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def base64png_to_pil_rgba(b64: str) -> Image.Image:
    """Decode base64 PNG into a PIL RGBA image (preserves alpha)."""
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGBA")

def pil_to_base64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def mask01_to_base64_png(mask01: np.ndarray) -> str:
    arr = np.asarray(mask01, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="L")
    return pil_to_base64_png(img)


def face_mask_to_full_mask01(mask_face: np.ndarray, bbox: dict, full_size_wh: tuple[int, int]) -> np.ndarray:
    fw, fh = int(full_size_wh[0]), int(full_size_wh[1])
    full = np.zeros((fh, fw), dtype=np.float32)
    bx, by = int(bbox["x"]), int(bbox["y"])
    bw, bh = int(bbox["w"]), int(bbox["h"])
    if bw <= 0 or bh <= 0:
        return full
    m = np.asarray(mask_face, dtype=np.float32)
    m_img = Image.fromarray((np.clip(m, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L")
    m_img = m_img.resize((bw, bh), resample=Image.Resampling.BILINEAR)
    m_arr = np.asarray(m_img, dtype=np.float32) / 255.0
    x2 = min(fw, bx + bw)
    y2 = min(fh, by + bh)
    w = max(0, x2 - bx)
    h = max(0, y2 - by)
    if w > 0 and h > 0:
        full[by:y2, bx:x2] = np.maximum(full[by:y2, bx:x2], m_arr[:h, :w])
    return full


def _get_concern_prob(results_obj: dict, label: str) -> float:
    """
    Extract concern probability from either:
    - new format: {"results": {"acne": {"probability": ...}}}
    - old format: {"acne": {"probability": ...}}
    """
    try:
        if isinstance(results_obj.get("results"), dict):
            v = results_obj["results"].get(label, {})
            if isinstance(v, dict) and "probability" in v:
                return float(v.get("probability") or 0.0)
        v2 = results_obj.get(label, {})
        if isinstance(v2, dict) and "probability" in v2:
            return float(v2.get("probability") or 0.0)
    except Exception:
        pass
    return 0.0


def _smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    if edge1 <= edge0:
        return np.zeros_like(x)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def simulate_expected_outcome(
    face_img: Image.Image,
    concerns: dict,
    yolo_boxes: Optional[list] = None,
    strength: float = 0.85,
) -> tuple[Image.Image, dict]:
    """
    Create a conservative cosmetic "expected outcome" simulation from predictions.
    This is a visual projection only (non-diagnostic).
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    img = face_img.convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    h, w = arr.shape[:2]
    if h < 4 or w < 4:
        return img, {"applied": False, "reason": "image_too_small"}

    # Lightweight skin-like mask from HSV + luma constraints.
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    lum = arr.mean(axis=2)
    skin_mask = ((sat > 0.08) & (sat < 0.68) & (lum > 0.08) & (lum < 0.96)).astype(np.float32)
    skin_mask = np.clip(
        np.asarray(
            Image.fromarray((skin_mask * 255).astype(np.uint8))
            .filter(ImageFilter.GaussianBlur(radius=6))
        ).astype(np.float32)
        / 255.0,
        0.0,
        1.0,
    )

    acne_p_raw = float(np.clip(concerns.get("acne", 0.0), 0.0, 1.0))
    red_p_raw = float(np.clip(concerns.get("redness", 0.0), 0.0, 1.0))
    pig_p_raw = float(np.clip(concerns.get("hyperpigmentation", 0.0), 0.0, 1.0))
    black_p_raw = float(np.clip(concerns.get("blackheads", 0.0), 0.0, 1.0))
    bags_p_raw = float(np.clip(concerns.get("bags", 0.0), 0.0, 1.0))

    # Non-linear boost so medium/low probabilities still produce visible (but bounded) preview changes.
    acne_p = float(np.sqrt(acne_p_raw))
    red_p = float(np.sqrt(red_p_raw))
    pig_p = float(np.sqrt(pig_p_raw))
    black_p = float(np.sqrt(black_p_raw))
    bags_p = float(np.sqrt(bags_p_raw))

    out = arr.copy()

    # Build localized acne/texture mask from YOLO boxes (or high-frequency fallback).
    acne_mask = np.zeros((h, w), dtype=np.float32)
    if isinstance(yolo_boxes, list):
        for b in yolo_boxes:
            cls = int(b.get("cls", -1))
            # Acne / blackheads / pores / whiteheads
            if cls not in {0, 1, 4, 8}:
                continue
            conf = float(b.get("conf", 0.0))
            if conf < 0.08:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in (b.get("box") or [0, 0, 0, 0])]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            acne_mask[y1:y2, x1:x2] = np.maximum(acne_mask[y1:y2, x1:x2], min(1.0, 0.35 + conf))
    if acne_mask.max() > 0:
        acne_mask = np.asarray(
            Image.fromarray((acne_mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=7))
        ).astype(np.float32) / 255.0
    else:
        # Fallback: estimate blemish-like high-frequency texture on skin only.
        blur_luma = np.asarray(
            Image.fromarray((lum * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.3))
        ).astype(np.float32) / 255.0
        highfreq = np.abs(lum - blur_luma)
        acne_mask = _smoothstep(highfreq, 0.02, 0.10) * skin_mask
        acne_mask = np.asarray(
            Image.fromarray((acne_mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=4))
        ).astype(np.float32) / 255.0

    # 1) Acne/texture improvement: targeted smoothing + tone evening only in acne mask.
    texture_level = np.clip((0.62 * acne_p + 0.45 * black_p) * strength, 0.0, 0.40)
    if (acne_p_raw > 0.08 or black_p_raw > 0.08) and texture_level < 0.08 * strength:
        texture_level = 0.08 * strength
    if texture_level > 1e-4:
        blur_rgb = np.asarray(
            Image.fromarray((arr * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=1.15))
        ).astype(np.float32) / 255.0
        texture_mask = np.clip(acne_mask * skin_mask, 0.0, 1.0)
        # Keep detail by using low blend and no global skin blur.
        smooth_w = (0.55 * texture_level) * texture_mask[..., None]
        out = out * (1.0 - smooth_w) + blur_rgb * smooth_w
        # Slightly even local contrast without washing out.
        local_luma = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
        out = out * (1.0 - 0.16 * texture_level * texture_mask[..., None]) + (
            local_luma[..., None] * (0.16 * texture_level * texture_mask[..., None])
        )

    # 2) Redness correction: reduce local saturation/red dominance without hue-casting.
    red_level = np.clip(red_p * strength * 0.46, 0.0, 0.34)
    if red_p_raw > 0.08 and red_level < 0.06 * strength:
        red_level = 0.06 * strength
    if red_level > 1e-4:
        r = out[..., 0]
        g = out[..., 1]
        b = out[..., 2]
        excess_red = np.clip(r - (0.52 * g + 0.48 * b), 0.0, 1.0)
        red_mask = _smoothstep(excess_red, 0.02, 0.22) * skin_mask
        # Pull toward local luminance to avoid green/purple color shift.
        local_luma = 0.299 * r + 0.587 * g + 0.114 * b
        sat_reduce = (0.62 * red_level) * red_mask
        out = out * (1.0 - sat_reduce[..., None]) + local_luma[..., None] * sat_reduce[..., None]
        # Slight brightness lift for "post-treatment" look.
        out = np.clip(out + (0.08 * red_level * red_mask[..., None]), 0.0, 1.0)

    # 3) Hyperpigmentation correction: lift darker areas while preserving chroma.
    pig_level = np.clip(pig_p * strength * 0.50, 0.0, 0.32)
    if pig_p_raw > 0.08 and pig_level < 0.06 * strength:
        pig_level = 0.06 * strength
    if pig_level > 1e-4:
        y = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
        dark_mask = _smoothstep(0.74 - y, 0.04, 0.33) * skin_mask
        dark_mask = np.asarray(
            Image.fromarray((dark_mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=5))
        ).astype(np.float32) / 255.0
        # Lighten toward white in a bounded way to keep skin tone realistic.
        lift = (pig_level * 0.55) * dark_mask[..., None]
        out = np.clip(out + lift * (1.0 - out), 0.0, 1.0)

    # 4) Under-eye softening based on ROI boxes when available (luma-only style).
    bags_level = np.clip(bags_p * strength * 0.32, 0.0, 0.22)
    if bags_p_raw > 0.08 and bags_level < 0.05 * strength:
        bags_level = 0.05 * strength
    if bags_level > 1e-4:
        try:
            rois = extract_rois_with_boxes(img)
        except Exception:
            rois = {}
        eye_mask = np.zeros((h, w), dtype=np.float32)
        for k in ("under_eye_left", "under_eye_right", "under_eye"):
            roi = rois.get(k)
            if not roi:
                continue
            x1, y1, x2, y2 = [int(v) for v in roi["box"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                eye_mask[y1:y2, x1:x2] = 1.0
        if eye_mask.max() > 0:
            eye_mask = np.asarray(
                Image.fromarray((eye_mask * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(radius=10))
            ).astype(np.float32) / 255.0
            # Gentle luma lift, avoid blur-heavy "beauty filter" look.
            local_luma = 0.299 * out[..., 0] + 0.587 * out[..., 1] + 0.114 * out[..., 2]
            w = (0.35 * bags_level) * eye_mask[..., None]
            out = out * (1.0 - w) + local_luma[..., None] * w
            out = np.clip(out + (0.08 * bags_level * eye_mask[..., None]), 0.0, 1.0)

    # Ensure a minimum visible difference for practical UX if concerns are present.
    mean_delta = float(np.mean(np.abs(out - arr)))
    concern_max = float(max(acne_p_raw, red_p_raw, pig_p_raw, black_p_raw, bags_p_raw))
    if concern_max >= 0.12 and mean_delta < 0.012:
        # No additional blur fallback: use tiny skin-only tonal lift to keep realism.
        out = np.clip(out + (0.010 * strength * skin_mask[..., None]), 0.0, 1.0)
        mean_delta = float(np.mean(np.abs(out - arr)))

    # Final detail-preserving sharpen so preview stays crisp, not "filtered".
    sim = Image.fromarray((np.clip(out, 0.0, 1.0) * 255).astype(np.uint8)).filter(
        ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3)
    )
    meta = {
        "applied": True,
        "strength": strength,
        "levels": {
            "texture": float(texture_level),
            "redness": float(red_level),
            "hyperpigmentation": float(pig_level),
            "bags": float(bags_level),
        },
        "mean_delta": mean_delta,
    }
    return sim, meta


@app.on_event("startup")
def on_startup():
    _load_week3_config()
    init_db()


# Allow all origins for now (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load models ===
# NOTE: these relative paths assume you run uvicorn from PROJECT ROOT.
# If you run from inside /api folder, you may need to adjust paths.

MODEL_PATH = "models/best_multilabel.pt"
THRESHOLDS_PATH = "models/per_class_thresholds.json"

SKIN_TYPE_MODEL_PATH = "models/skin_type_resnet18.pt"
SKIN_TONE_MODEL_PATH = "models/skin_tone_resnet18.pt"
FACE_PARSING_CKPT = "models/bisenet_face_parsing.pth"
YOLO_WEIGHTS_PATH = "models/yolo_skin_best.pt"

# Multilabel skin condition model (acne, redness, etc.)
model = SkinModel(
    model_path=MODEL_PATH,
    thresholds_path=THRESHOLDS_PATH,
)

# NEW: skin type model (normal/combination/oily/dry/sensitive)
skin_type_model = SkinTypeModel(
    checkpoint_path=SKIN_TYPE_MODEL_PATH
)

# NEW: skin tone model
skin_tone_model = SkinToneModel(
    checkpoint_path=SKIN_TONE_MODEL_PATH
)

# Face parsing (skin-only masking)
face_parser = FaceParser(ckpt_path=FACE_PARSING_CKPT)

# YOLO detector (localized lesions)
yolo_detector = YoloSkinDetector(YOLO_WEIGHTS_PATH, device="mps")


@app.get("/")
def root():
    return {"message": "Skin condition API is running."}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    user_id: int | None = Form(None),
):
    contents = await file.read()
    full_path = save_upload_image(contents, suffix="_full")  # save original bytes
    try:
        image_full = ImageOps.exif_transpose(Image.open(io.BytesIO(contents))).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    # === A) FACE CROP (kills background leakage) ===
    face_img, face_meta = crop_face(image_full, margin=0.35, min_confidence=0.6)
    if face_img is None:
        return {
            "ok": False,
            "face": {"face_found": False},
            "message": "No face detected. Please retake with your face centered and clearly visible.",
        }
    face_img_raw = face_img
    print("FULL:", image_full.size, "FACE:", face_img.size, "BBOX:", face_meta.get("bbox"))
    # --- Skin segmentation ---
    skin_mask = face_parser.skin_mask(face_img_raw)
    face_np = np.array(face_img_raw)
    skin_only = face_np * (skin_mask[..., None] > 0)
    skin_only_img = Image.fromarray(skin_only.astype("uint8"))
    # Skin tone prediction (internal use only)
    tone_out = skin_tone_model.predict(face_img_raw)
    # Save RAW face crop (unmasked) for ROI extraction/explainability
    buf_raw = io.BytesIO()
    face_img_raw.save(buf_raw, format="JPEG", quality=92)
    face_raw_bytes = buf_raw.getvalue()
    image_path_face_raw = save_upload_image(face_raw_bytes, suffix="_face_raw")

    # === NEW: Oval mask inside the cropped face region ===
    face_img = apply_oval_mask(skin_only_img, feather=18)

    # Save the CROPPED face image (masked; used for display)
    buf = io.BytesIO()
    face_img.save(buf, format="JPEG", quality=92)
    face_bytes = buf.getvalue()
    image_path = save_upload_image(face_bytes, suffix="_face")
    print("PREDICT saved image_path:", image_path)
    print("PREDICT exists:", os.path.exists(image_path))
    print("PREDICT saved image_path_face_raw:", image_path_face_raw)
    print("PREDICT exists raw:", os.path.exists(image_path_face_raw))
    print("UPLOAD_DIR:", UPLOAD_DIR)

    


    # Extract ROIs once (used by quality checks + inference + explainability).
    try:
        rois = extract_rois_with_boxes(face_img_raw)
    except Exception as e:
        print("ROI extraction failed:", e)
        rois = {}

    # === Phase 1 Quality Gate ===
    # Run on RAW face crop (not oval-masked) for more reliable metrics.
    quality = assess_image_quality(face_img_raw, QualityConfig(min_short_side=240))
    quality = _add_face_pose_quality_checks(quality, face_meta, image_full)
    quality = _add_face_surface_quality_checks(quality, face_img_raw, skin_mask, rois)
    retake_guidance = _retake_guidance_from_quality(quality)
    if not quality["passed"]:
        return {
            "ok": False,
            "quality": quality,
            "face": face_meta,
            "retake_guidance": retake_guidance,
            "message": "Image quality too low. Please retake using the guidance provided.",
            "image_path_full": full_path,
            "image_path_face": image_path,
            "image_path_face_raw": image_path_face_raw,
            "face_bbox": face_meta.get("bbox"),
        }

    # === Predict (ROI-aware, multi-scale) ===
    # Use RAW face crop for ROI extraction and inference (preserves texture detail)
    skin_type_out = skin_type_model.predict(face_img_raw)
    skin_type = skin_type_out["skin_type"]

    # Reuse precomputed ROIs
    exclusion_mask = rois.get("exclude_mouth_moustache_mask")
    # Base full-face prediction
    results_full = model.predict(
        skin_only_img,
        tone_group=tone_out.get("group"),
        tone_conf=tone_out.get("confidence"),
    )

    # Helper to match SkinModel threshold selection logic
    thr_map = model.thresholds
    if (
        model.thresholds_by_tone
        and tone_out.get("group")
        and tone_out.get("confidence") is not None
        and tone_out.get("confidence") >= model.tone_conf_min
        and tone_out.get("group") in model.thresholds_by_tone
    ):
        thr_map = model.thresholds_by_tone[tone_out.get("group")]
    elif model.thresholds_by_tone and "default" in model.thresholds_by_tone:
        thr_map = model.thresholds_by_tone["default"]

    # Aggregate probabilities using ROIs only for localized labels.
    results = {}
    roi_evidence = {}
    for label in LABELS:
        probs = [float(results_full[label]["probability"])]
        roi_probs = []

        for rk in LABEL_TO_ROIS.get(label, []):
            roi = rois.get(rk)
            if not roi:
                continue

            roi_img = roi["img"]
            if (
                exclusion_mask is not None
                and label in SENSITIVE_TARGETS
                and rk not in NO_EXCLUDE_ROIS
            ):
                x1, y1, x2, y2 = roi["box"]
                exclude_crop = exclusion_mask[y1:y2, x1:x2]
                roi_img = apply_exclusion_mask_pil(roi_img, exclude_crop)

            roi_pred = model.predict(
                roi_img,
                tone_group=tone_out["group"],
                tone_conf=tone_out["confidence"],
            )
            p_roi = float(roi_pred[label]["probability"])
            probs.append(p_roi)
            roi_probs.append(p_roi)

        if label in DIFFUSE:
            best_prob = float(np.mean(sorted(probs)[-2:]))
        else:
            best_prob = float(max(probs))

        thr = float(thr_map.get(label, 0.5))
        results[label] = {
            "probability": best_prob,
            "prediction": int(best_prob >= thr),
        }
        roi_evidence[label] = (float(max(roi_probs)) if roi_probs else None)

    # --- YOLO detection on face crop (for sanity + localization) ---
    try:
        yolo_boxes = yolo_detector.predict(
            face_img_raw,
            conf=0.10,
            iou=0.45,
            imgsz=512,
            max_det=100,
        )
    except Exception as e:
        print("YOLO predict failed:", e)
        yolo_boxes = []

    # Count detections by your labels
    det_count = {k: 0 for k in results.keys()}
    for b in yolo_boxes:
        for lbl, ids in YOLO_CLASS_IDS.items():
            if lbl in det_count and b["cls"] in ids and b["conf"] >= 0.10:
                det_count[lbl] += 1

    # Fusion: suppress acne/blackheads/bags when YOLO finds nothing
    for lbl in ["acne", "blackheads", "bags"]:
        if lbl in results and det_count.get(lbl, 0) == 0:
            results[lbl]["probability"] = float(results[lbl]["probability"] * 0.35)
            thr = float(thr_map.get(lbl, 0.5))
            results[lbl]["prediction"] = int(results[lbl]["probability"] >= thr)

    # Week 3: regional consistency penalties (ROI support + ignore-zone overlap)
    ignore_mask = rois.get("exclude_visual_ignore_mask")
    results, region_consistency = _apply_regional_consistency(
        results=results,
        thr_map=thr_map,
        roi_evidence=roi_evidence,
        det_count=det_count,
        yolo_boxes=yolo_boxes,
        ignore_mask=ignore_mask,
    )

    # Confidence/uncertainty gating to suppress weak detections
    results, uncertainty = _apply_uncertainty_gating(results, thr_map, det_count)

    mapped_skin_type = skin_type
    if mapped_skin_type in ["dry", "sensitive"]:
        mapped_skin_type = "normal"

    routine_raw = recommend_routine(results, skin_type=mapped_skin_type)

    # Phase 2 allergy safety filter (your existing code)
    allergies_text = None
    if user_id is not None:
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT allergies FROM users WHERE id=%s", (user_id,))
            row = cur.fetchone()
            cur.close()
            if conn.is_connected():
                conn.close()
            if row:
                allergies_text = row[0]
        except Exception as e:
            print("Failed to fetch allergies:", e)

    routine_filtered, safety = filter_routine_for_user(routine_raw, allergies_text)

    # Phase 3 outcome + risk (your existing function)
    phase3 = compute_outcome_and_risk(results, skin_type_out, routine_filtered)

    return {
        "ok": True,
        "face": face_meta,
        "quality": quality,
        "retake_guidance": retake_guidance,
        "results": results,
        "uncertainty": uncertainty,
        "region_consistency": region_consistency,
        "skin_type": skin_type_out,
        "routine_raw": routine_raw,
        "routine": routine_filtered,
        "safety": safety,
        "yolo": {
            "boxes": yolo_boxes,
            "counts": det_count,
        },
        **phase3,
        "image_path_full": full_path,
        "image_path_face": image_path,
        "image_path_face_raw": image_path_face_raw,
        "face_bbox": face_meta.get("bbox"),    
        }





# === Auth endpoints using MySQL ===

@app.post("/register")
def register(user: RegisterRequest):
    try:
        conn = get_connection()
        cur = conn.cursor()

        sql = """
        INSERT INTO users (name, email, password_hash, role, address, allergies)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cur.execute(
            sql,
            (
                user.name.strip(),
                user.email.strip().lower(),
                hash_password(user.password),
                "user",
                user.address,
                user.allergies,
            ),
        )

        conn.commit()
        user_id = cur.lastrowid

        cur.close()
        conn.close()

        return {
            "id": user_id,
            "name": user.name.strip(),
            "email": user.email.strip().lower(),
            "role": "user",
        }


    except IntegrityError:
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=400, detail="Email already registered")

    except Error as e:
        print("MySQL error in /register:", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")


@app.post("/login")
def login(data: LoginRequest):
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        sql = """
        SELECT id, name, email, password_hash, role
        FROM users
        WHERE email = %s
        """
        cur.execute(sql, (data.email.strip().lower(),))
        row = cur.fetchone()

        cur.close()
        if conn.is_connected():
            conn.close()

        if row is None:
            raise HTTPException(status_code=400, detail="Invalid email or password")

        user_id, name, email, pw_hash, role = row

        if pw_hash != hash_password(data.password):
            raise HTTPException(status_code=400, detail="Invalid email or password")

        return {"id": user_id, "name": name, "email": email, "role": role or "user"}

    except Error as e:
        print("MySQL error in /login:", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")


@app.post("/scans")
def save_scan(data: ScanCreate):
    """Save a scan result for a user."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        sql = """
        INSERT INTO scans (user_id, results_json)
        VALUES (%s, %s)
        """
        cur.execute(sql, (data.user_id, json.dumps(data.results)))
        conn.commit()
        scan_id = cur.lastrowid

        cur.close()
        if conn.is_connected():
            conn.close()

        return {"id": scan_id}

    except Error as e:
        print("MySQL error in /scans (POST):", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/scans")
def list_scans(user_id: int, limit: int = 10):
    """Return recent scans for a user (newest first)."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()

        sql = """
        SELECT id, created_at, results_json
        FROM scans
        WHERE user_id = %s
        ORDER BY created_at DESC
        LIMIT %s
        """
        cur.execute(sql, (user_id, limit))
        rows = cur.fetchall()

        cur.close()
        if conn.is_connected():
            conn.close()

        scans = []
        for row in rows:
            scan_id, created_at, results_json = row
            try:
                results = json.loads(results_json)
            except Exception:
                results = {}
            scans.append(
                {
                    "id": scan_id,
                    "created_at": created_at,
                    "results": results,
                }
            )

        return {"items": scans}

    except Error as e:
        print("MySQL error in /scans (GET):", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/scans/{scan_id}")
def get_scan(scan_id: int):
    """Return one scan record by id."""
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, user_id, created_at, results_json
            FROM scans
            WHERE id=%s
            LIMIT 1
            """,
            (scan_id,),
        )
        row = cur.fetchone()
        cur.close()
        if conn.is_connected():
            conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Scan not found")

        sid, uid, created_at, results_json = row
        try:
            results = json.loads(results_json or "{}")
        except Exception:
            results = {}

        return {"id": sid, "user_id": uid, "created_at": created_at, "results": results}

    except HTTPException:
        raise
    except Error as e:
        print("MySQL error in /scans/{scan_id}:", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/simulate_outcome")
def simulate_outcome(scan_id: int, user_id: int, strength: float = 0.9):
    """
    Returns a visual 'expected outcome' projection for a saved scan.
    This is a cosmetic simulation, not a medical prediction.
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, user_id, results_json
            FROM scans
            WHERE id=%s
            LIMIT 1
            """,
            (scan_id,),
        )
        row = cur.fetchone()
        cur.close()
        if conn.is_connected():
            conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Scan not found")

        sid, owner_id, results_json = row
        if int(owner_id) != int(user_id):
            raise HTTPException(status_code=403, detail="Not allowed")

        try:
            results_obj = json.loads(results_json or "{}")
        except Exception:
            results_obj = {}

        face_path = (
            results_obj.get("image_path_face_raw")
            or results_obj.get("image_path_face")
        )
        if not face_path or not os.path.exists(face_path):
            raise HTTPException(status_code=404, detail="Face image not found for this scan")

        face_img = ImageOps.exif_transpose(Image.open(face_path)).convert("RGB")
        concerns = {
            "acne": _get_concern_prob(results_obj, "acne"),
            "blackheads": _get_concern_prob(results_obj, "blackheads"),
            "redness": _get_concern_prob(results_obj, "redness"),
            "bags": _get_concern_prob(results_obj, "bags"),
            "hyperpigmentation": _get_concern_prob(results_obj, "hyperpigmentation"),
        }

        yolo_boxes = []
        try:
            yolo_boxes = (((results_obj or {}).get("yolo") or {}).get("boxes")) or []
        except Exception:
            yolo_boxes = []

        sim_img, sim_meta = simulate_expected_outcome(
            face_img=face_img,
            concerns=concerns,
            yolo_boxes=yolo_boxes,
            strength=strength,
        )

        return {
            "ok": True,
            "scan_id": sid,
            "type": "outcome_simulation",
            "disclaimer": "Visual simulation only. Not a diagnosis or guaranteed medical outcome.",
            "current_png_base64": pil_to_base64_png(face_img),
            "expected_png_base64": pil_to_base64_png(sim_img),
            "concerns": concerns,
            "meta": sim_meta,
        }
    except HTTPException:
        raise
    except Error as e:
        print("MySQL error in /simulate_outcome:", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")
    except Exception as e:
        print("Error in /simulate_outcome:", e)
        raise HTTPException(status_code=500, detail="Failed to generate outcome simulation")


@app.get("/admin/scan-image/{scan_id}")
def get_admin_scan_image(scan_id: int, user_id: int, kind: str = "face_raw"):
    """
    Returns scan image as base64 PNG for admin annotation UI.
    kind: face_raw | face | full
    """
    conn = None
    try:
        if not is_admin_user(user_id):
            raise HTTPException(status_code=403, detail="Admin access required")
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT results_json FROM scans WHERE id=%s", (scan_id,))
        row = cur.fetchone()
        cur.close()
        if conn.is_connected():
            conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Scan not found")

        results_obj = json.loads(row[0] or "{}")
        path_by_kind = {
            "face_raw": results_obj.get("image_path_face_raw"),
            "face": results_obj.get("image_path_face"),
            "full": results_obj.get("image_path_full"),
        }
        image_path = path_by_kind.get(kind) or results_obj.get("image_path_face_raw") or results_obj.get("image_path_face")
        if not image_path or not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Scan image not found on server")

        img = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        return {
            "ok": True,
            "scan_id": scan_id,
            "kind": kind,
            "image_png_base64": pil_to_base64_png(img),
            "size": {"w": img.size[0], "h": img.size[1]},
        }

    except HTTPException:
        raise
    except Exception as e:
        print("Error in /admin/scan-image:", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Failed to load scan image")


@app.get("/admin/users")
def list_admin_users(user_id: int, limit: int = 200):
    conn = None
    try:
        if not is_admin_user(user_id):
            raise HTTPException(status_code=403, detail="Admin access required")
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT
                u.id, u.name, u.email, u.role,
                COUNT(s.id) AS scan_count,
                MAX(s.created_at) AS last_scan_at
            FROM users u
            LEFT JOIN scans s ON s.user_id = u.id
            GROUP BY u.id, u.name, u.email, u.role
            ORDER BY last_scan_at DESC, u.id DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
        cur.close()
        if conn.is_connected():
            conn.close()

        items = []
        for r in rows:
            items.append(
                {
                    "id": r[0],
                    "name": r[1],
                    "email": r[2],
                    "role": r[3] or "user",
                    "scan_count": int(r[4] or 0),
                    "last_scan_at": r[5],
                }
            )
        return {"ok": True, "items": items}
    except HTTPException:
        raise
    except Error as e:
        print("MySQL error in /admin/users:", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/admin/users/{target_user_id}/scans")
def list_admin_user_scans(target_user_id: int, user_id: int, limit: int = 30):
    conn = None
    try:
        if not is_admin_user(user_id):
            raise HTTPException(status_code=403, detail="Admin access required")
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            SELECT id, name, email, role
            FROM users
            WHERE id=%s
            LIMIT 1
            """,
            (target_user_id,),
        )
        user_row = cur.fetchone()
        if not user_row:
            cur.close()
            if conn.is_connected():
                conn.close()
            raise HTTPException(status_code=404, detail="Target user not found")

        cur.execute(
            """
            SELECT id, created_at, results_json
            FROM scans
            WHERE user_id=%s
            ORDER BY created_at DESC
            LIMIT %s
            """,
            (target_user_id, limit),
        )
        rows = cur.fetchall()

        scan_ids = [int(r[0]) for r in rows]
        ann_by_scan = {}
        if scan_ids:
            placeholders = ",".join(["%s"] * len(scan_ids))
            cur.execute(
                f"""
                SELECT scan_id, annotations_json, notes, updated_at, user_id
                FROM scan_annotations
                WHERE scan_id IN ({placeholders})
                """,
                tuple(scan_ids),
            )
            for a in cur.fetchall():
                try:
                    ann_obj = json.loads(a[1] or "{}")
                except Exception:
                    ann_obj = {"boxes": []}
                ann_by_scan[int(a[0])] = {
                    "annotations": ann_obj,
                    "notes": a[2],
                    "updated_at": a[3],
                    "annotated_by_user_id": a[4],
                }

        cur.close()
        if conn.is_connected():
            conn.close()

        scans = []
        for row in rows:
            sid, created_at, results_json = row
            try:
                results = json.loads(results_json or "{}")
            except Exception:
                results = {}
            scans.append(
                {
                    "id": sid,
                    "created_at": created_at,
                    "results": results,
                    "annotation": ann_by_scan.get(int(sid)),
                }
            )

        return {
            "ok": True,
            "target_user": {
                "id": user_row[0],
                "name": user_row[1],
                "email": user_row[2],
                "role": user_row[3] or "user",
            },
            "items": scans,
        }
    except HTTPException:
        raise
    except Error as e:
        print("MySQL error in /admin/users/{target_user_id}/scans:", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")


@app.get("/admin/annotations")
def get_admin_annotations(scan_id: int, user_id: int):
    conn = None
    try:
        if not is_admin_user(user_id):
            raise HTTPException(status_code=403, detail="Admin access required")
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT scan_id, user_id, image_kind, annotations_json, notes, updated_at
            FROM scan_annotations
            WHERE scan_id=%s
            LIMIT 1
            """,
            (scan_id,),
        )
        row = cur.fetchone()
        cur.close()
        if conn.is_connected():
            conn.close()

        if not row:
            return {"ok": True, "scan_id": scan_id, "annotation": None}

        sid, uid, image_kind, annotations_json, notes, updated_at = row
        try:
            annotations = json.loads(annotations_json or "{}")
        except Exception:
            annotations = {"boxes": []}

        return {
            "ok": True,
            "scan_id": sid,
            "annotation": {
                "scan_id": sid,
                "user_id": uid,
                "image_kind": image_kind,
                "annotations": annotations,
                "notes": notes,
                "updated_at": updated_at,
            },
        }

    except Error as e:
        print("MySQL error in /admin/annotations (GET):", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")


@app.post("/admin/annotations")
def save_admin_annotations(data: AnnotationSave):
    conn = None
    try:
        if not is_admin_user(data.user_id):
            raise HTTPException(status_code=403, detail="Admin access required")
        conn = get_connection()
        cur = conn.cursor()

        annotations_json = json.dumps(
            {
                "boxes": data.boxes or [],
                "strokes": data.strokes or [],
            }
        )
        cur.execute(
            """
            INSERT INTO scan_annotations (scan_id, user_id, image_kind, annotations_json, notes)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                user_id = VALUES(user_id),
                image_kind = VALUES(image_kind),
                annotations_json = VALUES(annotations_json),
                notes = VALUES(notes)
            """,
            (data.scan_id, data.user_id, data.image_kind or "face_raw", annotations_json, data.notes),
        )

        conn.commit()
        cur.close()
        if conn.is_connected():
            conn.close()

        return {
            "ok": True,
            "scan_id": data.scan_id,
            "saved_boxes": len(data.boxes or []),
            "saved_strokes": len(data.strokes or []),
        }

    except Error as e:
        print("MySQL error in /admin/annotations (POST):", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")
    
@app.get("/explain")
def explain(scan_id: int, target: str, debug: bool = False):
    """
    Returns Grad-CAM overlay pasted onto the FULL image.
    target: acne | redness | blackheads | bags | hyperpigmentation
    """
    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT results_json FROM scans WHERE id=%s", (scan_id,))
        row = cur.fetchone()
        cur.close()
        if conn.is_connected():
            conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Scan not found")

        results_obj = json.loads(row[0] or "{}")

        full_path = results_obj.get("image_path_full")
        bbox = results_obj.get("face_bbox")
        print("EXPLAIN full_path:", full_path, "exists:", os.path.exists(full_path) if full_path else None)
        print("EXPLAIN face_raw_path:", results_obj.get("image_path_face_raw"), "exists:", os.path.exists(results_obj.get("image_path_face_raw")) if results_obj.get("image_path_face_raw") else None)
        print("EXPLAIN bbox:", bbox)
        print("CWD:", os.getcwd())
        

        if not full_path or not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Full scan image not found on server")
        if not bbox:
            raise HTTPException(status_code=404, detail="Face bbox missing for this scan")

        full_img = ImageOps.exif_transpose(Image.open(full_path)).convert("RGB")
        bx, by = int(bbox["x"]), int(bbox["y"])
        bw, bh = int(bbox["w"]), int(bbox["h"])
        # Always derive face crop from the full image + bbox to avoid oval-masked artifacts.
        face_img = full_img.crop((bx, by, bx + bw, by + bh)).convert("RGB")

        # If we have YOLO, use pixel-corrected localized overlays instead of Grad-CAM
        LOCALIZED = {"acne", "blackheads", "bags"}
        if target in LOCALIZED:
            yolo_boxes = yolo_detector.predict(
                face_img,
                conf=0.10,
                iou=0.45,
                imgsz=512,
                max_det=100,
            )
            ids = YOLO_CLASS_IDS.get(target, set())
            filtered = [b for b in yolo_boxes if b["cls"] in ids]
            detector_mask = detector_pixel_mask(face_img.size, filtered, expansion=0.20, blur_radius=2.2) if filtered else np.zeros((face_img.size[1], face_img.size[0]), dtype=np.float32)

            # Pixel-corrected mask: only detected bits get highlighted.
            # Still draw thin boxes for debugging/traceability.
            if filtered:
                mask_rgba = make_red_overlay_from_mask(detector_mask, strength=0.80)
                face_overlay = Image.alpha_composite(face_img.convert("RGBA"), mask_rgba)
                face_with_boxes = draw_boxes(
                    face_overlay.convert("RGB"),
                    filtered,
                    line_width=2,
                    box_color="red",
                    label_color="red",
                )
            else:
                face_with_boxes = face_img.convert("RGB")

            full_img_copy = full_img.copy()
            face_with_boxes_scaled = face_with_boxes.resize((bw, bh), Image.Resampling.LANCZOS)
            full_img_copy.paste(face_with_boxes_scaled, (bx, by))
            full_b64 = pil_to_base64_png(full_img_copy)
            
            # Also compute detection count for response
            face_w, face_h = face_img.size
            full_w, full_h = full_img.size
            detections = []
            for d in filtered:
                bf = clamp_box(d["box"], face_w, face_h)
                bfull = face_box_to_full(bf, (face_w, face_h), bbox)
                bfull = clamp_box(bfull, full_w, full_h)
                detections.append({
                    "label": d.get("label", f"class_{d['cls']}"),
                    "class_id": d["cls"],
                    "confidence": d["conf"],
                    "bbox_face": bf,
                    "bbox_full": bfull,
                })
            
            resp = {
                "ok": True,
                "target": target,
                "type": "boxes",
                "source": "detector_pixel",
                "image_png_base64": full_b64,
                "detections": detections,
                "face_size": {"w": face_w, "h": face_h},
                "full_size": {"w": full_w, "h": full_h},
                "detection_count": len(detections),
            }
            if debug:
                detector_full = face_mask_to_full_mask01(detector_mask, bbox, (full_w, full_h))
                resp["debug"] = {
                    "detector_mask_face_png_base64": mask01_to_base64_png(detector_mask),
                    "detector_mask_full_png_base64": mask01_to_base64_png(detector_full),
                }
            return resp

        skin_mask = face_parser.skin_mask(face_img)
        skin_only_img = apply_skin_mask(face_img, skin_mask)

        # 1) ROI-aware Grad-CAM (localized) or full-face Grad-CAM (diffuse)
        rois = extract_rois_with_boxes(face_img)
        exclusion_mask = rois.get("exclude_visual_ignore_mask")
        if exclusion_mask is None:
            exclusion_mask = rois.get("exclude_mouth_moustache_mask")
        roi_keys = LABEL_TO_ROIS.get(target, [])

        shadow_score = compute_shadow_score(face_img)
        shadow_present = shadow_score > 0.22

        # Build a transparent overlay canvas in face-image coordinates
        face_overlay_canvas = Image.new("RGBA", face_img.size, (0, 0, 0, 0))
        raw_cam_canvas = np.zeros((face_img.size[1], face_img.size[0]), dtype=np.float32)
        refined_canvas = np.zeros((face_img.size[1], face_img.size[0]), dtype=np.float32)

        if not roi_keys:
            explain_img = skin_only_img
            cam_exclude = None
            if exclusion_mask is not None and target in SENSITIVE_TARGETS:
                explain_img = apply_exclusion_mask_pil(explain_img, exclusion_mask)
                cam_exclude = exclusion_mask
            cam = gradcam_cam_array(model, explain_img, target_label=target)
            cfg = EXPLAIN_REFINE_CFG.get(target, {})
            refined = refine_gradcam_mask(
                cam,
                exclude_mask=cam_exclude,
                percentile=float(cfg.get("percentile", 86.0)),
                blur_radius=float(cfg.get("blur_radius", 2.0)),
            )
            raw_cam_canvas = np.maximum(raw_cam_canvas, cam)
            refined_canvas = np.maximum(refined_canvas, refined)
            face_overlay_canvas = make_red_overlay_from_mask(refined, strength=0.82).convert("RGBA")
        else:
            for rk in roi_keys:
                roi = rois.get(rk)
                if not roi:
                    continue
                x1, y1, x2, y2 = roi["box"]
                roi_img = skin_only_img.crop((x1, y1, x2, y2)).convert("RGB")
                exclude_crop = None
                if (
                    exclusion_mask is not None
                    and target in SENSITIVE_TARGETS
                    and rk not in NO_EXCLUDE_ROIS
                ):
                    exclude_crop = exclusion_mask[y1:y2, x1:x2]
                    roi_img = apply_exclusion_mask_pil(roi_img, exclude_crop)

                # Grad-CAM on ROI crop + refinement
                cam = gradcam_cam_array(model, roi_img, target_label=target)
                cfg = EXPLAIN_REFINE_CFG.get(target, {})
                refined = refine_gradcam_mask(
                    cam,
                    exclude_mask=exclude_crop,
                    percentile=float(cfg.get("percentile", 86.0)),
                    blur_radius=float(cfg.get("blur_radius", 2.0)),
                )
                roi_overlay = make_red_overlay_from_mask(refined, strength=0.82)

                # Resize overlay to ROI size and composite onto face canvas
                roi_w = max(1, int(x2 - x1))
                roi_h = max(1, int(y2 - y1))
                roi_overlay = roi_overlay.resize((roi_w, roi_h)).convert("RGBA")
                cam_resized = np.asarray(
                    Image.fromarray((np.clip(cam, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").resize(
                        (roi_w, roi_h), resample=Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                ) / 255.0
                refined_resized = np.asarray(
                    Image.fromarray((np.clip(refined, 0.0, 1.0) * 255.0).astype(np.uint8), mode="L").resize(
                        (roi_w, roi_h), resample=Image.Resampling.BILINEAR
                    ),
                    dtype=np.float32,
                ) / 255.0
                raw_cam_canvas[y1:y1 + roi_h, x1:x1 + roi_w] = np.maximum(
                    raw_cam_canvas[y1:y1 + roi_h, x1:x1 + roi_w], cam_resized
                )
                refined_canvas[y1:y1 + roi_h, x1:x1 + roi_w] = np.maximum(
                    refined_canvas[y1:y1 + roi_h, x1:x1 + roi_w], refined_resized
                )

                region_face = face_overlay_canvas.crop((x1, y1, x1 + roi_w, y1 + roi_h))
                region_face = Image.alpha_composite(region_face, roi_overlay)
                face_overlay_canvas.paste(region_face, (x1, y1))

        # Suppress hyperpigmentation overlays in strong shadow areas
        if shadow_present and target == "hyperpigmentation":
            gray = np.asarray(face_img.convert("L"))
            mean = float(gray.mean())
            shadow_mask = gray < (mean * 0.70)
            overlay_arr = np.array(face_overlay_canvas)
            overlay_arr[shadow_mask, 3] = 0
            face_overlay_canvas = Image.fromarray(overlay_arr, mode="RGBA")

        # 2) Paste the face overlay canvas back onto the FULL image using the stored face bbox
        full_rgba = full_img.convert("RGBA")
        face_overlay_resized = face_overlay_canvas.resize((bw, bh)).convert("RGBA")

        region_full = full_rgba.crop((bx, by, bx + bw, by + bh))
        region_full = Image.alpha_composite(region_full, face_overlay_resized)
        full_rgba.paste(region_full, (bx, by))

        # return base64 of composited full image overlay
        full_b64 = pil_to_base64_png(full_rgba)

        resp = {
            "ok": True,
            "target": target,
            "type": "overlay",
            "source": "classifier",
            "overlay_png_base64": full_b64,
        }
        if debug:
            full_w, full_h = full_img.size
            raw_full = face_mask_to_full_mask01(raw_cam_canvas, bbox, (full_w, full_h))
            refined_full = face_mask_to_full_mask01(refined_canvas, bbox, (full_w, full_h))
            dbg = {
                "raw_cam_face_png_base64": mask01_to_base64_png(raw_cam_canvas),
                "refined_mask_face_png_base64": mask01_to_base64_png(refined_canvas),
                "raw_cam_full_png_base64": mask01_to_base64_png(raw_full),
                "refined_mask_full_png_base64": mask01_to_base64_png(refined_full),
            }
            if exclusion_mask is not None:
                ex = np.clip(exclusion_mask.astype(np.float32), 0.0, 1.0)
                dbg["ignore_mask_face_png_base64"] = mask01_to_base64_png(ex)
                dbg["ignore_mask_full_png_base64"] = mask01_to_base64_png(
                    face_mask_to_full_mask01(ex, bbox, (full_w, full_h))
                )
            resp["debug"] = dbg
        return resp

    except HTTPException:
        raise
    except Exception as e:
        print("Error in /explain:", e)
        raise HTTPException(status_code=500, detail="Explainability error")
    finally:
        if conn is not None and getattr(conn, "is_connected", lambda: False)():
            conn.close()

@app.get("/me")
def get_me(user_id: int):
    try:
        conn = get_connection()
        cur = conn.cursor()

        sql = """
        SELECT id, name, email, role, phone, age, address, allergies
        FROM users
        WHERE id = %s
        """
        cur.execute(sql, (user_id,))
        row = cur.fetchone()

        cur.close()
        if conn.is_connected():
            conn.close()

        if row is None:
            raise HTTPException(status_code=404, detail="User not found")

        uid, name, email, role, phone, age, address, allergies = row
        return {
            "id": uid,
            "name": name,
            "email": email,
            "role": role or "user",
            "phone": phone,
            "age": age,
            "address": address,
            "allergies": allergies,
        }
    except Error as e:
        print("MySQL error in /me (GET):", e)
        raise HTTPException(status_code=500, detail="Database error")


@app.put("/me")
def update_me(user_id: int, data: ProfileUpdate):
    try:
        conn = get_connection()
        cur = conn.cursor()

        # Load existing so we can update only provided fields
        cur.execute(
            "SELECT name, email, role, phone, age, address, allergies FROM users WHERE id=%s",
            (user_id,),
        )
        row = cur.fetchone()
        if row is None:
            cur.close()
            if conn.is_connected():
                conn.close()
            raise HTTPException(status_code=404, detail="User not found")

        current = {
            "name": row[0],
            "email": row[1],
            "role": row[2] or "user",
            "phone": row[3],
            "age": row[4],
            "address": row[5],
            "allergies": row[6],
        }

        updated = {
            "name": current["name"],
            "email": current["email"],
            "phone": data.phone if data.phone is not None else current["phone"],
            "age": current["age"], 
            "address": data.address if data.address is not None else current["address"],
            "allergies": data.allergies if data.allergies is not None else current["allergies"],
        }

        # email uniqueness check if changed
        if updated["email"] != current["email"]:
            cur.execute("SELECT id FROM users WHERE email=%s AND id<>%s", (updated["email"], user_id))
            if cur.fetchone() is not None:
                cur.close()
                if conn.is_connected():
                    conn.close()
                raise HTTPException(status_code=400, detail="Email already in use")

        sql = """
        UPDATE users
        SET name=%s, email=%s, phone=%s, age=%s, address=%s, allergies=%s
        WHERE id=%s
        """
        cur.execute(
            sql,
            (
                updated["name"],
                updated["email"],
                updated["phone"],
                updated["age"],
                updated["address"],
                updated["allergies"],
                user_id,
            ),
        )
        conn.commit()

        cur.close()
        if conn.is_connected():
            conn.close()

        return {"ok": True, "user": {"id": user_id, "role": current["role"], **updated}}
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Email already in use")
    except Error as e:
        print("MySQL error in /me (PUT):", e)
        raise HTTPException(status_code=500, detail="Database error")


@app.post("/admin/bootstrap/promote")
def promote_user_to_admin(data: AdminPromoteRequest):
    """
    One-time/admin-only utility for local setup.
    Set ADMIN_PROMOTE_CODE in env and call this endpoint to promote a user by email.
    """
    expected = os.environ.get("ADMIN_PROMOTE_CODE")
    if not expected:
        raise HTTPException(status_code=500, detail="ADMIN_PROMOTE_CODE is not configured")
    if data.code != expected:
        raise HTTPException(status_code=403, detail="Invalid admin promotion code")

    conn = None
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET role='admin' WHERE email=%s",
            (data.email.strip().lower(),),
        )
        conn.commit()
        changed = cur.rowcount
        cur.close()
        if conn.is_connected():
            conn.close()

        if changed == 0:
            raise HTTPException(status_code=404, detail="User not found")
        return {"ok": True, "email": data.email.strip().lower(), "role": "admin"}
    except HTTPException:
        raise
    except Error as e:
        print("MySQL error in /admin/bootstrap/promote:", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")
