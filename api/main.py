from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from api.inference import SkinModel, SkinTypeModel, SkinToneModel, LABELS
from PIL import Image, ImageOps, ImageDraw, ImageFilter
from pydantic import BaseModel
from api.recommender import recommend_routine
from typing import Optional, List
from api.quality_gate import assess_image_quality, QualityConfig
from api.safety_filter import filter_routine_for_user
from api.outcome_risk import compute_outcome_and_risk
from api.explainability import gradcam_overlay_base64
from api.face_crop import crop_face, apply_oval_mask
from api.face_rois import extract_rois_with_boxes, apply_exclusion_mask_pil
from api.face_parse import FaceParser, apply_skin_mask
from api.yolo_detector import YoloSkinDetector
from datetime import datetime

import cv2
import numpy as np
import os
import io
import json
import base64

import hashlib
import bcrypt
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error, IntegrityError

# Load .env from project root (one level up from api/)
load_dotenv()

app = FastAPI()

# Rate limiter — unauthenticated callers: 10/min; authenticated (user_id present): 120/min
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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

        # Face angle check: a turned face has a narrower bounding box (width/height < 0.55)
        aspect = fw / max(fh, 1.0)
        out["metrics"]["face_aspect_ratio"] = round(aspect, 3)
        out["thresholds"]["face_aspect_ratio_min"] = 0.55
        if aspect < 0.55:
            out["reasons"].append(f"extreme_angle (face_aspect={aspect:.2f})")

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
            tips.append("Move to even, front-facing light. Avoid side windows or lamps.")

    # dedupe while preserving order
    seen = set()
    out = []
    for t in tips:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _apply_uncertainty_gating(results: dict, thr_map: dict, det_count: dict) -> tuple[dict, dict]:
    """
    Suppress weak positives and expose uncertainty diagnostics.
    """
    gated = {}
    uncertainty = {}
    for label, v in results.items():
        prob = float(v["probability"])
        thr = float(thr_map.get(label, 0.5))
        pred = int(v["prediction"])

        margin = abs(prob - thr)
        near_threshold = margin < 0.06
        # thr_map is the single source of truth — pred==1 already means prob>=thr,
        # so below_floor (prob<thr with pred==1) can never fire; removed.
        # weak_positive: suppress localized labels that only barely cleared the threshold.
        # Uses thr + 0.04 only — no hardcoded 0.45 floor that contradicted tuned thresholds.
        weak_positive = label in LOCAL and pred == 1 and prob < thr + 0.04
        yolo_disagree = label in LOCAL and pred == 1 and det_count.get(label, 0) == 0 and prob < 0.65

        suppressed = False
        reasons = []
        if near_threshold:
            reasons.append("near_threshold")
        if weak_positive:
            reasons.append("weak_positive")
        if yolo_disagree:
            reasons.append("yolo_disagree")

        suppress_reasons = {"weak_positive", "yolo_disagree"}
        if any(r in suppress_reasons for r in reasons):
            pred = 0
            suppressed = True

        gated[label] = {
            "probability": prob,
            "prediction": pred,
        }
        uncertainty[label] = {
            "threshold": thr,
            "floor": thr,
            "margin": margin,
            "suppressed": suppressed,
            "reasons": reasons,
            "confidence_band": (
                "high" if margin >= 0.15 else "medium" if margin >= 0.08 else "low"
            ),
        }
    return gated, uncertainty

# === MySQL CONFIG — loaded from .env ===
MYSQL_HOST = os.getenv("MYSQL_HOST", "localhost")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", "3306"))
MYSQL_USER = os.getenv("MYSQL_USER", "root")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD", "")
MYSQL_DB = os.getenv("MYSQL_DB", "aurai")




_db_pool: mysql.connector.pooling.MySQLConnectionPool | None = None


def _init_pool() -> mysql.connector.pooling.MySQLConnectionPool:
    return mysql.connector.pooling.MySQLConnectionPool(
        pool_name="aurai",
        pool_size=int(os.getenv("MYSQL_POOL_SIZE", "10")),
        pool_reset_session=True,
        host=MYSQL_HOST,
        port=MYSQL_PORT,
        user=MYSQL_USER,
        password=MYSQL_PASSWORD,
        database=MYSQL_DB,
    )


def get_connection():
    """Acquire a connection from the pool (lazy-initialises on first call)."""
    global _db_pool
    if _db_pool is None:
        _db_pool = _init_pool()
    return _db_pool.get_connection()


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
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against bcrypt hash. Falls back to SHA-256 for legacy accounts."""
    try:
        if stored_hash.startswith("$2b$") or stored_hash.startswith("$2a$"):
            return bcrypt.checkpw(password.encode("utf-8"), stored_hash.encode("utf-8"))
    except Exception:
        pass
    # Legacy SHA-256 fallback for accounts created before bcrypt migration
    import hashlib
    return hashlib.sha256(password.encode("utf-8")).hexdigest() == stored_hash


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
    Realistic cosmetic outcome simulation using OpenCV retouching techniques.
    - Acne/blackheads: cv2.inpaint (TELEA) fills blemishes with surrounding texture
    - Redness: HSV colorspace — reduce saturation only in red-hue skin pixels
    - Hyperpigmentation: LAB colorspace — lift L channel only (preserves chroma)
    - Bags: cv2.bilateralFilter on eye ROI (edge-preserving smooth)
    - Final: bilateral filter on skin area for natural smoothness
    Visual projection only — not a diagnosis.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    img = face_img.convert("RGB")
    img_np = np.array(img, dtype=np.uint8)          # HxWx3 RGB uint8
    h, iw = img_np.shape[:2]
    if h < 4 or iw < 4:
        return img, {"applied": False, "reason": "image_too_small"}

    # Work in BGR for all OpenCV ops; convert back at the end.
    bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # ── Skin mask (HSV + luma) used to restrict edits to skin regions ──────────
    arr_f = img_np.astype(np.float32) / 255.0
    mx = arr_f.max(axis=2)
    mn = arr_f.min(axis=2)
    sat_f = (mx - mn) / np.maximum(mx, 1e-6)
    lum_f = arr_f.mean(axis=2)
    skin_bool = (sat_f > 0.08) & (sat_f < 0.68) & (lum_f > 0.08) & (lum_f < 0.96)
    skin_mask_u8 = (skin_bool.astype(np.uint8)) * 255
    skin_mask_u8 = cv2.GaussianBlur(skin_mask_u8, (13, 13), 0)  # soft edges
    skin_w = skin_mask_u8.astype(np.float32) / 255.0             # 0-1 float

    # ── Concern probabilities (sqrt boost for visibility) ───────────────────────
    def _p(key):
        return float(np.sqrt(np.clip(concerns.get(key, 0.0), 0.0, 1.0)))
    def _p_raw(key):
        return float(np.clip(concerns.get(key, 0.0), 0.0, 1.0))

    acne_p_raw   = _p_raw("acne");       acne_p   = _p("acne")
    red_p_raw    = _p_raw("redness");    red_p    = _p("redness")
    pig_p_raw    = _p_raw("hyperpigmentation"); pig_p = _p("hyperpigmentation")
    black_p_raw  = _p_raw("blackheads"); black_p  = _p("blackheads")
    bags_p_raw   = _p_raw("bags");       bags_p   = _p("bags")

    texture_level = float(np.clip((0.62 * acne_p + 0.45 * black_p) * strength, 0.0, 0.40))
    red_level     = float(np.clip(red_p * strength * 0.46, 0.0, 0.34))
    pig_level     = float(np.clip(pig_p * strength * 0.50, 0.0, 0.32))
    bags_level    = float(np.clip(bags_p * strength * 0.32, 0.0, 0.22))

    if (acne_p_raw > 0.08 or black_p_raw > 0.08) and texture_level < 0.08 * strength:
        texture_level = 0.08 * strength
    if red_p_raw  > 0.08 and red_level   < 0.06 * strength: red_level   = 0.06 * strength
    if pig_p_raw  > 0.08 and pig_level   < 0.06 * strength: pig_level   = 0.06 * strength
    if bags_p_raw > 0.08 and bags_level  < 0.05 * strength: bags_level  = 0.05 * strength

    # ── 1) Acne / blackheads — cv2.inpaint (TELEA healing) ─────────────────────
    if texture_level > 1e-4:
        inpaint_mask = np.zeros((h, iw), dtype=np.uint8)
        has_yolo_boxes = False
        if isinstance(yolo_boxes, list):
            for b in yolo_boxes:
                cls = int(b.get("cls", -1))
                if cls not in {0, 1, 4, 8}:   # acne / blackheads / pores / whiteheads
                    continue
                conf = float(b.get("conf", 0.0))
                if conf < 0.08:
                    continue
                x1, y1, x2, y2 = [int(round(v)) for v in (b.get("box") or [0, 0, 0, 0])]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(iw, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    inpaint_mask[y1:y2, x1:x2] = 255
                    has_yolo_boxes = True

        if has_yolo_boxes:
            # Dilate slightly so inpainting fully covers each blemish edge.
            kernel = np.ones((7, 7), np.uint8)
            inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations=1)
            # Restrict inpainting to skin only — don't touch eyes/lips.
            inpaint_mask = cv2.bitwise_and(inpaint_mask, skin_mask_u8)
            # Blend: full inpaint at strength=1.0, partial at lower strengths.
            healed = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            blend_alpha = np.where(inpaint_mask > 0, texture_level, 0.0)[..., None]
            bgr = (bgr.astype(np.float32) * (1.0 - blend_alpha) +
                   healed.astype(np.float32) * blend_alpha).astype(np.uint8)
        else:
            # Fallback (no YOLO): gentle bilateral smooth on skin only.
            smooth = cv2.bilateralFilter(bgr, d=9, sigmaColor=50, sigmaSpace=50)
            alpha = (skin_w * texture_level * 0.6)[..., None]
            bgr = (bgr.astype(np.float32) * (1.0 - alpha) +
                   smooth.astype(np.float32) * alpha).astype(np.uint8)

    # ── 2) Redness — HSV: reduce S channel in red-hue skin pixels ───────────────
    if red_level > 1e-4:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        # OpenCV H range: 0-180. Red hue wraps: 0-15 and 165-180.
        h_ch = hsv[:, :, 0]
        red_px = ((h_ch <= 15) | (h_ch >= 165)) & (hsv[:, :, 1] > 40) & skin_bool
        # Reduce saturation — pull toward neutral without green/purple cast.
        hsv[:, :, 1][red_px] = np.clip(hsv[:, :, 1][red_px] * (1.0 - red_level * 0.65), 0, 255)
        # Slight brightness lift in those pixels for "calmed skin" look.
        hsv[:, :, 2][red_px] = np.clip(hsv[:, :, 2][red_px] * (1.0 + red_level * 0.08), 0, 255)
        bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # ── 3) Hyperpigmentation — LAB: lift L channel only in dark skin areas ──────
    if pig_level > 1e-4:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_ch = lab[:, :, 0]   # 0-255 in OpenCV LAB
        # Target uneven dark patches on skin (L < ~140 relative to skin baseline).
        dark_px = (l_ch < 140) & skin_bool
        # Smooth the selection to avoid hard edges.
        dark_f = dark_px.astype(np.float32)
        dark_f = cv2.GaussianBlur(dark_f, (11, 11), 0)
        lift_amount = pig_level * 0.55 * 60.0  # max ~33 L units at full strength
        lab[:, :, 0] = np.clip(l_ch + dark_f * lift_amount, 0, 255)
        bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    # ── 4) Under-eye bags — bilateral filter on eye ROI ─────────────────────────
    if bags_level > 1e-4:
        try:
            rois = extract_rois_with_boxes(img)
        except Exception:
            rois = {}
        for k in ("under_eye_left", "under_eye_right", "under_eye"):
            roi = rois.get(k)
            if not roi:
                continue
            ex1, ey1, ex2, ey2 = [int(v) for v in roi["box"]]
            ex1, ey1 = max(0, ex1), max(0, ey1)
            ex2, ey2 = min(iw, ex2), min(h, ey2)
            if ex2 <= ex1 or ey2 <= ey1:
                continue
            patch = bgr[ey1:ey2, ex1:ex2]
            # Bilateral preserves the fine lines/lashes; just softens dark circles.
            smooth_patch = cv2.bilateralFilter(patch, d=9, sigmaColor=60, sigmaSpace=60)
            alpha = float(bags_level) * 0.6
            bgr[ey1:ey2, ex1:ex2] = cv2.addWeighted(patch, 1.0 - alpha, smooth_patch, alpha, 0)

    # ── Final pass: light bilateral on skin for natural smoothness ───────────────
    smooth_final = cv2.bilateralFilter(bgr, d=7, sigmaColor=35, sigmaSpace=35)
    final_alpha = (skin_w * 0.30)[..., None]   # max 30% blend — keeps skin texture
    bgr = (bgr.astype(np.float32) * (1.0 - final_alpha) +
           smooth_final.astype(np.float32) * final_alpha).astype(np.uint8)

    # ── Convert back to PIL RGB and compute delta ────────────────────────────────
    out_np = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mean_delta = float(np.mean(np.abs(out_np.astype(np.float32) - img_np.astype(np.float32))) / 255.0)
    sim = Image.fromarray(out_np)
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
    init_db()
    # Warm up all models so first real request isn't slow (cold JIT / memory allocation)
    try:
        import numpy as np
        dummy = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        model.predict(dummy)
        skin_type_model.predict(dummy)
        skin_tone_model.predict(dummy)
        print("[startup] Model warm-up complete.")
    except Exception as e:
        print(f"[startup] Model warm-up failed (non-fatal): {e}")


# CORS — defaults to localhost dev ports; override via CORS_ORIGINS env var for production.
_cors_origins = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Load models ===
# NOTE: these relative paths assume you run uvicorn from PROJECT ROOT.
# If you run from inside /api folder, you may need to adjust paths.

MODEL_PATH = "models/best_multilabel_v2.pt"
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


def _predict_rate_key(request: Request) -> str:
    """Admins (role='admin' in DB) bypass the rate limit by returning a fixed key.
    Regular callers are keyed by IP address."""
    uid = request.query_params.get("user_id") or ""
    if uid:
        try:
            conn = get_connection()
            cur = conn.cursor()
            cur.execute("SELECT role FROM users WHERE id=%s", (int(uid),))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if row and str(row[0]).lower() == "admin":
                return "admin-unlimited"
        except Exception:
            pass
    return get_remote_address(request)


@app.post("/predict")
@limiter.limit("10/minute", key_func=_predict_rate_key)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    user_id: int | None = Form(None),
):
    contents = await file.read()

    # Reject oversized uploads before any processing
    MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # 15 MB
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image too large. Maximum size is 15 MB.")

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

    


    # === Phase 1 Quality Gate — must run on RAW face crop, not the oval-masked image ===
    # Masking artificially reduces contrast/blur variance and gives wrong readings.
    quality = assess_image_quality(face_img_raw, QualityConfig(min_short_side=240))
    quality = _add_face_pose_quality_checks(quality, face_meta, image_full)

    # Shadow gate: uneven left/right illumination degrades prediction quality
    shadow_score = compute_shadow_score(face_img_raw)
    quality["metrics"]["shadow_score"] = shadow_score
    quality["thresholds"]["shadow_max"] = 0.35
    if shadow_score > 0.35:
        quality["reasons"].append(f"uneven_lighting (shadow_score={shadow_score:.2f})")
        quality["passed"] = len(quality["reasons"]) == 0
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

    # Extract ROIs (auto from FaceMesh)
    try:
        rois = extract_rois_with_boxes(face_img_raw)
    except Exception as e:
        print("ROI extraction failed:", e)
        rois = {}
    exclusion_mask = rois.get("exclude_mouth_moustache_mask")
    # Base full-face prediction — TTA averages original + h-flip for stability
    results_full = model.predict_tta(
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
    for label in LABELS:
        probs = [float(results_full[label]["probability"])]

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

            roi_pred = model.predict_tta(
                roi_img,
                tone_group=tone_out["group"],
                tone_conf=tone_out["confidence"],
            )
            probs.append(float(roi_pred[label]["probability"]))

        if label in DIFFUSE:
            best_prob = float(np.mean(sorted(probs)[-2:]))
        else:
            best_prob = float(max(probs))

        thr = float(thr_map.get(label, 0.5))
        results[label] = {
            "probability": best_prob,
            "prediction": int(best_prob >= thr),
        }

    # --- YOLO detection on face crop (for sanity + localization) ---
    # conf=0.20: 0.10 was too noisy, producing false detections at 10% confidence
    try:
        yolo_boxes = yolo_detector.predict(
            face_img_raw,
            conf=0.20,
            iou=0.45,
            imgsz=512,
            max_det=100,
        )
    except Exception as e:
        print("YOLO predict failed:", e)
        yolo_boxes = []

    # Count detections by label
    det_count = {k: 0 for k in results.keys()}
    for b in yolo_boxes:
        for lbl, ids in YOLO_CLASS_IDS.items():
            if lbl in det_count and b["cls"] in ids and b["conf"] >= 0.20:
                det_count[lbl] += 1

    # Fusion: when YOLO finds nothing, reduce (not zero) the CNN probability.
    # Factor 0.50 (was 0.35) — less aggressive suppression to avoid missing real cases.
    for lbl in ["acne", "blackheads", "bags"]:
        if lbl in results and det_count.get(lbl, 0) == 0:
            results[lbl]["probability"] = float(results[lbl]["probability"] * 0.50)
            thr = float(thr_map.get(lbl, 0.5))
            results[lbl]["prediction"] = int(results[lbl]["probability"] >= thr)

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






    
@app.get("/explain")
def explain(scan_id: int, target: str):
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

        # If we have YOLO, use boxes for localized issues instead of Grad-CAM
        LOCALIZED = {"acne", "blackheads", "bags"}
        if target in LOCALIZED:
            yolo_boxes = yolo_detector.predict(
                face_img,
                conf=0.20,
                iou=0.45,
                imgsz=512,
                max_det=100,
            )
            ids = YOLO_CLASS_IDS.get(target, set())
            filtered = [b for b in yolo_boxes if b["cls"] in ids]
            
            # Draw boxes on face image and return as base64
            face_with_boxes = draw_boxes(
                face_img,
                filtered,
                line_width=3,
                box_color="red",
                label_color="red"
            )
            
            # Scale face image to fit within full image and overlay at face bbox location
            full_img_copy = full_img.copy()
            face_with_boxes_scaled = face_with_boxes.resize((bw, bh), Image.Resampling.LANCZOS)
            full_img_copy.paste(face_with_boxes_scaled, (bx, by))
            
            # Convert full image with boxes to base64
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
            
            return {
                "ok": True,
                "target": target,
                "type": "boxes",
                "source": "detector",
                "image_png_base64": full_b64,
                "detections": detections,
                "face_size": {"w": face_w, "h": face_h},
                "full_size": {"w": full_w, "h": full_h},
                "detection_count": len(detections),
            }

        skin_mask = face_parser.skin_mask(face_img)
        skin_only_img = apply_skin_mask(face_img, skin_mask)

        # 1) ROI-aware Grad-CAM (localized) or full-face Grad-CAM (diffuse)
        rois = extract_rois_with_boxes(face_img)
        exclusion_mask = rois.get("exclude_mouth_moustache_mask")
        roi_keys = LABEL_TO_ROIS.get(target, [])

        shadow_score = compute_shadow_score(face_img)
        shadow_present = shadow_score > 0.22

        # Build a transparent overlay canvas in face-image coordinates
        face_overlay_canvas = Image.new("RGBA", face_img.size, (0, 0, 0, 0))

        if not roi_keys:
            explain_img = skin_only_img
            if exclusion_mask is not None and target in SENSITIVE_TARGETS:
                explain_img = apply_exclusion_mask_pil(explain_img, exclusion_mask)
            out = gradcam_overlay_base64(model, explain_img, target_label=target, skin_mask=skin_mask)
            face_overlay_canvas = base64png_to_pil_rgba(out["overlay_png_base64"]).resize(
                face_img.size
            ).convert("RGBA")
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

                # Grad-CAM on ROI crop — pass cropped skin mask so overlay
                # stays on dermally-relevant pixels within the ROI
                roi_skin_mask = skin_mask[y1:y2, x1:x2]
                out = gradcam_overlay_base64(model, roi_img, target_label=target, skin_mask=roi_skin_mask)
                roi_overlay = base64png_to_pil_rgba(out["overlay_png_base64"])
                if exclude_crop is not None:
                    ov = np.array(roi_overlay)
                    h = min(ov.shape[0], exclude_crop.shape[0])
                    w = min(ov.shape[1], exclude_crop.shape[1])
                    ov[:h, :w, 3][exclude_crop[:h, :w].astype(bool)] = 0
                    roi_overlay = Image.fromarray(ov, mode="RGBA")

                # Resize overlay to ROI size and composite onto face canvas
                roi_w = max(1, int(x2 - x1))
                roi_h = max(1, int(y2 - y1))
                roi_overlay = roi_overlay.resize((roi_w, roi_h)).convert("RGBA")

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

        return {
            "ok": True,
            "target": target,
            "type": "overlay",
            "source": "classifier",
            "overlay_png_base64": full_b64,
        }

    except HTTPException:
        raise
    except Exception as e:
        print("Error in /explain:", e)
        raise HTTPException(status_code=500, detail="Explainability error")
    finally:
        if conn is not None and getattr(conn, "is_connected", lambda: False)():
            conn.close()


# === Route modules ===
from api.routes import auth, scans, users, admin

app.include_router(auth.router)
app.include_router(scans.router)
app.include_router(users.router)
app.include_router(admin.router)
