from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from api.inference import SkinModel, SkinTypeModel, SkinToneModel, LABELS
from PIL import Image, ImageOps
from pydantic import BaseModel
from api.recommender import recommend_routine
from typing import Optional, List
from api.quality_gate import assess_image_quality, QualityConfig
from api.safety_filter import filter_routine_for_user
from api.outcome_risk import compute_outcome_and_risk
from api.explainability import gradcam_overlay_base64
from api.face_crop import crop_face, apply_oval_mask
from api.face_rois import extract_rois_with_boxes  
from api.face_parse import FaceParser, apply_skin_mask
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

# --- ROI map for region-aware inference/explainability ---
LABEL_TO_ROIS = {
    "blackheads": ["nose", "t_zone"],
    "bags": ["under_eye_left", "under_eye_right"],
}

DIFFUSE = {"acne", "redness", "hyperpigmentation"}
LOCAL = {"blackheads", "bags"}

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
                phone VARCHAR(50) NULL,
                age INT NULL,
                address TEXT NULL,
                allergies TEXT NULL
            )
            """
        )

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


@app.on_event("startup")
def on_startup():
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

    


    # === Phase 1 Quality Gate should run on the face crop ===
    quality = assess_image_quality(face_img, QualityConfig(min_short_side=240))
    if not quality["passed"]:
        return {
            "ok": False,
            "quality": quality,
            "face": face_meta,
            "message": "Image quality too low. Please retake in better lighting and focus.",
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
    rois = extract_rois_with_boxes(face_img_raw)
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
    for label in LABELS:
        probs = [float(results_full[label]["probability"])]

        for rk in LABEL_TO_ROIS.get(label, []):
            roi = rois.get(rk)
            if not roi:
                continue

            roi_pred = model.predict(
                roi["img"],
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
        "results": results,
        "skin_type": skin_type_out,
        "routine_raw": routine_raw,
        "routine": routine_filtered,
        "safety": safety,
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
        INSERT INTO users (name, email, password_hash, address, allergies)
        VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(
            sql,
            (
                user.name.strip(),
                user.email.strip().lower(),
                hash_password(user.password),
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
        SELECT id, name, email, password_hash
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

        user_id, name, email, pw_hash = row

        if pw_hash != hash_password(data.password):
            raise HTTPException(status_code=400, detail="Invalid email or password")

        return {"id": user_id, "name": name, "email": email}

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
        face_path = results_obj.get("image_path_face")
        face_raw_path = results_obj.get("image_path_face_raw")
        bbox = results_obj.get("face_bbox")
        print("EXPLAIN full_path:", full_path, "exists:", os.path.exists(full_path) if full_path else None)
        print("EXPLAIN face_path:", face_path, "exists:", os.path.exists(face_path) if face_path else None)
        print("EXPLAIN bbox:", bbox)
        print("CWD:", os.getcwd())
        

        if not full_path or not os.path.exists(full_path):
            raise HTTPException(status_code=404, detail="Full scan image not found on server")
        # Prefer raw face crop for ROI extraction (avoid oval mask artifacts)
        if face_raw_path and os.path.exists(face_raw_path):
            face_path_to_use = face_raw_path
        else:
            face_path_to_use = face_path

        if not face_path_to_use or not os.path.exists(face_path_to_use):
            raise HTTPException(status_code=404, detail="Face scan image not found on server")
        if not bbox:
            raise HTTPException(status_code=404, detail="Face bbox missing for this scan")

        full_img = ImageOps.exif_transpose(Image.open(full_path)).convert("RGB")
        face_img = ImageOps.exif_transpose(Image.open(face_path_to_use)).convert("RGB")
        skin_mask = face_parser.skin_mask(face_img)
        skin_only_img = apply_skin_mask(face_img, skin_mask)

        # 1) ROI-aware Grad-CAM (localized) or full-face Grad-CAM (diffuse)
        rois = extract_rois_with_boxes(face_img)
        roi_keys = LABEL_TO_ROIS.get(target, [])

        shadow_score = compute_shadow_score(face_img)
        shadow_present = shadow_score > 0.22

        # Build a transparent overlay canvas in face-image coordinates
        face_overlay_canvas = Image.new("RGBA", face_img.size, (0, 0, 0, 0))

        if not roi_keys:
            out = gradcam_overlay_base64(model, skin_only_img, target_label=target)
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

                # Grad-CAM on ROI crop
                out = gradcam_overlay_base64(model, roi_img, target_label=target)
                roi_overlay = base64png_to_pil_rgba(out["overlay_png_base64"])

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
        bx, by = int(bbox["x"]), int(bbox["y"])
        bw, bh = int(bbox["w"]), int(bbox["h"])

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

@app.get("/me")
def get_me(user_id: int):
    try:
        conn = get_connection()
        cur = conn.cursor()

        sql = """
        SELECT id, name, email, phone, age, address, allergies
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

        uid, name, email, phone, age, address, allergies = row
        return {
            "id": uid,
            "name": name,
            "email": email,
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
            "SELECT name, email, phone, age, address, allergies FROM users WHERE id=%s",
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
            "phone": row[2],
            "age": row[3],
            "address": row[4],
            "allergies": row[5],
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

        return {"ok": True, "user": {"id": user_id, **updated}}
    except IntegrityError:
        raise HTTPException(status_code=400, detail="Email already in use")
    except Error as e:
        print("MySQL error in /me (PUT):", e)
        raise HTTPException(status_code=500, detail="Database error")
