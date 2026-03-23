"""
api/routes/admin.py — Admin routes:
  GET  /admin/scan-image/{scan_id}
  GET  /admin/users
  GET  /admin/users/{target_user_id}/scans
  GET  /admin/annotations
  POST /admin/annotations
  POST /admin/bootstrap/promote
"""
import os
import json
from fastapi import APIRouter, HTTPException
from mysql.connector import Error
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image, ImageOps

from api.main import get_connection, is_admin_user, pil_to_base64_png

router = APIRouter()


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


class RoleUpdateRequest(BaseModel):
    requester_id: int
    role: str  # "admin" or "user"


@router.get("/admin/scan-image/{scan_id}")
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


@router.get("/admin/users")
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


@router.get("/admin/users/{target_user_id}/scans")
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


@router.get("/admin/annotations")
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


@router.post("/admin/annotations")
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


@router.patch("/admin/users/{target_user_id}/role")
def update_user_role(target_user_id: int, data: RoleUpdateRequest):
    """Allow an admin to promote or demote another user's role."""
    conn = None
    if data.role not in ("admin", "user"):
        raise HTTPException(status_code=400, detail="Role must be 'admin' or 'user'")
    if target_user_id == data.requester_id:
        raise HTTPException(status_code=400, detail="Cannot change your own role")
    try:
        if not is_admin_user(data.requester_id):
            raise HTTPException(status_code=403, detail="Admin access required")
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM users WHERE id=%s LIMIT 1", (target_user_id,))
        if not cur.fetchone():
            raise HTTPException(status_code=404, detail="User not found")
        cur.execute("UPDATE users SET role=%s WHERE id=%s", (data.role, target_user_id))
        conn.commit()
        cur.close()
        if conn.is_connected():
            conn.close()
        return {"ok": True, "user_id": target_user_id, "role": data.role}
    except HTTPException:
        raise
    except Error as e:
        print("MySQL error in PATCH /admin/users/{id}/role:", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")


@router.post("/admin/bootstrap/promote")
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
