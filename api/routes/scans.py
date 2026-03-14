"""
api/routes/scans.py — Scan CRUD routes: POST /scans, GET /scans, GET /scans/{scan_id}
"""
from fastapi import APIRouter, HTTPException
from mysql.connector import Error
from pydantic import BaseModel
import json

from api.main import get_connection

router = APIRouter()


class ScanCreate(BaseModel):
    user_id: int
    results: dict


@router.post("/scans")
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


@router.get("/scans")
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


@router.get("/scans/{scan_id}")
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
