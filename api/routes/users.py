"""
api/routes/users.py — User profile routes: GET /me, PUT /me
"""
from fastapi import APIRouter, HTTPException
from mysql.connector import Error, IntegrityError
from pydantic import BaseModel
from typing import Optional

from api.main import get_connection, hash_password, verify_password

router = APIRouter()


class ProfileUpdate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    age: Optional[int] = None
    address: Optional[str] = None
    allergies: Optional[str] = None


@router.get("/me")
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


@router.put("/me")
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
            "age": data.age if data.age is not None else current["age"],
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


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@router.post("/me/change-password")
def change_password(user_id: int, data: ChangePasswordRequest):
    if len(data.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE id=%s", (user_id,))
        row = cur.fetchone()
        if row is None:
            cur.close()
            conn.close()
            raise HTTPException(status_code=404, detail="User not found")
        if not verify_password(data.current_password, row[0]):
            cur.close()
            conn.close()
            raise HTTPException(status_code=400, detail="Current password is incorrect.")
        new_hash = hash_password(data.new_password)
        cur.execute("UPDATE users SET password_hash=%s WHERE id=%s", (new_hash, user_id))
        conn.commit()
        cur.close()
        conn.close()
        return {"ok": True}
    except Error as e:
        print("MySQL error in /me/change-password:", e)
        raise HTTPException(status_code=500, detail="Database error")
