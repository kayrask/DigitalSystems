"""
api/routes/auth.py — Authentication routes: /register, /login
"""
from fastapi import APIRouter, HTTPException
from mysql.connector import Error, IntegrityError
from pydantic import BaseModel
from typing import Optional

from api.main import get_connection, hash_password, verify_password

router = APIRouter()


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str
    address: Optional[str] = None
    allergies: Optional[str] = None


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/register")
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


@router.post("/login")
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

        if not verify_password(data.password, pw_hash):
            raise HTTPException(status_code=400, detail="Invalid email or password")

        return {"id": user_id, "name": name, "email": email, "role": role or "user"}

    except Error as e:
        print("MySQL error in /login:", e)
        if conn is not None and conn.is_connected():
            conn.close()
        raise HTTPException(status_code=500, detail="Database error")
