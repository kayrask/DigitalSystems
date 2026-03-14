"""
api/deps.py — Shared auth dependencies for route modules.

JWT auth is not yet implemented (see CLAUDE.md planned improvements).
This module holds the constants and verify_token stub so route files can
import them in one place when JWT is added.
"""
import os

# These will be used once JWT auth is implemented.
SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-me-before-production")
ALGORITHM: str = "HS256"


def verify_token(token: str) -> dict:
    """
    Placeholder token verifier — JWT not yet implemented.

    When JWT auth is added, this should:
      1. Decode the token using SECRET_KEY + ALGORITHM.
      2. Validate expiry.
      3. Return the decoded payload dict (e.g. {"user_id": ..., "role": ...}).
      4. Raise HTTPException(401) on invalid / expired tokens.
    """
    raise NotImplementedError("JWT auth not yet implemented — see CLAUDE.md planned improvements")
