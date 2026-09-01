"""Password hashing and JWT sessions (S4-B2/B3).

- bcrypt for password_hash (module_1_data_collection.md A1).
- JWT bearer for every route except /auth/register and /auth/login
  (communication_protocol.md §3). No refresh tokens this phase — the named
  trade: users re-login when the token expires; accepted at friends scale.
  Token lifetime 7 days, chosen to make that re-login rare for a
  daily-use friends pool.
- A 401 anywhere means the session is dead; the UI's one interceptor routes
  to login.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated

import bcrypt
import jwt
from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.errors import ApiError
from app.models import User

TOKEN_LIFETIME = timedelta(days=7)
_ALGORITHM = "HS256"

_bearer = HTTPBearer(auto_error=False)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_token(user_id: uuid.UUID, secret: str) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {"sub": str(user_id), "iat": now, "exp": now + TOKEN_LIFETIME},
        secret,
        algorithm=_ALGORITHM,
    )


def _unauthenticated() -> ApiError:
    return ApiError(401, "unauthenticated", "You need to be signed in for this.")


async def get_current_user(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    session: Annotated[AsyncSession, Depends(get_session)],
) -> User:
    """The bearer dependency for every authenticated route. Expired, tampered,
    missing, or orphaned (user deleted) tokens all collapse to the same 401."""
    if credentials is None:
        raise _unauthenticated()
    try:
        payload = jwt.decode(
            credentials.credentials,
            request.app.state.settings.jwt_secret,
            algorithms=[_ALGORITHM],
        )
        user_id = uuid.UUID(payload["sub"])
    except (jwt.InvalidTokenError, KeyError, ValueError):
        raise _unauthenticated() from None
    user = await session.get(User, user_id)
    if user is None:
        raise _unauthenticated()
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
DbSession = Annotated[AsyncSession, Depends(get_session)]
