"""Creating a user — the ONE code path (S15-B5, §12, §16).

`POST /auth/register` and demo-profile seeding both come through here. A
demo profile is a real account created the way a person's is, with
`is_demo=True` as the only difference — there is no shortcut insert, because
a shortcut is a second registration path that the A1 validators, the log
line, and every later "how did this row get here" question do not cover.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date

from sqlalchemy.ext.asyncio import AsyncSession

from app.logging_setup import log_event
from app.models import User
from app.security import hash_password

logger = logging.getLogger("app.auth")


async def create_user(
    session: AsyncSession,
    *,
    email: str,
    password: str,
    display_name: str,
    birth_date: date,
    gender: str,
    interested_in: list[str],
    age_pref_min: int,
    age_pref_max: int,
    city: str | None = None,
    country: str | None = None,
    opt_in: bool = False,
    is_demo: bool = False,
) -> User:
    """Insert the account and log it. Validation (the A1 rules) is the
    caller's — pydantic at the endpoint, the seed loader for demo profiles."""
    user = User(
        email=email,
        password_hash=hash_password(password),
        display_name=display_name,
        birth_date=birth_date,
        gender=gender,
        interested_in=interested_in,
        age_pref_min=age_pref_min,
        age_pref_max=age_pref_max,
        city=city,
        country=country,
        opt_in=opt_in,
        is_demo=is_demo,
    )
    session.add(user)
    await session.commit()
    log_event(
        logger, "register", outcome="ok",
        user_id=str(user.id), opt_in=user.opt_in, gender=user.gender,
        is_demo=is_demo,
    )
    return user


def as_uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(value)
