"""POST /auth/register and POST /auth/login (S4-B1/B2) — the ONLY two
unauthenticated endpoints (communication_protocol.md §3).

Every A1 rule is mirrored as a pydantic validator so violations come back as
the 422 envelope with field-level detail (S4-B5) instead of a database error.
The database CHECKs remain the last line of defense, not the first.

Register returns a token (auto-login): the flow lands the new user directly
in onboarding without a second form.
"""

from __future__ import annotations

import logging
from datetime import date

from fastapi import APIRouter, Request
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.accounts import create_user
from app.errors import ApiError
from app.logging_setup import log_event
from app.models import User
from app.security import DbSession, create_token, hash_password, verify_password
from app.users import GENDER_VALUES, UserOut, compute_age

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger("app.auth")

# bcrypt hashes at most 72 BYTES. bcrypt 4 silently truncated; bcrypt 5 raises
# `ValueError` instead, which nothing caught, so a password-manager passphrase
# 500ed both register and login (audit 2026-09-02). Rejected at validation,
# with the unit named, because "36 accented characters" is already over.
MAX_PASSWORD_BYTES = 72

# Verified against when the email is unknown, so "no such account" and "wrong
# password" take the same time. The comment on login promised no account
# probing; the clock disagreed by ~200 ms (audit 2026-09-02).
_DUMMY_HASH = hash_password("timing-equaliser-not-a-real-password")


def check_password_bytes(value: str) -> str:
    if len(value.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise ValueError(
            f"a password can be at most {MAX_PASSWORD_BYTES} bytes long "
            "(accented letters and emoji count for more than one each)"
        )
    return value


class RegisterIn(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: str = Field(min_length=1, max_length=50)
    birth_date: date
    gender: str
    interested_in: list[str] = Field(min_length=1)
    age_pref_min: int = Field(default=18, ge=18, le=120)
    age_pref_max: int = Field(le=120)  # unbounded → int32 overflow → 500
    city: str | None = Field(default=None, max_length=100)
    country: str | None = Field(default=None, max_length=100)
    opt_in: bool = False  # default OFF (A1); registration never blocks on it

    # Strip the TEXT fields: a name of three spaces passed `min_length=1`
    # and the persona prompt then said "You are    ." (audit 2026-09-02).
    # Not the password — a model-wide strip hashed it stripped at
    # registration and compared it unstripped at login (review 2026-09-03).
    @field_validator("display_name", "city", "country", mode="before")
    @classmethod
    def _strip_text(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("password")
    @classmethod
    def _password_fits_bcrypt(cls, v: str) -> str:
        return check_password_bytes(v)

    @field_validator("birth_date")
    @classmethod
    def must_be_adult(cls, v: date) -> date:
        if compute_age(v) < 18:
            raise ValueError("you must be at least 18 to register")
        return v

    @field_validator("gender")
    @classmethod
    def known_gender(cls, v: str) -> str:
        if v not in GENDER_VALUES:
            raise ValueError(f"must be one of: {', '.join(GENDER_VALUES)}")
        return v

    @field_validator("interested_in")
    @classmethod
    def known_interests(cls, v: list[str]) -> list[str]:
        unknown = [g for g in v if g not in GENDER_VALUES]
        if unknown:
            raise ValueError(f"unknown values: {', '.join(unknown)}")
        return list(dict.fromkeys(v))  # dedupe, order kept

    @model_validator(mode="after")
    def sane_age_range(self) -> RegisterIn:
        if self.age_pref_max < self.age_pref_min:
            raise ValueError("the maximum age can't be below the minimum")
        return self


class AuthOut(BaseModel):
    token: str
    user: UserOut


class LoginIn(BaseModel):
    email: EmailStr
    password: str

    # Same bound as registration: an over-long password cannot match any
    # stored hash, and letting bcrypt raise turned "wrong password" into 500.
    @field_validator("password")
    @classmethod
    def _password_fits_bcrypt(cls, v: str) -> str:
        return check_password_bytes(v)


@router.post("/register", status_code=201)
async def register(payload: RegisterIn, request: Request, session: DbSession) -> AuthOut:
    # Case-insensitive on purpose: `Ana@x.com` and `ana@x.com` were two
    # accounts, and a phone keyboard capitalises the first letter (audit
    # 2026-09-02). Stored lowercased; matched lowercased so older rows still
    # resolve.
    email = payload.email.lower()
    existing = await session.scalar(
        select(User.id).where(func.lower(User.email) == email)
    )
    if existing is not None:
        log_event(logger, "register", outcome="email_taken", email=payload.email)
        raise ApiError(
            409, "email_taken", "That email is already registered — try signing in instead."
        )
    # The ONE creation path, shared with demo seeding (S15-B5, §16).
    try:
        user = await create_user(
            session,
            email=email, password=payload.password,
            display_name=payload.display_name, birth_date=payload.birth_date,
            gender=payload.gender, interested_in=payload.interested_in,
            age_pref_min=payload.age_pref_min, age_pref_max=payload.age_pref_max,
            city=payload.city, country=payload.country, opt_in=payload.opt_in,
        )
    except IntegrityError as exc:
        # Two registrations raced past the SELECT above (a double-tap, two
        # tabs). The first won; this one is the same 409 it would have got a
        # millisecond later, not a 500.
        await session.rollback()
        log_event(logger, "register", outcome="email_taken", email=email, raced=True)
        raise ApiError(
            409, "email_taken", "That email is already registered — try signing in instead."
        ) from exc
    token = create_token(user.id, request.app.state.settings.jwt_secret)
    return AuthOut(token=token, user=UserOut.from_user(user))


@router.post("/login")
async def login(payload: LoginIn, request: Request, session: DbSession) -> AuthOut:
    user = await session.scalar(
        select(User).where(func.lower(User.email) == payload.email.lower())
    )
    # One message for unknown email and wrong password — no account probing,
    # and the same amount of hashing either way.
    if user is None:
        verify_password(payload.password, _DUMMY_HASH)
    if user is None or not verify_password(payload.password, user.password_hash):
        log_event(
            logger, "login", level=logging.WARNING,
            outcome="invalid_credentials", email=payload.email,
        )
        raise ApiError(401, "invalid_credentials", "Email or password doesn't match.")
    log_event(logger, "login", outcome="ok", user_id=str(user.id))
    token = create_token(user.id, request.app.state.settings.jwt_secret)
    return AuthOut(token=token, user=UserOut.from_user(user))
