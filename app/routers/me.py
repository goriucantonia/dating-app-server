"""GET /me and PATCH /me (S4-B4) — profile fields, preferences, and the
opt_in toggle.

DELETE /me arrived with Step 15 (S15-B1..B3), once the whole cascade graph
existed to be verified — see `app/deletion.py`.

Every A1 rule PATCH can touch is re-validated here with the same validators'
semantics as registration; opt_in changes get their own log line (§8 — a flag
is decorative until it is watched being consulted).
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

from fastapi import APIRouter
from pydantic import BaseModel, Field, field_validator, model_validator

from app.deletion import delete_account
from app.errors import ApiError
from app.logging_setup import log_event
from app.security import CurrentUser, DbSession
from app.users import GENDER_VALUES, UserOut, compute_age

router = APIRouter(tags=["me"])
logger = logging.getLogger("app.me")


class MePatch(BaseModel):
    """Everything editable this phase. Email and password changes are out of
    scope until a phase with verification/reset flows exists
    (new_user_creation.md decision 2: no reset flow, friends pool)."""

    display_name: str | None = Field(default=None, min_length=1, max_length=50)
    birth_date: date | None = None
    gender: str | None = None
    interested_in: list[str] | None = Field(default=None, min_length=1)
    age_pref_min: int | None = Field(default=None, ge=18)
    age_pref_max: int | None = None
    city: str | None = None
    country: str | None = None
    opt_in: bool | None = None

    @field_validator("birth_date")
    @classmethod
    def must_be_adult(cls, v: date | None) -> date | None:
        if v is not None and compute_age(v) < 18:
            raise ValueError("you must be at least 18")
        return v

    @field_validator("gender")
    @classmethod
    def known_gender(cls, v: str | None) -> str | None:
        if v is not None and v not in GENDER_VALUES:
            raise ValueError(f"must be one of: {', '.join(GENDER_VALUES)}")
        return v

    @field_validator("interested_in")
    @classmethod
    def known_interests(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        unknown = [g for g in v if g not in GENDER_VALUES]
        if unknown:
            raise ValueError(f"unknown values: {', '.join(unknown)}")
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def sane_age_range(self) -> MePatch:
        if (
            self.age_pref_min is not None
            and self.age_pref_max is not None
            and self.age_pref_max < self.age_pref_min
        ):
            raise ValueError("the maximum age can't be below the minimum")
        return self


@router.get("/me")
async def get_me(user: CurrentUser) -> UserOut:
    return UserOut.from_user(user)


@router.patch("/me")
async def patch_me(payload: MePatch, user: CurrentUser, session: DbSession) -> UserOut:
    changes = payload.model_dump(exclude_unset=True)
    # The age-range CHECK spans two fields; when only one arrives, validate it
    # against the stored other half before the database has to reject it.
    new_min = changes.get("age_pref_min", user.age_pref_min)
    new_max = changes.get("age_pref_max", user.age_pref_max)
    if new_max < new_min:
        raise ApiError(
            422, "validation_error", "The maximum age can't be below the minimum.",
            fields=[{"field": "age_pref_max", "message": "below the minimum"}],
        )

    if "opt_in" in changes and changes["opt_in"] != user.opt_in:
        log_event(
            logger, "opt_in_changed",
            user_id=str(user.id), was=user.opt_in, now=changes["opt_in"],
        )
    for field_name, value in changes.items():
        setattr(user, field_name, value)
    user.updated_at = datetime.now(UTC)
    session.add(user)
    await session.commit()
    log_event(logger, "me_patched", user_id=str(user.id), fields=sorted(changes))
    return UserOut.from_user(user)


class DeletionReceipt(BaseModel):
    """The per-table counts logged BEFORE the cascade ran (§19), returned so
    the person can see what went — the deletion trace without the data."""

    deleted: dict[str, int]
    rows_removed: int


@router.delete("/me", response_model=DeletionReceipt)
async def delete_me(user: CurrentUser, session: DbSession) -> DeletionReceipt:
    """S15-B1/B2/B3. One transaction, the cascade graph, counts first.

    The two cross-user effects (data_hygiene.md §2) are in the counts under
    `…_as_candidate` and `…_as_match`: dates and chats where this person was
    the OTHER party disappear from someone else's history. Named trade —
    privacy beats history."""
    counts = await delete_account(session, user)
    return DeletionReceipt(deleted=counts, rows_removed=sum(counts.values()))
