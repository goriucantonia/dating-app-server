"""Shared account-domain pieces: the A1 value sets, age computation (age is
computed, never stored), and the user payload shape every /me-ish endpoint
returns. One place, so the registration validator, PATCH /me, and (later)
the matching hard filters can never disagree.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from app.models import User

GENDER_VALUES = ("man", "woman", "nonbinary", "other")


def compute_age(birth_date: date, today: date | None = None) -> int:
    today = today or date.today()
    had_birthday = (today.month, today.day) >= (birth_date.month, birth_date.day)
    return today.year - birth_date.year - (0 if had_birthday else 1)


class UserOut(BaseModel):
    """What the client sees of a user's own account. Never the password hash;
    age computed on the way out. is_demo is always present when a user is
    rendered (communication_protocol.md §6)."""

    id: str
    email: str
    display_name: str
    birth_date: date
    age: int
    gender: str
    interested_in: list[str]
    age_pref_min: int
    age_pref_max: int
    city: str | None
    country: str | None
    opt_in: bool
    is_demo: bool

    @classmethod
    def from_user(cls, user: User) -> UserOut:
        return cls(
            id=str(user.id),
            email=user.email,
            display_name=user.display_name,
            birth_date=user.birth_date,
            age=compute_age(user.birth_date),
            gender=user.gender,
            interested_in=list(user.interested_in),
            age_pref_min=user.age_pref_min,
            age_pref_max=user.age_pref_max,
            city=user.city,
            country=user.country,
            opt_in=user.opt_in,
            is_demo=user.is_demo,
        )
