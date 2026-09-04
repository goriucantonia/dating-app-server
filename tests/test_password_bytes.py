"""bcrypt's 72-byte ceiling, enforced at validation (audit 2026-09-02).

bcrypt 5 raises on a longer password where bcrypt 4 silently truncated. The
image carries 5, nothing caught the ValueError, and a password-manager
passphrase 500ed both register and login. These pin the boundary in BYTES —
the unit that matters and the one a character count hides.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.routers.auth import MAX_PASSWORD_BYTES, LoginIn, check_password_bytes


def test_exactly_72_ascii_bytes_passes():
    assert check_password_bytes("a" * 72) == "a" * 72


def test_73_ascii_bytes_rejected():
    with pytest.raises(ValueError, match="72 bytes"):
        check_password_bytes("a" * 73)


def test_bytes_not_characters():
    # 25 emoji are 25 characters and 100 bytes — over the limit the hash sees.
    emoji = "🔑" * 25
    assert len(emoji) < MAX_PASSWORD_BYTES
    with pytest.raises(ValueError):
        check_password_bytes(emoji)


def test_login_model_rejects_with_field_error():
    with pytest.raises(ValidationError) as info:
        LoginIn(email="a@example.com", password="x" * 80)
    errors = info.value.errors()
    assert errors and errors[0]["loc"] == ("password",)


def test_login_model_accepts_ordinary_password():
    assert LoginIn(email="a@example.com", password="correct horse").password == "correct horse"
