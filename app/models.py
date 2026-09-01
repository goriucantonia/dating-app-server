"""SQLAlchemy async models mirroring migration 0002 one-to-one (S3-B3).

Text + CHECK constraints rather than native enums, per the named trade in
module_1_data_collection.md A3 (ALTERing native enums during iteration is
painful). The database is the authority on constraints; these mirrors exist
so a model and its table can never drift silently.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    REAL,
    Boolean,
    CheckConstraint,
    Date,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
    text,
)

# The dialect ARRAY (not the generic one) — .contains() works only on it.
from sqlalchemy.dialects.postgresql import ARRAY, TIMESTAMP, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )


def _created_at() -> Mapped[datetime]:
    return mapped_column(
        TIMESTAMP(timezone=True), nullable=False, server_default=text("now()")
    )


class User(Base):
    __tablename__ = "users"
    __table_args__ = (
        CheckConstraint("char_length(display_name) BETWEEN 1 AND 50"),
        CheckConstraint("gender IN ('man','woman','nonbinary','other')"),
        CheckConstraint("cardinality(interested_in) >= 1"),
        CheckConstraint("age_pref_min >= 18"),
        CheckConstraint("age_pref_max >= age_pref_min"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    email: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    display_name: Mapped[str] = mapped_column(Text, nullable=False)
    birth_date: Mapped[date] = mapped_column(Date, nullable=False)
    gender: Mapped[str] = mapped_column(Text, nullable=False)
    interested_in: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    age_pref_min: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("18")
    )
    age_pref_max: Mapped[int] = mapped_column(Integer, nullable=False)
    city: Mapped[str | None] = mapped_column(Text)
    country: Mapped[str | None] = mapped_column(Text)
    opt_in: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    is_demo: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("FALSE")
    )
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _created_at()


class Trait(Base):
    __tablename__ = "traits"
    __table_args__ = (
        CheckConstraint(
            "category IN ('interest','quality','flaw','behavioral',"
            "'conversational_style','partner_preference')"
        ),
        CheckConstraint("confidence BETWEEN 0 AND 1"),
        CheckConstraint(
            "status IN ('inferred','confirmed','disputed','corrected','retracted')"
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    category: Mapped[str] = mapped_column(Text, nullable=False)
    label: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(REAL, nullable=False)
    status: Mapped[str] = mapped_column(
        Text, nullable=False, server_default=text("'inferred'")
    )
    source_answer_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False
    )
    extracted_by: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _created_at()


class Question(Base):
    __tablename__ = "questions"
    __table_args__ = (
        CheckConstraint("origin IN ('baseline','pool','dispute')"),
        CheckConstraint(
            "probe_area IN ('interests','partner_criteria','situational',"
            "'conversational','self_image')"
        ),
        CheckConstraint("(origin IN ('baseline','pool')) = (user_id IS NULL)"),
        CheckConstraint("(origin = 'pool') = (pool_order IS NOT NULL)"),
        CheckConstraint("(origin = 'dispute') = (trait_id IS NOT NULL)"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    origin: Mapped[str] = mapped_column(Text, nullable=False)
    code: Mapped[str | None] = mapped_column(Text, unique=True)
    pool_order: Mapped[int | None] = mapped_column(Integer, unique=True)
    probe_area: Mapped[str] = mapped_column(Text, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    trait_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("traits.id", ondelete="CASCADE")
    )
    created_at: Mapped[datetime] = _created_at()


class Answer(Base):
    __tablename__ = "answers"
    __table_args__ = (
        CheckConstraint("char_length(answer_text) >= 50"),
        UniqueConstraint("user_id", "question_id"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    question_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("questions.id", ondelete="CASCADE"), nullable=False
    )
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _created_at()


class TraitEvent(Base):
    __tablename__ = "trait_events"
    __table_args__ = (
        CheckConstraint(
            "event IN ('created','updated','disputed','corrected','confirmed','retracted')"
        ),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    trait_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("traits.id", ondelete="CASCADE"), nullable=False
    )
    event: Mapped[str] = mapped_column(Text, nullable=False)
    detail: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class ProfileEmbedding(Base):
    """The REVISED two-vector form (S3-B2): kind IN ('identity','preference'),
    PK (user_id, kind). The superseded single-vector version must never exist."""

    __tablename__ = "profile_embeddings"
    __table_args__ = (CheckConstraint("kind IN ('identity','preference')"),)

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    kind: Mapped[str] = mapped_column(Text, primary_key=True)
    embedding: Mapped[list[float]] = mapped_column(Vector(768), nullable=False)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    traits_hash: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[datetime] = _created_at()
