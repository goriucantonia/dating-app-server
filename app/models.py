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
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID
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


# --- Step 7: persona snapshots and calibration (migration 0004) -------------


class PersonaSnapshot(Base):
    """Immutable, per-user versioned. Recompiling creates v(n+1); old
    transcripts keep pointing at their own version, which is what lets a date
    stay explainable forever — "the agent said X because snapshot v3 said Y"
    (trait_persona.md §4). Nothing here is ever UPDATEd after 'ready'."""

    __tablename__ = "persona_snapshots"
    __table_args__ = (
        CheckConstraint("status IN ('compiling','ready','failed')"),
        UniqueConstraint("user_id", "version"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    # Nullable by design: the row is INSERTed as 'compiling' BEFORE the AI
    # call, so a compilation that dies mid-flight leaves evidence, not nothing.
    system_prompt: Mapped[str | None] = mapped_column(Text)
    schema_version: Mapped[str] = mapped_column(Text, nullable=False)
    traits_hash: Mapped[str] = mapped_column(Text, nullable=False)
    source_trait_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False
    )
    digest_model: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


class CalibrationSession(Base):
    """`snapshot_id` deliberately has NO cascade: a calibration session must
    always be able to name the persona version it was criticising."""

    __tablename__ = "calibration_sessions"

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("persona_snapshots.id"), nullable=False
    )
    created_at: Mapped[datetime] = _created_at()


class CalibrationMessage(Base):
    __tablename__ = "calibration_messages"
    __table_args__ = (
        CheckConstraint("sender IN ('user','persona')"),
        UniqueConstraint("session_id", "seq"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("calibration_sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    sender: Mapped[str] = mapped_column(Text, nullable=False)
    text_: Mapped[str] = mapped_column("text", Text, nullable=False)
    flagged: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    correction: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


# --- Step 9: analyses and candidates (migration 0005) ----------------------


class Analysis(Base):
    """ONE row per run, and the single object the UI polls for the whole
    journey. The status machine is owned jointly: matching drives
    `matching → matched | no_candidates | failed`; Step 11's simulation drives
    `matched → simulating → complete | failed`."""

    __tablename__ = "analyses"
    __table_args__ = (
        CheckConstraint(
            "status IN ('matching','matched','no_candidates','simulating',"
            "'complete','failed')"
        ),
        CheckConstraint("pool_status IN ('full','partial','empty')"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    status: Mapped[str] = mapped_column(Text, nullable=False)
    pool_status: Mapped[str | None] = mapped_column(Text)
    # Step 11's pipeline progress. `none_as_null=True` for the same reason as
    # `date_messages.state` (D-011): a NULL here means "no simulation has run",
    # and the JSON value `null` is not that.
    progress: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _created_at()


class AnalysisCandidate(Base):
    __tablename__ = "analysis_candidates"
    __table_args__ = (
        UniqueConstraint("analysis_id", "candidate_user_id"),
        UniqueConstraint("analysis_id", "rank"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    candidate_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    fit_forward: Mapped[float] = mapped_column(REAL, nullable=False)
    fit_backward: Mapped[float] = mapped_column(REAL, nullable=False)
    compatibility: Mapped[float] = mapped_column(REAL, nullable=False)
    shared_interests: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    reason_summary: Mapped[str] = mapped_column(Text, nullable=False)
    # No cascade, deliberately: this FREEZES the candidate's persona at match
    # time (trade #4), which only holds if the snapshot cannot vanish.
    snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("persona_snapshots.id"), nullable=False
    )
    created_at: Mapped[datetime] = _created_at()


# --- Step 11: dates and transcripts (migration 0006) -----------------------


class SimulatedDate(Base):
    """One simulated date. Named `SimulatedDate` rather than `Date` because
    `Date` is already a SQLAlchemy type in this file — a mirror class that
    shadows the column type it uses is a trap for the next reader.

    Both snapshot columns are frozen references with NO cascade, for the same
    reason `analysis_candidates.snapshot_id` is: a transcript must always be
    able to name the persona version that spoke it (date_simulation.md §3).
    """

    __tablename__ = "dates"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending','running','complete','incomplete','failed')"
        ),
        UniqueConstraint("analysis_id", "candidate_user_id", "ordinal"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("analyses.id", ondelete="CASCADE"), nullable=False
    )
    candidate_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    scenario: Mapped[dict] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    user_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("persona_snapshots.id"), nullable=False
    )
    candidate_snapshot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("persona_snapshots.id"), nullable=False
    )
    schema_version: Mapped[str] = mapped_column(Text, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()
    finished_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))


class DateMessage(Base):
    """The checkpoint row. `UNIQUE (date_id, seq)` is the invariant that makes
    resume idempotent: a re-launched date cannot write a second message at a
    seq it already wrote.

    `state` is NULL for `environment` rows — an event has no inner life, and a
    zeroed state block would read as an agent who felt nothing.
    """

    __tablename__ = "date_messages"
    __table_args__ = (
        CheckConstraint(
            "speaker IN ('user_agent','candidate_agent','environment')"
        ),
        UniqueConstraint("date_id", "seq"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    date_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("dates.id", ondelete="CASCADE"), nullable=False
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    speaker: Mapped[str] = mapped_column(Text, nullable=False)
    reply: Mapped[str] = mapped_column(Text, nullable=False)
    # `none_as_null=True` is load-bearing, not tidiness (D-011). Without it
    # SQLAlchemy writes Python `None` into a JSONB column as the JSON value
    # `null`, which is NOT SQL NULL: the row then fails `state IS NULL` while
    # looking null in every JSON payload and every Python read. The migration
    # and date_simulation.md §3 both say environment rows carry NULL here, and
    # Step 12's judging will filter spoken turns on exactly that.
    state: Mapped[dict | None] = mapped_column(JSONB(none_as_null=True))
    provider: Mapped[str | None] = mapped_column(Text)
    model_id: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = _created_at()


# --- Step 12: judging and scores (migration 0007) --------------------------


class DateEvaluation(Base):
    """One row per judged date. `date_id` is the PRIMARY KEY, which is what
    makes the judge pass idempotent: a relaunched pipeline that tries to judge
    an already-judged date collides instead of quietly scoring it twice.

    `criteria` (what the model said) and `date_score` (what the code computed)
    are both stored, deliberately. Either one alone is unauditable — see
    migration 0007."""

    __tablename__ = "date_evaluations"

    date_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("dates.id", ondelete="CASCADE"),
        primary_key=True,
    )
    criteria: Mapped[dict] = mapped_column(JSONB(none_as_null=True), nullable=False)
    clicked: Mapped[list] = mapped_column(JSONB(none_as_null=True), nullable=False)
    clashes: Mapped[list] = mapped_column(JSONB(none_as_null=True), nullable=False)
    per_peer: Mapped[dict] = mapped_column(JSONB(none_as_null=True), nullable=False)
    verdict: Mapped[str] = mapped_column(Text, nullable=False)
    date_score: Mapped[float] = mapped_column(REAL, nullable=False)
    is_partial: Mapped[bool] = mapped_column(
        Boolean, nullable=False, server_default=text("false")
    )
    judge_provider: Mapped[str] = mapped_column(Text, nullable=False)
    judge_model: Mapped[str] = mapped_column(Text, nullable=False)
    rubric_version: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = _created_at()


class CandidateScore(Base):
    """The per-candidate number, aggregated in code from the date scores.

    `dates_completed` and `dates_incomplete` sit beside `final_score` so the
    weighting is checkable by hand: an incomplete-but-judged date counts half
    (date_simulation.md, locked #3), and you cannot verify that from the score
    alone."""

    __tablename__ = "candidate_scores"

    analysis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("analyses.id", ondelete="CASCADE"),
        primary_key=True,
    )
    candidate_user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        primary_key=True,
    )
    final_score: Mapped[float] = mapped_column(REAL, nullable=False)
    dates_completed: Mapped[int] = mapped_column(Integer, nullable=False)
    dates_incomplete: Mapped[int] = mapped_column(Integer, nullable=False)
