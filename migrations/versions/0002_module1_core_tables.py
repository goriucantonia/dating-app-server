"""Module 1 core tables — verbatim from module_1_data_collection.md A3 (S3-B1).

One deliberate deviation from the DOCUMENT ORDER, none from the schema:
`traits` is created BEFORE `questions`, because `questions.trait_id`
references `traits(id)` — a forward reference in the document. The document
is correct as a specification; it is not an execution order (PICKUP trap).

`profile_embeddings` is the REVISED two-vector form (S3-B2) — the A3 text
already carries it, restated in candidate_matching.md §3. The superseded
single-vector version is never created.

Revision ID: 0002
Revises: 0001
Create Date: 2026-09-01
"""
from __future__ import annotations

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # One op.execute per statement — asyncpg cannot prepare multi-command SQL.
    op.execute(
        """
CREATE TABLE users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email           TEXT NOT NULL UNIQUE,
    password_hash   TEXT NOT NULL,
    display_name    TEXT NOT NULL CHECK (char_length(display_name) BETWEEN 1 AND 50),
    birth_date      DATE NOT NULL,
    gender          TEXT NOT NULL CHECK (gender IN ('man','woman','nonbinary','other')),
    interested_in   TEXT[] NOT NULL CHECK (cardinality(interested_in) >= 1),
    age_pref_min    INT  NOT NULL DEFAULT 18 CHECK (age_pref_min >= 18),
    age_pref_max    INT  NOT NULL,
    city            TEXT,
    country         TEXT,
    opt_in          BOOLEAN NOT NULL DEFAULT FALSE,
    is_demo         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (age_pref_max >= age_pref_min)
);
"""
    )
    op.execute(
        """
CREATE TABLE traits (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    category          TEXT NOT NULL CHECK (category IN
                        ('interest','quality','flaw','behavioral','conversational_style','partner_preference')),
    label             TEXT NOT NULL,
    description       TEXT NOT NULL,
    confidence        REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
    status            TEXT NOT NULL DEFAULT 'inferred' CHECK (status IN
                        ('inferred','confirmed','disputed','corrected','retracted')),
    source_answer_ids UUID[] NOT NULL,
    extracted_by      TEXT NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
    )
    op.execute(
        """
CREATE TABLE questions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID REFERENCES users(id) ON DELETE CASCADE,
    origin      TEXT NOT NULL CHECK (origin IN ('baseline','pool','dispute')),
    code        TEXT UNIQUE,
    pool_order  INT UNIQUE,
    probe_area  TEXT NOT NULL CHECK (probe_area IN
                  ('interests','partner_criteria','situational','conversational','self_image')),
    text        TEXT NOT NULL,
    trait_id    UUID REFERENCES traits(id) ON DELETE CASCADE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK ((origin IN ('baseline','pool')) = (user_id IS NULL)),
    CHECK ((origin = 'pool') = (pool_order IS NOT NULL)),
    CHECK ((origin = 'dispute') = (trait_id IS NOT NULL))
);
"""
    )
    op.execute(
        """
CREATE TABLE answers (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    question_id UUID NOT NULL REFERENCES questions(id) ON DELETE CASCADE,
    answer_text TEXT NOT NULL CHECK (char_length(answer_text) >= 200),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, question_id)
);
"""
    )
    op.execute(
        """
CREATE TABLE trait_events (
    id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trait_id   UUID NOT NULL REFERENCES traits(id) ON DELETE CASCADE,
    event      TEXT NOT NULL CHECK (event IN
                 ('created','updated','disputed','corrected','confirmed','retracted')),
    detail     TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""
    )
    op.execute(
        """
CREATE TABLE profile_embeddings (
    user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    kind            TEXT NOT NULL CHECK (kind IN ('identity','preference')),
    embedding       vector(768) NOT NULL,
    embedding_model TEXT NOT NULL,
    traits_hash     TEXT NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (user_id, kind)
);
"""
    )


def downgrade() -> None:
    for table in (
        "profile_embeddings", "trait_events", "answers",
        "questions", "traits", "users",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table}")
