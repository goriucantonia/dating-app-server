"""Candidate matching: embeddings, hard filters, two-vector scoring (S9-B2…B8, B11).

Three things in here are load-bearing and easy to quietly break:

**Reasons are COMPUTED, never generated** (trade #3, §10). `shared_interests`
is a real set intersection over `interest` trait labels, and `reason_summary`
is assembled in code from numbers we already hold. A model writing "you both
love hiking" about two people who never mentioned hiking is the canonical
failure this module exists to forbid — so no AI call touches the reasons, and
there is no place in this file for one to be added later without noticing.

**The pool is never padded.** 3+ eligible → top 3 (`full`); 1–2 → those
(`partial`); 0 → `no_candidates` with a plain sentence. An ineligible person
shown to fill a gap is a lie that costs someone a real evening.

**Every filter step logs its surviving count** (S9-B11), including on runs that
succeed. "Why am I getting no matches" is otherwise unanswerable, and §5 says
fix the blindness before the bug. On an empty pool the run also names WHICH
step emptied it.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.base import AIError
from app.ai.routing import TaskRouter
from app.logging_setup import log_event
from app.models import Analysis, AnalysisCandidate, ProfileEmbedding, Trait
from app.persona import get_current_snapshot
from app.traits_hash import compute_traits_hash

logger = logging.getLogger("app.matching")

MAX_CANDIDATES = 3

# The identity vector is who you are; the preference vector is who you want.
# Category order is fixed so the serialisation is byte-stable (S9-B2).
IDENTITY_CATEGORIES = (
    "interest",
    "quality",
    "flaw",
    "behavioral",
    "conversational_style",
)
PREFERENCE_CATEGORIES = ("partner_preference",)


@dataclass
class FilterFunnel:
    """The §7 evidence. Every step records what survived it, so a run that
    finds nobody can say which step was responsible instead of shrugging."""

    opted_in: int = 0
    gender_fit: int = 0
    age_fit: int = 0
    snapshot_ready: int = 0
    embedding_fresh: int = 0

    def emptied_at(self) -> str | None:
        """The first step whose output was zero — the answer to 'why no
        matches'. Order matters and mirrors the SQL."""
        for name, n in (
            ("opt_in", self.opted_in),
            ("mutual_gender", self.gender_fit),
            ("mutual_age", self.age_fit),
            ("ready_snapshot", self.snapshot_ready),
            ("fresh_embedding", self.embedding_fresh),
        ):
            if n == 0:
                return name
        return None

    def as_log_fields(self) -> dict:
        return {
            "pool_opt_in": self.opted_in,
            "pool_mutual_gender": self.gender_fit,
            "pool_mutual_age": self.age_fit,
            "pool_ready_snapshot": self.snapshot_ready,
            "pool_fresh_embedding": self.embedding_fresh,
        }


def serialise_traits(traits: list[Trait], categories: tuple[str, ...]) -> str:
    """S9-B2. Deterministic: the same trait set always produces byte-identical
    input, so an unchanged profile never re-embeds to a slightly different
    vector and never looks like it drifted.

    Category-ordered, then by label within a category — NOT by row id or
    created_at, because those change when a trait is retracted and re-added
    while the person has not changed at all.
    """
    lines: list[str] = []
    for category in categories:
        rows = sorted(
            (t for t in traits if t.category == category and t.status != "retracted"),
            key=lambda t: (t.label.casefold(), t.description.casefold()),
        )
        lines.extend(f"{t.label}: {t.description}" for t in rows)
    return "\n".join(lines)


def _tokens(label: str) -> set[str]:
    """Case-folded word tokens, minus filler. Used ONLY for shared interests,
    where a match must be defensible by hand (AC6)."""
    stop = {
        "and", "the", "a", "an", "of", "to", "in", "on", "with", "for", "at",
        "my", "his", "her", "their", "old", "new", "very", "really",
    }
    words = "".join(c if c.isalnum() or c.isspace() else " " for c in label).split()
    return {w.casefold() for w in words if len(w) > 2 and w.casefold() not in stop}


def shared_interests(a: list[Trait], b: list[Trait]) -> list[str]:
    """S9-B6. The intersection of `interest` trait labels by token overlap.

    Returns the REQUESTER's label wording for anything that genuinely overlaps.
    Every entry is checkable by hand against both users' interest traits, which
    is exactly what AC6 asks someone to do.
    """
    b_tokens: list[set[str]] = [
        _tokens(t.label) for t in b if t.category == "interest" and t.status != "retracted"
    ]
    out: list[str] = []
    for t in a:
        if t.category != "interest" or t.status == "retracted":
            continue
        mine = _tokens(t.label)
        if not mine:
            continue
        if any(mine & theirs for theirs in b_tokens):
            out.append(t.label)
    return out


def reason_summary(
    fit_forward: float, fit_backward: float, shared: list[str]
) -> str:
    """S9-B6. Assembled from numbers we already hold. Nothing here is
    generated, so nothing here can be plausibly wrong."""
    bits: list[str] = []
    if shared:
        listed = ", ".join(shared[:3])
        bits.append(f"You both mentioned {listed}.")
    if fit_forward >= fit_backward + 0.05:
        bits.append("They line up well with what you said you want.")
    elif fit_backward >= fit_forward + 0.05:
        bits.append("You line up well with what they said they want.")
    else:
        bits.append("What each of you wants points at the other about equally.")
    if not shared:
        # Said plainly rather than hidden: a match on preference alone is a
        # real match, and pretending otherwise would be the fabrication this
        # module forbids.
        bits.append("No overlapping interests yet — this one is about fit.")
    return " ".join(bits)


async def _load_traits(session: AsyncSession, user_id: uuid.UUID) -> list[Trait]:
    return list(
        (
            await session.execute(
                select(Trait).where(
                    Trait.user_id == user_id, Trait.status != "retracted"
                )
            )
        ).scalars()
    )


async def refresh_embeddings(
    session: AsyncSession, router: TaskRouter, user_id: uuid.UUID
) -> bool:
    """S9-B3. Re-embed BOTH vectors when the live traits_hash differs from the
    stored one. Returns True if the user now has fresh usable vectors.

    Never compares a stale vector (§11): a run that cannot refresh returns
    False and the user is simply not scored, rather than being scored against
    a vector describing who they used to be.

    Both vectors go in ONE embed call — the google free tier has a low
    per-minute cap and looping is how you meet it (PICKUP, Step 2).
    """
    traits = await _load_traits(session, user_id)
    live_hash = compute_traits_hash(traits)

    rows = {
        e.kind: e
        for e in (
            await session.execute(
                select(ProfileEmbedding).where(ProfileEmbedding.user_id == user_id)
            )
        ).scalars()
    }
    fresh = (
        len(rows) == 2
        and all(e.traits_hash == live_hash for e in rows.values())
    )
    if fresh:
        return True

    identity_text = serialise_traits(traits, IDENTITY_CATEGORIES)
    preference_text = serialise_traits(traits, PREFERENCE_CATEGORIES)
    if not identity_text:
        # No identity traits at all: there is nothing to represent. Not an
        # error — a profile that has not been read yet.
        log_event(
            logger, "embedding_skipped",
            user_id=str(user_id), reason="no identity traits to embed",
        )
        return False

    provider, model = router.resolve_embeddings()
    # `preference` may legitimately be empty (nobody has said what they want
    # yet). Embedding an empty string is meaningless, so it gets a stated
    # placeholder rather than being silently skipped — a missing row would make
    # the user un-scoreable and therefore invisible, which is a worse answer.
    vectors = await provider.embed(
        [identity_text, preference_text or "No stated preferences yet."], model
    )
    identity_vec, preference_vec = vectors[0], vectors[1]

    for kind, vec in (("identity", identity_vec), ("preference", preference_vec)):
        row = rows.get(kind)
        if row is None:
            session.add(ProfileEmbedding(
                user_id=user_id, kind=kind, embedding=vec,
                embedding_model=f"{provider.name}/{model}", traits_hash=live_hash,
            ))
        else:
            row.embedding = vec
            row.embedding_model = f"{provider.name}/{model}"
            row.traits_hash = live_hash
    await session.commit()

    log_event(
        logger, "embeddings_refreshed",
        user_id=str(user_id), model=f"{provider.name}/{model}",
        traits_hash=live_hash[:12], identity_chars=len(identity_text),
        preference_chars=len(preference_text),
    )
    return True


# S9-B4. The hard filter, in pure SQL, all five conditions. Age is COMPUTED
# from birth_date here rather than stored (module_1 A1) — a stored age is wrong
# the day after someone's birthday.
#
# Distance is deliberately NOT filtered: the columns exist and the decision is
# the owner's, deferred.
_FILTER_SQL = text(
    """
WITH me AS (
    SELECT id, gender, interested_in, age_pref_min, age_pref_max,
           date_part('year', age(birth_date))::int AS age
    FROM users WHERE id = :me
),
base AS (
    SELECT u.id, u.gender, u.interested_in, u.age_pref_min, u.age_pref_max,
           date_part('year', age(u.birth_date))::int AS age
    FROM users u, me
    WHERE u.id <> me.id AND u.opt_in = TRUE
),
gender_ok AS (
    SELECT b.* FROM base b, me
    WHERE b.gender = ANY(me.interested_in)
      AND me.gender = ANY(b.interested_in)
),
age_ok AS (
    SELECT g.* FROM gender_ok g, me
    WHERE g.age BETWEEN me.age_pref_min AND me.age_pref_max
      AND me.age BETWEEN g.age_pref_min AND g.age_pref_max
),
snap_ok AS (
    SELECT a.*, s.id AS snapshot_id
    FROM age_ok a
    JOIN LATERAL (
        SELECT ps.id FROM persona_snapshots ps
        WHERE ps.user_id = a.id AND ps.status = 'ready'
        ORDER BY ps.version DESC LIMIT 1
    ) s ON TRUE
)
SELECT s.id, s.snapshot_id FROM snap_ok s
"""
)

# The funnel counts are a SEPARATE query on purpose. Folded into the row
# query they would vanish exactly when the pool is EMPTY — which is the one
# time they are the whole point (S9-B11, AC3).
_FUNNEL_SQL = text(
    _FILTER_SQL.text.replace(
        "SELECT s.id, s.snapshot_id FROM snap_ok s",
        """SELECT
    (SELECT count(*) FROM base)      AS n_opt_in,
    (SELECT count(*) FROM gender_ok) AS n_gender,
    (SELECT count(*) FROM age_ok)    AS n_age,
    (SELECT count(*) FROM snap_ok)   AS n_snapshot""",
    )
)


@dataclass
class Scored:
    user_id: uuid.UUID
    snapshot_id: uuid.UUID
    fit_forward: float
    fit_backward: float
    shared: list[str] = field(default_factory=list)

    @property
    def compatibility(self) -> float:
        return (self.fit_forward + self.fit_backward) / 2


def cosine(a: list[float], b: list[float]) -> float:
    """S9-B5. Exact brute-force cosine — NO ANN index at this pool size
    (named trade #2). An approximate index at a few dozen users buys nothing
    and can silently drop the right answer."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def run_matching(
    session: AsyncSession, router: TaskRouter, analysis: Analysis
) -> Analysis:
    """The background job: refresh → filter → score → write."""
    me = analysis.user_id
    funnel = FilterFunnel()

    if not await refresh_embeddings(session, router, me):
        analysis.status = "no_candidates"
        analysis.pool_status = "empty"
        analysis.candidate_count = 0
        await session.commit()
        log_event(
            logger, "matching_done",
            level=logging.WARNING, user_id=str(me), analysis_id=str(analysis.id),
            outcome="no_candidates", emptied_at="requester_not_embeddable",
            note="the requester has no identity traits to embed yet",
            **funnel.as_log_fields(),
        )
        return analysis

    counts = (await session.execute(_FUNNEL_SQL, {"me": me})).one()
    funnel.opted_in = counts.n_opt_in
    funnel.gender_fit = counts.n_gender
    funnel.age_fit = counts.n_age
    funnel.snapshot_ready = counts.n_snapshot

    rows = (await session.execute(_FILTER_SQL, {"me": me})).all()
    eligible = [(r.id, r.snapshot_id) for r in rows]

    my_traits = await _load_traits(session, me)
    my_vectors = {
        e.kind: list(e.embedding)
        for e in (
            await session.execute(
                select(ProfileEmbedding).where(ProfileEmbedding.user_id == me)
            )
        ).scalars()
    }

    scored: list[Scored] = []
    for candidate_id, snapshot_id in eligible:
        if not await refresh_embeddings(session, router, candidate_id):
            continue
        vectors = {
            e.kind: list(e.embedding)
            for e in (
                await session.execute(
                    select(ProfileEmbedding).where(
                        ProfileEmbedding.user_id == candidate_id
                    )
                )
            ).scalars()
        }
        if "identity" not in vectors or "preference" not in vectors:
            continue
        funnel.embedding_fresh += 1
        their_traits = await _load_traits(session, candidate_id)
        scored.append(Scored(
            user_id=candidate_id,
            snapshot_id=snapshot_id,
            # fit(R→C): does C match what R wants?
            fit_forward=cosine(my_vectors["preference"], vectors["identity"]),
            # fit(C→R): does R match what C wants?
            fit_backward=cosine(vectors["preference"], my_vectors["identity"]),
            shared=shared_interests(my_traits, their_traits),
        ))

    scored.sort(key=lambda s: (-s.compatibility, str(s.user_id)))
    chosen = scored[:MAX_CANDIDATES]

    for rank, s in enumerate(chosen, start=1):
        session.add(AnalysisCandidate(
            analysis_id=analysis.id, candidate_user_id=s.user_id, rank=rank,
            fit_forward=s.fit_forward, fit_backward=s.fit_backward,
            compatibility=s.compatibility, shared_interests=s.shared,
            reason_summary=reason_summary(s.fit_forward, s.fit_backward, s.shared),
            snapshot_id=s.snapshot_id,
        ))

    # S9-B7. Never padded.
    if not chosen:
        analysis.status = "no_candidates"
        analysis.pool_status = "empty"
    else:
        analysis.status = "matched"
        analysis.pool_status = "full" if len(chosen) >= MAX_CANDIDATES else "partial"
    # S15-B3: remembered here, because nothing else will once a candidate
    # deletes their account and their row cascades away.
    analysis.candidate_count = len(chosen)
    await session.commit()

    log_event(
        logger,
        "matching_done",
        user_id=str(me), analysis_id=str(analysis.id),
        outcome=analysis.status, pool_status=analysis.pool_status,
        emptied_at=funnel.emptied_at() if not chosen else None,
        candidates=[
            {
                "user_id": str(s.user_id),
                "rank": i,
                "fit_forward": round(s.fit_forward, 4),
                "fit_backward": round(s.fit_backward, 4),
                "compatibility": round(s.compatibility, 4),
                "shared_interests": s.shared,
            }
            for i, s in enumerate(chosen, start=1)
        ],
        **funnel.as_log_fields(),
    )
    return analysis


async def start_and_run(
    session: AsyncSession, router: TaskRouter, analysis_id: uuid.UUID
) -> None:
    """Entry point for the background task. Owns failure: an analysis that
    dies must land in `failed` with its error, never sit in `matching`
    forever with a spinner on the other end."""
    analysis = (
        await session.execute(select(Analysis).where(Analysis.id == analysis_id))
    ).scalar_one()
    try:
        await run_matching(session, router, analysis)
    except (AIError, Exception) as exc:  # noqa: BLE001
        # Blind on purpose, same reason as the persona compiler: whatever the
        # failure is, the row must stop saying "matching".
        analysis.status = "failed"
        analysis.error = f"{type(exc).__name__}: {exc}"[:2000]
        await session.commit()
        log_event(
            logger, "matching_failed",
            level=logging.ERROR, analysis_id=str(analysis_id),
            error=analysis.error,
        )


async def snapshot_gate_ok(session: AsyncSession, user_id: uuid.UUID) -> bool:
    """The §11 gate, asked the same way the SQL asks it — used by the probe to
    prove the two agree."""
    return await get_current_snapshot(session, user_id) is not None
