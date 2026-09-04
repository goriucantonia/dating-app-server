"""Trait extraction: verdict-based, holistic reconciliation (S6-B1…B5, B9, B10).

Every run re-reads the user's FULL answer set and their EXISTING trait rows,
and converges the rows to the answers. It is a reconcile, not a patch — the
same shape `reconcile.py` uses for seeded questions, and for the same reason:
a reconcile cannot accumulate drift, and at 5–35 answers the extra tokens are
noise (A5.1, named trade).

The load-bearing property (decision log #10): a user's confirmations never
evaporate. The model returns an explicit `keep` / `update` / `retract` verdict
for each existing row, matched by id — so a rephrased description is an UPDATE
to the same row, and `confirmed` / `disputed` status, provenance, and dispute
history survive re-extraction. Nothing is ever silently deleted; a retraction
sets `status='retracted'` and the row stays in the table.

Ordering here is load-bearing (§19) and is enforced by the structure of
`run_extraction`, not by a comment: the trait write COMMITS first, and only
then is `traits_hash` recomputed. Staleness can never claim freshness. An
all-`keep` run leaves the hash byte-identical, which is what keeps embeddings
and the persona snapshot fresh instead of marking them stale for nothing.
"""

from __future__ import annotations

import asyncio
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.base import AIError, GenRequest, Message
from app.ai.routing import TaskRouter
from app.ai.structured import guarded_structured_call
from app.logging_setup import log_event
from app.models import Answer, Question, Trait, TraitEvent
from app.schemas.trait_extraction import TRAIT_CATEGORIES, TRAIT_EXTRACTION_V1
from app.traits_hash import compute_traits_hash

logger = logging.getLogger("app.extraction")

TASK = "trait_extraction"

# Generous on purpose. The original reason was google's thinking model spending
# a tight budget on reasoning and returning MAX_TOKENS with no text; that model
# is gone (owner decision 2026-09-01) but the budget stays, because this call
# returns a verdict per existing trait PLUS additions and can legitimately be
# long for a full 35-answer profile.
MAX_TOKENS = 8192

_CATEGORIES = set(TRAIT_CATEGORIES)

# `T1`, `BQ1`, `PQ07` — identifiers this call itself puts in front of the
# model. A model that echoes one into `label` has produced a row that is
# useless everywhere downstream: shared_interests intersects LABELS, the
# persona prompt lists them, and the profile screen shows them. Observed
# for real (D-009), so this is a guard and not a hypothetical.
_IDENTIFIER_LIKE = re.compile(r"^[A-Za-z]{1,3}[-_ ]?\d{1,3}$")

SYSTEM_PROMPT = """\
You read a person's own words about themselves and maintain a structured list \
of traits describing them. You are reconciling, not starting over: you are \
given the trait list as it stands, and you decide what each existing entry \
should become now that you have seen every answer again.

Rules that matter more than completeness:

- Ground every trait in what the person actually wrote. If you cannot point to \
the answer that supports it, it does not belong in the list.
- Keep what is still true. `keep` is the correct and most common verdict on a \
re-read where nothing changed. Do not rewrite a trait merely to phrase it \
better — a rewrite is an `update` and it costs the person something, so spend \
it only when the meaning actually changed.
- `retract` only when the answers no longer support the trait — because they \
were edited, or because you were wrong. Not because it seems unflattering.
- A `flaw` is a real category and an honest profile has some. Do not soften a \
person into someone with no rough edges; that is a less useful profile and a \
less true one.
- An answer that says little is allowed to produce nothing. Name it in \
declined_answer_ids. Inventing a trait to fill space is the worst thing you \
can do here.
- Write descriptions in the third person, about this person, concretely. \
Specific beats flattering, and specific beats vague.
- Aim for roughly 8-12 traits in total for a complete profile, and fewer \
while there are only a handful of answers. Thirty entries is not a sharper \
picture of someone, it is the same picture cut into more pieces: each one \
carries less, and the ones that actually matter stop standing out. If two \
entries would sit in the same category and lean on the same answer, they \
are one entry.

About adding — read this twice, because it is the rule most easily broken \
without noticing:

You are almost always RE-READING answers that have already been read, against \
a list that was already built from them. In that situation the correct number \
of additions is usually ZERO. Before you add anything, go through every \
existing entry and ask whether it already covers the ground. If it does, add \
nothing. If it nearly does, that is an `update` to that entry, not a new one.

A new entry that re-slices something an existing entry already covers — a \
finer shade of the same preference, one half of a trait already described, \
the same behaviour under a different name — is drift. It makes the list \
longer without making it truer, and because you will re-read again next time, \
it compounds. Add only when a real part of this person is described in the \
answers and is genuinely absent from every entry in the list."""


@dataclass
class ExtractionOutcome:
    """What a run did — the return value AND the shape of the §7 log line."""

    kept: int = 0
    updated: int = 0
    retracted: int = 0
    added: int = 0
    declined: list[str] = field(default_factory=list)
    hash_before: str = ""
    hash_after: str = ""
    model: str = ""
    answers_seen: int = 0
    ignored: list[str] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        """The one question the staleness cascade asks. Derived from the hash,
        never from the verdict counts — the hash is what downstream consumers
        actually compare, so anything else could disagree with them."""
        return self.hash_before != self.hash_after

    def as_log_fields(self) -> dict:
        return {
            "kept": self.kept,
            "updated": self.updated,
            "retracted": self.retracted,
            "added": self.added,
            "declined_count": len(self.declined),
            "declined": self.declined,
            "answers_seen": self.answers_seen,
            "traits_hash_before": self.hash_before[:12],
            "traits_hash_after": self.hash_after[:12],
            "changed": self.changed,
            "ignored": self.ignored,
        }


# --- S6-B5: one run per user, exactly one queued follow-up ------------------
#
# Named trade: this is per-process state. One uvicorn worker is the deployment
# for this phase (docker-compose.yml runs a single api container), so a
# per-process lock IS the whole system. On the day a second worker exists this
# must become a Postgres advisory lock keyed by user id — recorded here rather
# than discovered then. §17: give up rather than half-do, and a pile-up of
# concurrent extractions on one user is the half-doing this prevents.

_locks: dict[uuid.UUID, asyncio.Lock] = {}
_queued: set[uuid.UUID] = set()


def lock_for(user_id: uuid.UUID) -> asyncio.Lock:
    lock = _locks.get(user_id)
    if lock is None:
        lock = _locks[user_id] = asyncio.Lock()
    return lock


def is_running(user_id: uuid.UUID) -> bool:
    lock = _locks.get(user_id)
    return lock is not None and lock.locked()


def queue_follow_up(user_id: uuid.UUID) -> bool:
    """Register a single follow-up run. Returns False if one is ALREADY queued
    — the second, third and hundredth caller during one run all collapse into
    the same single follow-up. That collapsing is the whole point."""
    if user_id in _queued:
        return False
    _queued.add(user_id)
    return True


def take_queued(user_id: uuid.UUID) -> bool:
    """Consume the queued flag. True means a follow-up was owed and is now
    claimed by the caller."""
    if user_id in _queued:
        _queued.discard(user_id)
        return True
    return False


def _build_handles(traits: list[Trait]) -> tuple[dict[str, Trait], str]:
    """Short handles the model can echo without mistyping. See
    `app/schemas/trait_extraction.py` for why UUIDs never cross the wire."""
    by_handle = {f"T{i}": t for i, t in enumerate(traits, start=1)}
    if not by_handle:
        return {}, (
            "(EMPTY — no traits exist for this person yet, so there is "
            "nothing to return a verdict about. See the instruction at the "
            "end of this message.)"
        )
    lines = [
        f"{h}: [{t.category}] {t.label} — {t.description} "
        f"(status: {t.status}, confidence: {t.confidence:.2f})"
        for h, t in by_handle.items()
    ]
    return by_handle, "\n".join(lines)


def _build_answer_block(
    rows: list[tuple[Question, Answer]],
    handle_of: dict[uuid.UUID, str] | None = None,
) -> tuple[dict[str, uuid.UUID], str]:
    """Every answer needs a stable short name for provenance. Baseline and pool
    questions have codes already; dispute questions do not, so they get one.

    A dispute answer is also TOLD which trait it is about (audit 2026-09-02):
    the spec says the next extraction "corrects rather than re-infers", and
    that only happens if the model can connect the answer to the handle.
    """
    by_code: dict[str, uuid.UUID] = {}
    blocks: list[str] = []
    for i, (q, a) in enumerate(rows, start=1):
        code = q.code or f"D{i}"
        by_code[code] = a.id
        note = ""
        handle = (handle_of or {}).get(q.trait_id) if q.origin == "dispute" else None
        if handle:
            note = (
                f"\n(The person DISPUTED {handle} and this is their answer about "
                f"it. Treat it as the correction: `update` {handle} to what they "
                f"say, or `retract` it if they say it is simply not true.)"
            )
        blocks.append(f"--- {code} ({q.probe_area}){note}\nQ: {q.text}\nA: {a.answer_text}")
    return by_code, "\n\n".join(blocks)


async def _confirmed_and_unchanged(
    session: AsyncSession, traits: list[Trait], rows: list[tuple[Question, Answer]]
) -> frozenset[uuid.UUID]:
    """Traits the person confirmed and has said nothing new about since.

    "Confirmations never evaporate" (decision log #10) was enforced for
    `update` verdicts and NOT for `retract`, so a bad day at the model could
    strike a confirmed trait, hide it from every later run, and re-add it as
    a fresh guess with a new id (audit 2026-09-02). A retraction is honoured
    only when a source answer was edited AFTER the confirmation — the one
    case where the ground under the confirmation genuinely moved.
    """
    confirmed = [t for t in traits if t.status == "confirmed"]
    if not confirmed:
        return frozenset()
    latest_confirm: dict[uuid.UUID, datetime] = {}
    events = (
        await session.execute(
            select(TraitEvent.trait_id, TraitEvent.created_at).where(
                TraitEvent.trait_id.in_([t.id for t in confirmed]),
                TraitEvent.event == "confirmed",
            )
        )
    ).all()
    for trait_id, at in events:
        if trait_id not in latest_confirm or at > latest_confirm[trait_id]:
            latest_confirm[trait_id] = at
    answer_updated = {a.id: a.updated_at for _, a in rows}
    out: set[uuid.UUID] = set()
    for t in confirmed:
        confirmed_at = latest_confirm.get(t.id)
        if confirmed_at is None:
            # Confirmed with no event on record: nothing to compare against,
            # and the safe reading of a confirmation is to keep it.
            out.add(t.id)
            continue
        edited_since = any(
            (answer_updated.get(aid) or confirmed_at) > confirmed_at
            for aid in (t.source_answer_ids or [])
        )
        if not edited_since:
            out.add(t.id)
    return frozenset(out)


def _resolve_sources(
    codes: list[str] | None, by_code: dict[str, uuid.UUID]
) -> list[uuid.UUID]:
    """Model-supplied provenance, filtered to codes that actually exist. An
    unknown code is dropped rather than trusted: provenance that points at
    nothing is worse than short provenance (§9)."""
    out: list[uuid.UUID] = []
    for c in codes or []:
        key = str(c).strip()
        answer_id = by_code.get(key) or by_code.get(key.upper())
        if answer_id is not None and answer_id not in out:
            out.append(answer_id)
    return out


async def _load(
    session: AsyncSession, user_id: uuid.UUID
) -> tuple[list[tuple[Question, Answer]], list[Trait]]:
    answered = (
        await session.execute(
            select(Question, Answer)
            .join(Answer, Answer.question_id == Question.id)
            .where(Answer.user_id == user_id)
            .order_by(Question.origin, Question.pool_order, Question.code)
        )
    ).all()
    traits = list(
        (
            await session.execute(
                select(Trait)
                .where(Trait.user_id == user_id, Trait.status != "retracted")
                .order_by(Trait.created_at, Trait.id)
            )
        ).scalars()
    )
    return [(q, a) for q, a in answered], traits


async def run_extraction(
    session: AsyncSession, router: TaskRouter, user_id: uuid.UUID
) -> ExtractionOutcome:
    """One reconciliation pass. Callers must hold the user's lock — use
    `extract_once`, which is the only supported entry point."""
    rows, existing = await _load(session, user_id)
    if not rows:
        raise AIError("this person has not answered anything yet", task=TASK)

    by_handle, trait_block = _build_handles(existing)
    handle_of = {t.id: h for h, t in by_handle.items()}
    by_code, answer_block = _build_answer_block(rows, handle_of)
    protected = await _confirmed_and_unchanged(session, existing, rows)
    # A dispute question that has been ANSWERED is a correction waiting to be
    # applied: an `update` verdict on that trait closes the dispute.
    answered_disputes = {
        q.trait_id for q, _ in rows if q.origin == "dispute" and q.trait_id is not None
    }

    provider, model = router.resolve(TASK)
    outcome = ExtractionOutcome(model=f"{provider.name}/{model}", answers_seen=len(rows))
    outcome.hash_before = compute_traits_hash(existing)

    user_prompt = (
        f"THE TRAIT LIST AS IT STANDS\n{trait_block}\n\n"
        f"EVERYTHING THIS PERSON HAS WRITTEN\n\n{answer_block}\n\n"
        "Return a verdict for every handle above (one each, no more, no less), "
        "any genuinely new traits, and the codes of answers too thin to support "
        "anything.\n\n"
        + (
            "The list above was already built from these same answers. Unless an "
            "answer has changed, expect to return `keep` for everything and an "
            "EMPTY additions list. Adding here would mean the earlier reading "
            "missed something outright — not merely that you would have phrased "
            "it differently."
            if by_handle
            else (
                "THE TRAIT LIST IS EMPTY. This person has no traits yet.\n\n"
                "Therefore `verdicts` MUST be an empty array: [].\n\n"
                "There is nothing to return a verdict about — a verdict "
                "refers to a trait that already exists, and none do. Put "
                "every trait you find in `additions`.\n\n"
                "The codes above (BQ1, PQ07 and so on) name ANSWERS, not "
                "traits. Cite them in source_question_codes. NEVER use one "
                "as a label: a label is a short human phrase like "
                "\"restores old bicycles\", never a code or an identifier."
            )
        )
    )

    # §7: the input answer IDs are logged BEFORE the call, so a run that dies
    # mid-flight still says what it was working from.
    log_event(
        logger, "extraction_start",
        user_id=str(user_id), model=outcome.model,
        answer_codes=sorted(by_code), existing_traits=len(existing),
    )

    result = await guarded_structured_call(
        provider,
        GenRequest(
            task=TASK, model=model, system_prompt=SYSTEM_PROMPT,
            messages=[Message(role="user", content=user_prompt)],
            temperature=0.2, max_tokens=MAX_TOKENS,
        ),
        TRAIT_EXTRACTION_V1,
    )

    await _apply(
        session, result, outcome, by_handle, by_code, user_id,
        protected=protected, answered_disputes=answered_disputes,
    )

    # §19, and the reason this is not split into two functions: the trait write
    # COMMITS here, and only the lines below it recompute the hash. Recomputing
    # first would hash a state that might never reach the database.
    await session.commit()

    fresh = list(
        (
            await session.execute(
                select(Trait).where(
                    Trait.user_id == user_id, Trait.status != "retracted"
                )
            )
        ).scalars()
    )
    outcome.hash_after = compute_traits_hash(fresh)

    log_event(
        logger,
        "extraction_done" if outcome.changed else "extraction_done_no_change",
        user_id=str(user_id), model=outcome.model, **outcome.as_log_fields(),
    )
    if outcome.declined:
        # §10: a decline is a real, countable outcome, not an absence.
        log_event(
            logger, "extraction_declined_answers",
            user_id=str(user_id), model=outcome.model,
            codes=outcome.declined, count=len(outcome.declined),
        )
    return outcome


async def _apply(
    session: AsyncSession,
    result: dict,
    outcome: ExtractionOutcome,
    by_handle: dict[str, Trait],
    by_code: dict[str, uuid.UUID],
    user_id: uuid.UUID,
    *,
    protected: frozenset[uuid.UUID] = frozenset(),
    answered_disputes: set[uuid.UUID] | None = None,
) -> None:
    """S6-B3. Verdicts become row changes and `trait_events`. Bookkeeping only
    — the commit belongs to the caller, so the §19 ordering stays visible in
    one place instead of being spread across two."""
    seen: set[str] = set()

    for v in result.get("verdicts") or []:
        handle = str(v.get("trait_handle", "")).strip()
        trait = by_handle.get(handle)
        if trait is None:
            # An invented handle is dropped, loudly. Acting on it would mean
            # editing a row nobody returned a verdict about.
            outcome.ignored.append(f"{handle or '(blank)'}: unknown handle")
            continue
        if handle in seen:
            outcome.ignored.append(f"{handle}: duplicate verdict")
            continue
        seen.add(handle)

        verdict = v.get("verdict")
        reason = (v.get("reason") or "").strip()

        if verdict == "retract":
            if trait.id in protected:
                # The person said "yes, that's me" and nothing they wrote has
                # changed since. The model does not get to overrule that.
                log_event(
                    logger, "retract_declined_confirmed", level=logging.WARNING,
                    user_id=str(user_id), trait_id=str(trait.id), handle=handle,
                    model=outcome.model, reason=reason[:300],
                )
                outcome.ignored.append(f"{handle}: retract declined — confirmed by the person")
                outcome.kept += 1
                continue
            trait.status = "retracted"
            session.add(TraitEvent(
                trait_id=trait.id, event="retracted",
                detail=f"extraction ({outcome.model}): {reason}"[:2000],
            ))
            outcome.retracted += 1
            continue

        if verdict == "update":
            before = (
                trait.category, trait.label, trait.description,
                round(float(trait.confidence), 6),
            )
            if (label := (v.get("label") or "").strip()):
                trait.label = label
            if (description := (v.get("description") or "").strip()):
                trait.description = description
            if v.get("category") in _CATEGORIES:
                trait.category = v["category"]
            if isinstance(v.get("confidence"), int | float):
                trait.confidence = max(0.0, min(1.0, float(v["confidence"])))
            if (sources := _resolve_sources(v.get("source_question_codes"), by_code)):
                trait.source_answer_ids = sources
            after = (
                trait.category, trait.label, trait.description,
                round(float(trait.confidence), 6),
            )

            if before == after:
                # The model said `update` and changed nothing. Writing an event
                # here would inflate the drift alarm (A5.1) with edits that
                # never happened, so this counts as what it actually was.
                outcome.kept += 1
                continue

            # A `confirmed` row keeps its status through an update — that IS
            # decision log #10. Only the content moves.
            session.add(TraitEvent(
                trait_id=trait.id, event="updated",
                detail=f"extraction ({outcome.model}): {reason}"[:2000],
            ))
            if trait.status == "disputed" and trait.id in (answered_disputes or set()):
                # The person disputed it, answered the follow-up, and the
                # model rewrote the trait from that answer: the dispute is
                # closed. `corrected` existed in both CHECK constraints and
                # nothing ever wrote it (audit 2026-09-02) — "Being
                # corrected" was forever.
                trait.status = "corrected"
                session.add(TraitEvent(
                    trait_id=trait.id, event="corrected",
                    detail=f"extraction ({outcome.model}) applied the person's answer"[:2000],
                ))
            outcome.updated += 1
            continue

        # `keep`, and anything unrecognised, leaves the row untouched.
        outcome.kept += 1

    for handle in by_handle:
        if handle not in seen:
            # A verdict the model failed to return is NOT a retraction. Silence
            # keeps the row — the safe reading of an absent answer.
            outcome.kept += 1
            outcome.ignored.append(f"{handle}: no verdict returned, row kept")

    taken_labels = {
        (t.category, " ".join(t.label.casefold().split()))
        for t in by_handle.values() if t.status != "retracted"
    }
    for add in result.get("additions") or []:
        category = add.get("category")
        label = (add.get("label") or "").strip()
        description = (add.get("description") or "").strip()
        if category not in _CATEGORIES or not label or not description:
            outcome.ignored.append(f"add '{label or '(blank)'}': incomplete")
            continue
        if _IDENTIFIER_LIKE.match(label):
            # Dropped rather than stored: a trait labelled "T1" poisons the
            # trait block in every persona prompt and makes shared-interest
            # matching silently impossible. Loud beats quietly wrong.
            outcome.ignored.append(f"add '{label}': label is an identifier, not a label")
            continue
        key = (category, " ".join(label.casefold().split()))
        if key in taken_labels:
            # Same label (case- and space-insensitively) as a live row or an
            # addition earlier in this response: D-007 closed the compounding
            # variant by prompt alone; this is the guard in code.
            outcome.ignored.append(f"add '{label}': duplicates an existing trait")
            continue
        taken_labels.add(key)
        sources = _resolve_sources(add.get("source_question_codes"), by_code)
        if not sources:
            # §9: a trait with no provenance is an invention, and we do not
            # store inventions. Dropped rather than stored source-less.
            outcome.ignored.append(f"add '{label}': no valid provenance")
            continue
        confidence = add.get("confidence")
        confidence = float(confidence) if isinstance(confidence, int | float) else 0.5
        trait = Trait(
            user_id=user_id, category=category, label=label, description=description,
            confidence=max(0.0, min(1.0, confidence)),
            status="inferred", source_answer_ids=sources, extracted_by=outcome.model,
        )
        session.add(trait)
        # The event references trait.id, which the database assigns. Flush so
        # the id exists before it is used as a foreign key.
        await session.flush()
        session.add(TraitEvent(
            trait_id=trait.id, event="created",
            detail=f"extraction ({outcome.model})",
        ))
        outcome.added += 1

    outcome.declined = [
        str(c).strip() for c in (result.get("declined_answer_ids") or []) if str(c).strip()
    ]


async def extract_once(
    session: AsyncSession, router: TaskRouter, user_id: uuid.UUID
) -> ExtractionOutcome | None:
    """The only supported entry point (S6-B5). Runs under the user's lock, and
    on the way out runs the single queued follow-up if one was requested while
    this run held the lock.

    Returns the LAST outcome actually produced, or None if this call did
    nothing because another run already held the lock — the caller reports
    "queued", not "done", and that difference is the give-up being honest."""
    lock = lock_for(user_id)
    if lock.locked():
        queued = queue_follow_up(user_id)
        log_event(
            logger, "extraction_queued",
            user_id=str(user_id),
            note="a run is already in flight" if queued
                 else "a follow-up was already queued; collapsed into it",
            newly_queued=queued,
        )
        return None

    async with lock:
        try:
            outcome = await run_extraction(session, router, user_id)
        except BaseException:
            # The run died; the follow-up it owed dies with it, LOUDLY,
            # rather than leaking into the next run as an unrequested
            # second model call (audit 2026-09-02).
            if take_queued(user_id):
                log_event(
                    logger, "extraction_follow_up_dropped", level=logging.WARNING,
                    user_id=str(user_id), reason="the run it was queued behind failed",
                )
            raise
        if take_queued(user_id):
            log_event(logger, "extraction_follow_up_start", user_id=str(user_id))
            outcome = await run_extraction(session, router, user_id)
    return outcome
