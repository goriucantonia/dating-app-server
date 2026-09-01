"""`dispute_followup.v1` — the one follow-up question generated when a user
disputes a trait (S6-B7). Registered here; validated by the Guard.

This is the ONLY AI-generated question type left in the system: A5.2 replaced
the dynamic questionnaire generator with the curated pool, and the reason this
one survived is that a dispute question targets a specific trait on a specific
person and cannot be pre-written.

Exactly one question, by design. A dispute is the user saying "that's not me";
answering a pile of questions to prove it is a punishment, not a correction.
"""

from __future__ import annotations

from app.ai.base import VersionedSchema
from app.schemas import register

DISPUTE_FOLLOWUP_V1 = register(
    VersionedSchema(
        name="dispute_followup",
        version=1,
        json_schema={
            "type": "object",
            "required": ["question_text"],
            "properties": {
                "question_text": {
                    "type": "string",
                    "description": (
                        "One open question, addressed to the person as 'you', that "
                        "invites them to describe what is actually true instead of "
                        "the disputed trait. It must not argue for the trait, must "
                        "not ask them to justify disputing it, and must be "
                        "answerable in a few sentences."
                    ),
                }
            },
        },
    )
)
