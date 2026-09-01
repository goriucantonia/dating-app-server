"""`persona_digest.v1` — the ONE AI call in a persona compilation (S7-B3).

Everything else in a snapshot is a deterministic template: hard facts, trait
rows grouped by category, and the user's own sentences quoted verbatim. Only
this part needs a model, because "how does this person behave when a
conversation gets tense" is a synthesis across several situational answers
rather than a lookup.

The fields are the four moments a date actually turns on. They are behaviour
descriptions written TO the future agent, in the second person, because the
digest is dropped straight into a system prompt that says "you are this
person" — a third-person digest would have to be rewritten at use time, and
rewriting is exactly the paraphrase step trade #1 forbids.
"""

from __future__ import annotations

from app.ai.base import VersionedSchema
from app.schemas import register

PERSONA_DIGEST_V1 = register(
    VersionedSchema(
        name="persona_digest",
        version=1,
        json_schema={
            "type": "object",
            "required": [
                "in_tense_moments",
                "when_flirting",
                "when_supporting",
                "when_opening_up",
            ],
            "properties": {
                "in_tense_moments": {
                    "type": "string",
                    "description": (
                        "How you behave when a conversation gets difficult or "
                        "someone challenges you. Two or three sentences, "
                        "second person, grounded in what they wrote."
                    ),
                },
                "when_flirting": {
                    "type": "string",
                    "description": (
                        "How you show interest in someone you like — including "
                        "the awkward parts. Two or three sentences, second person."
                    ),
                },
                "when_supporting": {
                    "type": "string",
                    "description": (
                        "How you respond when someone needs comfort or help. "
                        "Two or three sentences, second person."
                    ),
                },
                "when_opening_up": {
                    "type": "string",
                    "description": (
                        "How readily you share something real about yourself, and "
                        "what it takes. Two or three sentences, second person."
                    ),
                },
            },
        },
    )
)
