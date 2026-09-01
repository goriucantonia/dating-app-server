"""`trait_extraction.v1` — the schema the holistic reconciliation call must obey
(S6-B2). Registered here; validated by the Structured Output Guard. No module
outside `app/ai/structured.py` parses model JSON (§16).

Two deliberate design choices in this schema, both about IDENTITY:

1. **The model never sees or echoes a UUID.** Existing trait rows are presented
   to it as short handles — `T1`, `T2`, … — and answers as their question codes
   (`BQ1`, `PQ07`, `D3` for dispute questions). The server holds the handle →
   row-id map for the duration of one call and translates back. A model asked
   to copy a 36-character UUID will eventually copy it wrong, and a mistyped id
   is indistinguishable from a verdict about a different trait. Handles are
   short enough to be copied reliably and are meaningless outside the call.

   This does NOT weaken A5.1's "matched by id, never by re-matching wording":
   the handle is a stand-in for the row id, assigned by us, never derived from
   the label or description. Wording still plays no part in identity.

2. **Declines are stated, not inferred from silence** (S6-B9). An answer too
   thin to support any trait is named in `declined_answer_ids`. If a decline
   were merely "no trait mentioned this answer", a model that simply forgot an
   answer would be indistinguishable from one that judged it thin — and §10
   requires the decline be logged and counted, which needs it to be observable.
"""

from __future__ import annotations

from app.ai.base import VersionedSchema
from app.schemas import register

TRAIT_CATEGORIES = [
    "interest",
    "quality",
    "flaw",
    "behavioral",
    "conversational_style",
    "partner_preference",
]

VERDICTS = ["keep", "update", "retract"]

TRAIT_EXTRACTION_V1 = register(
    VersionedSchema(
        name="trait_extraction",
        version=1,
        json_schema={
            "type": "object",
            "required": ["verdicts", "additions", "declined_answer_ids"],
            "properties": {
                "verdicts": {
                    "description": (
                        "Exactly one entry per existing trait handle you were "
                        "given. Never omit one, never invent a handle."
                    ),
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["trait_handle", "verdict", "reason"],
                        "properties": {
                            "trait_handle": {"type": "string"},
                            "verdict": {"type": "string", "enum": VERDICTS},
                            "reason": {
                                "type": "string",
                                "description": (
                                    "One short sentence. For 'keep', why the "
                                    "answers still support it; for 'retract', "
                                    "what withdrew the support."
                                ),
                            },
                            "label": {
                                "type": "string",
                                "description": "Only for 'update'. The revised short label.",
                            },
                            "description": {
                                "type": "string",
                                "description": "Only for 'update'. The revised full description.",
                            },
                            "category": {
                                "type": "string",
                                "enum": TRAIT_CATEGORIES,
                                "description": "Only for 'update', if the category changed.",
                            },
                            "confidence": {
                                "type": "number",
                                "description": "Only for 'update'. 0 to 1.",
                            },
                            "source_question_codes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "Only for 'update'. The question codes whose "
                                    "answers support the revised trait."
                                ),
                            },
                        },
                    },
                },
                "additions": {
                    "description": (
                        "Genuinely new traits only — never a restatement of an "
                        "existing handle. If nothing new is supported, return []."
                    ),
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": [
                            "category",
                            "label",
                            "description",
                            "confidence",
                            "source_question_codes",
                        ],
                        "properties": {
                            "category": {"type": "string", "enum": TRAIT_CATEGORIES},
                            "label": {
                                "type": "string",
                                "description": "Short, concrete: 'restores old cars'.",
                            },
                            "description": {
                                "type": "string",
                                "description": (
                                    "Full sentence or two, grounded in what the "
                                    "person actually wrote."
                                ),
                            },
                            "confidence": {"type": "number"},
                            "source_question_codes": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": (
                                    "The question codes whose answers support this "
                                    "trait. Never empty — a trait with no source "
                                    "is an invention."
                                ),
                            },
                        },
                    },
                },
                "declined_answer_ids": {
                    "description": (
                        "Question codes whose answers were too thin or evasive to "
                        "support any trait. Declining is correct and expected; "
                        "inventing a trait to avoid declining is not."
                    ),
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
        },
    )
)
