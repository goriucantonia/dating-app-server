"""`date_scenarios.v2` — the ONE AI call per candidate (S11-B2, B3).

**REVISED 2026-09-01 (owner decision): one date per candidate, so one setting
per call.** v1 asked for exactly two, because the caps were two dates per
candidate. The owner changed that to a single date each — A, B and C get one
evening apiece rather than two — and the schema follows: `minItems` and
`maxItems` are now `SETTINGS_PER_CANDIDATE`, which is 1.

Bumped to v2 rather than edited in place, per the registry's own rule. Nothing
stored referenced v1: `dates.schema_version` records `agent_response.v1` (the
conversation contract), and a generated setting is stored as a plain JSONB blob
on `dates.scenario`. So there is no migration and no unreadable old data — the
bump is bookkeeping that keeps the convention honest, not a rescue.

`SETTINGS_PER_CANDIDATE` is the ONE place the number lives. `app/simulation.py`
imports it as its `DATES_PER_CANDIDATE`, because one setting IS one date and
two constants that must agree are two constants that will eventually disagree.

`possible_events` is generated here, with the setting, rather than by a
mid-date generator (date_simulation.md trade #4). An event drawn from the
scenario is anchored — "a vintage Mustang pulls up" belongs to the car meet —
where a generic live generator produces "a waiter walks past" anywhere. It
also costs one fewer call per event, which at 0.15 per turn is not nothing.

`anchored_in_interest` exists for the **empty-intersection fallback** (S11-B3),
and it is the reason that fallback is checkable rather than hoped for. When the
two people share no interest labels at all, the setting has to be built around
ONE of them, and this field is where the model must name which interest it
built on. Without it, "anchored in at least one of the individuals" is a claim
nobody can verify after the fact — exactly the shape of failure principle §9
forbids: a derived value with no provenance.
"""

from __future__ import annotations

from app.ai.base import VersionedSchema
from app.schemas import register

# One setting per candidate, 4-6 events each (revised 2026-09-01; see above).
# `app/simulation.py` derives DATES_PER_CANDIDATE from this — one setting is
# one date, and the two numbers can never drift apart.
SETTINGS_PER_CANDIDATE = 1
MIN_EVENTS = 4
MAX_EVENTS = 6

DATE_SCENARIOS_V2 = register(
    VersionedSchema(
        name="date_scenarios",
        version=2,
        json_schema={
            "type": "object",
            "required": ["settings"],
            "properties": {
                "settings": {
                    "type": "array",
                    "minItems": SETTINGS_PER_CANDIDATE,
                    "maxItems": SETTINGS_PER_CANDIDATE,
                    "items": {
                        "type": "object",
                        "required": [
                            "setting_name",
                            "description",
                            "sensory_details",
                            "anchored_in_interest",
                            "possible_events",
                        ],
                        "properties": {
                            "setting_name": {
                                "type": "string",
                                "description": (
                                    "A short name for the place, e.g. 'the "
                                    "Sunday car meet at the docks'."
                                ),
                            },
                            "description": {
                                "type": "string",
                                "description": (
                                    "Two or three sentences: where they are, why "
                                    "they are there, and what they are doing. "
                                    "Written to be read by both people on the "
                                    "date, so no private detail about either."
                                ),
                            },
                            "sensory_details": {
                                "type": "string",
                                "description": (
                                    "What the place sounds, smells and looks "
                                    "like. One or two sentences — this is what "
                                    "gives the conversation something to catch on."
                                ),
                            },
                            "anchored_in_interest": {
                                "type": "string",
                                "description": (
                                    "The ONE interest this setting is built "
                                    "around, copied from the interest lists you "
                                    "were given, word for word. Not a summary, "
                                    "not a new phrase."
                                ),
                            },
                            "possible_events": {
                                "type": "array",
                                "minItems": MIN_EVENTS,
                                "maxItems": MAX_EVENTS,
                                "items": {
                                    "type": "string",
                                    "description": (
                                        "One thing that could happen around them, "
                                        "written as a sentence both of them would "
                                        "notice: 'the power cuts out and the room "
                                        "goes to candlelight'. Something in the "
                                        "world, never something either person "
                                        "does or says."
                                    ),
                                },
                            },
                        },
                    },
                }
            },
        },
    )
)
