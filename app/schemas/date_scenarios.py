"""`date_scenarios.v1` — the ONE AI call per candidate (S11-B2, B3).

Two settings come back in a single call rather than two, because the second
setting has to be DIFFERENT from the first and a model that cannot see the
first one will happily produce it twice. On a free tier it also halves the
cost of the cheapest stage of the pipeline.

`possible_events` is generated here, with the setting, rather than by a
mid-date generator (date_simulation.md trade #4). An event drawn from the
scenario is anchored — "a vintage Mustang pulls up" belongs to the car meet —
where a generic live generator produces "a waiter walks past" anywhere. It
also costs one fewer call per event, which at 0.15 per turn is not nothing.

`anchored_in_interest` exists for the **empty-intersection fallback** (S11-B3),
and it is the reason that fallback is checkable rather than hoped for. When the
two people share no interest labels at all, the prompt asks for one setting
built around HER interests and one around HIS — and this field is where the
model must name which interest it built on. Without it, "anchored in each
person's interests" is a claim nobody can verify after the fact, which is
exactly the shape of failure principle §9 forbids: a derived value with no
provenance.
"""

from __future__ import annotations

from app.ai.base import VersionedSchema
from app.schemas import register

# Locked by date_simulation.md §2: exactly 2 settings, 4-6 events each.
SETTINGS_PER_CANDIDATE = 2
MIN_EVENTS = 4
MAX_EVENTS = 6

DATE_SCENARIOS_V1 = register(
    VersionedSchema(
        name="date_scenarios",
        version=1,
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
