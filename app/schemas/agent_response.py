"""`agent_response.v1` — FROZEN, verbatim from trait_persona.md §3 (S7-B2).

Every AI agent that speaks as a person obeys this: calibration chat now, date
simulation in Step 11, match chat in Step 14. One schema, one registry entry,
one version string — stored alongside every snapshot and every transcript, so
a future v2 never breaks stored v1 data.

`wants_to_end` is the field that exists for a reason worth restating: without
it every simulated date runs to the hard message cap and every ending is an
awkward mid-conversation cut. It gives a date a natural way to finish
(trade #4, `date_simulation.md`).

The four non-`reply` fields are the agent's inner state. They are STORED but
not shown during calibration — the user is meeting their AI self, not reading
its telemetry (§6). Date results in Step 13 are where they surface.

This file replaces the loud stub that stood here from Step 1 (DEFECTS.md
D-001: an absent file hides better than a stubbed one). Anything that
imported it before today got a NotImplementedError naming this step.
"""

from __future__ import annotations

from app.ai.base import VersionedSchema
from app.schemas import register

SCHEMA_VERSION = "agent_response.v1"

AGENT_RESPONSE_V1 = register(
    VersionedSchema(
        name="agent_response",
        version=1,
        json_schema={
            "type": "object",
            "required": [
                "reply",
                "state_of_mind",
                "emotional_state",
                "connection",
                "satisfaction",
                "wants_to_end",
            ],
            "properties": {
                "reply": {
                    "type": "string",
                    "description": "The spoken message — what they actually say out loud.",
                },
                "state_of_mind": {
                    "type": "string",
                    "description": "One sentence: what they are thinking right now.",
                },
                "emotional_state": {
                    "type": "string",
                    "description": "One to three words, e.g. 'amused, a bit nervous'.",
                },
                "connection": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "Felt connection to the other person right now.",
                },
                "satisfaction": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": "How much they are enjoying this right now.",
                },
                "wants_to_end": {
                    "type": "boolean",
                    "description": (
                        "True when they would naturally wrap up the conversation "
                        "now — not a complaint, just a normal ending."
                    ),
                },
            },
        },
    )
)
