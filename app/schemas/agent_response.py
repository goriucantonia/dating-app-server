"""`agent_response.v1` — NOT YET IMPLEMENTED. Arrives in Step 7, frozen
verbatim from trait_persona.md §3: reply, state_of_mind, emotional_state,
connection (0-100), satisfaction (0-100), wants_to_end.

This stub exists on purpose (DEFECTS.md D-001): an absent file hides better
than a stubbed one. Importing `AGENT_RESPONSE_V1` before Step 7 fails loudly.
"""

from __future__ import annotations


def __getattr__(name: str):
    raise NotImplementedError(
        f"app.schemas.agent_response.{name}: agent_response.v1 is frozen and "
        "registered in Step 7 (trait_persona.md §3) — it does not exist yet"
    )
