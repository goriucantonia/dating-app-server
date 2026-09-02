"""`date_scenarios.v3` — the ONE AI call per ANALYSIS (S11-B2, B3; revised).

**REVISED 2026-09-02 (owner decision): one scenario per analysis, run
identically against every candidate.** v2 asked for one setting per CANDIDATE,
anchored in that pair's shared interests. Three candidates therefore got three
different evenings, and `candidate_scores` then ranked the three numbers as
though they were the same measurement. They were not. The requirement now is a
controlled comparison — same fixture, three different people — so the scenario
is drawn from `app/date_archetypes.py` and generated once.

Bumped rather than edited in place, per the registry's own rule, and this bump
is a genuine contract change: `anchored_in_interest` is GONE and `archetype`
has taken its place. Nothing stored breaks. A generated setting is a plain
JSONB blob on `dates.scenario` (and now `analyses.scenarios`), and
`dates.schema_version` records `agent_response.v1` — the conversation contract,
not this one. Old rows keep their `anchored_in_interest` key and simply have no
`archetype`; the API reads both with `.get`, so a pre-existing date still
renders.

**`archetype` is the provenance field, and it is the same idea
`anchored_in_interest` was.** The draw happens in code; the model is told which
archetype it drew and must copy the key back word for word. Without that,
"this analysis ran the cinema fixture" is a claim nobody can check after the
fact — the exact shape of failure principle §9 forbids, a derived value with no
provenance. `app/simulation.py` verifies the returned key against the one it
asked for and repairs it rather than trusting it.

**No names, no personal detail — enforced in the schema's own descriptions.**
This setting is read by three different pairs of people. A description that
says "Maya has been meaning to come here for weeks" would be false on two of
the three dates. The generated text has to work for any two strangers, and
that constraint is what makes the fixture fair rather than merely shared.

`possible_events` is still generated here, with the setting, rather than by a
mid-date generator (date_simulation.md trade #4). An event drawn from the
scenario is anchored — "the projector stutters and the reel is rethreaded"
belongs to the cinema — where a generic live generator produces "a waiter walks
past" anywhere. It also costs one fewer call per event. Under the shared
fixture it gains a second job: all three candidates now draw from the SAME
event list in the same order, so even the interruptions are held still.
"""

from __future__ import annotations

from app.ai.base import VersionedSchema
from app.schemas import register

# RENAMED as well as revalued (2026-09-02): it was `SETTINGS_PER_CANDIDATE`,
# and under the shared fixture that name states something false. One setting is
# now generated for the whole ANALYSIS and every candidate is run against it.
#
# `app/simulation.py` still derives `DATES_PER_CANDIDATE` from this, and the
# derivation is still exactly true: N fixtures means every candidate goes on N
# dates, one per fixture. Two constants that must agree are two constants that
# will eventually disagree.
SETTINGS_PER_ANALYSIS = 1
MIN_EVENTS = 4
MAX_EVENTS = 6

DATE_SCENARIOS_V3 = register(
    VersionedSchema(
        name="date_scenarios",
        version=3,
        json_schema={
            "type": "object",
            "required": ["settings"],
            "properties": {
                "settings": {
                    "type": "array",
                    "minItems": SETTINGS_PER_ANALYSIS,
                    "maxItems": SETTINGS_PER_ANALYSIS,
                    "items": {
                        "type": "object",
                        "required": [
                            "setting_name",
                            "description",
                            "sensory_details",
                            "archetype",
                            "possible_events",
                        ],
                        "properties": {
                            "setting_name": {
                                "type": "string",
                                "description": (
                                    "A short name for the place, e.g. 'the "
                                    "late show at the Roxy'. Name the PLACE. "
                                    "Never name a person."
                                ),
                            },
                            "description": {
                                "type": "string",
                                "description": (
                                    "Two or three sentences: where they are, "
                                    "why they are there, and what they are "
                                    "doing. Written for ANY two people who "
                                    "have just met — no names, and nothing "
                                    "that assumes either of them likes this "
                                    "kind of thing."
                                ),
                            },
                            "sensory_details": {
                                "type": "string",
                                "description": (
                                    "What the place sounds, smells and looks "
                                    "like. One or two sentences — this is what "
                                    "gives the conversation something to catch "
                                    "on."
                                ),
                            },
                            "archetype": {
                                "type": "string",
                                "description": (
                                    "The archetype key you were given, copied "
                                    "back word for word. Not a summary, not a "
                                    "new phrase, not the readable name."
                                ),
                            },
                            "possible_events": {
                                "type": "array",
                                "minItems": MIN_EVENTS,
                                "maxItems": MAX_EVENTS,
                                "items": {
                                    "type": "string",
                                    "description": (
                                        "One thing that could happen around "
                                        "them, written as a sentence both of "
                                        "them would notice: 'the power cuts "
                                        "out and the room goes to candlelight'. "
                                        "Something in the world, never "
                                        "something either person does or says, "
                                        "and never anything that names either "
                                        "of them."
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
