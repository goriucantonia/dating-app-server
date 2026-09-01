"""`judge_rubric.v1` — the judge's schema AND its rubric text (S12-B2, B4, B8).

**The rubric text lives here, next to the schema, and is versioned with it.**
Not in a prompt string inside the judging module, because the two cannot be
allowed to drift: `date_evaluations.rubric_version` claims that a stored score
was produced under a particular set of instructions, and that claim is only
worth something if the instructions and the shape they produce move together.
Changing a criterion's meaning is a v2, and old rows keep saying v1.

**The four criteria are scored; the final number is NOT asked for** (S12-B5,
trade #3). There is deliberately no `overall_score` field below. A model asked
for a headline number produces one that does not follow from its own
sub-scores, and then nobody can say why a date scored 71. The weights live in
code, apply identically to everyone, and can be recomputed by hand from the
stored `criteria` — which is exactly what `probe_judge.py` does.

**`clashes` can be empty, and that is a verdict** (S12-B8, §10). "Be specific"
without a way to refuse is an instruction to fabricate: a judge that must
produce a clash will invent one from two people who simply got on. So the
schema requires a citable `moment` for every clash — a quotation from the
transcript — and the rubric says plainly that an empty array is the right
answer when nothing clashed. The same applies to `clicked_subjects`.

**The judge never sees the personas** (trade #5). It receives the transcript
plus both people's trait LABELS, for attribution only. It scores what happened
on the date, not what the profiles predicted — and that separation is what
makes a surprising result informative rather than a bug.
"""

from __future__ import annotations

from app.ai.base import VersionedSchema
from app.schemas import register

RUBRIC_VERSION = "judge_rubric.v1"

# The instructions this version's scores were produced under. Stored by name on
# every evaluation; edit the meaning of anything here and it becomes v2.
JUDGE_SYSTEM_PROMPT = """\
You read the transcript of one date between two people and score it against \
four fixed criteria. You are not deciding whether they should meet again — you \
are describing what actually happened, in numbers someone else can check \
against the transcript.

Score each criterion 0-100:

TRAIT_ALIGNMENT — how much each person behaved like the traits listed for \
them. High when what they said matches who they are described as being; low \
when someone described as blunt spends the evening being diplomatic. This \
measures CONSISTENCY, not likeability.

CONVERSATIONAL_FLOW — whether it moved. High when they build on each other, \
change subject naturally, and neither is carrying it alone. Low when it is \
interview-and-answer, when one person keeps restarting it, or when replies \
stop connecting to what came before.

MUTUAL_ENGAGEMENT — whether BOTH of them were in it. This is the one that \
punishes an unbalanced date: one person delighted and one person politely \
waiting is a low score even when the transcript reads pleasantly. Look at who \
asks questions, who follows up, and who lets a subject drop.

CLASH_SEVERITY — how badly they grated on each other. **0 means they did not \
clash at all, and 0 is a completely normal score.** This is the only criterion \
where high is bad.

Then report:

- CLICKED_SUBJECTS — the things they genuinely connected over. An empty list \
is correct if nothing landed. Do not list a subject merely because it was \
mentioned.
- CLASHES — friction you can point at, each with the trait on each side and a \
short QUOTATION from the transcript as the moment. **If they did not clash, \
return an empty list.** Never invent a clash to fill the field, and never \
report one you cannot quote. A date with no clashes is a real outcome and \
saying so is the correct answer.
- PER_PEER summaries — one honest paragraph about each person's evening, \
written about them, not to them.
- VERDICT — two or three sentences on how the date actually went.

Stay inside the transcript. You may use the trait labels to name what someone \
was doing, but do not score them on traits that never came up: a trait nobody \
had a chance to show is not a failure to show it."""

JUDGE_RUBRIC_V1 = register(
    VersionedSchema(
        name="judge_rubric",
        version=1,
        json_schema={
            "type": "object",
            "required": [
                "trait_alignment",
                "conversational_flow",
                "mutual_engagement",
                "clash_severity",
                "clicked_subjects",
                "clashes",
                "per_peer_summary",
                "verdict_summary",
            ],
            "properties": {
                "trait_alignment": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": (
                        "0-100. How closely each person behaved like the traits "
                        "listed for them. Consistency, not likeability."
                    ),
                },
                "conversational_flow": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": (
                        "0-100. Whether the conversation moved and built, or "
                        "stalled and had to be restarted."
                    ),
                },
                "mutual_engagement": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": (
                        "0-100. Whether BOTH were in it. One person delighted "
                        "and one politely waiting scores low."
                    ),
                },
                "clash_severity": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": (
                        "0-100, and the ONLY criterion where high is bad. 0 "
                        "means they did not clash, which is a normal score."
                    ),
                },
                "clicked_subjects": {
                    "type": "array",
                    "description": (
                        "What they genuinely connected over. EMPTY is correct "
                        "when nothing landed."
                    ),
                    "items": {"type": "string"},
                },
                "clashes": {
                    "type": "array",
                    "description": (
                        "Friction you can quote. EMPTY is a valid verdict and "
                        "the right one when they did not clash."
                    ),
                    "items": {
                        "type": "object",
                        "required": ["user_trait", "candidate_trait", "moment"],
                        "properties": {
                            "user_trait": {
                                "type": "string",
                                "description": (
                                    "The first person's trait involved, from "
                                    "their listed labels."
                                ),
                            },
                            "candidate_trait": {
                                "type": "string",
                                "description": (
                                    "The other person's trait involved, from "
                                    "their listed labels."
                                ),
                            },
                            "moment": {
                                "type": "string",
                                "description": (
                                    "A short QUOTATION from the transcript "
                                    "where it happened. Without one, do not "
                                    "report the clash at all."
                                ),
                            },
                        },
                    },
                },
                "per_peer_summary": {
                    "type": "object",
                    "required": ["user", "candidate"],
                    "description": "One honest paragraph about each person's evening.",
                    "properties": {
                        "user": {"type": "string"},
                        "candidate": {"type": "string"},
                    },
                },
                "verdict_summary": {
                    "type": "string",
                    "description": "Two or three sentences on how the date went.",
                },
            },
        },
    )
)
