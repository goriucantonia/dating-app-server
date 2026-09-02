"""`judge_rubric.v2` — the judge's schema AND its rubric text (S12-B2, B4, B8).

**The rubric text lives here, next to the schema, and is versioned with it.**
Not in a prompt string inside the judging module, because the two cannot be
allowed to drift: `date_evaluations.rubric_version` claims that a stored score
was produced under a particular set of instructions, and that claim is only
worth something if the instructions and the shape they produce move together.
Changing a criterion's meaning is a new version, and old rows keep saying v1.

**v2 (2026-09-02, owner decision): every date is judged, however short, and the
judge reports how much it had to go on.** v1 was written for a judge that only
ever saw dates of at least ten agent turns — `app/judging.py` refused to call
it below that, the date was shown as failed, and its candidate's score was
computed as if the evening had not happened. The owner removed the threshold:
depth of evidence is a thing to REPORT, not a bar to clear. So v2 adds
`confidence` and `evidence_note`, and the rubric text now tells the judge what
to do with four turns instead of sixteen.

The four criteria and the weights are UNCHANGED. That matters: a v1 score and
a v2 score of the same transcript are the same measurement, so the version bump
does not silently re-base every number in the database. What is new is the
judge saying how sure it is.

**The four criteria are scored; the final number is NOT asked for** (S12-B5,
trade #3). There is deliberately no `overall_score` field below. A model asked
for a headline number produces one that does not follow from its own
sub-scores, and then nobody can say why a date scored 71. The weights live in
code, apply identically to everyone, and can be recomputed by hand from the
stored `criteria` — which is exactly what `probe_judge.py` does.

**`confidence` is reported, never multiplied into the score.** It would be easy
to scale `date_score` by it and call thin dates handled. That would make one
number mean two things — "how well it went" and "how much we saw" — and a
person reading a 40 could not tell which they were looking at. They stay
separate: the score says how the evening went, the confidence says how much
evening there was, and the results screen shows both.

**`clashes` can be empty, and that is a verdict** (S12-B8, §10). "Be specific"
without a way to refuse is an instruction to fabricate: a judge that must
produce a clash will invent one from two people who simply got on. So the
schema requires a citable `moment` for every clash — a quotation from the
transcript — and the rubric says plainly that an empty array is the right
answer when nothing clashed. The same applies to `clicked_subjects`, and it is
the same principle `confidence` serves at the level of the whole reading.

**The judge never sees the personas** (trade #5). It receives the transcript
plus both people's trait LABELS, for attribution only. It scores what happened
on the date, not what the profiles predicted — and that separation is what
makes a surprising result informative rather than a bug.
"""

from __future__ import annotations

from app.ai.base import VersionedSchema
from app.schemas import register

RUBRIC_VERSION = "judge_rubric.v2"

# The instructions this version's scores were produced under. Stored by name on
# every evaluation; edit the meaning of anything here and it becomes v3.
JUDGE_SYSTEM_PROMPT = """\
You read the transcript of one date between two people and score it against \
four fixed criteria. You are not deciding whether they should meet again — you \
are describing what actually happened, in numbers someone else can check \
against the transcript.

HOW MUCH THERE IS TO READ VARIES, AND THAT IS NORMAL

Some transcripts run a full evening. Others stop after a few lines, because \
something broke or because the two of them wound it up early. **You judge all \
of them.** A short date is not unjudgeable; it is thinly evidenced, and the \
difference matters:

- Scale what you CLAIM to what you saw. Sixteen turns will support a sentence \
like "she steered every subject back to herself". Four turns will support "she \
opened warmly and he matched it", and nothing more. Say the smaller thing.
- Score only what had a chance to appear. If a criterion had no opportunity to \
show itself in the lines you were given, put it near the middle rather than \
low — a thing that never came up is not a thing that went badly — and say so \
in your evidence note.
- Never pad. Do not extrapolate how the evening WOULD have gone, do not \
describe momentum you did not see, and do not write a paragraph's worth of \
character study off two lines.

Score each criterion 0-100:

TRAIT_ALIGNMENT — how much each person behaved like the traits listed for \
them. High when what they said matches who they are described as being; low \
when someone described as blunt spends the evening being diplomatic. This \
measures CONSISTENCY, not likeability.

CONVERSATIONAL_FLOW — whether it moved. High when they build on each other, \
change subject naturally, and neither is carrying it alone. Low when it is \
interview-and-answer, when one person keeps restarting it, or when replies \
stop connecting to what came before. A date that was CUT SHORT is not a date \
that stalled — do not score the cut as a failure of flow.

MUTUAL_ENGAGEMENT — whether BOTH of them were in it. This is the one that \
punishes an unbalanced date: one person delighted and one person politely \
waiting is a low score even when the transcript reads pleasantly. Look at who \
asks questions, who follows up, and who lets a subject drop.

CLASH_SEVERITY — how badly they grated on each other. **0 means they did not \
clash at all, and 0 is a completely normal score.** This is the only criterion \
where high is bad.

Then report:

- CONFIDENCE — 0-100, how much this transcript actually supported the reading \
you just gave. Base it on how much was said and how much of it was revealing, \
not on how much you liked them. A full evening where both people talked openly \
is high. A handful of polite opening lines is low, even if it was pleasant. \
This is you telling the reader how much weight to put on your own numbers, and \
a low value is a useful answer, not an admission of failure.
- EVIDENCE_NOTE — one sentence naming what this transcript could and could not \
show. "Long enough to see how they handle disagreement" or "stops before \
either of them says anything they had to think about". Write it about the \
TRANSCRIPT, not about the people.
- CLICKED_SUBJECTS — the things they genuinely connected over. An empty list \
is correct if nothing landed. Do not list a subject merely because it was \
mentioned.
- CLASHES — friction you can point at, each with the trait on each side and a \
short QUOTATION from the transcript as the moment. **If they did not clash, \
return an empty list.** Never invent a clash to fill the field, and never \
report one you cannot quote. A date with no clashes is a real outcome and \
saying so is the correct answer.
- PER_PEER summaries — one honest paragraph about each person's evening, \
written about them, not to them. On a thin transcript this is two or three \
sentences, not a paragraph stretched to look like one.
- VERDICT — two or three sentences on how the date actually went.

Stay inside the transcript. You may use the trait labels to name what someone \
was doing, but do not score them on traits that never came up: a trait nobody \
had a chance to show is not a failure to show it."""

JUDGE_RUBRIC_V2 = register(
    VersionedSchema(
        name="judge_rubric",
        version=2,
        json_schema={
            "type": "object",
            "required": [
                "trait_alignment",
                "conversational_flow",
                "mutual_engagement",
                "clash_severity",
                "confidence",
                "evidence_note",
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
                        "stalled and had to be restarted. A transcript that was "
                        "cut off is not one that stalled."
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
                "confidence": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 100,
                    "description": (
                        "0-100. How much this transcript supported the reading "
                        "above — how much was said, and how revealing it was. "
                        "A short or guarded date scores LOW here even when it "
                        "went pleasantly. This is not a score of the date."
                    ),
                },
                "evidence_note": {
                    "type": "string",
                    "description": (
                        "One sentence on what this transcript could and could "
                        "not show. About the transcript, not about the people."
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
                    "description": (
                        "One honest paragraph about each person's evening — "
                        "shorter when the transcript is short."
                    ),
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
