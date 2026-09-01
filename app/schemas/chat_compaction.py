"""`chat_compaction.v1` — the one structured call that keeps an unbounded chat
inside a bounded context (S14-B7, `chat.md` §2).

Dates are capped at 16 turns; chats are not. The window is the last 40
messages verbatim plus a running summary of everything older, and when the
window overflows this call folds the oldest 20 messages into the summary. It
asks for ONE field on purpose: a summary is prose, and a schema with slots
for "topics" and "plans" and "mood" would make the model invent a plan every
time the window rolls (§10 — "be specific" without a way to refuse is an
instruction to fabricate).

The rules the summary must obey are in the description so the model reads
them on every call rather than in a prompt that can drift from the schema:
third person, facts either side actually stated, nothing inferred, and the
simulated-date framing preserved word for word — a summary that quietly
turns "our simulated date" into "our date" would undo the anti-gaslighting
rule twenty messages at a time.
"""

from __future__ import annotations

from app.ai.base import VersionedSchema
from app.schemas import register

COMPACTION_VERSION = "chat_compaction.v1"

CHAT_COMPACTION_V1 = register(
    VersionedSchema(
        name="chat_compaction",
        version=1,
        json_schema={
            "type": "object",
            "required": ["summary"],
            "properties": {
                "summary": {
                    "type": "string",
                    "description": (
                        "A running summary of the conversation so far, "
                        "replacing the previous summary. Third person, plain "
                        "facts only: what each person said about themselves, "
                        "what they asked, any plans or preferences either of "
                        "them stated. Keep everything from the previous summary "
                        "that is still true. Do not infer feelings or intentions "
                        "that were not said. If the simulated date is mentioned, "
                        "keep calling it 'the simulated date' — never 'their "
                        "date'. At most 300 words."
                    ),
                },
            },
        },
    )
)
