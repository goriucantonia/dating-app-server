# Module Plan — Chat Module

Status: planning locked 2026-09-01. Depends on: `date_simulation.md` (results, `DateDigest`), `trait_persona.md` (snapshots, `agent_response.v1`). Final server module before Data Hygiene.

---

## 1. Purpose

After the user selects their one match from a completed analysis, this module runs the ongoing conversation between the human user and the AI persona of that match. Direct human-to-human chat stays out of scope; the selected person is not notified.

## 2. Core features and async operation

- **Selection creates the session.** `POST /analyses/{id}/select {candidate_user_id}` — valid only on a `complete` analysis, for one of its candidates, once per analysis. Creates the session, pins the candidate's **matched snapshot** (the same one the dates ran against — the user chats with the person they read transcripts of, not a possibly-recompiled newer persona), and compiles the date digest once via `DateDigest` (no AI call; assembled from evaluations).
- **Turn handling.** `POST /chat/sessions/{id}/messages` is a plain async request–response: persist the user message, one AI call, persist and return the persona reply. Seconds, not minutes — no background job, no polling. *(Trade: no token streaming this phase; a reply arrives whole after a few seconds. Accepted — streaming across three Flutter platforms is polish, not function. Revisit if replies feel dead.)*
- **Same schema, hidden metadata.** Persona replies go through the Structured Output Guard with `agent_response.v1` — identical to dates — so internal state keeps being tracked and stored. The chat UI does **not** display the metadata *(trade: less spectacle; accepted — a live conversation where you watch the other side's connection meter turns chatting into gaming a gauge; the data still exists for a future "relationship insights" view)*.
- **Simulated history, honestly framed.** The system prompt extends the persona snapshot with:
  1. the date digest ("in a simulated date at the car meet, you two clicked about engine swaps; a vintage Mustang pulled up…");
  2. the standing instruction: these dates were simulations the human wasn't present for — refer to them as "our simulated date," never as a lived shared memory, and never invent details beyond the digest.
- **Unbounded-length chats need compaction** (dates are capped at 30 messages; chats aren't). Rolling window: the last 40 messages verbatim + a running summary of everything older. When the window overflows, one structured AI call folds the oldest 20 into the summary (stored on the session, versioned by `compacted_upto_seq`). Compaction runs inline before the reply call when needed — one extra call every 20 messages, invisible at chat pace.
- **Session lifecycle.** User can end the chat (status `ended`), go answer more questions, or start a new analysis. Multiple sessions may exist over time; only the persona snapshot pinned at selection is ever used per session — a new analysis + selection makes a new session with fresher everything.

## 3. Data flow and database

```
POST /analyses/{id}/select ─► chat_sessions (pins snapshot_id, digest)
POST /chat/sessions/{id}/messages
   ─► INSERT user msg ─► [compact if window full] ─► AI call (snapshot prompt + digest
        + summary + last-40 window) ─► INSERT persona msg (with hidden state) ─► return
```

```sql
CREATE TABLE chat_sessions (
    id                 UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id            UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    match_user_id      UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    analysis_id        UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    snapshot_id        UUID NOT NULL REFERENCES persona_snapshots(id),  -- the matched snapshot
    date_digest        TEXT NOT NULL,
    summary            TEXT,                        -- compacted history, NULL until first compaction
    compacted_upto_seq INT NOT NULL DEFAULT 0,
    status             TEXT NOT NULL CHECK (status IN ('active','ended')),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    ended_at           TIMESTAMPTZ,
    UNIQUE (analysis_id)                            -- one selection per analysis
);

CREATE TABLE chat_messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    seq         INT  NOT NULL,
    sender      TEXT NOT NULL CHECK (sender IN ('user','persona')),
    text        TEXT NOT NULL,
    state       JSONB,                              -- persona internal state; NULL for user msgs
    provider    TEXT, model_id TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, seq)
);
```

Deletion note for Data Hygiene: `match_user_id` cascade means if the *matched friend* deletes their account, the session and messages vanish from the selecting user's app too — consistent with the deletion philosophy (their persona is their data). Named again in `data_hygiene.md`.

## 4. Interfaces and endpoints

```python
class ChatService(Protocol):
    async def select_match(self, analysis_id: UUID, candidate_user_id: UUID) -> UUID:  # session_id
    async def send_message(self, session_id: UUID, text: str) -> ChatReply:            # persona reply
    async def end_session(self, session_id: UUID) -> None:
```

| Endpoint | Behavior |
|---|---|
| `POST /analyses/{id}/select` | Create the session; 409 if analysis not `complete` or already selected. |
| `GET /chat/sessions` | The user's sessions, active first. |
| `GET /chat/sessions/{id}/messages?after_seq=` | Paged history (no `state` field in the response — metadata stays server-side). |
| `POST /chat/sessions/{id}/messages` | Send + receive, as above. |
| `POST /chat/sessions/{id}/end` | Mark ended. |

Logging obligations (§7): every reply logs session/seq/provider/model/attempt/outcome; compaction logs the folded range and summary length; a Structured-Output give-up returns a explicit "couldn't reply, try again" error to the client — never a silently degraded plain-text fallback — and logs the raw output.

## Locked by this document

1. Selection pins the matched snapshot; one selection per analysis; sessions persist.
2. Replies use `agent_response.v1`; metadata stored, hidden from the chat UI.
3. Simulated-history framing rules, verbatim in the system prompt.
4. Compaction: last-40 verbatim window + running summary folded every 20.
5. Synchronous request–response chat; no streaming this phase.
