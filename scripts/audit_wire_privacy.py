"""Wire privacy audit against RAW response bodies (S16-B7,
`communication_protocol.md` §6). Widgets are not consulted; the bytes are.

The five rules, each checked on the body text of the endpoints where it
could be broken:

1. Persona system prompts never cross — `/persona/current`, every analysis,
   every chat payload, every transcript.
2. Another user's raw answers and full trait DESCRIPTIONS never cross — the
   candidate payloads carry labels only.
3. Chat internal-state metadata never crosses — no `"state"` key in any chat
   payload.
4. `is_demo` ALWAYS crosses wherever a user is rendered — `/me`, candidates,
   chat matches.
5. Non-candidates appear in no payload — the only user ids on the wire
   belong to the caller, their candidates, or their matches.

The one deliberate exposure — both agents' state on a TRANSCRIPT — is
asserted to be present there and nowhere else.

    docker compose exec api python scripts/audit_wire_privacy.py <email> [password]

Uses an account that owns at least one complete analysis and one chat.
"""

from __future__ import annotations

import json
import re
import sys

import httpx

API = "http://localhost:8000"
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")

checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    checks.append((label, ok))
    print(f"  {'PASS' if ok else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    email = sys.argv[1]
    password = sys.argv[2] if len(sys.argv) > 2 else "probe-password"
    with httpx.Client(base_url=API, timeout=30) as c:
        r = c.post("/auth/login", json={"email": email, "password": password})
        if r.status_code != 200:
            print(f"login failed: {r.status_code} {r.text[:200]}")
            return 2
        h = {"Authorization": f"Bearer {r.json()['token']}"}
        me = r.json()["user"]
        my_id = me["id"]

        bodies: dict[str, str] = {}
        bodies["/me"] = c.get("/me", headers=h).text
        bodies["/persona/current"] = c.get("/persona/current", headers=h).text
        bodies["/traits"] = c.get("/traits", headers=h).text
        analyses_raw = c.get("/analyses", headers=h).text
        bodies["/analyses"] = analyses_raw
        analyses = json.loads(analyses_raw)["analyses"]
        candidate_ids: set[str] = set()
        for a in analyses:
            for cand in a["candidates"]:
                candidate_ids.add(cand["candidate_user_id"])
            bodies[f"/analyses/{a['id'][:8]}"] = c.get(f"/analyses/{a['id']}", headers=h).text
            dates_raw = c.get(f"/analyses/{a['id']}/dates", headers=h).text
            bodies[f"/analyses/{a['id'][:8]}/dates"] = dates_raw
            for d in json.loads(dates_raw)["dates"]:
                bodies[f"/dates/{d['date_id'][:8]}/transcript"] = c.get(
                    f"/dates/{d['date_id']}/transcript", headers=h).text
        sessions_raw = c.get("/chat/sessions", headers=h).text
        bodies["/chat/sessions"] = sessions_raw
        match_ids: set[str] = set()
        for s in json.loads(sessions_raw)["sessions"]:
            match_ids.add(s["match"]["user_id"])
            sid = s["session_id"]
            bodies[f"/chat/sessions/{sid[:8]}"] = c.get(f"/chat/sessions/{sid}", headers=h).text
            bodies[f"/chat/sessions/{sid[:8]}/messages"] = c.get(
                f"/chat/sessions/{sid}/messages", headers=h).text

    print(f"audited {len(bodies)} raw bodies for {email}")

    # Rule 1: no system prompt anywhere.
    check("rule 1: no `system_prompt` in any body",
          all('"system_prompt"' not in b for b in bodies.values()))

    # Rule 2: candidates carry labels only — no `description`, no `answer_text`
    # in analysis payloads. (`description` on a DATE is the scenario's, so the
    # check is scoped to the analysis payloads where a trait description could
    # hide.) `/traits` is the caller's OWN profile and legitimately has them.
    analysis_bodies = {k: v for k, v in bodies.items()
                       if k.startswith("/analyses") and not k.endswith("/dates")}
    check("rule 2: no trait `description` or `answer_text` in analysis payloads",
          all('"description"' not in b and '"answer_text"' not in b
              for b in analysis_bodies.values()))
    check("rule 2: candidates carry `trait_labels`",
          all('"trait_labels"' in b for k, b in analysis_bodies.items()
              if '"candidates": [{' in b.replace(" ", "")))

    # Rule 3: no `state` in any chat payload.
    chat_bodies = {k: v for k, v in bodies.items() if k.startswith("/chat")}
    check("rule 3: no `\"state\"` in any chat payload",
          all('"state"' not in b for b in chat_bodies.values()), f"{len(chat_bodies)} bodies")

    # Rule 4: is_demo everywhere a user is rendered.
    check("rule 4: `is_demo` on /me", '"is_demo"' in bodies["/me"])
    check("rule 4: `is_demo` on every candidate",
          all(cand.get("is_demo") in (True, False) for a in analyses for cand in a["candidates"]))
    # Only the bodies that render a person: the list and the detail. A page of
    # messages renders no user and must not be asked for a label it has no
    # one to attach to.
    check("rule 4: `is_demo` on every chat match",
          all('"is_demo"' in b for b in chat_bodies.values() if '"match"' in b))

    # Rule 5: only the caller, their candidates, and their matches appear as user ids.
    allowed = {my_id} | candidate_ids | match_ids
    seen_users: set[str] = set()
    for b in bodies.values():
        for key in ("candidate_user_id", "user_id", "match_user_id"):
            for m in re.finditer(rf'"{key}":\s*"({UUID_RE.pattern})"', b):
                seen_users.add(m.group(1))
    strangers = seen_users - allowed
    check("rule 5: no user id on the wire outside caller/candidates/matches",
          not strangers, f"{len(seen_users)} user ids seen")

    # The deliberate exposure: state ONLY on transcripts.
    transcripts = {k: v for k, v in bodies.items() if k.endswith("/transcript")}
    check("transcripts DO carry `state` (the one deliberate exposure)",
          all('"state"' in b for b in transcripts.values()), f"{len(transcripts)} transcripts")
    check("`connection`/`satisfaction` appear ONLY on transcripts",
          all(('"connection"' not in b and '"satisfaction"' not in b)
              for k, b in bodies.items() if not k.endswith("/transcript")))

    failed = [l for l, ok in checks if not ok]
    print(f"\naudit_wire_privacy: {'GREEN' if not failed else 'RED'} — "
          f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
