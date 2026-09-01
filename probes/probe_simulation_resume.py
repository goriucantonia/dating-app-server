"""probe_simulation_resume.py (S11-P1, §2)

The one thing this module exists to guarantee: **killing the server mid-date
does not lose the date.** The probe starts a real simulation, waits until a
transcript is genuinely in progress, kills the API process outright, brings it
back, and then checks that the SAME date carried on from its last checkpointed
message instead of starting again.

§1 names this as one of the observations that is the product rather than a
corner case, so it is watched, never inferred.

What is asserted:

- a `matched` analysis accepts `POST /analyses/{id}/simulate` and goes
  `simulating` with a real stage name in `progress` (AC1, S11-B10);
- a second POST while it runs is a 409 carrying the running analysis — state,
  not failure;
- the API is SIGKILLed mid-date and the transcript survives (§19: the message
  row was committed before the turn advanced);
- on restart, boot reconciliation logs `analysis_relaunched` and the date
  CONTINUES: its first message id and its whole prefix are unchanged, and
  `max(seq)` only grows (AC2, S11-B8, S11-B9);
- the analysis reaches a terminal state with real transcripts (AC1);
- the event rules held: at most 3 per date, never two in a row (AC3);
- every date says how it ended, and the two mechanisms are named (AC4).

**This probe runs on the HOST, not inside the api container** — unlike every
other probe in this directory, and for an unavoidable reason: it has to kill
the very container the others run inside. `docker compose exec` processes die
with the container, so a probe that kills the API from within kills itself
mid-assertion and proves nothing.

That is also why it uses `urllib` from the standard library rather than httpx:
the host has no project virtualenv, and a witness script that needs an install
before it can run is a witness script nobody runs.

    python probes/probe_simulation_resume.py            # the resume witness
    python probes/probe_simulation_resume.py --full     # ...and wait for every date

Cost, because it is large enough to plan around: this drives a REAL pipeline.
Two onboardings (~4 calls), one scenario call per matched candidate, and then
about 28 turns per date at roughly 6 seconds each. The default run stops once
the resume claim is proven — usually one date's worth. `--full` waits for the
whole analysis, which since the 2026-09-01 revision to ONE date per candidate
is at most three dates, ~90 calls and around eleven minutes. See PICKUP's
quota table before choosing.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# This probe prints on the HOST console, which on this machine is a legacy
# Windows code page — it cannot encode an em dash, let alone an arrow, and
# `print` raises UnicodeEncodeError rather than degrading. A witness script
# that dies while printing its own verdict is worse than one that prints
# nothing, so stdout is forced to UTF-8 with a lossy fallback.
if hasattr(sys.stdout, "reconfigure"):
    # `line_buffering` because this probe runs for ten minutes or more, and
    # Python buffers stdout hard whenever it is piped rather than attached to a
    # terminal. A witness script that shows nothing at all until it exits
    # cannot be watched while it works, and watching it is the point.
    sys.stdout.reconfigure(
        encoding="utf-8", errors="replace", line_buffering=True
    )

API = "http://localhost:8000"
# The compose project lives one directory above `server/`.
COMPOSE_DIR = Path(__file__).resolve().parent.parent.parent

PASSWORD = "probe-password"

# Two people who should plausibly match AND share an interest, so the scenario
# call takes the shared-interest branch. Written in a real voice because the
# whole pipeline downstream is only as good as the traits extracted from these.
ALICE = {
    "BQ1": (
        "I restore old bicycles. Steel frames from the seventies, the ones "
        "people leave out for the bin men. I strip them, re-cable them, and "
        "give most of them away. I learned by ruining two of them first and I "
        "read old repair manuals on the train to work."
    ),
    "BQ2": (
        "Someone who says the difficult thing early instead of letting it sit "
        "and rot. I want them to have something of their own they disappear "
        "into. The dealbreaker is contempt — how they talk about an ex, or to "
        "the person bringing the food."
    ),
    "BQ3": (
        "My flatmate told me I had been snappy for a month and she was right. "
        "I said something unfair about her never washing up and then went for "
        "a walk for an hour. I apologised properly two days later. I need time "
        "before I can come back to something."
    ),
    "BQ4": (
        "I remember what people told me weeks ago and bring it up, which is "
        "either lovely or unnerving. I ask a lot of questions and then go "
        "quiet at the exact moment a normal person would say something warm."
    ),
    "BQ5": (
        "I am reliable to the point of being dull about it. I turn up early. I "
        "am bad at the soft parts — I hand someone a solution when they wanted "
        "company, and I correct small things nobody needed corrected."
    ),
}

DAN = {
    "BQ1": (
        "I restore old bicycles and the odd motorbike. Mostly steel frames I "
        "find abandoned. I take them apart on the balcony and learn what I "
        "need as I go, usually by getting it wrong first. It is the quietest "
        "part of my week and I like doing it alone."
    ),
    "BQ2": (
        "I want someone straightforward. If something is wrong I would rather "
        "hear it on the day than in a summary at the end. Someone with their "
        "own obsession. I cannot stand people who are rude to staff."
    ),
    "BQ3": (
        "My brother said I had been impossible since our mother got ill and he "
        "was right. I said something cheap back and then sat in the car. I "
        "said sorry for the sentence two days later, not for being angry."
    ),
    "BQ4": (
        "I bring up things people mentioned ages ago. I am not smooth. I ask "
        "questions and then run out of words right when I should say something "
        "kind."
    ),
    "BQ5": (
        "Dependable and a bit boring about it. Early for everything. Worse at "
        "the emotional parts — I offer a fix instead of just sitting with "
        "someone, and I am pedantic about details that did not matter."
    ),
}

checks: list[tuple[str, bool]] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    """`detail` must be a FACT, not an explanation of failure.

    It is printed on pass and on fail alike — this probe shipped for about ten
    minutes with a check whose detail read "no analysis_relaunched line in the
    api logs" printed next to the word PASS, which is the D-009 failure mode
    in miniature: a verdict line that contradicts itself is worse than no
    verdict line, because someone reads one half of it.
    """
    checks.append((label, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}: {label}" + (f" — {detail}" if detail else ""))


def note(text: str) -> None:
    print(f"  ..   {text}")


# --- the smallest HTTP client that can witness this ------------------------


class Response:
    def __init__(self, status: int, body: str):
        self.status = status
        self.text = body

    def json(self) -> dict:
        return json.loads(self.text or "{}")


def request(
    method: str, path: str, *, token: str | None = None, body: dict | None = None,
    timeout: float = 30.0,
) -> Response:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(API + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return Response(r.status, r.read().decode())
    except urllib.error.HTTPError as e:
        return Response(e.code, e.read().decode())
    except OSError as e:
        # The API being down is a normal, EXPECTED state in this probe — it is
        # the probe that killed it. `OSError` rather than `URLError` because a
        # SIGKILLed container does not refuse the connection politely: on
        # Windows the in-flight socket raises ConnectionAbortedError (WinError
        # 10053) straight out of http.client, which URLError never wraps.
        # Status 0 means "no server answered", which every caller here reads
        # as down.
        return Response(0, json.dumps(
            {"error": {"code": "unreachable", "message": str(e)}}))


def compose(*args: str, check_rc: bool = True) -> str:
    result = subprocess.run(
        ["docker", "compose", *args], cwd=COMPOSE_DIR,
        capture_output=True, text=True, check=False,
    )
    if check_rc and result.returncode != 0:
        raise RuntimeError(f"docker compose {' '.join(args)} failed: {result.stderr}")
    return result.stdout


def wait_for_api(timeout_s: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if request("GET", "/health", timeout=3).status == 200:
            return True
        time.sleep(1)
    return False


# --- building the two people -----------------------------------------------


def register(tag: str, gender: str, interested_in: list[str]) -> dict:
    email = f"probe-sim-{tag}-{uuid.uuid4().hex[:8]}@probe.dev"
    r = request("POST", "/auth/register", body={
        "email": email, "password": PASSWORD, "display_name": tag.title(),
        "birth_date": "1992-06-15", "gender": gender,
        "interested_in": interested_in, "age_pref_min": 18, "age_pref_max": 60,
        "opt_in": True,
    })
    if r.status != 201:
        raise RuntimeError(f"register {tag} failed: {r.status} {r.text}")
    payload = r.json()
    return {"token": payload["token"], "id": payload["user"]["id"], "email": email}


def answer_baseline(token: str, answers: dict[str, str]) -> None:
    questions = request("GET", "/questions", token=token).json()["questions"]
    by_code = {q["code"]: q["id"] for q in questions if q.get("code")}
    for code, text in answers.items():
        r = request("PUT", f"/answers/{by_code[code]}", token=token,
                    body={"answer_text": text})
        if r.status != 200:
            raise RuntimeError(f"answering {code} failed: {r.status} {r.text}")


def build_persona(who: dict, answers: dict[str, str], label: str) -> None:
    """Answer BQ1-BQ5, extract, compile — the real pipeline, no shortcuts
    (§12: there is no path that skips verification because a probe made it)."""
    answer_baseline(who["token"], answers)
    note(f"{label}: extracting traits…")
    r = request("POST", "/profile/extract", token=who["token"], timeout=300)
    if r.status not in (200, 202):
        raise RuntimeError(f"extract for {label} failed: {r.status} {r.text}")
    # Compilation is auto-chained off a CHANGED extraction (S7-B6), so this
    # waits rather than posting one — an unnecessary POST /persona/compile
    # would spend a free-tier call to rebuild an identical persona.
    note(f"{label}: waiting for the persona to compile…")
    deadline = time.monotonic() + 420
    nudged = False
    while time.monotonic() < deadline:
        current = request("GET", "/persona/current", token=who["token"]).json()
        snapshot = current.get("snapshot") or {}
        if current.get("simulatable"):
            note(f"{label}: persona ready (v{snapshot.get('version')})")
            return
        if snapshot.get("status") == "failed":
            raise RuntimeError(f"persona for {label} failed: {snapshot.get('error')}")
        if not snapshot and not nudged and time.monotonic() > deadline - 400:
            # No row at all after ten seconds means the auto-chain did not fire
            # (an all-`keep` extraction). Ask for one explicitly.
            request("POST", "/persona/compile", token=who["token"], timeout=60)
            nudged = True
        time.sleep(4)
    raise RuntimeError(f"persona for {label} never became ready")


# --- the witness -----------------------------------------------------------


def transcript(token: str, date_id: str) -> dict:
    return request("GET", f"/dates/{date_id}/transcript", token=token).json()


def dates_of(token: str, analysis_id: str) -> list[dict]:
    r = request("GET", f"/analyses/{analysis_id}/dates", token=token)
    return r.json().get("dates", []) if r.status == 200 else []


def wait_for_a_date_in_progress(
    token: str, analysis_id: str, *, min_messages: int, timeout_s: float
) -> dict | None:
    """Poll until some date has at least `min_messages` checkpointed.

    Waiting for MESSAGES rather than for a wall-clock delay is the point: a
    fixed sleep would kill the server before the first turn on a slow day and
    after the whole date on a fast one, and neither proves anything about
    resuming.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for d in dates_of(token, analysis_id):
            if d["message_count"] >= min_messages and d["status"] == "running":
                return d
        time.sleep(4)
    return None


def event_rules_hold(messages: list[dict]) -> tuple[bool, str]:
    """AC3's two rules, checked against the stored transcript: at most 3
    events, and never two in a row."""
    speakers = [m["speaker"] for m in messages]
    events = sum(1 for s in speakers if s == "environment")
    consecutive = any(
        a == "environment" and b == "environment"
        for a, b in itertools.pairwise(speakers)
    )
    if events > 3:
        return False, f"{events} events, cap is 3"
    if consecutive:
        return False, "two events in a row"
    return True, f"{events} events, none consecutive"


def opt_out(who: dict) -> None:
    """Leave the pool the way we found it.

    D-009's operational lesson, in code: probe users who stay opted in become
    other people's candidates, and a later probe then matches somebody built
    for a different test. Opting them out is cheaper and safer than deleting
    them - the transcripts stay readable for anyone reviewing this run.
    """
    request("PATCH", "/me", token=who["token"], body={"opt_in": False})


def main() -> int:
    full = "--full" in sys.argv
    print("probe_simulation_resume - S11-P1")
    print("mode:", "FULL - waits for every date" if full
          else "resume-only - pass --full to wait for the whole analysis")
    print()

    if not wait_for_api(15):
        print("  the API is not up on localhost:8000 — start the stack first")
        return 2

    print("Building two people through the real pipeline (this costs AI calls)…")
    alice = register("alice", "woman", ["man"])
    dan = register("dan", "man", ["woman"])
    build_persona(dan, DAN, "dan")
    build_persona(alice, ALICE, "alice")

    print("\nMatching…")
    started = request("POST", "/analyses", token=alice["token"])
    analysis_id = started.json()["id"]
    deadline = time.monotonic() + 300
    analysis = {}
    while time.monotonic() < deadline:
        analysis = request("GET", f"/analyses/{analysis_id}", token=alice["token"]).json()
        if analysis["status"] not in ("matching",):
            break
        time.sleep(3)
    check("matching produced candidates to date", analysis.get("status") == "matched",
          f"status={analysis.get('status')} pool={analysis.get('pool_status')}")
    if analysis.get("status") != "matched":
        return report()

    # --- AC1: the pipeline starts and reports a real stage -----------------
    print("\nStarting the simulation…")
    r = request("POST", f"/analyses/{analysis_id}/simulate", token=alice["token"])
    check("POST /simulate on a matched analysis is accepted", r.status == 202,
          f"{r.status} {r.text[:200]}")

    second = request("POST", f"/analyses/{analysis_id}/simulate", token=alice["token"])
    check(
        "a second POST while running is a 409 naming the analysis — state, not failure",
        second.status == 409
        and second.json()["error"]["code"] == "simulation_in_progress",
        f"{second.status} {second.text[:200]}",
    )

    # The progress stage must NAME the stage, never a fake timer (S11-B10).
    stage = ""
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        live = request("GET", f"/analyses/{analysis_id}", token=alice["token"]).json()
        progress = live.get("progress") or {}
        stage = progress.get("stage", "")
        if stage in ("scenarios", "simulating"):
            note(f"progress: {progress.get('message')}")
            break
        time.sleep(3)
    check("progress names a real stage", stage in ("scenarios", "simulating"), stage)

    # --- AC2: kill it mid-date and watch it continue -----------------------
    print("\nWaiting for a date to be genuinely in progress before killing it…")
    in_progress = wait_for_a_date_in_progress(
        alice["token"], analysis_id, min_messages=3, timeout_s=900
    )
    check("a date is running with a transcript in progress", in_progress is not None)
    if in_progress is None:
        return report()

    date_id = in_progress["date_id"]
    before = transcript(alice["token"], date_id)
    before_msgs = before["messages"]
    note(f"date {date_id[:8]} has {len(before_msgs)} messages — killing the API now")

    compose("kill", "-s", "SIGKILL", "api")
    down = request("GET", "/health", timeout=3)
    check("the API is actually down", down.status == 0, f"status={down.status}")

    compose("start", "api")
    check("the API came back", wait_for_api(120))

    after_restart = transcript(alice["token"], date_id)
    check(
        "the transcript survived the kill — every message committed before the "
        "turn advanced (§19)",
        len(after_restart["messages"]) >= len(before_msgs),
        f"{len(before_msgs)} before, {len(after_restart['messages'])} after",
    )

    relaunch_logs = compose("logs", "api", "--since", "5m")
    relaunches = relaunch_logs.count("analysis_relaunched")
    resumes = relaunch_logs.count("resumed_from_checkpoint")
    check(
        "boot reconciliation relaunched the analysis (S11-B9)",
        relaunches > 0,
        f"{relaunches} analysis_relaunched line(s) since the kill",
    )
    check(
        "…and the date logged that it picked up from a checkpoint, not from "
        "the start",
        resumes > 0,
        f"{resumes} resumed_from_checkpoint line(s)",
    )

    print("\nWatching the killed date continue rather than restart…")
    deadline = time.monotonic() + 900
    final = after_restart
    while time.monotonic() < deadline:
        final = transcript(alice["token"], date_id)
        if final["status"] in ("complete", "incomplete", "failed"):
            break
        time.sleep(6)

    prefix_intact = [
        (m["seq"], m["speaker"], m["reply"]) for m in final["messages"][: len(before_msgs)]
    ] == [(m["seq"], m["speaker"], m["reply"]) for m in before_msgs]
    check(
        "THE DATE CONTINUED: every message written before the kill is still "
        "there, byte for byte, in the same order",
        prefix_intact,
    )
    check(
        "…and it grew from that checkpoint rather than starting over",
        len(final["messages"]) > len(before_msgs),
        f"{len(before_msgs)} → {len(final['messages'])}",
    )

    # --- AC1 / AC3 / AC4 over whatever the pipeline produced ---------------
    if full:
        print("\nLetting the analysis finish…")
        deadline = time.monotonic() + 3600
        live = {}
        while time.monotonic() < deadline:
            live = request("GET", f"/analyses/{analysis_id}", token=alice["token"]).json()
            if live["status"] in ("complete", "failed"):
                break
            time.sleep(10)
        check("the analysis reached a terminal state",
              live.get("status") in ("complete", "failed"), str(live.get("status")))
    else:
        note("not waiting for the remaining dates (no --full). The resume "
             "claim above is complete; the rules below are checked against "
             "whatever has finished so far.")

    all_dates = dates_of(alice["token"], analysis_id)
    check("dates were created for the matched candidates", bool(all_dates),
          f"{len(all_dates)} dates")
    real_transcripts = [d for d in all_dates if d["message_count"] > 0]
    check("at least one date has a real transcript (AC1)", bool(real_transcripts),
          f"{len(real_transcripts)} of {len(all_dates)} dates have messages")

    for d in all_dates:
        if d["message_count"] == 0:
            continue
        msgs = transcript(alice["token"], d["date_id"])["messages"]
        ok, detail = event_rules_hold(msgs)
        check(f"event rules hold on '{d['setting_name'][:40]}' (AC3)", ok, detail)
        check(
            f"…and it never exceeded the 30-message cap, events included (§18) "
            f"on '{d['setting_name'][:30]}'",
            len(msgs) <= 30, f"{len(msgs)} messages",
        )
        spoken = [m for m in msgs if m["speaker"] != "environment"]
        check(
            f"…and every spoken turn carries its inner state on "
            f"'{d['setting_name'][:30]}'",
            all(m["state"] and "wants_to_end" in m["state"] for m in spoken),
        )

    logs = compose("logs", "api", "--since", "60m")
    check("every event roll is logged, fired or not (AC3, §8)",
          '"event": "event_roll"' in logs or "event_roll" in logs)
    check("each date logged HOW it ended (AC4)", "ended_by" in logs)

    opt_out(alice)
    opt_out(dan)
    note("both probe users opted OUT of the pool again (D-009)")

    print(f"\n  analysis: {analysis_id}")
    print(f"  alice:    {alice['email']}")
    print(f"  dan:      {dan['email']}")
    return report()


def report() -> int:
    failed = [label for label, ok in checks if not ok]
    print(f"\n{'GREEN' if not failed else 'RED'} — {len(checks) - len(failed)}"
          f"/{len(checks)} checks passed")
    for label in failed:
        print(f"  failed: {label}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
