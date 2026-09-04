"""Every promised route is registered, on the handler it belongs to.

A decorator that lands on the wrong function is invisible to the suite: the
module imports, the tests that call handlers directly pass, and the route is
simply gone (`PATCH /me` on 2026-09-03 — the guard helper inserted above it
took the decorator, and the real handler was never registered). This pins the
(method, path, handler name) table so that cannot happen silently again.
"""

from __future__ import annotations

from app.routers import analyses, auth, chat, me, persona, questions, simulation, traits

EXPECTED = {
    ("POST", "/auth/register", "register"),
    ("POST", "/auth/login", "login"),
    ("GET", "/me", "get_me"),
    ("PATCH", "/me", "patch_me"),
    ("DELETE", "/me", "delete_me"),
    ("GET", "/questions", "list_questions"),
    ("GET", "/questions/next-batch", "next_batch"),
    ("PUT", "/answers/{question_id}", "upsert_answer"),
    ("POST", "/profile/extract", "extract"),
    ("GET", "/profile/extract/status", "extract_status"),
    ("GET", "/traits", "list_traits"),
    ("POST", "/traits/{trait_id}/confirm", "confirm_trait"),
    ("POST", "/traits/{trait_id}/dispute", "dispute_trait"),
    ("POST", "/persona/compile", "compile_endpoint"),
    ("GET", "/persona/current", "current"),
    ("POST", "/calibration/sessions", "start_session"),
    ("POST", "/calibration/sessions/{session_id}/messages", "send_message"),
    ("GET", "/calibration/sessions/{session_id}/messages", "list_messages"),
    ("POST", "/calibration/messages/{message_id}/flag", "flag_message"),
    ("GET", "/calibration/flags/count", "flag_count"),
    ("POST", "/analyses", "create_analysis"),
    ("GET", "/analyses", "list_analyses"),
    ("GET", "/analyses/{analysis_id}", "get_analysis"),
    ("POST", "/analyses/{analysis_id}/candidates/{candidate_user_id}/reject", "reject_candidate"),
    ("POST", "/analyses/{analysis_id}/simulate", "simulate"),
    ("GET", "/analyses/{analysis_id}/dates", "list_dates"),
    ("GET", "/dates/{date_id}/transcript", "transcript"),
    ("POST", "/analyses/{analysis_id}/select", "select_endpoint"),
    ("GET", "/chat/sessions", "list_sessions"),
    ("GET", "/chat/sessions/{session_id}", "session_detail"),
    ("GET", "/chat/sessions/{session_id}/messages", "list_messages"),
    ("POST", "/chat/sessions/{session_id}/messages", "send_message"),
    ("POST", "/chat/sessions/{session_id}/end", "end_endpoint"),
}


def _table() -> set[tuple[str, str, str]]:
    out: set[tuple[str, str, str]] = set()
    for module in (auth, me, questions, traits, persona, analyses, simulation, chat):
        for route in module.router.routes:
            for method in route.methods:
                out.add((method, route.path, route.endpoint.__name__))
    return out


def test_every_promised_route_is_on_its_handler():
    table = _table()
    missing = sorted(EXPECTED - table)
    assert not missing, f"routes missing or on the wrong handler: {missing}"


def test_no_helper_took_a_decorator():
    # A route whose handler name starts with an underscore is a helper that
    # caught a decorator meant for the function below it.
    strays = sorted(t for t in _table() if t[2].startswith("_"))
    assert not strays, strays


def test_no_duplicate_method_path_pairs():
    pairs = [(m, p) for m, p, _ in _table()]
    dupes = sorted({x for x in pairs if pairs.count(x) > 1})
    assert not dupes, dupes
