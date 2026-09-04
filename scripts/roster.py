"""Owner's roster: every person in the database, rendered as one static page.

    docker compose exec -T api python scripts/roster.py > roster.html

**Why this is a script and not an endpoint.** `communication_protocol.md` §6
rule 5 — audited by `scripts/audit_wire_privacy.py` — says non-candidates
appear in NO payload: the only user ids that may cross the wire belong to the
caller, their candidates, or their matches. A "list everyone" endpoint breaks
that invariant for every signed-in client, permanently, to serve one person
looking at their own database. So this reads the database directly and writes
a file. Nothing is served, no route is added, and the audit still passes
(owner decision, 2026-09-03).

**What it shows, and what it deliberately does not.** Identity basics (the A1
form fields) and trait LABELS. Not trait descriptions, not answers, not
persona snapshots, not system prompts — labels are the same field the
candidate payloads are already allowed to carry (§6 rule 2), so this page is
no more revealing than a ranking screen the app itself renders. Pipeline
readiness is shown because a person with no vector is invisible to matching,
and that is the thing you actually want to see at a glance.

Read-only by construction: it opens a session, selects, and renders. There is
no write path in this file and no link on the page that leads to one.
"""

from __future__ import annotations

import asyncio
import html
import sys
from collections import Counter
from datetime import UTC, datetime

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import get_settings
from app.db import create_engine
from app.models import Answer, PersonaSnapshot, ProfileEmbedding, Trait, User
from app.users import compute_age

# The Modernist tokens, transcribed from ux/lib/app/theme.dart so this page and
# the app are visibly the same system. Kept as literals: this script must not
# import Flutter anything, and the two drifting apart is a cosmetic problem,
# not a correctness one.
CSS = """
:root {
  --bg: #f3f2f2; --surface: #eae9e9; --ink: #201e1d; --muted: #605d5d;
  --accent: #ec3013; --accent100: #fff2ef; --accent200: #ffe0d9;
  --rule: rgba(32,30,29,0.4); --hairline: rgba(32,30,29,0.14);
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--ink);
  font-family: Archivo, "Helvetica Neue", Helvetica, Arial, sans-serif;
  font-size: 14px; line-height: 1.45;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 40px 24px 96px; }
.kicker {
  font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--muted); margin: 0 0 8px;
}
h1 { font-size: 40px; line-height: 1.05; margin: 0 0 16px; letter-spacing: -0.02em; }
.lede { color: var(--muted); max-width: 68ch; margin: 0 0 28px; }
.lede.small { font-size: 12.5px; margin: 10px 0 6px; }
.lede code { font-size: 12px; background: var(--surface); padding: 1px 4px; }
.tally {
  display: flex; flex-wrap: wrap; gap: 0; border-top: 2px solid var(--ink);
  border-bottom: 2px solid var(--ink); margin: 0 0 40px;
}
.tally div { padding: 14px 28px 14px 0; margin-right: 28px; }
.tally .n { font-size: 26px; font-variant-numeric: tabular-nums; display: block; }
.tally .l {
  font-size: 10px; letter-spacing: 0.11em; text-transform: uppercase; color: var(--muted);
}
h2 {
  font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--muted); font-weight: 600;
  border-bottom: 2px solid var(--ink); padding-bottom: 8px; margin: 44px 0 0;
}
.person { border-bottom: 1px solid var(--hairline); padding: 18px 0; }
.person:last-child { border-bottom: 0; }
.row { display: flex; gap: 20px; align-items: baseline; flex-wrap: wrap; }
.idx {
  font-variant-numeric: tabular-nums; color: var(--muted); font-size: 12px;
  width: 30px; flex: none;
}
.name { font-size: 19px; font-weight: 600; letter-spacing: -0.01em; }
.facts { color: var(--muted); font-size: 13px; }
.facts b { color: var(--ink); font-weight: 500; }
.tag {
  display: inline-block; font-size: 10px; letter-spacing: 0.09em;
  text-transform: uppercase; border: 1px solid var(--rule);
  padding: 2px 7px; margin-left: 8px; color: var(--muted); white-space: nowrap;
}
.tag.demo { border-color: var(--accent); color: var(--accent); }
.tag.real { border-color: var(--ink); color: var(--ink); }
.tag.test { border-color: var(--hairline); color: var(--muted); }
.tag.warn { border-color: var(--accent); color: #fff; background: var(--accent); }
.labels { margin: 10px 0 0 50px; }
.chip {
  display: inline-block; background: var(--accent100); border: 1px solid var(--accent200);
  color: var(--ink); font-size: 12px; padding: 3px 9px; margin: 0 6px 6px 0;
}
.chip.flaw { background: var(--surface); border-color: var(--hairline); color: var(--muted); }
.none { margin: 10px 0 0 50px; color: var(--accent); font-size: 12px; }
footer {
  margin-top: 56px; border-top: 2px solid var(--ink); padding-top: 14px;
  color: var(--muted); font-size: 11px; letter-spacing: 0.04em;
}
@media (max-width: 640px) {
  h1 { font-size: 30px; } .wrap { padding: 24px 16px 72px; }
  .labels, .none { margin-left: 0; }
}
"""


def esc(v: object) -> str:
    return html.escape(str(v if v is not None else ""))


# The only split this page makes is people vs probe wreckage (owner, 2026-09-03:
# real-vs-demo "is not something that matters" — a seeded profile and a
# registered one are both just someone in the app, and the page says so by not
# distinguishing them). Probe accounts stay separate because they are not
# people: they are litter from test runs, they are named `Pool Probe`, and
# folding them in buries the fifty profiles that matter.
TEST_DOMAINS = ("probe.dev", "dating-test.dev")


def origin(email: str, is_demo: bool) -> str:
    """`test` beats `is_demo`: a probe that set the flag is still a probe."""
    domain = email.rsplit("@", 1)[-1].lower()
    return "test" if domain in TEST_DOMAINS else "person"


async def collect() -> tuple[list[dict], dict]:
    engine = create_engine(get_settings().database_url)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    people: list[dict] = []
    async with factory() as session:
        users = (
            await session.scalars(select(User).order_by(User.is_demo, User.display_name))
        ).all()
        for u in users:
            traits = (
                await session.scalars(
                    select(Trait)
                    .where(Trait.user_id == u.id, Trait.status != "retracted")
                    .order_by(Trait.category, Trait.label)
                )
            ).all()
            snaps = (
                await session.scalars(
                    select(PersonaSnapshot.status).where(PersonaSnapshot.user_id == u.id)
                )
            ).all()
            people.append({
                "name": u.display_name,
                "email": u.email,
                "origin": origin(u.email, u.is_demo),
                "age": compute_age(u.birth_date),
                "gender": u.gender,
                "interested_in": list(u.interested_in),
                "city": u.city,
                "country": u.country,
                "pref": (u.age_pref_min, u.age_pref_max),
                "is_demo": u.is_demo,
                "opt_in": u.opt_in,
                "answers": await session.scalar(
                    select(func.count()).select_from(Answer).where(Answer.user_id == u.id)
                ),
                "embeddings": await session.scalar(
                    select(func.count())
                    .select_from(ProfileEmbedding)
                    .where(ProfileEmbedding.user_id == u.id)
                ),
                # Labels only. Descriptions are another user's private text (§6 rule 2)
                # and have no business on a page that exists to be glanced at.
                "labels": [(t.label, t.category) for t in traits],
                "ready": any(s == "ready" for s in snaps),
            })
    await engine.dispose()
    # The headline numbers describe the PEOPLE. Probe wreckage is counted
    # separately and never folded in, or every tally here would be a lie about
    # how many people the app actually has.
    pool = [p for p in people if p["origin"] == "person"]
    matchable = sum(1 for p in pool if p["labels"] and p["ready"] and p["embeddings"] >= 2)
    summary = {
        "total": len(people),
        "pool": len(pool),
        "test": sum(1 for p in people if p["origin"] == "test"),
        "matchable": matchable,
        "not_matchable": len(pool) - matchable,
        "traits": sum(len(p["labels"]) for p in pool),
        "cities": len({p["city"] for p in pool if p["city"]}),
        "answers": sum(p["answers"] for p in pool),
        "genders": Counter(p["gender"] for p in pool),
    }
    return people, summary


def render_person(i: int, p: dict) -> str:
    where = ", ".join(x for x in (p["city"], p["country"]) if x) or "no location"
    tags = []
    if p["origin"] == "test":
        tags.append('<span class="tag test">probe</span>')
    if not p["opt_in"]:
        tags.append('<span class="tag">not opted in</span>')
    if not (p["labels"] and p["ready"] and p["embeddings"] >= 2):
        missing = []
        if not p["labels"]:
            missing.append("no traits")
        if not p["ready"]:
            missing.append("no ready persona")
        if p["embeddings"] < 2:
            missing.append("no vectors")
        tags.append(f'<span class="tag warn">{esc(" · ".join(missing))}</span>')
    chips = "".join(
        f'<span class="chip{" flaw" if c == "flaw" else ""}">{esc(label)}</span>'
        for label, c in p["labels"]
    )
    body = (
        f'<div class="labels">{chips}</div>' if chips
        else '<div class="none">no traits extracted — this person is invisible to matching</div>'
    )
    return f"""    <div class="person">
      <div class="row">
        <span class="idx">{i:02d}</span>
        <span class="name">{esc(p["name"])}</span>
        <span class="facts"><b>{p["age"]}</b> · {esc(p["gender"])} · seeking
          {esc(", ".join(p["interested_in"]))} · aged {p["pref"][0]}&ndash;{p["pref"][1]}
          · {esc(where)} · {p["answers"]} answers</span>
        {"".join(tags)}
      </div>
      {body}
    </div>"""


def render(people: list[dict], s: dict) -> str:
    gender_line = " · ".join(f"{n} {g}" for g, n in s["genders"].most_common())
    groups = [
        ("person", "People",
         ("Everyone in the app, in one list. Some registered through the app and some "
          "were seeded from a fixture, and the page does not distinguish them &mdash; "
          "they all went through the same registration path and the same pipeline.")),
        ("test", "Probe &amp; test accounts",
         ("Left behind by probe runs (<code>@probe.dev</code>, "
          "<code>@dating-test.dev</code>). Not people; excluded from every count "
          "above. Safe to delete.")),
    ]
    sections = ""
    for key, title, blurb in groups:
        rows = [p for p in people if p["origin"] == key]
        if not rows:
            continue
        sections += (
            f'\n<h2>{title} &middot; {len(rows)}</h2>\n<p class="lede small">{blurb}</p>\n'
            + "\n".join(render_person(i, p) for i, p in enumerate(rows, 1))
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Roster &middot; everyone in the database</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Archivo:wght@400;500;600&display=swap" rel="stylesheet">
<style>{CSS}</style></head>
<body><div class="wrap">
  <p class="kicker">Owner view &middot; read only</p>
  <h1>Everyone in the database</h1>
  <p class="lede">Every account that exists, with the identity fields they registered
  with and the trait labels the extraction pipeline produced for them. Nothing here
  is interactive: there is no way to message, rank or choose anyone from this page.
  Generated by <code>scripts/roster.py</code> straight from the database &mdash; not
  from an API, so no user ids crossed the wire to build it.</p>
  <div class="tally">
    <div><span class="n">{s["pool"]}</span><span class="l">people</span></div>
    <div><span class="n">{s["matchable"]}</span><span class="l">matchable</span></div>
    <div><span class="n">{s["not_matchable"]}</span><span class="l">not matchable</span></div>
    <div><span class="n">{s["traits"]}</span><span class="l">live traits</span></div>
    <div><span class="n">{s["answers"]}</span><span class="l">answers</span></div>
    <div><span class="n">{s["cities"]}</span><span class="l">cities</span></div>
    <div><span class="n">{s["test"]}</span><span class="l">probe leftovers</span></div>
  </div>
{sections}
  <footer>{gender_line} &middot; generated {datetime.now(UTC).strftime("%Y-%m-%d %H:%M")} UTC
  &middot; trait labels only, never descriptions or answers (communication_protocol.md &sect;6)</footer>
</div></body></html>
"""


async def main() -> None:
    people, summary = await collect()
    sys.stdout.write(render(people, summary))


if __name__ == "__main__":
    asyncio.run(main())
