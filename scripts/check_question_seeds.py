"""Question-seed fidelity tool (S3-B4 / Step 3 AC6).

The 35 seeded questions are LOCKED verbatim in module_1_data_collection.md
(A2: BQ1-BQ5, A5.3: PQ01-PQ30). Hand-transcription is how a curly apostrophe
silently becomes a straight one, so:

    --write   parses the module plan and (re)generates seeds/questions.yaml
    (default) diffs module plan <-> seeds/questions.yaml <-> database
              and prints a human-readable verdict (the AC6 witness)

Baseline probe_area mapping: A2's descriptive areas ("interests + approach",
"situational (flirty/supportive) + conversational", ...) are wider than the
schema's single-value CHECK. Mapping chosen 2026-09-01 (routine call, §25):
BQ1=interests, BQ2=partner_criteria, BQ3=situational, BQ4=situational
(situational is its primary framing; the pool carries six dedicated
conversational questions), BQ5=self_image.

Run inside the api container:
    docker compose exec api python scripts/check_question_seeds.py [--write]
"""

from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path

import yaml

SERVER_ROOT = Path(__file__).resolve().parent.parent
DOC = SERVER_ROOT / "module_1_data_collection.md"
SEEDS = SERVER_ROOT / "seeds" / "questions.yaml"

BASELINE_PROBE_AREA = {
    "BQ1": "interests",
    "BQ2": "partner_criteria",
    "BQ3": "situational",
    "BQ4": "situational",
    "BQ5": "self_image",
}

_BASELINE_ROW = re.compile(r'^\|\s*`(BQ\d)`\s*\|[^|]*\|\s*"(.*)"\s*\|\s*$')
_POOL_ROW = re.compile(r'^\|\s*(PQ\d\d)\s*\|\s*(\w+)\s*\|\s*"(.*)"\s*\|\s*$')


def parse_doc() -> dict:
    baseline, pool = [], []
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if m := _BASELINE_ROW.match(line):
            code, text = m.groups()
            baseline.append(
                {"code": code, "probe_area": BASELINE_PROBE_AREA[code], "text": text}
            )
        elif m := _POOL_ROW.match(line):
            code, area, text = m.groups()
            pool.append(
                {
                    "code": code,
                    "pool_order": int(code[2:]),
                    "probe_area": area,
                    "text": text,
                }
            )
    if len(baseline) != 5 or len(pool) != 30:
        raise SystemExit(
            f"RED: parsed {len(baseline)} baseline + {len(pool)} pool questions "
            "from the module plan; expected 5 + 30. The doc format changed — fix me."
        )
    return {"baseline": baseline, "pool": pool}


def write_seeds(parsed: dict) -> None:
    SEEDS.parent.mkdir(exist_ok=True)
    header = (
        "# GENERATED VERBATIM from module_1_data_collection.md (A2 + A5.3) by\n"
        "# scripts/check_question_seeds.py --write. Do not edit by hand — the\n"
        "# module plan is the source of truth and the checker diffs against it.\n"
    )
    body = yaml.safe_dump(parsed, allow_unicode=True, sort_keys=False, width=1000)
    SEEDS.write_text(header + body, encoding="utf-8", newline="\n")
    print(f"wrote {SEEDS.relative_to(SERVER_ROOT)}: 5 baseline + 30 pool questions")


def _diff(label_a: str, a: dict[str, dict], label_b: str, b: dict[str, dict]) -> list[str]:
    problems = []
    for code in sorted(set(a) | set(b)):
        if code not in a:
            problems.append(f"{code}: missing from {label_a}")
        elif code not in b:
            problems.append(f"{code}: missing from {label_b}")
        elif a[code] != b[code]:
            for key in a[code]:
                if a[code].get(key) != b[code].get(key):
                    problems.append(
                        f"{code}.{key}: {label_a}={a[code].get(key)!r} "
                        f"!= {label_b}={b[code].get(key)!r}"
                    )
    return problems


def _by_code(entries: list[dict]) -> dict[str, dict]:
    return {e["code"]: e for e in entries}


async def check() -> int:
    parsed = parse_doc()
    doc_map = _by_code(parsed["baseline"]) | _by_code(parsed["pool"])

    seeds = yaml.safe_load(SEEDS.read_text(encoding="utf-8"))
    seed_map = _by_code(seeds["baseline"]) | _by_code(seeds["pool"])

    problems = _diff("doc", doc_map, "seeds.yaml", seed_map)

    from sqlalchemy import text as sql_text

    from app.config import get_settings
    from app.db import create_engine

    engine = create_engine(get_settings().database_url)
    async with engine.connect() as conn:
        rows = (
            await conn.execute(
                sql_text(
                    "SELECT code, origin, pool_order, probe_area, text "
                    "FROM questions WHERE user_id IS NULL"
                )
            )
        ).mappings().all()
    await engine.dispose()

    db_map: dict[str, dict] = {}
    for r in rows:
        entry = {"code": r["code"], "probe_area": r["probe_area"], "text": r["text"]}
        if r["origin"] == "pool":
            entry["pool_order"] = r["pool_order"]
        db_map[r["code"]] = entry
    problems += _diff("doc", doc_map, "database", db_map)

    print(f"doc: {len(doc_map)} questions · seeds.yaml: {len(seed_map)} · database: {len(db_map)}")
    if problems:
        print("VERDICT: RED — drift found:")
        for p in problems:
            print(f"  {p}")
        return 1
    print("VERDICT: GREEN — module plan, seeds fixture, and database are "
          "character-for-character identical (35 questions).")
    return 0


if __name__ == "__main__":
    if "--write" in sys.argv:
        write_seeds(parse_doc())
        sys.exit(0)
    sys.exit(asyncio.run(check()))
