"""`/docs` drift check (S16-B6, `communication_protocol.md` §7).

Every endpoint named in a module plan's endpoint table must exist in the
generated OpenAPI document with the same method and path. Any drift is a
defect (§14: every promise greppable to code) — this script prints it as a
verdict rather than leaving it to a reader's memory.

    docker compose exec api python scripts/check_docs_drift.py

Reads the plans from the repository root (bind-mounted), and the OpenAPI
document from the running server, so it checks what is actually served.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
PLANS = [
    "module_1_data_collection.md", "trait_persona.md", "candidate_matching.md",
    "date_simulation.md", "chat.md", "data_hygiene.md",
]
# `METHOD /path` inside backticks, as the plans write them. Query strings and
# `{id}`-style placeholders are normalised to FastAPI's `{name}` form.
ENDPOINT_RE = re.compile(r"`(GET|POST|PUT|PATCH|DELETE)\s+(/[A-Za-z0-9_{}/\-]+)")

# Endpoints the plans name that are deliberately not served, with the reason
# — listed so the check stays honest instead of silently skipping them.
DELIBERATELY_ABSENT: dict[str, str] = {}


def normalise(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{x}", path.rstrip("/"))


def main() -> int:
    spec = httpx.get("http://localhost:8000/openapi.json", timeout=10).json()
    served = {
        (method.upper(), normalise(path))
        for path, methods in spec["paths"].items()
        for method in methods
    }

    promised: dict[tuple[str, str], list[str]] = {}
    for plan in PLANS:
        text = (ROOT / plan).read_text(encoding="utf-8")
        for method, path in ENDPOINT_RE.findall(text):
            promised.setdefault((method, normalise(path)), []).append(plan)

    drift = [(k, v) for k, v in sorted(promised.items()) if k not in served]
    extra = sorted(served - set(promised))

    print(f"promised in plans: {len(promised)} · served by /docs: {len(served)}")
    print("\n== promised but NOT served (drift):")
    for (method, path), plans in drift:
        note = DELIBERATELY_ABSENT.get(f"{method} {path}")
        print(f"  {method} {path}  ← {', '.join(sorted(set(plans)))}"
              + (f"  [deliberate: {note}]" if note else ""))
    if not drift:
        print("  none")
    print("\n== served but not named in a plan's endpoint table (additive, allowed by §7):")
    for method, path in extra:
        print(f"  {method} {path}")
    real_drift = [d for d in drift if f"{d[0][0]} {d[0][1]}" not in DELIBERATELY_ABSENT]
    print(f"\ncheck_docs_drift: {'GREEN' if not real_drift else 'RED'} — "
          f"{len(real_drift)} drifting endpoint(s)")
    return 0 if not real_drift else 1


if __name__ == "__main__":
    sys.exit(main())
