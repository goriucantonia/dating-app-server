"""The traits_hash helper (S3-B7) — a deterministic hash over a user's trait
set, computed in exactly ONE place.

§19: this hash is bumped only AFTER the trait write commits, so staleness can
never claim freshness. Nothing consumes it yet (Step 6 writes it, Step 9
compares it); it exists here because the trait tables do.

What is hashed: the rows downstream consumers actually consume — every trait
whose status is not 'retracted' — by (id, category, label, description,
status, confidence), sorted by id. An all-`keep` extraction run therefore
leaves the hash untouched (A5.1, decision log #10), and a retraction changes
it (the active set shrank).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Protocol


class TraitLike(Protocol):
    id: object
    category: str
    label: str
    description: str
    status: str
    confidence: float


def compute_traits_hash(traits: Iterable[TraitLike]) -> str:
    rows = [
        {
            "id": str(t.id),
            "category": t.category,
            "label": t.label,
            "description": t.description,
            "status": t.status,
            "confidence": round(float(t.confidence), 6),
        }
        for t in traits
        if t.status != "retracted"
    ]
    rows.sort(key=lambda r: r["id"])
    payload = json.dumps(rows, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
