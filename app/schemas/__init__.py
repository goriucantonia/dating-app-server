"""The schema registry (S2-B9): every VersionedSchema the app validates
against, held by `name.vN`. Stored artifacts record the version they were
produced against, so old data stays readable after a schema evolves.

`agent_response.v1` is registered here in Step 7 (frozen in trait_persona.md §3).
"""

from __future__ import annotations

from app.ai.base import VersionedSchema

_REGISTRY: dict[str, VersionedSchema] = {}


def register(schema: VersionedSchema) -> VersionedSchema:
    if schema.full_name in _REGISTRY:
        raise ValueError(f"schema {schema.full_name} is already registered")
    _REGISTRY[schema.full_name] = schema
    return schema


def get(full_name: str) -> VersionedSchema:
    return _REGISTRY[full_name]


def all_schemas() -> dict[str, VersionedSchema]:
    return dict(_REGISTRY)
