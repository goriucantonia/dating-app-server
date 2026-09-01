"""Structured logging substrate (S1-B3, principle §7).

One line per decision, machine-parseable fields, stdout (Docker collects it).
Every later §7 obligation writes through `log_event` — the event name says what
was decided and the fields say why. Refusal and failure paths log through the
same door.

Usage:
    logger = logging.getLogger("app.health")
    log_event(logger, "health_check", outcome="ok", db="connected")
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

_FIELDS_KEY = "event_fields"


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        line: dict = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, _FIELDS_KEY, None)
        if fields:
            line.update(fields)
        if record.exc_info and record.exc_info[0] is not None:
            line["exception"] = self.formatException(record.exc_info)
        return json.dumps(line, ensure_ascii=False, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLineFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # uvicorn's own loggers keep their line format out of our JSON stream
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, event: str, *, level: int = logging.INFO, **fields) -> None:
    """One structured line: what was decided, and the inputs that decided it."""
    logger.log(level, event, extra={_FIELDS_KEY: fields})
