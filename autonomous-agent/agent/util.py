from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


def new_id() -> str:
    return uuid.uuid4().hex[:8]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def truncate(text: str, limit: int) -> str:
    """Keep the head and (longer) tail: errors and results usually sit at the end."""
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.4)
    tail = limit - head
    omitted = len(text) - limit
    return f"{text[:head]}\n... [{omitted} chars omitted] ...\n{text[-tail:]}"


def redact(text: str, secrets) -> str:
    for secret in secrets:
        if secret and len(secret) >= 6:
            text = text.replace(secret, "[REDACTED]")
    return text


def is_secret_name(name: str) -> bool:
    upper = name.upper()
    return any(marker in upper for marker in SECRET_MARKERS)


def secret_values() -> list[str]:
    """Values of secret-looking environment variables (for log redaction)."""
    return [v for k, v in os.environ.items() if is_secret_name(k) and len(v) >= 8]
