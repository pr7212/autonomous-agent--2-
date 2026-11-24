from __future__ import annotations

import re

_SENSITIVE_APP_MARKERS = {
    "chat", "discord", "gmail", "imessage", "mail", "messenger", "messages",
    "outlook", "signal", "slack", "teams", "telegram", "thunderbird", "whatsapp",
}


def is_sensitive_application(process_name: str, window_title: str = "") -> bool:
    """Classify private messaging/mail windows without retaining their titles."""
    words = set(re.findall(r"[a-z0-9]+", f"{process_name} {window_title}".lower()))
    return bool(words & _SENSITIVE_APP_MARKERS)
