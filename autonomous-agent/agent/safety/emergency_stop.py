from __future__ import annotations

import threading
from pathlib import Path


class StopRequested(Exception):
    """Raised at the next safe point after an emergency stop."""


class EmergencyStop:
    """Two triggers: an in-process flag (Ctrl-C, dashboard button) and a STOP file
    that a separate terminal can create with `python -m agent stop`."""

    def __init__(self, stop_file: Path | None = None):
        self._event = threading.Event()
        self._reason = ""
        self.stop_file = Path(stop_file) if stop_file else None

    def trigger(self, reason: str = "manual") -> None:
        self._reason = reason
        self._event.set()

    def clear(self) -> None:
        self._event.clear()
        self._reason = ""
        if self.stop_file and self.stop_file.exists():
            self.stop_file.unlink()

    def request_via_file(self) -> None:
        if self.stop_file:
            self.stop_file.parent.mkdir(parents=True, exist_ok=True)
            self.stop_file.write_text("stop", encoding="utf-8")

    def is_set(self) -> bool:
        return self._event.is_set() or bool(self.stop_file and self.stop_file.exists())

    @property
    def reason(self) -> str:
        if self._event.is_set():
            return self._reason or "manual"
        return "STOP file present" if self.is_set() else ""

    def check(self) -> None:
        if self.is_set():
            raise StopRequested(self.reason)
