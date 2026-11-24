from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from agent.memory.task_history import TaskHistory
from agent.util import now_iso, redact


class ActionLog:
    """Every action, decision and error goes to SQLite and to logs/<task>.jsonl.
    Secrets are scrubbed on the way in, so neither store ever holds a credential."""

    def __init__(self, history: TaskHistory, log_dir: Path, secrets=(), sink: Callable[[str], None] | None = None):
        self.history = history
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.secrets = [s for s in secrets if s]
        self.sink = sink
        self.task_id: str | None = None

    def bind(self, task_id: str | None) -> None:
        self.task_id = task_id

    def event(self, kind: str, message: str, **data) -> None:
        message = redact(str(message), self.secrets)
        if data:
            data = json.loads(redact(json.dumps(data, default=str), self.secrets))
        if self.task_id:
            self.history.log(self.task_id, kind, message, data)
            line = json.dumps({"ts": now_iso(), "kind": kind, "message": message, "data": data})
            with open(self.log_dir / f"{self.task_id}.jsonl", "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        if self.sink:
            short = message if len(message) <= 220 else message[:217] + "..."
            self.sink(f"[{kind}] {short}")
