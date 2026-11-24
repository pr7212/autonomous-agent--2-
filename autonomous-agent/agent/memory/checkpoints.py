from __future__ import annotations

import json

from agent.memory.database import Database
from agent.models import Task
from agent.util import now_iso


class Checkpoints:
    """Full task snapshots after the plan and after every verified step.
    A crash or a Ctrl-C loses at most the step in flight."""

    KEEP = 25

    def __init__(self, db: Database):
        self.db = db

    def save(self, task: Task, label: str = "") -> None:
        self.db.execute(
            "INSERT INTO checkpoints (task_id, ts, label, state_json) VALUES (?, ?, ?, ?)",
            (task.id, now_iso(), label, json.dumps(task.to_dict())),
        )
        self.db.execute(
            "DELETE FROM checkpoints WHERE task_id = ? AND id NOT IN "
            "(SELECT id FROM checkpoints WHERE task_id = ? ORDER BY id DESC LIMIT ?)",
            (task.id, task.id, self.KEEP),
        )

    def latest(self, task_id: str) -> Task | None:
        rows = self.db.query(
            "SELECT state_json FROM checkpoints WHERE task_id = ? ORDER BY id DESC LIMIT 1", (task_id,)
        )
        return Task.from_dict(json.loads(rows[0]["state_json"])) if rows else None

    def count(self, task_id: str) -> int:
        return self.db.query("SELECT COUNT(*) AS n FROM checkpoints WHERE task_id = ?", (task_id,))[0]["n"]
