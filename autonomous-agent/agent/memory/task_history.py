from __future__ import annotations

import json

from agent.memory.database import Database
from agent.models import Task
from agent.util import now_iso

UPSERT = """
INSERT INTO tasks (id, objective, status, summary, error, state_json, created_at, updated_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
    status = excluded.status,
    summary = excluded.summary,
    error = excluded.error,
    state_json = excluded.state_json,
    updated_at = excluded.updated_at
"""


class TaskHistory:
    def __init__(self, db: Database):
        self.db = db

    def save(self, task: Task) -> None:
        task.updated_at = now_iso()
        state = task.to_dict()
        for step in state["steps"]:
            sensitive = step.get("sensitive_args", ())
            if not sensitive:
                continue
            step["description"] = f"{step['tool']} (sensitive input redacted)"
            step["expected"] = ""
            step["verify"] = {}
            for key in sensitive:
                if key in step["args"]:
                    step["args"][key] = "[REDACTED]"
                    if step["status"] != "verified":
                        step["deferred"] = True
        self.db.execute(
            UPSERT,
            (
                task.id, task.objective, task.status.value, task.summary, task.error,
                json.dumps(state), task.created_at, task.updated_at,
            ),
        )

    def get(self, task_id: str) -> Task | None:
        rows = self.db.query("SELECT state_json FROM tasks WHERE id = ?", (task_id,))
        return Task.from_dict(json.loads(rows[0]["state_json"])) if rows else None

    def find(self, id_or_prefix: str) -> Task | None:
        rows = self.db.query("SELECT id FROM tasks WHERE id LIKE ?", (id_or_prefix + "%",))
        if len(rows) > 1:
            raise ValueError(f"ambiguous task id {id_or_prefix!r}: {', '.join(r['id'] for r in rows)}")
        return self.get(rows[0]["id"]) if rows else None

    def list(self, limit: int = 20) -> list[Task]:
        rows = self.db.query("SELECT state_json FROM tasks ORDER BY rowid DESC LIMIT ?", (limit,))
        return [Task.from_dict(json.loads(r["state_json"])) for r in rows]

    def log(self, task_id: str, kind: str, message: str, data: dict | None = None) -> None:
        self.db.execute(
            "INSERT INTO events (task_id, ts, kind, message, data_json) VALUES (?, ?, ?, ?, ?)",
            (task_id, now_iso(), kind, message, json.dumps(data or {}, default=str)),
        )

    def events(self, task_id: str) -> list[dict]:
        rows = self.db.query("SELECT ts, kind, message, data_json FROM events WHERE task_id = ? ORDER BY id", (task_id,))
        return [
            {"ts": r["ts"], "kind": r["kind"], "message": r["message"], "data": json.loads(r["data_json"])}
            for r in rows
        ]

    def context_for(self, exclude: str = "", limit: int = 3) -> str:
        """Recent finished work, so a new plan can build on it instead of rediscovering it."""
        rows = self.db.query(
            "SELECT objective, summary FROM tasks WHERE status = 'completed' AND id != ? "
            "ORDER BY rowid DESC LIMIT ?",
            (exclude, limit),
        )
        return "\n".join(f"- {r['objective']} -> {r['summary']}" for r in rows)
