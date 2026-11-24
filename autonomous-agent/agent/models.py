from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from enum import Enum

from agent.util import now_iso


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    NEEDS_ATTENTION = "needs_attention"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    VERIFIED = "verified"
    FAILED = "failed"


@dataclass
class Step:
    index: int
    description: str
    tool: str
    args: dict = field(default_factory=dict)
    expected: str = ""
    verify: dict = field(default_factory=dict)
    deferred: bool = False  # args are decided at run time, from earlier observations
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    output: str = ""
    error: str = ""
    verified_by: str = ""
    sensitive_args: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in d.items() if k in known}
        data["status"] = StepStatus(data.get("status", "pending"))
        data["sensitive_args"] = tuple(data.get("sensitive_args", ()))
        return cls(**data)


@dataclass
class Task:
    id: str
    objective: str
    status: TaskStatus = TaskStatus.PENDING
    steps: list[Step] = field(default_factory=list)
    summary: str = ""
    error: str = ""
    replans: int = 0
    actions: int = 0
    warnings: list[str] = field(default_factory=list)
    report: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)

    # --- plan bookkeeping -----------------------------------------------
    def next_pending(self) -> Step | None:
        for step in self.steps:
            if step.status == StepStatus.PENDING:
                return step
        return None

    def verified_steps(self) -> list[Step]:
        return [s for s in self.steps if s.status == StepStatus.VERIFIED]

    def renumber(self) -> None:
        for i, step in enumerate(self.steps, 1):
            step.index = i

    def keep_verified_and_append(self, new_steps: list[Step]) -> None:
        """Replanning keeps proven work and replaces everything else."""
        self.steps = self.verified_steps() + new_steps
        self.renumber()

    # --- (de)serialisation ----------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["steps"] = [s.to_dict() for s in self.steps]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        known = {f.name for f in fields(cls)}
        data = {k: v for k, v in d.items() if k in known}
        data["status"] = TaskStatus(data.get("status", "pending"))
        data["steps"] = [Step.from_dict(s) for s in data.get("steps", [])]
        return cls(**data)
