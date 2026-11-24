from __future__ import annotations

import re
from dataclasses import dataclass

from agent.config import Config
from agent.llm import LLMError, Router
from agent.models import Step, StepStatus, Task
from agent.safety import Workspace
from agent.tools import ToolRegistry
from agent.util import truncate

VERIFY_TYPES = {"file_exists", "file_contains", "exit_code", "output_contains", "none"}
DESKTOP_TOOLS = {"observe_desktop", "move_mouse", "click", "type_text", "press_key", "scroll", "wait"}
_DESKTOP_OBJECTIVE = re.compile(
    r"\b(desktop|foreground|screen|window|click|mouse|keyboard|gui|interface)\b"
    r"|\b(?:open|use|interact with|control|in)\s+(?:the\s+)?"
    r"(?:(?:current|active|foreground|open)\s+)?(?:app|application)\b"
    r"|\b(?:current|active|foreground|open)\s+(?:app|application)\b",
    re.I,
)

COMMON_RULES = """\
- All file paths are relative to the workspace. Never use paths outside it.
- Never send messages, make purchases, change security settings, or submit applications or
  assessments. If the objective requires that, plan up to that point and report it in
  "blocked_reason".
- Never try to impersonate a human or evade bot detection, platform rules or assessments.
- For desktop work, observe the foreground application before acting and re-observe after each action.
- Never inspect or type into password fields. Do not automate submissions, purchases, sending messages,
  account/security changes, or any consequential action; stop for human approval instead.
- If the objective cannot be achieved with the available tools, return {"steps": [], "blocked_reason": "<why>"}.
"""

STEP_SCHEMA = """\
Each step is an object:
  {"description": str, "tool": str, "args": {...}, "expected": str, "verify": {...}, "deferred": bool}
- "expected": the observable evidence that the step worked.
- "verify": ONE deterministic check, preferred over prose. Options:
    {"type": "file_exists", "path": p} | {"type": "file_contains", "path": p, "text": s} |
    {"type": "exit_code", "code": 0} | {"type": "output_contains", "text": s} | {"type": "none"}
- "deferred": true when the args depend on what an earlier step will reveal (for example editing a
  file you have not read yet). Then leave "args" as {} - they are decided when the step runs.
"""

PLAN_SYSTEM = """ROLE: PLAN
You are the planning module of an autonomous desktop agent. Break the OBJECTIVE into the fewest
concrete, verifiable steps that use ONLY the listed tools, at most __MAX__ steps.
Return one JSON object: {"steps": [...], "blocked_reason": ""}
""" + STEP_SCHEMA + COMMON_RULES

REPLAN_SYSTEM = """ROLE: REPLAN
You are the planning module of an autonomous desktop agent. Work has stalled or is incomplete.
Return ONLY the steps still needed, as one JSON object: {"steps": [...], "blocked_reason": ""}
Do not repeat steps that are already verified. Change approach if the same action already failed.
At most __MAX__ steps.
""" + STEP_SCHEMA + COMMON_RULES

RESOLVE_SYSTEM = """ROLE: RESOLVE_ARGS
You are the action module of an autonomous desktop agent. Decide the exact arguments for ONE step,
using what earlier steps observed. Return one JSON object: {"args": {...}} containing every
required argument of the tool. File paths are relative to the workspace."""


class PlanningError(Exception):
    pass


@dataclass
class PlanResult:
    steps: list[Step]
    blocked_reason: str = ""


class Planner:
    def __init__(self, router: Router, registry: ToolRegistry, workspace: Workspace, cfg: Config):
        self.router = router
        self.registry = registry
        self.workspace = workspace
        self.cfg = cfg

    # --- public ---------------------------------------------------------
    def make_plan(self, objective: str, memory: str = "") -> PlanResult:
        user = (
            f"OBJECTIVE:\n{objective}\n\nTOOLS:\n{self.registry.describe()}\n\n"
            f"WORKSPACE FILES:\n{self._listing()}\n"
        )
        if memory:
            user += f"\nRECENT FINISHED TASKS (for context only):\n{memory}\n"
        return self._ask(
            "plan", self._system(PLAN_SYSTEM), user, start_index=1,
            local_only=bool(_DESKTOP_OBJECTIVE.search(objective)),
        )

    def replan(self, task: Task, reason: str, failed: Step | None) -> PlanResult:
        done = task.verified_steps()
        user = f"OBJECTIVE:\n{task.objective}\n\nTOOLS:\n{self.registry.describe()}\n\n"
        user += "VERIFIED STEPS SO FAR:\n" + (
            "\n".join(f"{s.index}. {s.description} -> {truncate(s.output, 300)}" for s in done) or "(none)"
        )
        if failed is not None:
            safe_args = {
                key: ("[REDACTED]" if key in failed.sensitive_args else value)
                for key, value in failed.args.items()
            }
            user += (
                f"\n\nFAILED STEP: {failed.description}\n  tool: {failed.tool}\n  args: {safe_args}\n"
                f"  after {failed.attempts} attempt(s)"
            )
        user += f"\n\nPROBLEM:\n{truncate(reason, 1500)}"
        pending = [s for s in task.steps if s.status != StepStatus.VERIFIED and s is not failed]
        if pending:
            user += "\n\nSTEPS THAT WERE STILL PLANNED (they will be discarded):\n" + "\n".join(
                f"- {s.description}" for s in pending
            )
        user += f"\n\nWORKSPACE FILES:\n{self._listing()}\n"
        local_only = bool(_DESKTOP_OBJECTIVE.search(task.objective)) or any(
            step.tool in DESKTOP_TOOLS for step in task.steps
        )
        return self._ask(
            "replan", self._system(REPLAN_SYSTEM), user, start_index=len(done) + 1,
            local_only=local_only,
        )

    def resolve_args(self, task: Task, step: Step) -> dict:
        tool = self.registry.get(step.tool)
        if tool is None:
            raise PlanningError(f"unknown tool {step.tool!r}")
        spec = "\n".join(
            f"  {name}{'*' if name in tool.required else ''}: {desc}" for name, desc in tool.params.items()
        )
        seen = "\n".join(
            f"{s.index}. {s.description} -> {truncate(s.output, 1500)}" for s in task.verified_steps()[-5:]
        )
        user = (
            f"OBJECTIVE:\n{task.objective}\n\nSTEP TO EXECUTE:\n{step.description}\nExpected: {step.expected}\n"
            f"Tool: {tool.name}\n{spec}\n\nWHAT EARLIER STEPS OBSERVED:\n{seen or '(nothing yet)'}\n\n"
            f"WORKSPACE FILES:\n{self._listing()}\n"
        )
        try:
            local_only = step.tool in DESKTOP_TOOLS or any(s.tool in DESKTOP_TOOLS for s in task.steps)
            data = self.router.complete_json(
                "step", RESOLVE_SYSTEM, user, required=("args",), local_only=local_only
            )
        except LLMError as e:
            raise PlanningError(f"no model could decide the arguments: {e}") from None
        args = data["args"]
        if not isinstance(args, dict):
            raise PlanningError("args must be an object")
        missing = self.registry.missing_args(step.tool, args)
        if missing:
            raise PlanningError(f"model did not provide required args: {', '.join(missing)}")
        return args

    # --- internals ------------------------------------------------------
    def _system(self, template: str) -> str:
        return template.replace("__MAX__", str(self.cfg.max_plan_steps))

    def _listing(self) -> str:
        files = self.workspace.snapshot()
        return "\n".join(files) if files else "(empty)"

    def _ask(self, role: str, system: str, user: str, start_index: int, local_only: bool = False) -> PlanResult:
        feedback = ""
        errors: list[str] = []
        for _ in range(2):
            try:
                data = self.router.complete_json(role, system, user + feedback, local_only=local_only)
            except LLMError as e:
                raise PlanningError(str(e)) from None
            blocked = str(data.get("blocked_reason") or "").strip()
            steps, errors = self._parse(data.get("steps"), start_index)
            if not errors:
                return PlanResult(steps, blocked)
            feedback = (
                "\n\nYour previous plan was rejected:\n- " + "\n- ".join(errors) + "\nReturn a corrected plan."
            )
        raise PlanningError("plan rejected: " + "; ".join(errors))

    def _parse(self, raw, start_index: int) -> tuple[list[Step], list[str]]:
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            return [], ["'steps' must be a list"]
        errors: list[str] = []
        steps: list[Step] = []
        if len(raw) > self.cfg.max_plan_steps:
            errors.append(f"too many steps ({len(raw)}); the maximum is {self.cfg.max_plan_steps}")
        for i, item in enumerate(raw, 1):
            if not isinstance(item, dict):
                errors.append(f"step {i}: must be an object")
                continue
            tool = item.get("tool")
            if tool not in self.registry.names():
                errors.append(f"step {i}: unknown tool {tool!r}; available: {', '.join(self.registry.names())}")
                continue
            args = item.get("args") or {}
            if not isinstance(args, dict):
                errors.append(f"step {i}: args must be an object")
                continue
            verify = item.get("verify") or {}
            if not isinstance(verify, dict) or (verify and verify.get("type") not in VERIFY_TYPES):
                errors.append(f"step {i}: verify.type must be one of {sorted(VERIFY_TYPES)}")
                continue
            deferred = bool(item.get("deferred")) or bool(self.registry.missing_args(tool, args))
            sensitive_args = tuple(self.registry.get(tool).sensitive_args)
            steps.append(
                Step(
                    index=start_index + len(steps),
                    description=(
                        f"{tool} (sensitive input redacted)"
                        if sensitive_args else str(item.get("description") or tool)
                    ),
                    tool=tool,
                    args=args,
                    expected="" if sensitive_args else str(item.get("expected") or ""),
                    verify={} if sensitive_args else verify,
                    deferred=deferred,
                    sensitive_args=sensitive_args,
                )
            )
        return steps, errors
