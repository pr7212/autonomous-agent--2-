from __future__ import annotations

import hashlib
from dataclasses import dataclass

from agent.brain.planner import DESKTOP_TOOLS
from agent.llm import LLMError, Router
from agent.models import Step, Task
from agent.safety import SandboxError
from agent.tools import ToolContext, ToolResult
from agent.util import truncate

FINAL_SYSTEM = """ROLE: FINAL_VERIFY
You audit the work of an autonomous agent. Decide whether the OBJECTIVE is fully achieved, judging only
from the evidence given (step results and workspace files). Be strict: an action that ran is not proof
that the goal was met. Return one JSON object:
{"complete": true|false, "reason": "one or two sentences", "gap": "what is still missing (empty if complete)"}"""


@dataclass
class Verdict:
    passed: bool
    reason: str
    method: str  # which evidence convinced us: readback, file_exists, exit_code, model, ...


class Verifier:
    """Clicking Save is an action; finding the saved file is verification.
    Steps are verified deterministically; the whole objective gets a model audit on top."""

    def __init__(self, ctx: ToolContext, router: Router | None = None):
        self.ctx = ctx
        self.router = router

    # --- per step -------------------------------------------------------
    def check_step(self, step: Step, result: ToolResult) -> Verdict:
        if not result.ok:
            return Verdict(False, result.error or "tool failed", "tool_status")
        methods: list[str] = []
        try:
            builtin = self._builtin(step, result)
            if builtin is not None:
                if not builtin.passed:
                    return builtin
                methods.append(builtin.method)
            spec = step.verify or {}
            kind = spec.get("type", "none")
            if kind != "none":
                verdict = self._spec(kind, spec, result)
                if not verdict.passed:
                    return verdict
                methods.append(kind)
        except SandboxError as e:
            return Verdict(False, str(e), "sandbox")
        if "returncode" in result.data and "exit_code" not in methods:
            methods.append("exit_code")
        return Verdict(True, "ok", "+".join(methods) or "tool_status")

    def _builtin(self, step: Step, result: ToolResult) -> Verdict | None:
        ws = self.ctx.workspace
        path = result.data.get("path")
        if step.tool in {"click", "type_text", "press_key", "scroll"}:
            if result.data.get("post_action_observed") is True:
                return Verdict(True, "foreground application was re-observed after the action", "desktop_observation")
            return Verdict(False, "desktop action has no post-action observation", "desktop_observation")
        if step.tool == "write_file" and "sha256" in result.data:
            target = ws.resolve(path)
            if not target.is_file():
                return Verdict(False, f"{path} is missing after write", "readback")
            if hashlib.sha256(target.read_bytes()).hexdigest() != result.data["sha256"]:
                return Verdict(False, f"{path} on disk differs from what was written", "readback")
            return Verdict(True, "ok", "readback")
        if step.tool == "delete_file" and result.data.get("deleted"):
            if ws.resolve(path).exists():
                return Verdict(False, f"{path} still exists after delete", "readback")
            return Verdict(True, "ok", "readback")
        if step.tool == "make_dir" and result.data.get("is_dir"):
            if not ws.resolve(path).is_dir():
                return Verdict(False, f"{path} is not a directory after mkdir", "readback")
            return Verdict(True, "ok", "readback")
        return None

    def _spec(self, kind: str, spec: dict, result: ToolResult) -> Verdict:
        ws = self.ctx.workspace
        if kind == "file_exists":
            path = spec.get("path", "")
            ok = bool(path) and ws.resolve(path).exists()
            return Verdict(ok, "ok" if ok else f"expected file not found: {path}", kind)
        if kind == "file_contains":
            path, text = spec.get("path", ""), str(spec.get("text", ""))
            target = ws.resolve(path) if path else None
            if target is None or not target.is_file():
                return Verdict(False, f"expected file not found: {path}", kind)
            ok = text in target.read_text(encoding="utf-8", errors="replace")
            return Verdict(ok, "ok" if ok else f"{path} does not contain {text!r}", kind)
        if kind == "exit_code":
            want = int(spec.get("code", 0))
            got = result.data.get("returncode")
            ok = got == want
            return Verdict(ok, "ok" if ok else f"exit code {got}, expected {want}", kind)
        if kind == "output_contains":
            text = str(spec.get("text", ""))
            ok = text in result.output
            return Verdict(ok, "ok" if ok else f"output does not contain {text!r}", kind)
        return Verdict(False, f"unknown verification type {kind!r}", "invalid")

    # --- whole objective -----------------------------------------------
    def check_objective(self, task: Task) -> Verdict:
        evidence = "\n".join(
            f"{s.index}. [{s.status.value}; verified by {s.verified_by or 'n/a'}] {s.description}\n"
            f"   result: {truncate(s.output, 600)}"
            for s in task.steps
        )
        files = "\n".join(self.ctx.workspace.snapshot(limit=100)) or "(empty)"
        user = f"OBJECTIVE:\n{task.objective}\n\nSTEPS:\n{evidence}\n\nWORKSPACE FILES:\n{files}\n"
        if self.router is None:
            return Verdict(True, "all steps verified deterministically; no model available for an objective audit", "deterministic_only")
        try:
            data = self.router.complete_json(
                "final_verify",
                FINAL_SYSTEM,
                user,
                required=("complete",),
                local_only=any(step.tool in DESKTOP_TOOLS for step in task.steps),
            )
        except LLMError as e:
            return Verdict(
                True,
                f"all steps verified deterministically; model audit unavailable ({truncate(str(e), 120)})",
                "deterministic_only",
            )
        complete = data["complete"]
        if isinstance(complete, str):
            complete = complete.strip().lower() == "true"
        reason = str(data.get("reason") or "")
        if complete:
            return Verdict(True, reason or "objective met", "model")
        return Verdict(False, str(data.get("gap") or reason or "objective not met"), "model")
