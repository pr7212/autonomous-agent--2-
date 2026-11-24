from __future__ import annotations

import time
import traceback
from typing import Callable

from agent.actionlog import ActionLog
from agent.brain.planner import Planner, PlanningError
from agent.brain.recovery import Recovery, decide
from agent.brain.verifier import Verdict, Verifier
from agent.config import Config
from agent.memory import Checkpoints, TaskHistory
from agent.models import StepStatus, Task, TaskStatus
from agent.reporting import write_report
from agent.safety import EmergencyStop, StopRequested
from agent.tools import ToolRegistry, ToolResult
from agent.util import new_id, truncate


class _Finish(Exception):
    """Internal control flow: end the run with this status."""

    def __init__(self, status: TaskStatus, message: str = ""):
        super().__init__(message)
        self.status = status
        self.message = message


class TaskManager:
    """The observe -> plan -> act -> verify -> recover loop.

    A step only becomes VERIFIED after independent evidence. A task only becomes COMPLETED after
    every step is verified and the objective itself has been audited."""

    def __init__(
        self,
        cfg: Config,
        registry: ToolRegistry,
        planner: Planner,
        verifier: Verifier,
        history: TaskHistory,
        checkpoints: Checkpoints,
        log: ActionLog,
        stop: EmergencyStop,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        activity=None,
    ):
        self.cfg = cfg
        self.registry = registry
        self.planner = planner
        self.verifier = verifier
        self.history = history
        self.checkpoints = checkpoints
        self.log = log
        self.stop = stop
        self.clock = clock
        self.sleep = sleep
        self.activity = activity
        self._deadline = 0.0
        self._run_actions = 0

    # --- public ---------------------------------------------------------
    def create_task(self, objective: str) -> Task:
        task = Task(id=new_id(), objective=objective.strip())
        self.history.save(task)
        return task

    def run(self, task: Task) -> Task:
        self._deadline = self.clock() + self.cfg.max_task_seconds
        self._run_actions = 0
        self.log.bind(task.id)
        task.status = TaskStatus.RUNNING
        task.error = ""
        self.history.save(task)
        self.log.event("task_started", task.objective)
        try:
            if self.activity:
                self.activity.start_task(task.id)
            if not task.steps:
                self._plan(task)
            while True:
                step = task.next_pending()
                if step is None:
                    self._finalize(task)
                    if task.status == TaskStatus.COMPLETED:
                        break
                    continue
                if not self._execute_step(task, step):
                    self._replan(task, step, step.error)
        except _Finish as f:
            task.status = f.status
            task.error = "" if f.status == TaskStatus.COMPLETED else f.message
        except StopRequested as e:
            task.status = TaskStatus.CANCELLED
            task.error = f"emergency stop: {e}"
            for s in task.steps:
                if s.status == StepStatus.RUNNING:
                    s.status = StepStatus.PENDING
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"unexpected error: {type(e).__name__}: {e}"
            if self.activity:
                self.activity.record("error_encountered", f"agent error ({type(e).__name__})", error_type=type(e).__name__)
            self.log.event("crash", traceback.format_exc())
        finally:
            try:
                self._conclude(task)
            finally:
                if self.activity:
                    self.activity.stop_task()
        return task

    def resume(self, task_id: str) -> Task:
        task = self.checkpoints.latest(task_id) or self.history.get(task_id)
        if task is None:
            raise KeyError(f"no task {task_id!r}")
        if task.status == TaskStatus.COMPLETED:
            return task
        for step in task.steps:
            if step.status in (StepStatus.RUNNING, StepStatus.FAILED):
                step.status = StepStatus.PENDING
                step.attempts = 0
                step.error = ""
        task.replans = 0
        self.log.bind(task.id)
        self.log.event("task_resumed", f"{len(task.verified_steps())}/{len(task.steps)} steps already verified")
        return self.run(task)

    # --- planning -------------------------------------------------------
    def _plan(self, task: Task) -> None:
        self._guard()
        memory = self.history.context_for(exclude=task.id)
        try:
            result = self.planner.make_plan(task.objective, memory)
        except PlanningError as e:
            if self.activity:
                self.activity.record("error_encountered", "planning failed", error_type=type(e).__name__)
            raise _Finish(TaskStatus.FAILED, f"planning failed: {e}") from None
        if result.blocked_reason or not result.steps:
            raise _Finish(TaskStatus.NEEDS_ATTENTION, result.blocked_reason or "the planner produced no steps")
        task.steps = result.steps
        task.renumber()
        self.checkpoints.save(task, "planned")
        self.history.save(task)
        for s in task.steps:
            self.log.event("plan_step", f"{s.index}. {s.description}", tool=s.tool)

    def _replan(self, task: Task, failed, reason: str) -> None:
        if task.replans >= self.cfg.max_replans:
            raise _Finish(TaskStatus.FAILED, f"gave up after {task.replans} replans: {truncate(reason, 600)}")
        self._guard()
        task.replans += 1
        if self.activity:
            self.activity.record(
                "recovery_attempted", "planner replan started", failed_tool=failed.tool if failed else ""
            )
        self.log.event("replan", truncate(reason, 400))
        try:
            result = self.planner.replan(task, reason, failed)
        except PlanningError as e:
            if self.activity:
                self.activity.record("error_encountered", "replanning failed", error_type=type(e).__name__)
            raise _Finish(TaskStatus.FAILED, f"replanning failed: {e}") from None
        if result.blocked_reason:
            raise _Finish(TaskStatus.NEEDS_ATTENTION, result.blocked_reason)
        if not result.steps:
            raise _Finish(TaskStatus.FAILED, f"no viable plan after: {truncate(reason, 600)}")
        task.keep_verified_and_append(result.steps)
        self.checkpoints.save(task, "replanned")
        self.history.save(task)
        for s in task.steps:
            if s.status == StepStatus.PENDING:
                self.log.event("plan_step", f"{s.index}. {s.description}", tool=s.tool)

    # --- acting + verifying ----------------------------------------------
    def _execute_step(self, task: Task, step) -> bool:
        step.status = StepStatus.RUNNING
        self.history.save(task)
        self.log.event("step_started", f"step {step.index}: {step.description}", tool=step.tool)
        tool = self.registry.get(step.tool)
        while True:
            self._guard()
            step.attempts += 1
            task.actions += 1
            self._run_actions += 1

            result = self._act(task, step)
            step.output = result.output
            step.error = result.error
            self.log.event(
                "step_result",
                f"step {step.index} attempt {step.attempts}: {'ok' if result.ok else result.kind + ': ' + truncate(result.error, 200)}",
                tool=step.tool, args={
                    key: ("[REDACTED]" if key in step.sensitive_args else value)
                    for key, value in step.args.items()
                },
            )
            if result.kind == "human":
                step.status = StepStatus.FAILED
                raise _Finish(TaskStatus.NEEDS_ATTENTION, result.error)
            if not result.ok and self.activity:
                self.activity.record(
                    "error_encountered",
                    f"{step.tool} failed ({result.kind})",
                    tool=step.tool,
                    error_kind=result.kind,
                )

            verdict = self.verifier.check_step(step, result) if result.ok else Verdict(False, result.error, "tool_status")
            if verdict.passed:
                step.status = StepStatus.VERIFIED
                step.verified_by = verdict.method
                self.log.event("step_verified", f"step {step.index} verified by {verdict.method}")
                self.checkpoints.save(task, f"step {step.index} verified")
                self.history.save(task)
                return True

            step.error = verdict.reason
            kind = result.kind if not result.ok else "verification"
            self.log.event("step_failed", f"step {step.index}: {truncate(verdict.reason, 300)}", failure=kind)
            if decide(tool, kind, step.attempts, self.cfg.max_step_retries) == Recovery.RETRY:
                if self.activity:
                    self.activity.record(
                        "recovery_attempted", "idempotent tool retry", tool=step.tool, attempt=step.attempts + 1
                    )
                self.sleep(self.cfg.retry_delay * (2 ** (step.attempts - 1)))
                continue
            step.status = StepStatus.FAILED
            return False

    def _act(self, task: Task, step) -> ToolResult:
        if step.deferred or self.registry.missing_args(step.tool, step.args):
            try:
                step.args = self.planner.resolve_args(task, step)
                step.deferred = False
            except PlanningError as e:
                return ToolResult(False, error=f"could not decide arguments: {e}", kind="invalid")
        return self.registry.execute(step.tool, step.args)

    # --- finishing --------------------------------------------------------
    def _finalize(self, task: Task) -> None:
        self._guard()
        verdict = self.verifier.check_objective(task)
        self.log.event("objective_check", f"{'PASS' if verdict.passed else 'FAIL'} ({verdict.method}): {verdict.reason}")
        if verdict.passed:
            task.status = TaskStatus.COMPLETED
            methods = sorted({s.verified_by for s in task.steps if s.verified_by})
            task.summary = (
                f"{len(task.verified_steps())}/{len(task.steps)} steps verified ({', '.join(methods) or 'n/a'}). "
                f"{verdict.reason}"
            )
            if verdict.method == "deterministic_only":
                task.warnings.append(verdict.reason)
            return
        self._replan(task, None, f"The objective is not yet met. Missing: {verdict.reason}")

    def _guard(self) -> None:
        self.stop.check()
        if self.clock() > self._deadline:
            raise _Finish(TaskStatus.TIMED_OUT, f"exceeded the {self.cfg.max_task_seconds}s time limit")
        if self._run_actions >= self.cfg.max_actions:
            raise _Finish(TaskStatus.FAILED, f"action budget of {self.cfg.max_actions} exhausted")

    def _conclude(self, task: Task) -> None:
        self.history.save(task)  # refresh updated_at before the report reads it
        task.report = str(write_report(task, self.cfg.reports_dir))
        self.history.save(task)
        self.checkpoints.save(task, f"final: {task.status.value}")
        self.log.event("task_finished", f"{task.status.value}: {task.summary or task.error}")
