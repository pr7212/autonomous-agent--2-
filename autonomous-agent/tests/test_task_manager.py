import json
import unittest
from pathlib import Path

from agent.llm import LLMError
from agent.models import StepStatus, TaskStatus
from agent.safety import Action
from agent.tools import Tool, ToolResult
from tests.helpers import AgentCase, plan, step

V = StepStatus.VERIFIED


def write(path, content, verify=None):
    return step("write_file", {"path": path, "content": content}, verify=verify)


class HappyPathTests(AgentCase):
    def test_plan_act_verify_report(self):
        self.make({"PLAN": [plan(
            write("hello.py", "print('hi')\n", {"type": "file_exists", "path": "hello.py"}),
            step("run_command", {"command": "python hello.py"}, verify={"type": "output_contains", "text": "hi"}),
        )]})
        task = self.run_task("make hello")
        self.assertEqual(task.status, TaskStatus.COMPLETED, task.error)
        self.assertEqual([s.status for s in task.steps], [V, V])
        self.assertTrue((self.ws / "hello.py").exists())
        self.assertIn("readback+file_exists", task.steps[0].verified_by)
        self.assertIn("COMPLETED", Path(task.report).read_text(encoding="utf-8"))
        kinds = {e["kind"] for e in self.agent.history.events(task.id)}
        self.assertTrue({"task_started", "plan_step", "step_verified", "objective_check", "task_finished"} <= kinds)
        self.assertGreaterEqual(self.agent.checkpoints.count(task.id), 4)
        self.assertEqual(self.agent.history.get(task.id).status, TaskStatus.COMPLETED)

    def test_finished_work_becomes_context_for_the_next_plan(self):
        self.make({"PLAN": [plan(write("a.txt", "x")), plan(write("b.txt", "y"))]})
        first = self.run_task("first job")
        self.run_task("second job")
        self.assertIn("first job", self.llm.users("PLAN")[1])
        self.assertEqual(first.status, TaskStatus.COMPLETED)

    def test_model_chooses_arguments_at_run_time_from_observations(self):
        self.make({
            "PLAN": [plan(step("read_file", {"path": "note.txt"}), step("write_file", deferred=True, desc="copy the note"))],
            "RESOLVE_ARGS": [{"args": {"path": "copy.txt", "content": "alpha"}}],
        })
        (self.ws / "note.txt").write_text("alpha", encoding="utf-8")
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.COMPLETED, task.error)
        self.assertEqual((self.ws / "copy.txt").read_text(encoding="utf-8"), "alpha")
        self.assertIn("alpha", self.llm.users("RESOLVE_ARGS")[0])  # it saw step 1's output

    def test_secrets_never_reach_logs(self):
        key = "sk-test-secret-999"
        self.make({"PLAN": [plan(write("leak.txt", f"key={key}"))]}, anthropic_api_key=key)
        task = self.run_task()
        dump = json.dumps(self.agent.history.events(task.id))
        dump += (self.agent.cfg.log_dir / f"{task.id}.jsonl").read_text(encoding="utf-8")
        self.assertNotIn(key, dump)
        self.assertIn("[REDACTED]", dump)


class PlanningTests(AgentCase):
    def test_invalid_plan_is_sent_back_with_the_reason(self):
        self.make({"PLAN": [{"steps": [{"tool": "teleport", "args": {}}]}, plan(write("a.txt", "x"))]})
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        plans = self.llm.users("PLAN")
        self.assertEqual(len(plans), 2)
        self.assertIn("unknown tool", plans[1])

    def test_blocked_objective_needs_attention(self):
        self.make({"PLAN": [{"steps": [], "blocked_reason": "needs a login I do not have"}]})
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.NEEDS_ATTENTION)
        self.assertIn("login", task.error)

    def test_model_outage_fails_cleanly(self):
        self.make({"PLAN": [LLMError("down"), LLMError("down")]})
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("planning failed", task.error)


class RecoveryTests(AgentCase):
    def counting_tool(self, name, fails, idempotent):
        calls = {"n": 0}

        def func(ctx, args):
            calls["n"] += 1
            if calls["n"] <= fails:
                return ToolResult(False, error="transient glitch")
            return ToolResult(True, "done")

        self.agent.registry.register(Tool(name, "test tool", {}, func, idempotent=idempotent))
        return calls

    def test_idempotent_step_is_retried_in_place(self):
        self.make({"PLAN": [plan(step("flaky"))]})
        calls = self.counting_tool("flaky", fails=1, idempotent=True)
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual((calls["n"], task.steps[0].attempts, task.replans), (2, 2, 0))

    def test_non_idempotent_failure_replans_instead_of_repeating(self):
        self.make({"PLAN": [plan(step("once"))], "REPLAN": [plan(write("fixed.txt", "ok"))]})
        calls = self.counting_tool("once", fails=99, idempotent=False)
        task = self.run_task()
        self.assertEqual(calls["n"], 1)
        self.assertEqual(task.replans, 1)
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertTrue((self.ws / "fixed.txt").exists())

    def test_failing_command_triggers_replan_that_keeps_verified_work(self):
        self.make({
            "PLAN": [plan(write("broken.py", "raise SystemExit(3)\n"), step("run_command", {"command": "python broken.py"}))],
            "REPLAN": [plan(write("ok.py", "print('fixed')\n"), step("run_command", {"command": "python ok.py"}))],
        })
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.COMPLETED, task.error)
        self.assertEqual(task.replans, 1)
        self.assertEqual([s.index for s in task.steps], [1, 2, 3])
        self.assertTrue(all(s.status == V for s in task.steps))
        replan_prompt = self.llm.users("REPLAN")[0]
        self.assertIn("exit code 3", replan_prompt)
        self.assertIn("broken.py", replan_prompt)

    def test_replan_budget_is_enforced(self):
        failing = plan(step("run_command", {"command": "python missing.py"}))
        self.make({"PLAN": [failing], "REPLAN": [failing, failing]}, max_replans=1)
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("gave up after 1 replans", task.error)

    def test_objective_audit_failure_replans_with_the_gap(self):
        self.make({
            "PLAN": [plan(write("a.txt", "x"))],
            "FINAL_VERIFY": [{"complete": False, "reason": "incomplete", "gap": "README missing"},
                             {"complete": True, "reason": "now complete", "gap": ""}],
            "REPLAN": [plan(write("README.md", "# hi"))],
        })
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.COMPLETED)
        self.assertEqual(task.replans, 1)
        self.assertIn("README missing", self.llm.users("REPLAN")[0])
        self.assertTrue((self.ws / "README.md").exists())


class SafetyBoundaryTests(AgentCase):
    def delete_plan(self):
        return {"PLAN": [plan(write("a.txt", "x"), step("delete_file", {"path": "a.txt"}))]}

    def test_denied_deletion_stops_for_a_human_and_keeps_the_file(self):
        self.make({**self.delete_plan(), "REPLAN": [{"steps": [], "blocked_reason": "deleting a.txt needs approval"}]})
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.NEEDS_ATTENTION)
        self.assertTrue((self.ws / "a.txt").exists())
        self.assertIn("approval", task.error)

    def test_approved_deletion_is_executed_and_verified(self):
        self.make(self.delete_plan(), approve=True)
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.COMPLETED, task.error)
        self.assertFalse((self.ws / "a.txt").exists())
        self.assertEqual(task.steps[1].verified_by, "readback")

    def test_human_only_actions_halt_immediately_and_are_not_retried(self):
        self.make({"PLAN": [plan(step("submit_application"), write("after.txt", "x"))]}, approve=True)
        calls = {"n": 0}

        def submit(ctx, args):
            calls["n"] += 1
            ctx.policy.require(Action.SUBMIT_CONSEQUENTIAL, "graduate scholarship application")
            return ToolResult(True, "submitted")

        self.agent.registry.register(Tool("submit_application", "submit", {}, submit, idempotent=True))
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.NEEDS_ATTENTION)
        self.assertEqual((calls["n"], task.replans), (1, 0))
        self.assertFalse((self.ws / "after.txt").exists())

    def test_sandbox_escape_attempt_is_a_failure_not_a_write(self):
        self.make({
            "PLAN": [plan(write("../outside.txt", "x"))],
            "REPLAN": [{"steps": [], "blocked_reason": "cannot write outside the workspace"}],
        })
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.NEEDS_ATTENTION)
        self.assertFalse((self.ws.parent / "outside.txt").exists())


class LimitTests(AgentCase):
    def test_time_limit(self):
        ticks = {"t": 0}

        def clock():
            ticks["t"] += 400
            return ticks["t"]

        self.make({"PLAN": [plan(*[write(f"f{i}.txt", "x") for i in range(4)])]}, clock=clock, max_task_seconds=900)
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.TIMED_OUT)
        self.assertIn("time limit", task.error)
        self.assertTrue(0 < len(task.verified_steps()) < 4)

    def test_action_budget(self):
        self.make({"PLAN": [plan(*[write(f"f{i}.txt", "x") for i in range(3)])]}, max_actions=2)
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.FAILED)
        self.assertIn("action budget", task.error)
        self.assertEqual(len(task.verified_steps()), 2)

    def test_emergency_stop_then_resume_skips_finished_steps(self):
        self.make({"PLAN": [plan(write("a.txt", "1"), step("trip"), write("b.txt", "2"))]})
        calls = {"trip": 0}

        def trip(ctx, args):
            calls["trip"] += 1
            ctx.stop.trigger("test button")
            return ToolResult(True, "tripped")

        self.agent.registry.register(Tool("trip", "pull the emergency stop", {}, trip))
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.CANCELLED)
        self.assertIn("test button", task.error)
        self.assertEqual([s.status for s in task.steps], [V, V, StepStatus.PENDING])
        self.assertFalse((self.ws / "b.txt").exists())

        self.agent.stop.clear()
        resumed = self.agent.manager.resume(task.id)
        self.assertEqual(resumed.status, TaskStatus.COMPLETED, resumed.error)
        self.assertTrue((self.ws / "b.txt").exists())
        self.assertEqual(calls["trip"], 1)             # finished work was not repeated
        self.assertEqual(len(self.llm.users("PLAN")), 1)  # and nothing was re-planned

    def test_stop_file_from_another_terminal_halts_the_task(self):
        self.make({"PLAN": [plan(write("a.txt", "1"), write("b.txt", "2"))]})
        self.agent.stop.request_via_file()
        task = self.run_task()
        self.assertEqual(task.status, TaskStatus.CANCELLED)
        self.assertFalse((self.ws / "a.txt").exists())

    def test_resume_of_a_completed_task_is_a_noop(self):
        self.make({"PLAN": [plan(write("a.txt", "1"))]})
        task = self.run_task()
        again = self.agent.manager.resume(task.id)
        self.assertEqual(again.status, TaskStatus.COMPLETED)
        self.assertEqual(len(self.llm.users("PLAN")), 1)


if __name__ == "__main__":
    unittest.main()
