import unittest

from agent.llm import LLMError
from agent.models import Step, StepStatus, Task
from agent.tools import ToolResult
from tests.helpers import AgentCase


class StepVerificationTests(AgentCase):
    def setUp(self):
        super().setUp()
        self.make()
        self.verify = self.agent.verifier.check_step

    def written(self, content="print('x')\n", path="a.py", spec=None):
        step = Step(1, "write", "write_file", {"path": path, "content": content}, verify=spec or {})
        return step, self.agent.registry.execute("write_file", step.args)

    def test_write_is_confirmed_by_reading_it_back(self):
        step, result = self.written()
        verdict = self.verify(step, result)
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.method, "readback")

    def test_tampered_file_fails_readback(self):
        step, result = self.written()
        (self.ws / "a.py").write_text("something else")
        verdict = self.verify(step, result)
        self.assertFalse(verdict.passed)
        self.assertIn("differs", verdict.reason)

    def test_file_deleted_after_write_fails(self):
        step, result = self.written()
        (self.ws / "a.py").unlink()
        self.assertFalse(self.verify(step, result).passed)

    def test_tool_failure_is_never_verified(self):
        step = Step(1, "read", "read_file", {"path": "missing"})
        verdict = self.verify(step, ToolResult(False, error="not found"))
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.method, "tool_status")

    def test_file_exists_and_contains_specs(self):
        step, result = self.written("alpha beta", "n.txt", {"type": "file_contains", "path": "n.txt", "text": "beta"})
        self.assertTrue(self.verify(step, result).passed)
        step.verify = {"type": "file_contains", "path": "n.txt", "text": "gamma"}
        self.assertFalse(self.verify(step, result).passed)
        step.verify = {"type": "file_exists", "path": "other.txt"}
        self.assertFalse(self.verify(step, result).passed)

    def test_command_specs(self):
        self.agent.registry.execute("write_file", {"path": "p.py", "content": "print('ready')\n"})
        result = self.agent.registry.execute("run_command", {"command": "python p.py"})
        step = Step(1, "run", "run_command", {"command": "python p.py"}, verify={"type": "output_contains", "text": "ready"})
        self.assertTrue(self.verify(step, result).passed)
        step.verify = {"type": "output_contains", "text": "never printed"}
        self.assertFalse(self.verify(step, result).passed)
        step.verify = {"type": "exit_code", "code": 7}
        self.assertFalse(self.verify(step, result).passed)

    def test_spec_paths_cannot_leave_the_workspace(self):
        step, result = self.written(spec={"type": "file_exists", "path": "../escape.txt"})
        verdict = self.verify(step, result)
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.method, "sandbox")


class ObjectiveVerificationTests(AgentCase):
    def task(self):
        return Task(
            id="t1", objective="build the thing",
            steps=[Step(1, "write a", "write_file", output="wrote", status=StepStatus.VERIFIED, verified_by="readback")],
        )

    def test_model_can_reject_an_objective_and_names_the_gap(self):
        self.make({"FINAL_VERIFY": [{"complete": False, "reason": "partial", "gap": "tests missing"}]})
        verdict = self.agent.verifier.check_objective(self.task())
        self.assertFalse(verdict.passed)
        self.assertEqual(verdict.reason, "tests missing")

    def test_string_booleans_are_understood(self):
        self.make({"FINAL_VERIFY": [{"complete": "false", "reason": "no", "gap": "x"}]})
        self.assertFalse(self.agent.verifier.check_objective(self.task()).passed)

    def test_model_outage_degrades_to_deterministic_only_and_says_so(self):
        self.make({"FINAL_VERIFY": [LLMError("down")]})
        verdict = self.agent.verifier.check_objective(self.task())
        self.assertTrue(verdict.passed)
        self.assertEqual(verdict.method, "deterministic_only")


if __name__ == "__main__":
    unittest.main()
