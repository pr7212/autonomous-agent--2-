import os
import sys
import unittest
from unittest import mock

from tests.helpers import AgentCase


class FileToolTests(AgentCase):
    def setUp(self):
        super().setUp()
        self.make()
        self.run_tool = self.agent.registry.execute

    def test_write_read_roundtrip(self):
        w = self.run_tool("write_file", {"path": "notes/a.txt", "content": "héllo\nworld\n"})
        self.assertTrue(w.ok, w.error)
        self.assertEqual(w.data["path"], "notes/a.txt")
        r = self.run_tool("read_file", {"path": "notes/a.txt"})
        self.assertEqual(r.output, "héllo\nworld\n")

    def test_errors_are_typed(self):
        self.assertEqual(self.run_tool("read_file", {"path": "nope.txt"}).kind, "error")
        self.assertEqual(self.run_tool("read_file", {"path": "../../etc/passwd"}).kind, "sandbox")
        self.assertEqual(self.run_tool("write_file", {"path": "a", "content": 5}).kind, "invalid")
        self.assertEqual(self.run_tool("write_file", {"path": "a"}).kind, "invalid")
        self.assertEqual(self.run_tool("teleport", {}).kind, "unknown_tool")

    def test_list_dir_and_make_dir(self):
        self.run_tool("make_dir", {"path": "src"})
        self.run_tool("write_file", {"path": "a.txt", "content": "x"})
        out = self.run_tool("list_dir", {}).output.splitlines()
        self.assertEqual(out, ["src/", "a.txt"])

    def test_delete_needs_approval(self):
        self.run_tool("write_file", {"path": "keep.txt", "content": "x"})
        denied = self.run_tool("delete_file", {"path": "keep.txt"})  # approver says no
        self.assertEqual(denied.kind, "denied")
        self.assertTrue((self.ws / "keep.txt").exists())

    def test_delete_when_approved_and_root_is_protected(self):
        self.make(approve=True)
        self.agent.registry.execute("write_file", {"path": "gone.txt", "content": "x"})
        self.assertTrue(self.agent.registry.execute("delete_file", {"path": "gone.txt"}).ok)
        self.assertFalse((self.ws / "gone.txt").exists())
        self.assertEqual(self.agent.registry.execute("delete_file", {"path": "."}).kind, "invalid")

    def test_long_output_is_truncated(self):
        self.run_tool("write_file", {"path": "big.txt", "content": "x" * 50_000})
        out = self.run_tool("read_file", {"path": "big.txt"}).output
        self.assertLess(len(out), 9_000)
        self.assertIn("omitted", out)


class TerminalToolTests(AgentCase):
    def setUp(self):
        super().setUp()
        self.make()
        self.run_tool = self.agent.registry.execute

    def script(self, name, code):
        self.run_tool("write_file", {"path": name, "content": code})

    def test_python_runs_in_the_workspace(self):
        self.script("hi.py", "import os; print('cwd=' + os.path.basename(os.getcwd()))\n")
        r = self.run_tool("run_command", {"command": "python hi.py"})
        self.assertTrue(r.ok, r.error)
        self.assertIn("cwd=workspace", r.output)
        self.assertEqual(r.data["returncode"], 0)

    def test_nonzero_exit_is_a_failure_with_output(self):
        self.script("bad.py", "import sys; print('boom'); sys.exit(3)\n")
        r = self.run_tool("run_command", {"command": "python bad.py"})
        self.assertFalse(r.ok)
        self.assertEqual(r.data["returncode"], 3)
        self.assertIn("boom", r.error)

    def test_shell_operators_are_rejected(self):
        r = self.run_tool("run_command", {"command": "python a.py && python b.py"})
        self.assertEqual(r.kind, "invalid")
        self.assertIn("no shell", r.error)

    def test_dangerous_and_unlisted_commands_are_blocked(self):
        self.assertEqual(self.run_tool("run_command", {"command": "rm -rf ."}).kind, "denied")
        self.assertEqual(self.run_tool("run_command", {"command": "bash script.sh"}).kind, "denied")
        self.assertEqual(self.run_tool("run_command", {"command": "git push origin main"}).kind, "denied")
        self.assertEqual(self.run_tool("run_command", {"command": "sudo whoami"}).kind, "denied")

    def test_script_outside_workspace_is_refused(self):
        r = self.run_tool("run_command", {"command": "python ../outside.py"})
        self.assertEqual(r.kind, "sandbox")

    def test_timeout_kills_the_process(self):
        self.script("slow.py", "import time; time.sleep(30)\n")
        r = self.run_tool("run_command", {"command": "python slow.py", "timeout": 1})
        self.assertEqual(r.kind, "timeout")
        self.assertFalse(r.ok)

    def test_credentials_do_not_leak_into_child_processes(self):
        self.script("env.py", "import os; print('token=' + str(os.environ.get('MY_SECRET_TOKEN')))\n")
        with mock.patch.dict(os.environ, {"MY_SECRET_TOKEN": "supersecretvalue123"}):
            r = self.run_tool("run_command", {"command": "python env.py"})
        self.assertIn("token=None", r.output)
        self.assertNotIn("supersecretvalue123", r.output)

    def test_run_tests_reports_pass_and_fail(self):
        self.script("test_ok.py", "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(1 + 1, 2)\n")
        self.assertTrue(self.run_tool("run_tests", {"target": "test_ok.py"}).ok)
        self.script("test_bad.py", "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        self.assertEqual(1, 2)\n")
        bad = self.run_tool("run_tests", {"target": "test_bad.py"})
        self.assertFalse(bad.ok)
        self.assertEqual(bad.data["returncode"], 1)

    def test_unknown_executable_runs_only_with_approval(self):
        self.make(approve=True)
        r = self.agent.registry.execute("run_command", {"command": "definitely-not-a-real-program"})
        self.assertEqual(r.kind, "error")  # approved, then "command not found"
        self.assertIn("not found", r.error)


if __name__ == "__main__":
    unittest.main()
