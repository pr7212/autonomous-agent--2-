import os
import shutil
import tempfile
import unittest
from pathlib import Path

from agent.safety import (
    Action,
    Decision,
    EmergencyStop,
    NeedsHuman,
    PermissionDenied,
    PermissionPolicy,
    SandboxError,
    StopRequested,
    Workspace,
    classify_command,
    has_shell_operators,
)


class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.ws = Workspace(self.tmp / "ws")

    def test_inside_paths_resolve(self):
        self.assertEqual(self.ws.resolve("a/b.txt"), self.ws.root / "a" / "b.txt")
        self.assertEqual(self.ws.rel(self.ws.root / "a" / "b.txt"), "a/b.txt")

    def test_dotdot_escape_is_blocked(self):
        for bad in ("../x", "a/../../x", "a/b/../../../etc/passwd"):
            with self.assertRaises(SandboxError, msg=bad):
                self.ws.resolve(bad)

    def test_absolute_outside_is_blocked(self):
        with self.assertRaises(SandboxError):
            self.ws.resolve(str(self.tmp / "outside.txt"))

    def test_absolute_inside_is_allowed(self):
        self.assertEqual(self.ws.resolve(str(self.ws.root / "ok.txt")), self.ws.root / "ok.txt")

    def test_empty_and_null_paths(self):
        for bad in ("", "  ", "a\x00b"):
            with self.assertRaises(SandboxError):
                self.ws.resolve(bad)

    @unittest.skipUnless(os.name == "posix", "symlinks need privileges on Windows")
    def test_symlink_escape_is_blocked(self):
        outside = self.tmp / "secret"
        outside.mkdir()
        (self.ws.root / "link").symlink_to(outside)
        with self.assertRaises(SandboxError):
            self.ws.resolve("link/data.txt")

    def test_snapshot_skips_noise(self):
        (self.ws.root / ".git").mkdir()
        (self.ws.root / ".git" / "config").write_text("x")
        (self.ws.root / "src").mkdir()
        (self.ws.root / "src" / "app.py").write_text("x")
        self.assertEqual(self.ws.snapshot(), ["src/app.py"])


class PermissionTests(unittest.TestCase):
    def test_policy_table_matches_the_design(self):
        expected = {
            Decision.ALLOW: [Action.READ_FILE, Action.WRITE_FILE, Action.RUN_TESTS, Action.NAVIGATE, Action.USE_API],
            Decision.ASK: [Action.DELETE_FILE, Action.SEND_MESSAGE, Action.PURCHASE, Action.CHANGE_SECURITY],
            Decision.HUMAN: [Action.SUBMIT_CONSEQUENTIAL],
            Decision.DENY: [Action.BLOCKED_COMMAND],
        }
        for decision, actions in expected.items():
            for action in actions:
                self.assertEqual(PermissionPolicy.decide(action), decision, action)
        self.assertEqual(PermissionPolicy.decide(Action.DESKTOP_OBSERVE), Decision.ASK)
        self.assertEqual(PermissionPolicy.decide(Action.DESKTOP_CONTROL), Decision.ASK)

    def test_ask_needs_a_yes(self):
        asked = []
        yes = PermissionPolicy(lambda a, d: asked.append((a, d)) or True)
        yes.require(Action.DELETE_FILE, "x.txt")
        self.assertEqual(asked, [("delete_file", "x.txt")])
        no = PermissionPolicy(lambda a, d: False)
        with self.assertRaises(PermissionDenied):
            no.require(Action.PURCHASE, "laptop")

    def test_allow_never_asks(self):
        policy = PermissionPolicy(lambda a, d: self.fail("should not ask"))
        policy.require(Action.READ_FILE, "x")

    def test_human_only_actions_cannot_be_approved(self):
        policy = PermissionPolicy(lambda a, d: True)  # even a yes-man approver
        with self.assertRaises(NeedsHuman):
            policy.require(Action.SUBMIT_CONSEQUENTIAL, "exam")

    def test_audit_trail(self):
        seen = []
        policy = PermissionPolicy(lambda a, d: False, audit=lambda kind, msg, **kw: seen.append(kind))
        with self.assertRaises(PermissionDenied):
            policy.require(Action.DELETE_FILE, "x")
        self.assertEqual(seen, ["approval_denied"])

    def test_command_classification(self):
        cases = {
            ("python", "app.py"): Action.RUN_CODE,
            ("python3.12", "-m", "pytest"): Action.RUN_TESTS,
            ("pytest", "-q"): Action.RUN_TESTS,
            ("git", "status"): Action.SAFE_COMMAND,
            ("git", "push"): Action.RUN_COMMAND,
            ("rm", "-rf", "x"): Action.BLOCKED_COMMAND,
            ("bash", "-c", "x"): Action.BLOCKED_COMMAND,
            ("powershell.exe", "x"): Action.BLOCKED_COMMAND,
            ("sudo", "x"): Action.CHANGE_SECURITY,
            ("reg", "add", "x"): Action.CHANGE_SECURITY,
            ("curl", "http://x"): Action.RUN_COMMAND,
        }
        for argv, action in cases.items():
            self.assertEqual(classify_command(list(argv))[0], action, argv)

    def test_shell_operators_detected(self):
        for cmd in ("a && b", "a | b", "a; b", "a > f", "a `b`", "a $(b)"):
            self.assertTrue(has_shell_operators(cmd), cmd)
        self.assertFalse(has_shell_operators("python app.py --flag value"))


class EmergencyStopTests(unittest.TestCase):
    def test_flag_and_file_triggers(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        stop = EmergencyStop(tmp / "STOP")
        stop.check()  # not set: no error
        stop.trigger("button")
        with self.assertRaises(StopRequested):
            stop.check()
        stop.clear()
        self.assertFalse(stop.is_set())
        stop.request_via_file()
        self.assertTrue(stop.is_set())
        self.assertEqual(stop.reason, "STOP file present")
        stop.clear()
        self.assertFalse((tmp / "STOP").exists())


if __name__ == "__main__":
    unittest.main()
