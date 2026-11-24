import shutil
import tempfile
import unittest
from pathlib import Path

from agent.activity_monitor import ActivityMonitor
from agent.system_monitor import SystemMonitor


class FakeLog:
    def __init__(self):
        self.task_id = ""
        self.items = []

    def event(self, kind, message, **data):
        self.items.append({"kind": kind, "message": message, "data": data})


class ActivityMonitorTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.log = FakeLog()
        self.apps = iter([
            {"pid": 300, "name": "newapp.exe", "sensitive": False},
            {"pid": 200, "name": "existing.exe", "sensitive": False},
        ])
        self.monitor = ActivityMonitor(
            self.log,
            SystemMonitor(),
            self.root,
            interval=10,
            foreground=lambda: next(self.apps, None),
            process_ids=lambda: {100, 200},
        )

    def tearDown(self):
        self.monitor.stop_task()

    def test_application_changes_are_task_scoped_and_metadata_only(self):
        self.log.task_id = "task1"
        self.monitor.start_task("task1")
        self.monitor.poll_once()
        self.monitor.poll_once()
        kinds = [event["kind"] for event in self.log.items]
        self.assertIn("application_opened", kinds)
        self.assertIn("application_switched", kinds)
        app_event = next(event for event in self.log.items if event["kind"] == "application_opened")
        self.assertNotIn("title", app_event["data"])
        self.assertIn("newapp.exe", app_event["message"])

        before = len(self.log.items)
        self.monitor.set_monitoring(False)
        self.monitor.record("keyboard_shortcut_executed", "keyboard action executed", key="ctrl+s")
        self.monitor.poll_once()
        self.assertEqual(len(self.log.items), before + 1)  # monitoring_stopped only

    def test_sensitive_app_and_raw_input_are_not_recorded(self):
        self.log.task_id = "task1"
        self.monitor._foreground = lambda: {
            "pid": None,
            "name": "Sensitive application excluded",
            "sensitive": True,
        }
        self.monitor.start_task("task1")
        self.monitor.poll_once()
        self.monitor.record(
            "text_entry_executed", "text entered (content omitted)", character_count=19
        )
        dump = repr(self.log.items)
        self.assertNotIn("private message", dump)
        self.assertEqual(self.monitor.status()["current_app"], "Sensitive application excluded")
        self.assertFalse(any(event["kind"] in ("application_opened", "application_switched") for event in self.log.items))

    def test_workspace_changes_are_metadata_only(self):
        self.log.task_id = "task1"
        self.monitor.start_task("task1")
        target = self.root / "note.txt"
        target.write_text("first", encoding="utf-8")
        self.monitor.poll_once()
        target.write_text("a longer update", encoding="utf-8")
        self.monitor.poll_once()
        kinds = [event["kind"] for event in self.log.items]
        self.assertIn("file_created", kinds)
        self.assertIn("file_modified", kinds)
        self.assertNotIn("first", repr(self.log.items))


if __name__ == "__main__":
    unittest.main()
