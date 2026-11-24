import json
import shutil
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from agent.dashboard import DashboardServer
from agent.memory import Database, TaskHistory
from agent.models import Step, StepStatus, Task, TaskStatus
from agent.safety import EmergencyStop


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        self.db = Database(self.root / "agent.db")
        self.addCleanup(self.db.close)
        self.history = TaskHistory(self.db)
        self.stop = EmergencyStop(self.root / "STOP")
        self.desktop = SimpleNamespace(current_app="Test app", latest_screenshot=b"redacted")
        task = Task(
            id="dash1234",
            objective="Inspect the app",
            status=TaskStatus.RUNNING,
            steps=[Step(index=1, description="Observe", tool="observe_desktop", status=StepStatus.VERIFIED)],
        )
        self.history.save(task)
        self.server = DashboardServer(self.history, self.stop, self.desktop, port=0)
        self.url = self.server.start()
        self.addCleanup(self.server.close)

    def test_state_and_in_memory_screenshot_are_available(self):
        with urllib.request.urlopen(self.url + "api/state") as response:
            state = json.loads(response.read())
        self.assertEqual(state["current_app"], "Test app")
        self.assertEqual(state["task"]["verified"], 1)
        self.assertEqual(state["task"]["percent"], 100)
        with urllib.request.urlopen(self.url + "api/screenshot") as response:
            self.assertEqual(response.read(), b"redacted")

    def test_stop_requires_same_origin_and_triggers_emergency_stop(self):
        request = urllib.request.Request(
            self.url + "api/stop",
            data=b"",
            headers={"Origin": "http://127.0.0.1:1"},
            method="POST",
        )
        with self.assertRaises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        self.assertEqual(error.exception.code, 403)
        error.exception.close()
        self.assertFalse(self.stop.is_set())

        origin = self.url.rstrip("/")
        request = urllib.request.Request(
            self.url + "api/stop",
            data=b"",
            headers={"Origin": origin},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            self.assertEqual(response.status, 200)
        self.assertTrue(self.stop.is_set())

    def test_monitoring_controls_and_system_telemetry_are_same_origin(self):
        class FakeActivity:
            enabled = True

            def status(self):
                return {"active_task": True, "enabled": self.enabled, "current_app": "demo.exe"}

            def system_snapshot(self):
                return {"cpu": {"usage_percent": 12.5}} if self.enabled else None

            def set_monitoring(self, enabled):
                if type(enabled) is not bool:
                    raise ValueError("enabled must be a boolean")
                self.enabled = enabled
                return self.status()

        activity = FakeActivity()
        server = DashboardServer(
            self.history, self.stop, self.desktop, port=0, activity=activity
        )
        url = server.start()
        self.addCleanup(server.close)
        with urllib.request.urlopen(url + "api/state") as response:
            state = json.loads(response.read())
        self.assertEqual(state["system"]["cpu"]["usage_percent"], 12.5)
        self.assertTrue(state["monitoring"]["enabled"])

        request = urllib.request.Request(
            url + "api/monitoring",
            data=b'{"enabled":false}',
            headers={"Origin": url.rstrip("/"), "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request) as response:
            self.assertFalse(json.loads(response.read())["enabled"])

        with urllib.request.urlopen(url + "api/state") as response:
            state = json.loads(response.read())
        self.assertIsNone(state["system"])
        self.assertFalse(state["screenshot"])


if __name__ == "__main__":
    unittest.main()
