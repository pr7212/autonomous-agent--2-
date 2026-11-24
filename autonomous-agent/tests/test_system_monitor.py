import os
import unittest
from unittest import mock
from types import SimpleNamespace

from agent.privacy import is_sensitive_application
from agent.system_monitor import SystemMonitor


class SystemMonitorTests(unittest.TestCase):
    def test_system_snapshot_contains_requested_resource_and_network_fields(self):
        snapshot = SystemMonitor().snapshot(os.getpid())
        self.assertIn("name", snapshot["os"])
        self.assertIn("version", snapshot["os"])
        self.assertGreaterEqual(snapshot["hardware"]["logical_cores"], 1)
        self.assertGreaterEqual(snapshot["cpu"]["usage_percent"], 0)
        self.assertIn("usage_percent", snapshot["memory"])
        self.assertIn("usage_percent", snapshot["disk"])
        self.assertIn("connected", snapshot["network"])
        self.assertIsInstance(snapshot["network"]["interfaces"], list)
        self.assertEqual(snapshot["application"]["pid"], os.getpid())
        self.assertIn("memory_mb", snapshot["application"])

    def test_sensitive_apps_are_classified_without_returning_window_text(self):
        self.assertTrue(is_sensitive_application("chrome.exe", "Inbox - Gmail"))
        self.assertTrue(is_sensitive_application("Slack.exe"))
        self.assertFalse(is_sensitive_application("notepad.exe", "notes.txt"))

    def test_foreground_process_is_reused_for_cpu_sampling(self):
        samples = iter((0.0, 12.5))

        class ProcessStub:
            def name(self):
                return "test-process"

            def status(self):
                return "running"

            def memory_info(self):
                return SimpleNamespace(rss=1024)

            def cpu_percent(self, interval=None):
                return next(samples)

        monitor = SystemMonitor()
        process = ProcessStub()
        with mock.patch.object(monitor._psutil, "Process", return_value=process) as process_factory:
            monitor.snapshot(os.getpid())
            second = monitor.snapshot(os.getpid())

        process_factory.assert_called_once_with(os.getpid())
        self.assertEqual(second["application"]["cpu_percent"], 12.5)


if __name__ == "__main__":
    unittest.main()
