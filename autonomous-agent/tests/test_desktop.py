import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from agent.config import Config
from agent.memory import Database, TaskHistory
from agent.models import Step, Task
from agent.tools.desktop import DesktopRuntime, TOOLS, _check_coordinate, _is_password_control, _process_name


class DesktopToolTests(unittest.TestCase):
    def test_tools_are_registered_without_loading_platform_packages(self):
        names = {tool.name for tool in TOOLS}
        self.assertTrue({
            "observe_desktop", "move_mouse", "click", "type_text",
            "press_key", "scroll", "wait",
        }.issubset(names))
        typing = next(tool for tool in TOOLS if tool.name == "type_text")
        self.assertEqual(typing.sensitive_args, ("text",))

    def test_screenshots_and_ocr_default_off(self):
        config = Config(home=Path(tempfile.mkdtemp()))
        self.addCleanup(shutil.rmtree, config.home, True)
        runtime = DesktopRuntime(config)
        self.assertFalse(runtime.screenshot_enabled)
        self.assertFalse(runtime.ocr_enabled)

    @unittest.skipUnless(os.name == "nt", "Windows process inspection only")
    def test_application_label_uses_process_name_not_window_title(self):
        self.assertTrue(_process_name(os.getpid()))

    def test_coordinates_are_confined_to_window_and_integer(self):
        runtime = DesktopRuntime()
        runtime._window_rect = (10, 20, 100, 200)
        self.assertEqual(_check_coordinate(runtime, 20, 30), (20, 30))
        for x, y in ((9, 30), (20, 200), (20.5, 30), (True, 30)):
            with self.subTest(x=x, y=y), self.assertRaises(ValueError):
                _check_coordinate(runtime, x, y)

    def test_password_detection_fails_closed_for_unknown_edit_controls(self):
        unknown_edit = SimpleNamespace(element_info=SimpleNamespace(control_type="Edit", is_password=None))
        normal_edit = SimpleNamespace(element_info=SimpleNamespace(control_type="Edit", is_password=False))
        tagged_custom = SimpleNamespace(
            element_info=SimpleNamespace(control_type="Custom", is_password=False, automation_id="PasswordEntry")
        )
        self.assertTrue(_is_password_control(unknown_edit))
        self.assertFalse(_is_password_control(normal_edit))
        self.assertTrue(_is_password_control(tagged_custom))

    def test_sensitive_desktop_args_are_redacted_when_persisted(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        db = Database(root / "agent.db")
        self.addCleanup(db.close)
        history = TaskHistory(db)
        step = Step(
            index=1,
            description="Type ordinary user-provided text",
            tool="type_text",
            expected="The ordinary user-provided text appears",
            args={"text": "ordinary user-provided text"},
            verify={"type": "output_contains", "text": "ordinary user-provided text"},
            sensitive_args=("text",),
        )
        task = Task(id="desktop1", objective="Type text", steps=[step])
        history.save(task)

        raw = db.query("SELECT state_json FROM tasks WHERE id = ?", ("desktop1",))[0]["state_json"]
        self.assertNotIn("ordinary user-provided text", raw)
        self.assertEqual(task.steps[0].args["text"], "ordinary user-provided text")
        restored = history.get("desktop1")
        self.assertIn("sensitive input redacted", restored.steps[0].description)
        self.assertEqual(restored.steps[0].expected, "")
        self.assertEqual(restored.steps[0].verify, {})
        self.assertEqual(restored.steps[0].args["text"], "[REDACTED]")
        self.assertTrue(restored.steps[0].deferred)


if __name__ == "__main__":
    unittest.main()
