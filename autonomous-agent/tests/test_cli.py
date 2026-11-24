import contextlib
import io
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.main import main


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        patcher = mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "", "OLLAMA_HOST": "http://127.0.0.1:9"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def cli(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = main(["--home", str(self.tmp), *argv])
        return code, out.getvalue()

    def test_history_starts_empty(self):
        code, out = self.cli("history")
        self.assertEqual((code, out.strip()), (0, "No tasks yet."))

    def test_stop_creates_the_flag_file(self):
        code, _ = self.cli("stop")
        self.assertEqual(code, 0)
        self.assertTrue((self.tmp / "data" / "STOP").exists())

    def test_doctor_flags_missing_models(self):
        code, out = self.cli("doctor")
        self.assertEqual(code, 1)
        self.assertIn("[!!] at least one model usable", out)
        self.assertIn("[ok] workspace writable", out)

    def test_run_without_a_model_explains_instead_of_crashing(self):
        code, out = self.cli("run", "write", "a", "file", "--mode", "local")
        self.assertEqual(code, 1)
        self.assertIn("No model available", out)

    def test_unknown_task_id(self):
        self.assertEqual(self.cli("show", "deadbeef")[0], 1)
        self.assertEqual(self.cli("resume", "deadbeef")[0], 1)


if __name__ == "__main__":
    unittest.main()
