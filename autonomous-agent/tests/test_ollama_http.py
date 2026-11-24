"""End-to-end through the real OllamaProvider, against a fake Ollama HTTP server."""
import json
import re
import shutil
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from agent.bootstrap import build_agent
from agent.config import Config
from agent.llm import LLMError, OllamaProvider
from agent.models import TaskStatus

PLAN = {
    "steps": [
        {"description": "write the script", "tool": "write_file",
         "args": {"path": "hello.py", "content": "print('hello from the agent')\n"},
         "expected": "file exists", "verify": {"type": "file_exists", "path": "hello.py"}},
        {"description": "run it", "tool": "run_command", "args": {"command": "python hello.py"},
         "expected": "greeting printed", "verify": {"type": "output_contains", "text": "hello from the agent"}},
    ],
    "blocked_reason": "",
}


class FakeOllama(BaseHTTPRequestHandler):
    requests: list = []

    def log_message(self, *args):
        pass

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send({"models": [{"name": "fake-model:1"}]})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append(payload)
        system = payload["messages"][0]["content"]
        role = re.search(r"ROLE: (\w+)", system).group(1)
        reply = PLAN if role == "PLAN" else {"complete": True, "reason": "script ran and printed the greeting", "gap": ""}
        self._send({"message": {"role": "assistant", "content": json.dumps(reply)}})


class OllamaHttpTests(unittest.TestCase):
    def setUp(self):
        FakeOllama.requests = []
        self.server = HTTPServer(("127.0.0.1", 0), FakeOllama)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.host = f"http://127.0.0.1:{self.server.server_port}"
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_provider_talks_to_ollama(self):
        provider = OllamaProvider(self.host, "fake-model:1", timeout=5)
        self.assertTrue(provider.available())
        self.assertEqual(provider.models(), ["fake-model:1"])
        out = provider.chat("ROLE: FINAL_VERIFY", "hi", json_mode=True)
        self.assertTrue(json.loads(out)["complete"])
        sent = FakeOllama.requests[0]
        self.assertEqual((sent["model"], sent["format"], sent["stream"]), ("fake-model:1", "json", False))

    def test_unreachable_server_raises_llm_error(self):
        with self.assertRaises(LLMError):
            OllamaProvider("http://127.0.0.1:9", "x", timeout=2).chat("s", "u")

    def test_full_task_through_the_real_provider(self):
        cfg = Config(home=self.tmp, llm_mode="local", ollama_host=self.host, ollama_model="fake-model:1", retry_delay=0)
        agent = build_agent(cfg)
        self.addCleanup(agent.close)
        task = agent.manager.run(agent.manager.create_task("print a greeting"))
        self.assertEqual(task.status, TaskStatus.COMPLETED, task.error)
        self.assertTrue((cfg.workspace_dir / "hello.py").exists())
        self.assertEqual(agent.router.last_provider, "ollama")
        self.assertIn("hello from the agent", task.steps[1].output)


if __name__ == "__main__":
    unittest.main()
