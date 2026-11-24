from __future__ import annotations

import re
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from agent.bootstrap import build_agent
from agent.config import Config
from agent.llm import LLMError, LLMProvider


class ScriptedLLM(LLMProvider):
    """A fake model. `script` maps a role tag (PLAN, REPLAN, RESOLVE_ARGS, FINAL_VERIFY) to the
    replies it gives, in order. FINAL_VERIFY passes by default; any other exhausted role errors."""

    name = "scripted"

    def __init__(self, script: dict | None = None):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.calls: list[tuple[str, str]] = []

    def chat(self, system, user, *, json_mode=False, temperature=0.2):
        role = re.search(r"ROLE: (\w+)", system).group(1)
        self.calls.append((role, user))
        queue = self.script.get(role)
        if queue:
            reply = queue.pop(0)
        elif role == "FINAL_VERIFY":
            reply = {"complete": True, "reason": "objective met", "gap": ""}
        else:
            raise LLMError(f"script exhausted for {role}")
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, str):
            return reply
        import json
        return json.dumps(reply)

    def users(self, role: str) -> list[str]:
        return [u for r, u in self.calls if r == role]


def step(tool, args=None, desc="", verify=None, deferred=False, expected=""):
    return {
        "description": desc or f"{tool} {args or ''}".strip(),
        "tool": tool,
        "args": args or {},
        "expected": expected,
        "verify": verify or {"type": "none"},
        "deferred": deferred,
    }


def plan(*steps):
    return {"steps": list(steps), "blocked_reason": ""}


class AgentCase(unittest.TestCase):
    """Fresh agent in a temp home, wired to a scripted model."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def make(self, script=None, approve=False, clock=time.monotonic, **cfg):
        cfg.setdefault("llm_mode", "local")
        config = Config(home=self.tmp, retry_delay=0, **cfg)
        self.llm = ScriptedLLM(script)
        self.agent = build_agent(
            config, approver=lambda action, detail: approve, providers=(self.llm, None),
            clock=clock, sleep=lambda s: None,
        )
        self.addCleanup(self.agent.close)
        self.ws = self.agent.cfg.workspace_dir
        return self.agent

    def run_task(self, objective="test objective"):
        task = self.agent.manager.create_task(objective)
        return self.agent.manager.run(task)
