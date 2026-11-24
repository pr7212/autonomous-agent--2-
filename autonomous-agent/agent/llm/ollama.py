from __future__ import annotations

import json
import time
import urllib.request

from agent.llm.base import LLMError, LLMProvider, post_json


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, host: str, model: str, timeout: float = 180.0):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._probe: tuple[float, bool] = (0.0, False)

    def chat(self, system: str, user: str, *, json_mode: bool = False, temperature: float = 0.2) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"temperature": temperature},
        }
        if json_mode:
            payload["format"] = "json"
        data = post_json(f"{self.host}/api/chat", payload, {}, self.timeout)
        try:
            return data["message"]["content"]
        except (KeyError, TypeError):
            raise LLMError(f"unexpected Ollama response: {str(data)[:200]}") from None

    def models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as response:
                return [m["name"] for m in json.loads(response.read().decode("utf-8")).get("models", [])]
        except Exception:
            return []

    def available(self) -> bool:
        stamp, value = self._probe
        if time.monotonic() - stamp < 30 and stamp:
            return value
        value = bool(self.models())
        self._probe = (time.monotonic(), value)
        return value
