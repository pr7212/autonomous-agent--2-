from __future__ import annotations

from agent.llm.base import LLMError, LLMProvider, post_json

API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"


class ClaudeProvider(LLMProvider):
    name = "claude"

    def __init__(self, api_key: str, model: str, timeout: float = 180.0):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def chat(self, system: str, user: str, *, json_mode: bool = False, temperature: float = 0.2) -> str:
        if not self.api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set")
        if json_mode:
            system += "\n\nRespond with a single JSON object and nothing else."
        payload = {
            "model": self.model,
            "max_tokens": 4096,
            "temperature": temperature,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {"x-api-key": self.api_key, "anthropic-version": API_VERSION}
        data = post_json(API_URL, payload, headers, self.timeout)
        try:
            return "".join(block.get("text", "") for block in data["content"] if block.get("type") == "text")
        except (KeyError, TypeError):
            raise LLMError(f"unexpected Claude response: {str(data)[:200]}") from None
