from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod


class LLMError(Exception):
    """Transport or provider failure (not a malformed answer)."""


class LLMProvider(ABC):
    name = "llm"

    @abstractmethod
    def chat(self, system: str, user: str, *, json_mode: bool = False, temperature: float = 0.2) -> str: ...

    def available(self) -> bool:
        return True


def extract_json(text: str) -> dict:
    """Pull one JSON object out of a model reply, tolerating fences and chatter."""
    text = text.strip()
    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except ValueError:
            pass
        for match in re.finditer(r"\{", candidate):
            try:
                value, _ = decoder.raw_decode(candidate[match.start():])
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("no JSON object found in model reply")


def post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:300]
        raise LLMError(f"HTTP {e.code} from {url}: {body}") from None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise LLMError(f"cannot reach {url}: {e}") from None
    except ValueError as e:
        raise LLMError(f"non-JSON response from {url}: {e}") from None
