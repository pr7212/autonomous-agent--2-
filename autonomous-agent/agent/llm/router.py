from __future__ import annotations

from agent.llm.base import LLMError, LLMProvider, extract_json

# Planning and judging the whole objective need the strongest model; resolving the
# arguments of one step is cheap enough for a local model.
HEAVY_ROLES = {"plan", "replan", "final_verify"}

NUDGE = "\n\nYour last reply was not a valid JSON object with the required fields. Reply with ONLY the JSON object."


class Router:
    """Hybrid routing: local first for light work, cloud first for heavy work.
    Local-only calls never escalate to a cloud provider."""

    def __init__(self, local: LLMProvider | None, cloud: LLMProvider | None, mode: str = "hybrid"):
        self.local = local
        self.cloud = cloud
        self.mode = mode
        self.last_provider = ""

    def order(self, role: str, *, local_only: bool = False) -> list[LLMProvider]:
        if local_only or self.mode == "local":
            chain = [self.local]
        elif self.mode == "cloud":
            chain = [self.cloud]
        elif role in HEAVY_ROLES:
            chain = [self.cloud, self.local]
        else:
            chain = [self.local, self.cloud]
        return [p for p in chain if p is not None and p.available()]

    def available(self) -> bool:
        return bool(self.order("plan"))

    def complete_json(
        self, role: str, system: str, user: str, required: tuple[str, ...] = (), *, local_only: bool = False
    ) -> dict:
        providers = self.order(role, local_only=local_only)
        if not providers:
            scope = "local Ollama" if local_only else f"mode={self.mode!r}"
            raise LLMError(f"no model available for {scope} (is Ollama running / is the API key set?)")
        errors: list[str] = []
        for provider in providers:
            for attempt in range(2):
                try:
                    reply = provider.chat(system, user if attempt == 0 else user + NUDGE, json_mode=True)
                    data = extract_json(reply)
                    missing = [k for k in required if k not in data]
                    if missing:
                        raise ValueError(f"missing fields {missing}")
                except LLMError as e:
                    errors.append(f"{provider.name}: {e}")
                    break  # transport problem: no point retrying, escalate
                except ValueError as e:
                    errors.append(f"{provider.name}: {e}")
                    continue
                self.last_provider = provider.name
                return data
        raise LLMError("; ".join(errors))
