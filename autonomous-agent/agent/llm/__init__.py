from agent.llm.base import LLMError, LLMProvider, extract_json
from agent.llm.claude import ClaudeProvider
from agent.llm.ollama import OllamaProvider
from agent.llm.router import Router


def make_providers(cfg):
    """(local, cloud) providers from config; either may be unavailable at run time."""
    local = OllamaProvider(cfg.ollama_host, cfg.ollama_model, cfg.llm_timeout)
    cloud = ClaudeProvider(cfg.anthropic_api_key, cfg.anthropic_model, cfg.llm_timeout)
    return local, cloud


__all__ = [
    "ClaudeProvider",
    "LLMError",
    "LLMProvider",
    "OllamaProvider",
    "Router",
    "extract_json",
    "make_providers",
]
