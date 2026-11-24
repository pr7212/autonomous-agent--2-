from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from agent.util import secret_values

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LLM_MODES = ("local", "cloud", "hybrid")
APPROVAL_MODES = ("interactive", "deny")


def load_dotenv(path: Path) -> None:
    """Minimal .env loader. Real environment variables win."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass
class Config:
    home: Path = PROJECT_ROOT
    llm_mode: str = "hybrid"
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5-5"
    llm_timeout: float = 180.0
    max_task_seconds: int = 900
    max_actions: int = 40
    max_step_retries: int = 2
    max_replans: int = 2
    max_plan_steps: int = 12
    command_timeout: int = 60
    retry_delay: float = 0.5
    max_output_chars: int = 8000
    approval_mode: str = "interactive"
    desktop_screenshots: bool = False
    desktop_ocr: bool = False

    def __post_init__(self) -> None:
        self.home = Path(self.home).resolve()
        if self.llm_mode not in LLM_MODES:
            raise ValueError(f"llm_mode must be one of {LLM_MODES}, got {self.llm_mode!r}")
        if self.approval_mode not in APPROVAL_MODES:
            raise ValueError(f"approval_mode must be one of {APPROVAL_MODES}, got {self.approval_mode!r}")

    # --- locations -------------------------------------------------------
    @property
    def workspace_dir(self) -> Path:
        return self.home / "workspace"

    @property
    def data_dir(self) -> Path:
        return self.home / "data"

    @property
    def log_dir(self) -> Path:
        return self.home / "logs"

    @property
    def reports_dir(self) -> Path:
        return self.home / "logs" / "reports"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "agent.db"

    @property
    def stop_file(self) -> Path:
        return self.data_dir / "STOP"

    def ensure_dirs(self) -> None:
        for d in (self.workspace_dir, self.data_dir, self.log_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)

    def secrets(self) -> list[str]:
        return [self.anthropic_api_key, *secret_values()]

    @classmethod
    def from_env(cls, home: str | Path | None = None, **overrides) -> "Config":
        home_path = Path(home or os.environ.get("AGENT_HOME") or PROJECT_ROOT)
        load_dotenv(home_path / ".env")
        env = os.environ.get
        values = dict(
            home=home_path,
            llm_mode=env("AGENT_LLM_MODE", "hybrid"),
            ollama_host=env("OLLAMA_HOST", "http://localhost:11434"),
            ollama_model=env("OLLAMA_MODEL", "qwen2.5-coder:7b"),
            anthropic_api_key=env("ANTHROPIC_API_KEY", ""),
            anthropic_model=env("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            max_task_seconds=int(env("AGENT_MAX_TASK_SECONDS", "900")),
            max_actions=int(env("AGENT_MAX_ACTIONS", "40")),
            max_step_retries=int(env("AGENT_MAX_RETRIES", "2")),
            max_replans=int(env("AGENT_MAX_REPLANS", "2")),
            command_timeout=int(env("AGENT_COMMAND_TIMEOUT", "60")),
            desktop_screenshots=env("AGENT_DESKTOP_SCREENSHOTS", "0").lower() in ("1", "true", "yes"),
            desktop_ocr=env("AGENT_DESKTOP_OCR", "0").lower() in ("1", "true", "yes"),
        )
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)
