from __future__ import annotations

import os
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", ".pytest_cache"}


class SandboxError(Exception):
    """A path tried to leave the approved workspace."""


class Workspace:
    """All file access goes through here. Symlinks and '..' are resolved first,
    so a path is judged by where it actually lands, not by how it is spelled."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, path: str | Path) -> Path:
        text = str(path)
        if not text.strip() or "\x00" in text:
            raise SandboxError("empty or invalid path")
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = self.root / candidate
        resolved = candidate.resolve()
        try:
            resolved.relative_to(self.root)
        except ValueError:
            raise SandboxError(f"path escapes the workspace: {text}") from None
        return resolved

    def rel(self, path: str | Path) -> str:
        rel = Path(path).resolve().relative_to(self.root).as_posix()
        return rel or "."

    def snapshot(self, limit: int = 60, depth: int = 2) -> list[str]:
        """Relative file listing (shallow) for planner and verifier context."""
        out: list[str] = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            rel = Path(dirpath).relative_to(self.root)
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            if len(rel.parts) >= depth:
                dirnames[:] = []
            for name in sorted(filenames):
                out.append((rel / name).as_posix())
            if len(out) >= limit:
                break
        return out[:limit]
