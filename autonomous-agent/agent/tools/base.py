from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from agent.config import Config
from agent.safety import EmergencyStop, NeedsHuman, PermissionDenied, PermissionPolicy, SandboxError, StopRequested, Workspace


@dataclass
class ToolResult:
    ok: bool
    output: str = ""
    error: str = ""
    # ok | error | denied | human | sandbox | timeout | invalid | unknown_tool
    kind: str = ""
    data: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            self.kind = "ok" if self.ok else "error"


@dataclass
class ToolContext:
    workspace: Workspace
    policy: PermissionPolicy
    config: Config
    stop: EmergencyStop | None = None
    desktop: Any = None
    activity: Any = None


@dataclass
class Tool:
    name: str
    description: str
    params: dict[str, str]  # arg name -> what it means
    func: Callable[[ToolContext, dict], ToolResult]
    required: tuple[str, ...] = ()
    idempotent: bool = False  # safe to run again if it failed or failed verification
    sensitive_args: tuple[str, ...] = ()


class ToolRegistry:
    def __init__(self, ctx: ToolContext):
        self.ctx = ctx
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def missing_args(self, name: str, args: dict) -> list[str]:
        tool = self._tools[name]
        return [r for r in tool.required if args.get(r) is None]

    def describe(self) -> str:
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
            for pname, pdesc in tool.params.items():
                mark = "*" if pname in tool.required else ""
                lines.append(f"    {pname}{mark}: {pdesc}")
        return "\n".join(lines)

    def execute(self, name: str, args: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, error=f"unknown tool {name!r}; available: {', '.join(self.names())}", kind="unknown_tool")
        if not isinstance(args, dict):
            return ToolResult(False, error="args must be an object", kind="invalid")
        missing = self.missing_args(name, args)
        if missing:
            return ToolResult(False, error=f"missing required args: {', '.join(missing)}", kind="invalid")
        if self.ctx.stop:
            self.ctx.stop.check()  # StopRequested propagates to the task manager
        try:
            return tool.func(self.ctx, args)
        except StopRequested:
            raise
        except NeedsHuman as e:
            return ToolResult(False, error=str(e), kind="human")
        except PermissionDenied as e:
            return ToolResult(False, error=str(e), kind="denied")
        except SandboxError as e:
            return ToolResult(False, error=str(e), kind="sandbox")
        except Exception as e:  # a tool bug must never take the agent down
            return ToolResult(False, error=f"{type(e).__name__}: {e}", kind="error")
