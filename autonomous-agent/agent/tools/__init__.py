from agent.tools import desktop, files, terminal
from agent.tools.base import Tool, ToolContext, ToolRegistry, ToolResult


def default_registry(ctx: ToolContext) -> ToolRegistry:
    registry = ToolRegistry(ctx)
    for tool in files.TOOLS + terminal.TOOLS + desktop.TOOLS:
        registry.register(tool)
    return registry


__all__ = ["Tool", "ToolContext", "ToolRegistry", "ToolResult", "default_registry"]
