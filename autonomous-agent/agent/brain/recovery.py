from __future__ import annotations

from enum import Enum

from agent.tools import Tool


class Recovery(str, Enum):
    RETRY = "retry"    # same action again: only for idempotent tools and transient-looking failures
    REPLAN = "replan"  # ask the planner for a different approach


# Running the same thing again cannot change these outcomes.
NON_RETRYABLE = {"denied", "sandbox", "unknown_tool", "invalid", "timeout"}


def decide(tool: Tool | None, kind: str, attempts: int, max_retries: int) -> Recovery:
    """`attempts` counts tries so far. Non-idempotent tools (commands, deletes) are never blindly
    repeated: a failing test run needs a code change, not a second run."""
    if kind in NON_RETRYABLE:
        return Recovery.REPLAN
    if tool is not None and tool.idempotent and attempts <= max_retries:
        return Recovery.RETRY
    return Recovery.REPLAN
