from __future__ import annotations

from pathlib import Path

from agent.models import Task


def _cell(text: str, limit: int = 90) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_report(task: Task) -> str:
    verified = len(task.verified_steps())
    lines = [
        f"# Task {task.id}: {task.status.value.upper()}",
        "",
        f"**Objective:** {task.objective}",
        "",
        f"- Steps verified: {verified}/{len(task.steps)}",
        f"- Actions executed: {task.actions}",
        f"- Replans: {task.replans}",
        f"- Started: {task.created_at}",
        f"- Finished: {task.updated_at}",
        "",
    ]
    if task.summary:
        lines += ["## Outcome", "", task.summary, ""]
    if task.error:
        lines += ["## Needs attention" if task.status.value == "needs_attention" else "## Problem", "", task.error, ""]
    if task.warnings:
        lines += ["## Warnings", ""] + [f"- {w}" for w in task.warnings] + [""]
    if task.steps:
        lines += ["## Steps", "", "| # | Step | Tool | Status | Verified by |", "|---|------|------|--------|-------------|"]
        for s in task.steps:
            lines.append(f"| {s.index} | {_cell(s.description)} | {s.tool} | {s.status.value} | {s.verified_by or '-'} |")
        lines.append("")
    return "\n".join(lines)


def write_report(task: Task, reports_dir: Path) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    path = reports_dir / f"{task.id}.md"
    path.write_text(build_report(task), encoding="utf-8")
    return path
