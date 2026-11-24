from __future__ import annotations

import hashlib
import shutil

from agent.safety import Action
from agent.tools.base import Tool, ToolContext, ToolResult
from agent.util import truncate

MAX_READ_BYTES = 1_000_000
MAX_LIST_ENTRIES = 200


def _read_file(ctx: ToolContext, args: dict) -> ToolResult:
    ctx.policy.require(Action.READ_FILE, str(args["path"]))
    p = ctx.workspace.resolve(args["path"])
    if not p.exists():
        return ToolResult(False, error=f"not found: {args['path']}")
    if p.is_dir():
        return ToolResult(False, error=f"{args['path']} is a directory; use list_dir")
    size = p.stat().st_size
    if size > MAX_READ_BYTES:
        return ToolResult(False, error=f"file too large to read ({size} bytes)", kind="invalid")
    text = p.read_text(encoding="utf-8", errors="replace")
    return ToolResult(True, truncate(text, ctx.config.max_output_chars), data={"path": ctx.workspace.rel(p), "bytes": size})


def _write_file(ctx: ToolContext, args: dict) -> ToolResult:
    content = args["content"]
    if not isinstance(content, str):
        return ToolResult(False, error="content must be a string", kind="invalid")
    ctx.policy.require(Action.WRITE_FILE, str(args["path"]))
    p = ctx.workspace.resolve(args["path"])
    if p == ctx.workspace.root or p.is_dir():
        return ToolResult(False, error=f"{args['path']} is a directory", kind="invalid")
    created = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = content.encode("utf-8")
    p.write_bytes(payload)  # bytes, so Windows does not rewrite newlines behind our back
    rel = ctx.workspace.rel(p)
    if ctx.activity:
        ctx.activity.record_file_change(rel, created=created)
    return ToolResult(
        True, f"wrote {len(payload)} bytes to {rel}",
        data={"path": rel, "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)},
    )


def _list_dir(ctx: ToolContext, args: dict) -> ToolResult:
    path = args.get("path") or "."
    ctx.policy.require(Action.LIST_DIR, str(path))
    p = ctx.workspace.resolve(path)
    if not p.is_dir():
        return ToolResult(False, error=f"not a directory: {path}")
    entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    names = [e.name + ("/" if e.is_dir() else "") for e in entries[:MAX_LIST_ENTRIES]]
    more = f"\n... and {len(entries) - MAX_LIST_ENTRIES} more" if len(entries) > MAX_LIST_ENTRIES else ""
    return ToolResult(True, ("\n".join(names) or "(empty)") + more, data={"path": ctx.workspace.rel(p), "count": len(entries)})


def _make_dir(ctx: ToolContext, args: dict) -> ToolResult:
    ctx.policy.require(Action.WRITE_FILE, str(args["path"]))
    p = ctx.workspace.resolve(args["path"])
    p.mkdir(parents=True, exist_ok=True)
    return ToolResult(True, f"directory ready: {ctx.workspace.rel(p)}", data={"path": ctx.workspace.rel(p), "is_dir": True})


def _delete_file(ctx: ToolContext, args: dict) -> ToolResult:
    p = ctx.workspace.resolve(args["path"])
    if p == ctx.workspace.root:
        return ToolResult(False, error="refusing to delete the workspace root", kind="invalid")
    if not p.exists():
        return ToolResult(False, error=f"not found: {args['path']}")
    rel = ctx.workspace.rel(p)
    detail = f"delete {rel}"
    is_directory = p.is_dir()
    if is_directory:
        detail += f" (directory with {sum(1 for _ in p.rglob('*'))} items)"
    ctx.policy.require(Action.DELETE_FILE, detail)
    if is_directory:
        shutil.rmtree(p)
    else:
        p.unlink()
    if ctx.activity and not is_directory:
        ctx.activity.record("file_deleted", f"file deleted: {rel}", path=rel)
    return ToolResult(True, f"deleted {rel}", data={"path": rel, "deleted": True})


TOOLS = [
    Tool("read_file", "Read a UTF-8 text file from the workspace.",
         {"path": "file path relative to the workspace"}, _read_file, required=("path",), idempotent=True),
    Tool("write_file", "Create or overwrite a text file in the workspace (parent folders are created).",
         {"path": "file path relative to the workspace", "content": "full text content of the file"},
         _write_file, required=("path", "content"), idempotent=True),
    Tool("list_dir", "List a folder in the workspace.",
         {"path": "folder relative to the workspace (default '.')"}, _list_dir, idempotent=True),
    Tool("make_dir", "Create a folder in the workspace.",
         {"path": "folder relative to the workspace"}, _make_dir, required=("path",), idempotent=True),
    Tool("delete_file", "Delete a file or folder in the workspace. Requires the person's approval.",
         {"path": "path relative to the workspace"}, _delete_file, required=("path",)),
]
