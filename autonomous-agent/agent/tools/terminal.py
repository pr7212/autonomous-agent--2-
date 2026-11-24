from __future__ import annotations

import importlib.util
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time

from agent.safety import (
    Action,
    SandboxError,
    StopRequested,
    classify_command,
    has_shell_operators,
)
from agent.safety.permissions import is_python
from agent.tools.base import Tool, ToolContext, ToolResult
from agent.util import is_secret_name, truncate

# -E: ignore PYTHON* variables, -s: no user site-packages, utf8: stable encoding on Windows.
PY_FLAGS = ["-E", "-s", "-X", "utf8"]
MAX_TIMEOUT = 600


def clean_env() -> dict:
    """Child processes never inherit credentials from the agent's environment."""
    env = {k: v for k, v in os.environ.items() if not is_secret_name(k)}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _limits(timeout: int):
    def apply() -> None:  # POSIX only: cap CPU time and the size of any single file written
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (timeout + 5, timeout + 5))
        resource.setrlimit(resource.RLIMIT_FSIZE, (50_000_000, 50_000_000))
    return apply


def _kill(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _execute(ctx: ToolContext, argv: list[str], timeout: int) -> ToolResult:
    kwargs: dict = dict(
        cwd=str(ctx.workspace.root), env=clean_env(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace",
    )
    if os.name == "posix":
        kwargs["start_new_session"] = True
        kwargs["preexec_fn"] = _limits(timeout)
    else:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    try:
        proc = subprocess.Popen(argv, **kwargs)
    except FileNotFoundError:
        return ToolResult(False, error=f"command not found: {argv[0]}")
    except OSError as e:
        return ToolResult(False, error=f"could not start {argv[0]}: {e}")

    deadline = time.monotonic() + timeout
    timed_out = False
    while True:
        try:
            out, _ = proc.communicate(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            if ctx.stop and ctx.stop.is_set():
                _kill(proc)
                proc.communicate()
                raise StopRequested(ctx.stop.reason) from None
            if time.monotonic() > deadline:
                _kill(proc)
                out, _ = proc.communicate()
                timed_out = True
                break

    out = truncate(out or "", ctx.config.max_output_chars)
    if timed_out:
        return ToolResult(False, output=out, error=f"timed out after {timeout}s\n{truncate(out, 1500)}", kind="timeout")
    rc = proc.returncode
    return ToolResult(
        rc == 0,
        output=f"exit code {rc}\n{out}".rstrip(),
        error="" if rc == 0 else f"exit code {rc}\n{truncate(out, 1500)}",
        data={"returncode": rc},
    )


def _python_argv(ctx: ToolContext, rest: list[str]) -> list[str]:
    """Run python under the interpreter the agent itself uses; a script must live in the workspace."""
    i = 0
    while i < len(rest) and rest[i].startswith("-"):
        if rest[i] in ("-m", "-c"):
            return [sys.executable, *PY_FLAGS, *rest]
        i += 1
    if i < len(rest):
        ctx.workspace.resolve(rest[i])  # raises SandboxError if the script is outside
    return [sys.executable, *PY_FLAGS, *rest]


def _timeout(ctx: ToolContext, args: dict) -> int:
    try:
        value = int(args.get("timeout") or ctx.config.command_timeout)
    except (TypeError, ValueError):
        value = ctx.config.command_timeout
    return max(1, min(value, MAX_TIMEOUT))


def _run_command(ctx: ToolContext, args: dict) -> ToolResult:
    command = args["command"]
    if not isinstance(command, str) or not command.strip():
        return ToolResult(False, error="command must be a non-empty string", kind="invalid")
    if has_shell_operators(command):
        return ToolResult(
            False, kind="invalid",
            error="shell operators (; & | > < `) are not supported because no shell is used. "
                  "Run one program per call, or put the logic in a script file.",
        )
    try:
        argv = shlex.split(command, posix=(os.name != "nt"))
    except ValueError as e:
        return ToolResult(False, error=f"cannot parse command: {e}", kind="invalid")
    argv = [a[1:-1] if len(a) > 1 and a[0] == a[-1] and a[0] in "\"'" else a for a in argv]
    if not argv:
        return ToolResult(False, error="empty command", kind="invalid")

    action, why = classify_command(argv)
    ctx.policy.require(action, f"{command}  ({why})")

    if is_python(argv[0]):
        argv = _python_argv(ctx, argv[1:])
    else:
        resolved = shutil.which(argv[0])
        if resolved is None:
            return ToolResult(False, error=f"command not found: {argv[0]}")
        argv[0] = resolved
    return _execute(ctx, argv, _timeout(ctx, args))


def _run_tests(ctx: ToolContext, args: dict) -> ToolResult:
    target = args.get("target") or "."
    path = ctx.workspace.resolve(target)
    ctx.policy.require(Action.RUN_TESTS, str(target))
    rel = ctx.workspace.rel(path)
    if importlib.util.find_spec("pytest") is not None:
        argv = [sys.executable, *PY_FLAGS, "-m", "pytest", "-q", "-p", "no:cacheprovider", rel]
    elif path.is_dir():
        argv = [sys.executable, *PY_FLAGS, "-m", "unittest", "discover", "-s", rel]
    else:
        argv = [sys.executable, *PY_FLAGS, "-m", "unittest", rel]
    return _execute(ctx, argv, _timeout(ctx, args))


TOOLS = [
    Tool("run_command",
         "Run ONE program in the workspace (no shell, no pipes). `python script.py` runs inside the workspace; "
         "other programs need the person's approval; shells and destructive commands are blocked.",
         {"command": "program and arguments, e.g. 'python app.py'", "timeout": "seconds, optional"},
         _run_command, required=("command",)),
    Tool("run_tests", "Run the project's tests (pytest if installed, otherwise unittest).",
         {"target": "test file or folder relative to the workspace (default '.')", "timeout": "seconds, optional"},
         _run_tests),
]
