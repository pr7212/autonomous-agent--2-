# Autonomous Agent

Observe → plan → act → **verify** → recover → report. The task engine uses the standard library and
`psutil` for live system telemetry.

Windows desktop tools now plug into the existing agent registry and task loop. The engine discovers
controls with Windows UI Automation, can optionally use OCR/screenshots, and moves/types/clicks through
PyAutoGUI. Ollama and the existing model router provide planning; verified work and events use the same
SQLite history and checkpoints as other tools.

## Run it (Windows, VS Code terminal)

```powershell
cd autonomous-agent
copy .env.example .env          # then edit it
python -m agent doctor          # checks Python, workspace, Ollama, API key
python -m agent run "Create hello.py that prints a greeting, then run it"
```

Hybrid mode needs at least one of:

- **Local:** install Ollama, then `ollama pull qwen2.5-coder:7b`
- **Cloud:** set `ANTHROPIC_API_KEY` in `.env`

Commands: `run`, `resume <id>`, `history`, `show <id>`, `stop`, `doctor`.
`stop` creates a STOP file, so you can kill a running task from a second terminal. Ctrl-C also stops
gracefully; press it twice to force quit.

## Windows desktop control

Install the optional desktop packages in the same Python environment as the agent:

```powershell
python -m pip install -r requirements.txt -r requirements-desktop.txt
```

For OCR, install the Tesseract executable separately and make it available on `PATH`. Desktop tools
are registered in the same tool list as file and terminal tools. Before each action, the agent must
observe the current foreground app. Actions are restricted to that window, move the pointer smoothly,
and refresh UI Automation state after acting. Every observation and control operation requests explicit
approval; use `--approval interactive` in a terminal.

Launch the GUI dashboard and a task from Explorer or a terminal:

```powershell
.\start-agent.bat "Use the open application to complete the authorized task"
```

The task-scoped Activity Monitor starts only when a task runs. It records foreground process switches,
authorized mouse/shortcut metadata, workspace file changes, errors, and recovery events; it never
records raw typed text, window titles, or message contents. Known mail/chat applications are excluded
from desktop interaction and monitoring. The dashboard can pause/resume monitoring; system/network
snapshots are shown live and are not written to task history.

Tasks that use desktop controls are planned, resolved, replanned, and audited locally through Ollama;
the router will not send screen-derived context to a cloud fallback. Ensure Ollama is available for
desktop tasks, including when `--mode hybrid` is selected.

Editable values are not read by UI Automation. Password fields cannot be typed into; password,
editable, and combo-box regions are masked in screenshots. Sensitive controls are blocked, and app submissions,
assessments, and other consequential actions remain outside the agent's remit. Type-text arguments are
redacted in action logs, replanning context, and persisted task checkpoints.

Screen capture and OCR are **off by default**. Enable only for a trusted foreground application:

```powershell
$env:AGENT_DESKTOP_SCREENSHOTS = "1" # redacted active-window image, memory only
$env:AGENT_DESKTOP_OCR = "1"         # OCR text is sent to the model and stored in task evidence
python -m agent run --dashboard "Use the open application to ..."
```

The optional dashboard is served only on loopback (default `http://127.0.0.1:8765/`) and opens in the
default browser. It shows task progress, current app, system resources, network interfaces, a timeline,
live SQLite events, and the latest redacted in-memory screenshot. It has separate cooperative-stop,
emergency-stop, and monitoring controls. Use `--dashboard-port` to select another port.

## The loop (agent/brain/task_manager.py)

1. **Plan**: the planner returns steps, each with a tool, args, and a deterministic `verify` check.
2. **Act**: a step whose args depend on earlier results is `deferred`; the model decides its args at run time from what earlier steps observed.
3. **Verify**: a step is `VERIFIED` only with independent evidence: write_file is re-read and hashed, deletes are confirmed gone, files are checked to exist or contain text, exit codes and output are inspected.
4. **Recover**: idempotent tools are retried in place; anything else (a failing test run, a denied action) goes to a **replan**, which keeps verified work and replaces the rest.
5. **Audit**: when all steps are verified, a model checks the *objective* against the evidence. If it finds a gap, that gap triggers another replan.
6. **Report**: `logs/reports/<id>.md`, plus a full event trail in SQLite and `logs/<id>.jsonl`.

A task is `COMPLETED` only after steps 3 and 5. If no model is reachable for the audit, it still completes,
but the report carries a warning that only deterministic checks ran.

## Hybrid routing (agent/llm/router.py)

`hybrid`: ordinary planning, replanning and the final audit go to the cloud model first; run-time arg
decisions go to the local model first. Desktop objectives and all screen-derived context remain local
and never escalate to a cloud provider. `local` and `cloud` pin a provider for non-desktop work.

## Safety (agent/safety/)

| Action | Policy |
|---|---|
| Read files, write in workspace, run tests, run workspace Python | Automatic |
| Delete files, unlisted programs, send messages, purchases, security changes | Asks you (denied automatically when not at a terminal) |
| Desktop observation and control | Asks you for each operation |
| Submit applications or assessments | Never done by the agent: task stops as `needs_attention` |
| Shells (`bash`, `cmd`, `powershell`), `rm`/`del`/`format`... | Blocked |

Also: all paths resolve through a sandbox (`..` and symlinks included); commands run without a shell;
child processes get a copy of the environment with secrets stripped; logs redact secrets; limits on task
time (900 s), total actions (40), retries (2) and replans (2); emergency stop.

**Be honest about the sandbox.** Python run by the agent is contained by working directory, interpreter
flags, a scrubbed environment, a timeout and (on Linux/macOS) CPU/file-size limits. That is *process
hygiene, not a security boundary*: a malicious script can still touch your files. For untrusted tasks, run
the whole agent inside Windows Sandbox, a VM or a container.

## Layout

```
agent/brain/      planner, verifier, recovery policy, task manager
agent/tools/      registry + file, terminal, and Windows desktop tools
agent/llm/        Ollama + Claude providers, hybrid router
agent/safety/     sandbox, permission policy, emergency stop
agent/memory/     SQLite database, task history, checkpoints
agent/system_monitor.py  on-demand OS, hardware, process, and network metrics
agent/activity_monitor.py  task-scoped activity events and monitoring controls
agent/dashboard.py  local live activity dashboard
agent/main.py     CLI        agent/bootstrap.py  wiring
tests/            unittest suite; scripted models plus a fake Ollama server
```

Run tests: `python -m unittest discover -s tests -t .`

## Adding a tool (Phase 2+)

```python
from agent.tools import Tool, ToolResult

def my_tool(ctx, args):
    ctx.policy.require(Action.NAVIGATE, args["url"])   # ask the policy first
    ...
    return ToolResult(True, "output", data={"k": "v"})

registry.register(Tool("my_tool", "what it does", {"url": "..."}, my_tool,
                       required=("url",), idempotent=True))
```

Set `idempotent=True` only if running it twice is harmless. Add a branch to `Verifier._builtin` if the tool
can be checked independently.

## Roadmap

| Phase | Status |
|---|---|
| 1 Planner, task manager, verifier, safety, memory | **done** |
| 2 Windows desktop control, UI Automation, optional OCR/screenshots | **implemented** |
| 3 Browser (Playwright) + official API tools | |
| 4 Richer code/debug loop | |
| 5 Scheduling, smarter memory | |
| 6 Dashboard and task-scoped system monitoring | **implemented** |
