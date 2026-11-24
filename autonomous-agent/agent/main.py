from __future__ import annotations

import argparse
import signal
import sys
import webbrowser

from agent import __version__
from agent.bootstrap import build_agent
from agent.config import APPROVAL_MODES, LLM_MODES, Config
from agent.dashboard import DashboardServer
from agent.llm import OllamaProvider
from agent.models import TaskStatus
from agent.reporting import build_report
from agent.safety import deny_approver, interactive_approver

EXIT_CODES = {
    TaskStatus.COMPLETED: 0,
    TaskStatus.FAILED: 1,
    TaskStatus.NEEDS_ATTENTION: 2,
    TaskStatus.TIMED_OUT: 3,
    TaskStatus.CANCELLED: 130,
}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent", description="Autonomous agent with Windows desktop controls, verification, safety, and memory.")
    p.add_argument("--home", help="project home (workspace/, data/, logs/ live here); default: AGENT_HOME or the project folder")
    p.add_argument("--version", action="version", version=f"agent {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="give the agent an objective")
    run.add_argument("objective", nargs="+")
    run.add_argument("--mode", choices=LLM_MODES, help="local | cloud | hybrid (default from AGENT_LLM_MODE)")
    run.add_argument("--approval", choices=APPROVAL_MODES, help="interactive prompts you; deny refuses anything that needs approval")
    run.add_argument("--max-seconds", type=int, help="hard time limit for this task")
    run.add_argument("--dashboard", action="store_true", help="serve the local live activity dashboard")
    run.add_argument("--dashboard-port", type=int, default=8765, help="dashboard port (default 8765)")

    resume = sub.add_parser("resume", help="continue an interrupted, cancelled or failed task")
    resume.add_argument("task_id")
    resume.add_argument("--approval", choices=APPROVAL_MODES)
    resume.add_argument("--dashboard", action="store_true", help="serve the local live activity dashboard")
    resume.add_argument("--dashboard-port", type=int, default=8765, help="dashboard port (default 8765)")

    hist = sub.add_parser("history", help="list recent tasks")
    hist.add_argument("-n", type=int, default=10)

    show = sub.add_parser("show", help="show a task's report and event log")
    show.add_argument("task_id")

    sub.add_parser("stop", help="emergency stop: ask a running agent to halt at the next safe point")
    sub.add_parser("doctor", help="check the environment")
    return p


def _approver(mode: str):
    if mode == "interactive" and sys.stdin.isatty():
        return interactive_approver
    return deny_approver


def _install_sigint(agent) -> None:
    def handler(signum, frame):
        if agent.stop.is_set():
            raise KeyboardInterrupt  # second Ctrl-C: leave now
        agent.stop.trigger("Ctrl-C")
        print("\nStopping at the next safe point (Ctrl-C again to force quit)...")

    signal.signal(signal.SIGINT, handler)


def _finish(agent, task) -> int:
    print(f"\n{task.status.value.upper()}: {task.summary or task.error}")
    if task.report:
        print(f"Report: {task.report}")
    return EXIT_CODES.get(task.status, 1)


def _dashboard(agent, enabled: bool, port: int):
    if not enabled:
        return None
    server = DashboardServer(
        agent.history,
        agent.stop,
        agent.ctx.desktop,
        agent.cfg.secrets(),
        port=port,
        activity=agent.activity,
    )
    try:
        url = server.start()
        print(f"Dashboard: {url}")
        try:
            webbrowser.open(url)
        except webbrowser.Error as exc:
            print(f"Open the dashboard manually: {exc}")
    except OSError as exc:
        server.close()
        print(f"Could not start dashboard: {exc}")
        return False
    return server


def cmd_run(args) -> int:
    cfg = Config.from_env(args.home, llm_mode=args.mode, approval_mode=args.approval, max_task_seconds=args.max_seconds)
    agent = build_agent(cfg, approver=_approver(cfg.approval_mode), sink=print)
    dashboard = None
    try:
        if not agent.router.available():
            print("No model available. Start Ollama (and `ollama pull` the model), or set ANTHROPIC_API_KEY. Try: python -m agent doctor")
            return 1
        agent.stop.clear()
        dashboard = _dashboard(agent, args.dashboard, args.dashboard_port)
        if dashboard is False:
            return 1
        _install_sigint(agent)
        task = agent.manager.create_task(" ".join(args.objective))
        print(f"Task {task.id} (mode={cfg.llm_mode})")
        return _finish(agent, agent.manager.run(task))
    finally:
        if dashboard:
            dashboard.close()
        agent.close()


def cmd_resume(args) -> int:
    cfg = Config.from_env(args.home, approval_mode=args.approval)
    agent = build_agent(cfg, approver=_approver(cfg.approval_mode), sink=print)
    dashboard = None
    try:
        found = agent.history.find(args.task_id)
        if found is None:
            print(f"No task matching {args.task_id!r}")
            return 1
        agent.stop.clear()
        dashboard = _dashboard(agent, args.dashboard, args.dashboard_port)
        if dashboard is False:
            return 1
        _install_sigint(agent)
        return _finish(agent, agent.manager.resume(found.id))
    finally:
        if dashboard:
            dashboard.close()
        agent.close()


def cmd_history(args) -> int:
    agent = build_agent(Config.from_env(args.home))
    try:
        tasks = agent.history.list(args.n)
        if not tasks:
            print("No tasks yet.")
        for t in tasks:
            print(f"{t.id}  {t.status.value:<16} {len(t.verified_steps())}/{len(t.steps)} steps  {t.objective[:70]}")
        return 0
    finally:
        agent.close()


def cmd_show(args) -> int:
    agent = build_agent(Config.from_env(args.home))
    try:
        task = agent.history.find(args.task_id)
        if task is None:
            print(f"No task matching {args.task_id!r}")
            return 1
        print(build_report(task))
        print("## Events\n")
        for e in agent.history.events(task.id):
            print(f"{e['ts']}  {e['kind']:<18} {e['message'][:160]}")
        return 0
    finally:
        agent.close()


def cmd_stop(args) -> int:
    cfg = Config.from_env(args.home)
    cfg.ensure_dirs()
    cfg.stop_file.write_text("stop", encoding="utf-8")
    print(f"Stop requested ({cfg.stop_file}). A running agent halts at its next safe point; the next `run` clears the flag.")
    return 0


def cmd_doctor(args) -> int:
    cfg = Config.from_env(args.home)
    cfg.ensure_dirs()
    ok = True

    def line(good: bool, text: str, fatal: bool = True) -> None:
        nonlocal ok
        print(f"  [{'ok' if good else '!!'}] {text}")
        if not good and fatal:
            ok = False

    print(f"agent {__version__}  home={cfg.home}")
    line(sys.version_info >= (3, 10), f"Python {sys.version.split()[0]} (3.10+ required, 3.12+ recommended)")
    try:
        probe = cfg.workspace_dir / ".write_test"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        line(True, f"workspace writable: {cfg.workspace_dir}")
    except OSError as e:
        line(False, f"workspace not writable: {e}")
    ollama = OllamaProvider(cfg.ollama_host, cfg.ollama_model)
    models = ollama.models()
    local_ok = cfg.ollama_model in models or any(m.startswith(cfg.ollama_model.split(":")[0] + ":") and cfg.ollama_model in m for m in models)
    line(bool(models), f"Ollama reachable at {cfg.ollama_host} ({len(models)} models)", fatal=False)
    if models:
        line(local_ok, f"model {cfg.ollama_model} installed" + ("" if local_ok else f"  -> run: ollama pull {cfg.ollama_model}"), fatal=False)
    line(bool(cfg.anthropic_api_key), "ANTHROPIC_API_KEY set", fatal=False)
    usable = (bool(models) and cfg.llm_mode in ("local", "hybrid")) or (bool(cfg.anthropic_api_key) and cfg.llm_mode in ("cloud", "hybrid"))
    line(usable, f"at least one model usable for mode={cfg.llm_mode}")
    print("\nReady." if ok else "\nFix the [!!] items above.")
    return 0 if ok else 1


COMMANDS = {
    "run": cmd_run, "resume": cmd_resume, "history": cmd_history,
    "show": cmd_show, "stop": cmd_stop, "doctor": cmd_doctor,
}


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = _parser().parse_args(argv)
    try:
        return COMMANDS[args.command](args)
    except KeyboardInterrupt:
        print("\nAborted.")
        return 130
    except ValueError as e:
        print(f"error: {e}")
        return 1
