from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from agent.activity_monitor import ActivityMonitor
from agent.actionlog import ActionLog
from agent.brain import Planner, TaskManager, Verifier
from agent.config import Config
from agent.llm import Router, make_providers
from agent.memory import Checkpoints, Database, TaskHistory
from agent.safety import EmergencyStop, PermissionPolicy, Workspace, deny_approver
from agent.safety.permissions import Approver
from agent.system_monitor import SystemMonitor
from agent.tools import ToolContext, ToolRegistry, default_registry
from agent.tools.desktop import DesktopRuntime


@dataclass
class Agent:
    cfg: Config
    db: Database
    history: TaskHistory
    checkpoints: Checkpoints
    log: ActionLog
    stop: EmergencyStop
    activity: ActivityMonitor
    ctx: ToolContext
    registry: ToolRegistry
    router: Router
    planner: Planner
    verifier: Verifier
    manager: TaskManager

    def close(self) -> None:
        self.db.close()


def build_agent(
    cfg: Config,
    approver: Approver | None = None,
    sink: Callable[[str], None] | None = None,
    providers: tuple | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Agent:
    """Wire every layer together. `providers=(local, cloud)` lets tests inject scripted models."""
    cfg.ensure_dirs()
    db = Database(cfg.db_path)
    history = TaskHistory(db)
    checkpoints = Checkpoints(db)
    log = ActionLog(history, cfg.log_dir, cfg.secrets(), sink)
    stop = EmergencyStop(cfg.stop_file)
    policy = PermissionPolicy(approver or deny_approver, audit=log.event)
    workspace = Workspace(cfg.workspace_dir)
    system = SystemMonitor()
    desktop = DesktopRuntime(cfg)
    activity = ActivityMonitor(log, system, workspace.root, stop, desktop)
    ctx = ToolContext(workspace, policy, cfg, stop, desktop, activity)
    registry = default_registry(ctx)
    local, cloud = providers if providers is not None else make_providers(cfg)
    router = Router(local, cloud, cfg.llm_mode)
    planner = Planner(router, registry, workspace, cfg)
    verifier = Verifier(ctx, router)
    manager = TaskManager(
        cfg, registry, planner, verifier, history, checkpoints, log, stop, clock, sleep,
        activity=activity,
    )
    return Agent(cfg, db, history, checkpoints, log, stop, activity, ctx, registry, router, planner, verifier, manager)
