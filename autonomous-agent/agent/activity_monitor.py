from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Callable

from agent.actionlog import ActionLog
from agent.safety import EmergencyStop
from agent.safety.sandbox import SKIP_DIRS
from agent.system_monitor import SystemMonitor, foreground_application

MAX_WORKSPACE_FILES = 5000


def _process_ids() -> set[int]:
    import psutil

    return set(psutil.pids())


class ActivityMonitor:
    """Records metadata only while an authorized task is running and monitoring is enabled."""

    def __init__(
        self,
        log: ActionLog,
        system: SystemMonitor,
        workspace: str | Path,
        stop: EmergencyStop | None = None,
        desktop=None,
        *,
        interval: float = 1.0,
        foreground: Callable[[], dict | None] = foreground_application,
        process_ids: Callable[[], set[int]] = _process_ids,
    ):
        self.log = log
        self.system = system
        self.workspace = Path(workspace).resolve()
        self.stop = stop
        self.desktop = desktop
        self.interval = max(0.25, float(interval))
        self._foreground = foreground
        self._process_ids = process_ids
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._task_id: str | None = None
        self._enabled = False
        self._current_app = ""
        self._current_pid: int | None = None
        self._previous_pid: int | None = None
        self._known_pids: set[int] = set()
        self._workspace_state: dict[str, tuple[int, int]] = {}

    def start_task(self, task_id: str) -> None:
        with self._lock:
            if self._task_id is not None:
                raise RuntimeError("activity monitor already has an active task")
            self._task_id = task_id
            self._enabled = True
            try:
                self._known_pids = self._process_ids()
            except Exception as exc:
                self._known_pids = set()
                self.log.event("monitor_error", f"process baseline unavailable ({type(exc).__name__})")
            self._workspace_state = self._scan_workspace()
            self._previous_pid = None
            self._current_pid = None
            self._current_app = ""
            self._stop_event.clear()
            self.log.event("monitoring_started", "Task-scoped activity monitoring started")
            self._start_thread_locked()

    def stop_task(self) -> None:
        with self._lock:
            active = self._task_id is not None
            enabled = self._enabled
        if active and enabled:
            self._poll_workspace()
            self.set_monitoring(False)
        with self._lock:
            self._task_id = None
            self._current_pid = None
            self._current_app = ""
            self._previous_pid = None
            if self.desktop:
                self.desktop.latest_screenshot = None

    def set_monitoring(self, enabled: bool) -> dict:
        if type(enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        thread_to_join = None
        with self._lock:
            if self._task_id is None:
                raise RuntimeError("monitoring can only be changed during an active task")
            if enabled == self._enabled:
                return self._status_locked()
            self._enabled = enabled
            if enabled:
                try:
                    self._known_pids = self._process_ids()
                except Exception as exc:
                    self._known_pids = set()
                    self.log.event("monitor_error", f"process baseline unavailable ({type(exc).__name__})")
                self._workspace_state = self._scan_workspace()
                self._previous_pid = None
                self._stop_event.clear()
                self.log.event("monitoring_started", "Task-scoped activity monitoring started")
                self._start_thread_locked()
            else:
                self._stop_event.set()
                thread_to_join = self._thread
                self._thread = None
                self._current_app = ""
                self._current_pid = None
                self._previous_pid = None
                if self.desktop:
                    self.desktop.latest_screenshot = None
                self.log.event("monitoring_stopped", "Task-scoped activity monitoring stopped")
        if thread_to_join and thread_to_join is not threading.current_thread():
            thread_to_join.join(timeout=max(2.0, self.interval + 1.0))
        return self.status()

    def record(self, kind: str, message: str, **metadata) -> None:
        with self._lock:
            if not self._task_id or not self._enabled or self.log.task_id != self._task_id:
                return
            self.log.event(kind, message, **metadata)

    def record_file_change(self, relative_path: str, *, created: bool) -> None:
        try:
            target = (self.workspace / relative_path).resolve()
            target.relative_to(self.workspace)
        except (OSError, ValueError):
            return
        try:
            stat = target.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
        except OSError:
            signature = None
        with self._lock:
            if not self._task_id or not self._enabled or self.log.task_id != self._task_id:
                return
            key = target.relative_to(self.workspace).as_posix()
            if signature is not None:
                self._workspace_state[key] = signature
            kind = "file_created" if created else "file_modified"
            self.log.event(kind, f"{kind.replace('_', ' ')}: {key}", path=key)

    def poll_once(self) -> None:
        with self._lock:
            if not self._task_id or not self._enabled:
                return
        try:
            foreground = self._foreground()
        except Exception as exc:
            self.record("monitor_error", f"foreground application sampling failed ({type(exc).__name__})")
            foreground = None

        if foreground and foreground.get("sensitive"):
            with self._lock:
                self._current_app = "Sensitive application excluded"
                self._current_pid = None
                self._previous_pid = None
        elif foreground and foreground.get("pid"):
            pid = int(foreground["pid"])
            name = str(foreground.get("name") or f"Application {pid}")[:120]
            with self._lock:
                previous = self._previous_pid
                self._current_app = name
                self._current_pid = pid
                self._previous_pid = pid
                is_new = pid not in self._known_pids
                self._known_pids.add(pid)
            if is_new:
                self.record("application_opened", f"Application opened: {name}", application=name, pid=pid)
            elif previous is not None and previous != pid:
                self.record("application_switched", f"Application switched to: {name}", application=name, pid=pid)
        else:
            with self._lock:
                self._current_app = ""
                self._current_pid = None
                self._previous_pid = None

        self._poll_workspace()

    def system_snapshot(self) -> dict | None:
        with self._lock:
            if not self._task_id or not self._enabled:
                return None
            pid = self._current_pid
        try:
            return self.system.snapshot(pid)
        except RuntimeError as exc:
            return {"error": str(exc)}

    def status(self) -> dict:
        with self._lock:
            return self._status_locked()

    def _status_locked(self) -> dict:
        return {
            "active_task": self._task_id is not None,
            "enabled": self._enabled,
            "current_app": self._current_app,
            "interval_seconds": self.interval,
        }

    def _start_thread_locked(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="agent-activity-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval):
            if self.stop and self.stop.is_set():
                with self._lock:
                    task_active = self._task_id is not None
                    enabled = self._enabled
                if task_active and enabled:
                    self.set_monitoring(False)
                return
            self.poll_once()

    def _poll_workspace(self) -> None:
        current = self._scan_workspace()
        with self._lock:
            previous = self._workspace_state
            self._workspace_state = current
        for path, signature in current.items():
            old = previous.get(path)
            if old is None:
                self.record("file_created", f"file created: {path}", path=path)
            elif old != signature:
                self.record("file_modified", f"file modified: {path}", path=path)
        if len(current) < MAX_WORKSPACE_FILES:
            for path in previous.keys() - current.keys():
                self.record("file_deleted", f"file deleted: {path}", path=path)

    def _scan_workspace(self) -> dict[str, tuple[int, int]]:
        result: dict[str, tuple[int, int]] = {}
        if not self.workspace.is_dir():
            return result
        for root, dirs, filenames in os.walk(self.workspace, followlinks=False):
            dirs[:] = sorted(
                name for name in dirs
                if name not in SKIP_DIRS and not (Path(root) / name).is_symlink()
            )
            for name in sorted(filenames):
                path = Path(root) / name
                if path.is_symlink():
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                relative = path.relative_to(self.workspace).as_posix()
                result[relative] = (stat.st_mtime_ns, stat.st_size)
                if len(result) >= MAX_WORKSPACE_FILES:
                    return result
        return result
