from __future__ import annotations

import os
import platform
import socket
from pathlib import Path

from agent.privacy import is_sensitive_application


def foreground_application() -> dict | None:
    """Return foreground process metadata, never its window title or command line."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        import psutil

        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.GetForegroundWindow.restype = wintypes.HWND
        user32.GetWindowThreadProcessId.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
        user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        user32.GetWindowTextLengthW.argtypes = (wintypes.HWND,)
        user32.GetWindowTextLengthW.restype = ctypes.c_int
        user32.GetWindowTextW.argtypes = (wintypes.HWND, wintypes.LPWSTR, ctypes.c_int)
        user32.GetWindowTextW.restype = ctypes.c_int
        handle = user32.GetForegroundWindow()
        if not handle:
            return None
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(handle, ctypes.byref(pid))
        if not pid.value:
            return None
        title_buffer = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(handle) + 1)
        user32.GetWindowTextW(handle, title_buffer, len(title_buffer))
        process_name = psutil.Process(pid.value).name()
        if is_sensitive_application(process_name, title_buffer.value):
            return {"name": "Sensitive application excluded", "pid": None, "window": None, "sensitive": True}
        return {
            "name": process_name,
            "pid": int(pid.value),
            "window": int(handle),
            "sensitive": False,
        }
    except Exception as exc:
        raise RuntimeError(f"foreground process inspection failed ({type(exc).__name__})") from None


class SystemMonitor:
    """On-demand system telemetry. Snapshots are returned to the dashboard, not persisted."""

    def __init__(self):
        try:
            import psutil
        except ImportError as exc:
            raise RuntimeError("system monitoring requires psutil; install requirements.txt") from exc
        self._psutil = psutil
        self._active_process_pid: int | None = None
        self._active_process = None

    def snapshot(self, active_pid: int | None = None) -> dict:
        psutil = self._psutil
        if active_pid is None:
            self._active_process_pid = None
            self._active_process = None
        errors: list[str] = []
        result = {
            "os": {
                "name": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
            },
            "hardware": {
                "processor": platform.processor() or "Unknown",
                "architecture": platform.machine() or "Unknown",
                "logical_cores": psutil.cpu_count(logical=True),
                "physical_cores": psutil.cpu_count(logical=False),
            },
            "cpu": None,
            "memory": None,
            "disk": None,
            "network": None,
            "application": None,
            "errors": errors,
        }
        try:
            result["cpu"] = {"usage_percent": round(float(psutil.cpu_percent(interval=None)), 1)}
        except (psutil.Error, OSError, TypeError, ValueError) as exc:
            errors.append(f"CPU metrics unavailable ({type(exc).__name__})")
        try:
            memory = psutil.virtual_memory()
            result["memory"] = {
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "available_gb": round(memory.available / (1024 ** 3), 2),
                "usage_percent": round(float(memory.percent), 1),
            }
        except (psutil.Error, OSError, AttributeError, TypeError, ValueError) as exc:
            errors.append(f"RAM metrics unavailable ({type(exc).__name__})")
        try:
            root = os.environ.get("SystemDrive") or Path.home().anchor or os.path.abspath(os.sep)
            usage = psutil.disk_usage(root)
            result["disk"] = {
                "path": root,
                "total_gb": round(usage.total / (1024 ** 3), 2),
                "free_gb": round(usage.free / (1024 ** 3), 2),
                "usage_percent": round(float(usage.percent), 1),
            }
        except (psutil.Error, OSError, AttributeError, TypeError, ValueError) as exc:
            errors.append(f"Disk metrics unavailable ({type(exc).__name__})")
        try:
            result["network"] = self._network_snapshot()
        except (psutil.Error, OSError, AttributeError, TypeError, ValueError) as exc:
            errors.append(f"Network metrics unavailable ({type(exc).__name__})")
        if active_pid is not None:
            try:
                if active_pid != self._active_process_pid:
                    self._active_process = psutil.Process(active_pid)
                    self._active_process_pid = active_pid
                process = self._active_process
                memory = process.memory_info()
                result["application"] = {
                    "name": process.name(),
                    "pid": int(active_pid),
                    "status": process.status(),
                    "cpu_percent": round(float(process.cpu_percent(interval=None)), 1),
                    "memory_mb": round(memory.rss / (1024 ** 2), 1),
                }
            except (psutil.Error, OSError, AttributeError, TypeError, ValueError) as exc:
                errors.append(f"Foreground process metrics unavailable ({type(exc).__name__})")
                if active_pid == self._active_process_pid:
                    self._active_process_pid = None
                    self._active_process = None
        return result

    def _network_snapshot(self) -> dict:
        psutil = self._psutil
        stats = psutil.net_if_stats()
        addresses = psutil.net_if_addrs()
        interfaces = []
        connected = False
        for name, info in sorted(stats.items()):
            ip_addresses = [
                address.address.split("%", 1)[0]
                for address in addresses.get(name, [])
                if address.family in (socket.AF_INET, socket.AF_INET6)
                and address.address.split("%", 1)[0] not in ("127.0.0.1", "::1")
            ]
            if info.isup and ip_addresses:
                connected = True
            interfaces.append({
                "name": name,
                "is_up": bool(info.isup),
                "speed_mbps": int(info.speed),
                "mtu": int(info.mtu),
                "addresses": ip_addresses,
            })
        return {
            "connected": connected,
            "status": "connected" if connected else "disconnected",
            "interfaces": interfaces,
        }
