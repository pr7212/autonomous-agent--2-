from __future__ import annotations

import io
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from agent.privacy import is_sensitive_application
from agent.safety import Action
from agent.tools.base import Tool, ToolContext, ToolResult
from agent.util import truncate

MAX_CONTROLS = 100
OBSERVATION_TTL = 12.0
BLOCKED_CONTROL_WORDS = (
    "password", "passcode", "submit", "application", "assessment", "exam",
    "purchase", "checkout", "payment", "pay now", "send", "delete", "security",
)
SAFE_KEYS = {
    "enter", "tab", "esc", "backspace", "delete", "left", "right", "up", "down",
    "home", "end", "pageup", "pagedown", "space", "f1", "f2", "f3", "f4",
    "f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12",
}
SAFE_SHORTCUT_KEYS = SAFE_KEYS | {"a", "f", "s", "y", "z"}


@dataclass
class DesktopRuntime:
    """Transient desktop observation state; screenshots are never written to disk."""

    screenshot_enabled: bool = False
    ocr_enabled: bool = False
    current_app: str = ""
    current_pid: int | None = None
    latest_screenshot: bytes | None = None
    last_observed_at: float = 0.0
    last_window_handle: int | None = None
    _window_rect: tuple[int, int, int, int] | None = field(default=None, repr=False)

    def __init__(self, config=None):
        self.screenshot_enabled = bool(config and config.desktop_screenshots)
        self.ocr_enabled = bool(config and config.desktop_ocr)
        self.current_app = ""
        self.current_pid = None
        self.latest_screenshot = None
        self.last_observed_at = 0.0
        self.last_window_handle = None
        self._window_rect = None

    def observe(self, expected_handle: int | None = None) -> dict:
        if os.name != "nt":
            raise RuntimeError("desktop automation is supported only on Windows")
        try:
            from pywinauto import Desktop
        except ImportError as exc:
            raise RuntimeError("Windows UI Automation requires pywinauto; install the desktop extras") from exc

        window = Desktop(backend="uia").get_active()
        if window is None or not window.is_visible():
            raise RuntimeError("no visible foreground application window")

        handle = int(window.handle)
        if expected_handle is not None and handle != expected_handle:
            raise RuntimeError("foreground application changed during approval; no screen data was captured")
        process_id = int(window.process_id())
        process_name = _process_name(process_id)
        window_title = str(window.window_text() or "")
        if is_sensitive_application(process_name, window_title):
            raise SensitiveApplicationError("desktop interaction and monitoring are disabled for this sensitive application")
        controls = []
        sensitive_rects = []
        control_errors = 0
        error_types: set[str] = set()
        try:
            descendants = window.descendants()
        except Exception as exc:
            raise RuntimeError(f"could not inspect the active window: {exc}") from exc

        for control in descendants[:MAX_CONTROLS * 4]:
            try:
                control_type = str(control.element_info.control_type or "")
                is_password = _is_password_control(control)
                if control_type not in {
                    "Button", "Edit", "ComboBox", "CheckBox", "RadioButton",
                    "MenuItem", "TabItem",
                } and not is_password:
                    continue
                rect_obj = control.rectangle()
                rect = (int(rect_obj.left), int(rect_obj.top), int(rect_obj.right), int(rect_obj.bottom))
                if rect[2] <= rect[0] or rect[3] <= rect[1]:
                    continue
                if is_password or control_type in ("Edit", "ComboBox"):
                    sensitive_rects.append(rect)
                if control_type not in {
                    "Button", "Edit", "ComboBox", "CheckBox", "RadioButton",
                    "MenuItem", "TabItem",
                }:
                    continue
                # Do not expose values from editable controls or password controls.
                name = (
                    "" if is_password or control_type in ("Edit", "ComboBox")
                    else str(control.window_text() or "")[:120]
                )
                controls.append({
                    "type": control_type,
                    "name": name,
                    "automation_id": str(control.element_info.automation_id or "")[:100],
                    "enabled": bool(control.is_enabled()),
                    "rect": rect,
                    "password": is_password,
                })
            except Exception as exc:
                control_errors += 1
                if len(error_types) < 5:
                    error_types.add(type(exc).__name__)
                continue

        rect_obj = window.rectangle()
        bounds = (int(rect_obj.left), int(rect_obj.top), int(rect_obj.right), int(rect_obj.bottom))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise RuntimeError("active window has invalid screen bounds")

        self.current_app = process_name or f"Application (PID {process_id})"
        self.current_pid = process_id
        self.last_window_handle = handle
        self._window_rect = bounds
        self.last_observed_at = time.monotonic()
        self.latest_screenshot = None
        snapshot = {
            "application": self.current_app,
            "window_handle": handle,
            "controls": controls[:MAX_CONTROLS],
            "control_count": len(controls),
            "screenshot_available": False,
            "ocr_text": "",
            "warnings": [],
        }
        if self.screenshot_enabled or self.ocr_enabled:
            image_bytes = self._capture(bounds, sensitive_rects)
            if self.screenshot_enabled:
                self.latest_screenshot = image_bytes
                snapshot["screenshot_available"] = True
        if self.ocr_enabled:
            snapshot["ocr_text"] = self._ocr(image_bytes)
        if control_errors:
            snapshot["warnings"].append(
                f"{control_errors} UIA control(s) could not be inspected ({', '.join(sorted(error_types))})."
            )
        return snapshot

    def _capture(self, bounds: tuple[int, int, int, int], sensitive_rects: list[tuple[int, int, int, int]]) -> bytes:
        try:
            import pyautogui
            from PIL import ImageDraw
        except ImportError as exc:
            raise RuntimeError("screenshots require pyautogui and Pillow; install the desktop extras") from exc

        left, top, right, bottom = bounds
        image = pyautogui.screenshot(region=(left, top, right - left, bottom - top))
        draw = ImageDraw.Draw(image)
        for l, t, r, b in sensitive_rects:
            draw.rectangle((max(0, l - left), max(0, t - top), min(image.width, r - left), min(image.height, b - top)), fill="black")
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    def _ocr(self, image_bytes: bytes) -> str:
        try:
            import pytesseract
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("OCR requires pytesseract and Pillow; install the desktop extras") from exc
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return truncate(pytesseract.image_to_string(image, timeout=5), 3000)
        except Exception as exc:
            raise RuntimeError(f"OCR failed: {exc}") from exc

    def require_fresh_observation(self) -> None:
        if time.monotonic() - self.last_observed_at > OBSERVATION_TTL:
            raise RuntimeError("the screen observation is stale; call observe_desktop before acting")
        try:
            import pyautogui
        except ImportError as exc:
            raise RuntimeError("desktop control requires pyautogui; install the desktop extras") from exc
        if self.last_window_handle is None:
            raise RuntimeError("observe_desktop must run before desktop actions")
        try:
            from pywinauto import Desktop
            active = Desktop(backend="uia").get_active()
            if active is None or int(active.handle) != self.last_window_handle:
                raise RuntimeError("the foreground application changed; call observe_desktop before acting")
            if time.monotonic() - self.last_observed_at > OBSERVATION_TTL:
                raise RuntimeError("the screen observation became stale; call observe_desktop before acting")
        except ImportError as exc:
            raise RuntimeError("Windows UI Automation requires pywinauto; install the desktop extras") from exc

    @staticmethod
    def focused_control(expected_handle: int):
        try:
            from pywinauto import Desktop
            active = Desktop(backend="uia").get_active()
            if active is None or int(active.handle) != expected_handle:
                raise RuntimeError("the foreground application changed")
            return active.get_focus()
        except Exception as exc:
            raise RuntimeError(f"could not safely inspect the focused control: {exc}") from exc


def _is_password_control(control) -> bool:
    info = control.element_info
    control_type = str(getattr(info, "control_type", "") or "")
    is_password = getattr(info, "is_password", None)
    if is_password is True:
        return True
    if is_password is None and control_type in ("Edit", "ComboBox"):
        return True
    metadata = " ".join(
        str(getattr(info, name, "") or "")
        for name in ("automation_id", "class_name")
    ).lower()
    return any(marker in metadata for marker in ("password", "passcode", "passwd"))


def _runtime(ctx: ToolContext) -> DesktopRuntime:
    if not isinstance(ctx.desktop, DesktopRuntime):
        raise RuntimeError("desktop runtime is unavailable")
    return ctx.desktop


class SensitiveApplicationError(RuntimeError):
    pass


def _process_name(process_id: int) -> str:
    """Return only the executable basename, never a document/window title."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.QueryFullProcessImageNameW.argtypes = (
            wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
        )
        kernel.QueryFullProcessImageNameW.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x1000, False, process_id)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ""
        try:
            buffer = ctypes.create_unicode_buffer(32768)
            size = wintypes.DWORD(len(buffer))
            if kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                return Path(buffer.value).name[:120]
        finally:
            kernel.CloseHandle(handle)
    except (AttributeError, OSError, TypeError, ValueError):
        return ""
    return ""


def _observe(ctx: ToolContext, args: dict) -> ToolResult:
    runtime = _runtime(ctx)
    if os.name != "nt":
        return ToolResult(False, error="desktop automation is supported only on Windows", kind="invalid")
    try:
        from pywinauto import Desktop
        foreground = Desktop(backend="uia").get_active()
        if foreground is None:
            return ToolResult(False, error="no visible foreground application window", kind="invalid")
        expected_handle = int(foreground.handle)
        process_name = _process_name(int(foreground.process_id()))
        if is_sensitive_application(process_name, str(foreground.window_text() or "")):
            return ToolResult(
                False,
                error="desktop interaction and monitoring are disabled for this sensitive application",
                kind="human",
            )
    except ImportError as exc:
        return ToolResult(False, error=f"Windows UI Automation requires pywinauto: {exc}", kind="error")
    capture_modes = []
    if runtime.screenshot_enabled:
        capture_modes.append("redacted screenshot")
    if runtime.ocr_enabled:
        capture_modes.append("OCR text")
    detail = "inspect the foreground application's interactive controls"
    if capture_modes:
        detail += " and capture " + " and ".join(capture_modes)
    ctx.policy.require(Action.DESKTOP_OBSERVE, detail)
    if ctx.stop:
        ctx.stop.check()
    try:
        snapshot = runtime.observe(expected_handle=expected_handle)
    except SensitiveApplicationError as exc:
        return ToolResult(False, error=str(exc), kind="human")
    controls = "\n".join(
        f"- {c['type']} {c['name']!r} id={c['automation_id']!r} enabled={c['enabled']} at={c['rect']}"
        for c in snapshot["controls"]
    ) or "(no supported interactive controls found)"
    output = (
        f"Active application: {snapshot['application']}\n"
        f"Interactive controls ({len(snapshot['controls'])} shown, {snapshot['control_count']} found):\n{controls}"
    )
    if snapshot["ocr_text"]:
        output += "\n\nOCR from the active window (editable/password regions masked):\n" + snapshot["ocr_text"]
    if snapshot["screenshot_available"]:
        output += "\nA redacted, in-memory screenshot is available to the local dashboard."
    if snapshot["warnings"]:
        output += "\nUI Automation warnings: " + " ".join(snapshot["warnings"])
    return ToolResult(True, output, data={
        "application": snapshot["application"],
        "control_count": snapshot["control_count"],
        "screenshot_available": snapshot["screenshot_available"],
    })


def _get_pyautogui():
    if os.name != "nt":
        raise RuntimeError("desktop automation is supported only on Windows")
    try:
        import pyautogui
    except ImportError as exc:
        raise RuntimeError("desktop control requires pyautogui; install the desktop extras") from exc
    return pyautogui


def _check_action(ctx: ToolContext, detail: str) -> DesktopRuntime:
    if ctx.stop:
        ctx.stop.check()
    runtime = _runtime(ctx)
    runtime.require_fresh_observation()
    ctx.policy.require(Action.DESKTOP_CONTROL, detail)
    if ctx.stop:
        ctx.stop.check()
    runtime.require_fresh_observation()
    return runtime


def _check_coordinate(runtime: DesktopRuntime, x, y) -> tuple[int, int]:
    try:
        if isinstance(x, bool) or isinstance(y, bool):
            raise ValueError()
        px, py = int(x), int(y)
        if isinstance(x, float) and not x.is_integer():
            raise ValueError()
        if isinstance(y, float) and not y.is_integer():
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError("x and y must be integer screen coordinates") from None
    if runtime._window_rect is None:
        raise RuntimeError("observe_desktop must run before desktop actions")
    left, top, right, bottom = runtime._window_rect
    if not (left <= px < right and top <= py < bottom):
        raise ValueError("coordinates must fall inside the observed foreground application")
    return px, py


def _sensitive_control_at(runtime: DesktopRuntime, x: int, y: int) -> bool:
    if runtime.last_window_handle is None:
        return True
    from pywinauto import Desktop
    active = Desktop(backend="uia").get_active()
    if active is None or int(active.handle) != runtime.last_window_handle:
        raise RuntimeError("the foreground application changed; call observe_desktop before acting")
    for control in active.descendants():
        try:
            rect = control.rectangle()
            if rect.left <= x < rect.right and rect.top <= y < rect.bottom:
                name = str(control.window_text() or "").lower()
                is_password = _is_password_control(control)
                if is_password:
                    return True
                if any(word in name for word in BLOCKED_CONTROL_WORDS):
                    return True
        except Exception:
            return True
    return False


def _refresh_after_action(runtime: DesktopRuntime) -> dict | ToolResult:
    # The explicit action approval also covers this immediate post-action verification snapshot.
    try:
        return runtime.observe()
    except Exception as exc:
        return ToolResult(
            False,
            error=f"action may have taken effect, but post-action observation failed: {type(exc).__name__}: {exc}",
            kind="human",
        )


def _control_summary(snapshot: dict) -> str:
    controls = snapshot["controls"]
    lines = "\n".join(
        f"- {c['type']} {c['name']!r} id={c['automation_id']!r} enabled={c['enabled']} at={c['rect']}"
        for c in controls
    )
    return f"\nUpdated UIA controls ({len(controls)} shown):\n{lines or '(none found)'}"


def _move(ctx: ToolContext, args: dict) -> ToolResult:
    runtime = _check_action(ctx, f"move the pointer to ({args['x']}, {args['y']}) in the observed application")
    x, y = _check_coordinate(runtime, args["x"], args["y"])
    try:
        duration = float(args.get("duration", 0.35))
    except (TypeError, ValueError):
        return ToolResult(False, error="duration must be a number", kind="invalid")
    if not math.isfinite(duration):
        return ToolResult(False, error="duration must be a finite number", kind="invalid")
    duration = max(0.0, min(duration, 1.5))
    pyautogui = _get_pyautogui()
    pyautogui.moveTo(x, y, duration=duration, tween=pyautogui.easeInOutQuad)
    if ctx.stop:
        ctx.stop.check()
    if ctx.activity:
        ctx.activity.record("mouse_action_executed", "pointer moved", x=x, y=y, application=runtime.current_app)
    return ToolResult(True, f"Pointer moved to ({x}, {y})", data={"x": x, "y": y})


def _click(ctx: ToolContext, args: dict) -> ToolResult:
    x, y = args["x"], args["y"]
    button = str(args.get("button", "left")).lower()
    clicks = int(args.get("clicks", 1))
    if button not in ("left", "right", "middle") or clicks not in (1, 2):
        return ToolResult(False, error="button must be left/right/middle and clicks must be 1 or 2", kind="invalid")
    runtime = _check_action(
        ctx, f"{button} click {clicks} time(s) at ({x}, {y}) in the observed foreground application"
    )
    x, y = _check_coordinate(runtime, x, y)
    if _sensitive_control_at(runtime, x, y):
        return ToolResult(False, error="click blocked on password or sensitive controls", kind="denied")
    runtime.require_fresh_observation()
    pyautogui = _get_pyautogui()
    pyautogui.moveTo(x, y, duration=0.25, tween=pyautogui.easeInOutQuad)
    runtime.require_fresh_observation()
    pyautogui.click(x, y, clicks=clicks, interval=0.15, button=button)
    if ctx.stop:
        ctx.stop.check()
    if ctx.activity:
        ctx.activity.record(
            "mouse_action_executed", "click executed", x=x, y=y, button=button,
            clicks=clicks, application=runtime.current_app,
        )
    snapshot = _refresh_after_action(runtime)
    if isinstance(snapshot, ToolResult):
        return snapshot
    return ToolResult(True, f"Clicked at ({x}, {y}); re-observed {runtime.current_app}."
                      + _control_summary(snapshot),
                      data={"x": x, "y": y, "application": runtime.current_app,
                            "post_action_observed": True})


def _type_text(ctx: ToolContext, args: dict) -> ToolResult:
    text = args["text"]
    if not isinstance(text, str) or not text:
        return ToolResult(False, error="text must be a non-empty string", kind="invalid")
    if len(text) > 4000:
        return ToolResult(False, error="text exceeds the 4000 character limit", kind="invalid")
    runtime = _check_action(ctx, f"type {len(text)} user-provided characters in the focused control")
    focused = runtime.focused_control(runtime.last_window_handle)
    if _is_password_control(focused):
        return ToolResult(False, error="typing into password fields is blocked", kind="denied")
    if str(focused.element_info.control_type or "") not in ("Edit", "ComboBox"):
        return ToolResult(False, error="typing is allowed only in a focused editable control", kind="denied")
    runtime.require_fresh_observation()
    try:
        interval = float(args.get("interval", 0.04))
    except (TypeError, ValueError):
        return ToolResult(False, error="interval must be a number", kind="invalid")
    if not math.isfinite(interval):
        return ToolResult(False, error="interval must be a finite number", kind="invalid")
    interval = max(0.0, min(interval, 0.25))
    try:
        _get_pyautogui().write(text, interval=interval)
    except Exception as exc:
        return ToolResult(
            False,
            error=f"typing failed ({type(exc).__name__}); input may be partially entered",
            kind="human",
        )
    if ctx.stop:
        ctx.stop.check()
    if ctx.activity:
        ctx.activity.record(
            "text_entry_executed", "text entered (content omitted)",
            character_count=len(text), application=runtime.current_app,
        )
    snapshot = _refresh_after_action(runtime)
    if isinstance(snapshot, ToolResult):
        return snapshot
    return ToolResult(True, f"Typed {len(text)} characters; input content is not included in logs."
                      + _control_summary(snapshot),
                      data={"character_count": len(text), "application": runtime.current_app,
                            "post_action_observed": True})


def _press_key(ctx: ToolContext, args: dict) -> ToolResult:
    key = str(args["key"]).strip().lower()
    parts = key.split("+")
    modifiers = {"ctrl", "shift", "alt"}
    safe = (len(parts) == 1 and parts[0] in SAFE_KEYS) or (
        len(parts) == 2 and parts[0] in modifiers and parts[1] in SAFE_SHORTCUT_KEYS
    )
    if not safe:
        return ToolResult(False, error="key is not in the supported safe keyboard shortcut set", kind="denied")
    detail = f"press {key} in the observed foreground application"
    if key == "enter":
        detail += " (Enter may submit the current form)"
    runtime = _check_action(ctx, detail)
    focused = runtime.focused_control(runtime.last_window_handle)
    if _is_password_control(focused):
        return ToolResult(False, error="keyboard actions in password fields are blocked", kind="denied")
    focused_name = str(focused.window_text() or "").lower()
    if any(word in focused_name for word in BLOCKED_CONTROL_WORDS):
        return ToolResult(False, error="keyboard action blocked on a sensitive control", kind="denied")
    runtime.require_fresh_observation()
    pyautogui = _get_pyautogui()
    if len(parts) == 1:
        pyautogui.press(parts[0])
    else:
        pyautogui.hotkey(*parts, interval=0.08)
    if ctx.stop:
        ctx.stop.check()
    if ctx.activity:
        ctx.activity.record(
            "keyboard_shortcut_executed", "keyboard action executed",
            key=key, application=runtime.current_app,
        )
    snapshot = _refresh_after_action(runtime)
    if isinstance(snapshot, ToolResult):
        return snapshot
    return ToolResult(True, f"Pressed {key}; re-observed {runtime.current_app}."
                      + _control_summary(snapshot),
                      data={"key": key, "application": runtime.current_app, "post_action_observed": True})


def _scroll(ctx: ToolContext, args: dict) -> ToolResult:
    direction = str(args["direction"]).lower()
    if direction not in ("up", "down"):
        return ToolResult(False, error="direction must be up or down", kind="invalid")
    try:
        amount = max(1, min(int(args.get("amount", 3)), 10))
    except (TypeError, ValueError):
        return ToolResult(False, error="amount must be an integer", kind="invalid")
    runtime = _check_action(
        ctx, f"scroll {direction} {amount} notch(es) at ({args['x']}, {args['y']}) in the observed application"
    )
    x, y = _check_coordinate(runtime, args["x"], args["y"])
    _get_pyautogui().moveTo(x, y, duration=0.2)
    _get_pyautogui().scroll(amount if direction == "up" else -amount, x=x, y=y)
    if ctx.stop:
        ctx.stop.check()
    if ctx.activity:
        ctx.activity.record(
            "mouse_action_executed", "scroll executed", direction=direction,
            amount=amount, x=x, y=y, application=runtime.current_app,
        )
    snapshot = _refresh_after_action(runtime)
    if isinstance(snapshot, ToolResult):
        return snapshot
    return ToolResult(True, f"Scrolled {direction} at ({x}, {y}); re-observed {runtime.current_app}."
                      + _control_summary(snapshot),
                      data={"direction": direction, "application": runtime.current_app,
                            "post_action_observed": True})


def _wait(ctx: ToolContext, args: dict) -> ToolResult:
    try:
        seconds = float(args.get("seconds", 1))
    except (TypeError, ValueError):
        return ToolResult(False, error="seconds must be a number", kind="invalid")
    if not math.isfinite(seconds) or seconds < 0 or seconds > 30:
        return ToolResult(False, error="wait duration must be between 0 and 30 seconds", kind="invalid")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if ctx.stop:
            ctx.stop.check()
        time.sleep(min(0.1, deadline - time.monotonic()))
    return ToolResult(True, f"Waited {seconds:g} seconds")


TOOLS = [
    Tool("observe_desktop", "Request approval, then inspect the foreground Windows application and its interactive UIA controls. Editable values are never read.",
         {}, _observe, idempotent=True),
    Tool("move_mouse", "Smoothly move the pointer inside the most recently observed foreground application.",
         {"x": "screen x coordinate inside the observed application", "y": "screen y coordinate inside the observed application",
          "duration": "movement time in seconds (0-1.5, default 0.35)"},
         _move, required=("x", "y"), idempotent=True),
    Tool("click", "Click one or two times within the observed application; password and sensitive controls are blocked.",
         {"x": "screen x coordinate", "y": "screen y coordinate", "button": "left, right, or middle",
          "clicks": "1 or 2"},
         _click, required=("x", "y")),
    Tool("type_text", "Type user-provided text into the focused non-password control. The content is not included in action logs.",
         {"text": "text to type (never a password)", "interval": "delay between keystrokes, seconds"},
         _type_text, required=("text",), sensitive_args=("text",)),
    Tool("press_key", "Press a supported navigation key or safe Ctrl/Shift/Alt shortcut; password fields are blocked.",
         {"key": "key or supported shortcut, e.g. tab, enter, ctrl+s"},
         _press_key, required=("key",)),
    Tool("scroll", "Scroll inside the observed foreground application.",
         {"direction": "up or down", "x": "screen x coordinate", "y": "screen y coordinate",
          "amount": "scroll notches (1-10, default 3)"},
         _scroll, required=("direction", "x", "y"), idempotent=True),
    Tool("wait", "Wait briefly for an application response while checking the emergency stop.",
         {"seconds": "duration from 0 to 30 seconds"}, _wait, idempotent=True),
]
