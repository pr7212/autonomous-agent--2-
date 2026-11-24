from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Callable


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"      # needs an explicit yes from the person
    HUMAN = "human"  # the agent must not do this at all; a person does it
    DENY = "deny"


class Action(str, Enum):
    READ_FILE = "read_file"
    WRITE_FILE = "write_file"
    LIST_DIR = "list_dir"
    RUN_TESTS = "run_tests"
    RUN_CODE = "run_code"
    SAFE_COMMAND = "safe_command"
    NAVIGATE = "navigate"
    USE_API = "use_api"
    DELETE_FILE = "delete_file"
    SEND_MESSAGE = "send_message"
    PURCHASE = "purchase"
    CHANGE_SECURITY = "change_security"
    RUN_COMMAND = "run_command"
    SUBMIT_CONSEQUENTIAL = "submit_consequential"
    BLOCKED_COMMAND = "blocked_command"
    DESKTOP_OBSERVE = "desktop_observe"
    DESKTOP_CONTROL = "desktop_control"


# The permission table from the design doc, plus the command classes this phase adds.
POLICY: dict[Action, Decision] = {
    Action.READ_FILE: Decision.ALLOW,
    Action.WRITE_FILE: Decision.ALLOW,  # confined to the workspace by the sandbox
    Action.LIST_DIR: Decision.ALLOW,
    Action.RUN_TESTS: Decision.ALLOW,
    Action.RUN_CODE: Decision.ALLOW,
    Action.SAFE_COMMAND: Decision.ALLOW,
    Action.NAVIGATE: Decision.ALLOW,
    Action.USE_API: Decision.ALLOW,
    Action.DELETE_FILE: Decision.ASK,
    Action.SEND_MESSAGE: Decision.ASK,
    Action.PURCHASE: Decision.ASK,
    Action.CHANGE_SECURITY: Decision.ASK,
    Action.RUN_COMMAND: Decision.ASK,  # unknown executable
    Action.SUBMIT_CONSEQUENTIAL: Decision.HUMAN,
    Action.BLOCKED_COMMAND: Decision.DENY,
    Action.DESKTOP_OBSERVE: Decision.ASK,
    Action.DESKTOP_CONTROL: Decision.ASK,
}


class PermissionDenied(Exception):
    pass


class NeedsHuman(PermissionDenied):
    """The action is reserved for a person (applications, assessments)."""


Approver = Callable[[str, str], bool]


def deny_approver(action: str, detail: str) -> bool:
    return False


def interactive_approver(action: str, detail: str) -> bool:
    print(f"\n  APPROVAL NEEDED [{action}] {detail}")
    try:
        answer = input("  Allow? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


class PermissionPolicy:
    def __init__(self, approver: Approver = deny_approver, audit: Callable[..., None] | None = None):
        self.approver = approver
        self.audit = audit

    @staticmethod
    def decide(action: Action) -> Decision:
        return POLICY.get(action, Decision.ASK)  # unknown action: fail toward asking

    def require(self, action: Action, detail: str = "") -> None:
        decision = self.decide(action)
        if decision == Decision.ALLOW:
            return
        if decision == Decision.DENY:
            self._audit("permission_blocked", f"{action.value}: {detail}")
            raise PermissionDenied(f"{action.value} is blocked by policy: {detail}")
        if decision == Decision.HUMAN:
            self._audit("permission_human", f"{action.value}: {detail}")
            raise NeedsHuman(f"{action.value} must be done by a person: {detail}")
        approved = self.approver(action.value, detail)
        self._audit("approval_granted" if approved else "approval_denied", f"{action.value}: {detail}")
        if not approved:
            raise PermissionDenied(f"approval denied for {action.value}: {detail}")

    def _audit(self, kind: str, message: str) -> None:
        if self.audit:
            self.audit(kind, message)


# --- command classification -------------------------------------------------

_SHELLS = {"bash", "sh", "zsh", "fish", "cmd", "powershell", "pwsh", "wsl"}
_DESTRUCTIVE = {
    "rm", "rmdir", "rd", "del", "erase", "format", "mkfs", "dd", "diskpart",
    "fdisk", "shutdown", "reboot", "halt", "poweroff",
}
_SECURITY = {
    "sudo", "su", "chmod", "chown", "chgrp", "icacls", "takeown", "reg", "regedit",
    "netsh", "net", "schtasks", "sc", "setx", "usermod", "passwd", "ufw", "iptables",
    "bcdedit",
}
_SAFE = {"echo", "pwd", "hostname", "whoami"}
_GIT_READ_ONLY = {"status", "log", "diff", "show", "branch", "rev-parse"}
_PYTHON = re.compile(r"^(python|python3|py|pythonw)(\d+(\.\d+)*)?$")


def executable_name(token: str) -> str:
    name = Path(token).name.lower()
    return re.sub(r"\.(exe|cmd|bat|com)$", "", name)


def is_python(token: str) -> bool:
    return bool(_PYTHON.match(executable_name(token)))


def has_shell_operators(command: str) -> bool:
    """Commands run without a shell, so ; && | > < ` $( would be passed literally.
    Rejecting them beats silently doing something the planner did not intend."""
    return bool(re.search(r"[;&|<>`\n]|\$\(", command))


def classify_command(argv: list[str]) -> tuple[Action, str]:
    exe = executable_name(argv[0])
    rest = argv[1:]
    if exe in _SHELLS:
        return Action.BLOCKED_COMMAND, "shells would bypass argument checks; run the program directly"
    if exe in _DESTRUCTIVE:
        return Action.BLOCKED_COMMAND, "destructive command; use the delete_file tool (needs approval)"
    if exe in _SECURITY:
        return Action.CHANGE_SECURITY, "may change security or system settings"
    if is_python(argv[0]):
        if rest[:2] in (["-m", "pytest"], ["-m", "unittest"]):
            return Action.RUN_TESTS, "test run"
        return Action.RUN_CODE, "python in the workspace"
    if exe == "pytest":
        return Action.RUN_TESTS, "test run"
    if exe == "git" and rest and rest[0] in _GIT_READ_ONLY:
        return Action.SAFE_COMMAND, "read-only git"
    if exe in _SAFE:
        return Action.SAFE_COMMAND, "harmless command"
    return Action.RUN_COMMAND, "executable not on the allowlist"
