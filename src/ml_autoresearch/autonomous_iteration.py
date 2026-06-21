"""Bounded autonomous iteration loop orchestration."""

from __future__ import annotations

import base64
import json
import re
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic
from typing import Callable

from ml_autoresearch.agent_handoffs import _discover_uningested_primary_handoffs
from ml_autoresearch.autonomy_step import AutonomyStepResult, run_autonomy_step

RESULT_FILENAME = "autonomous-iteration-result.json"


class AutonomousIterationError(ValueError):
    """Raised when a bounded autonomous iteration cannot proceed safely."""


@dataclass(frozen=True)
class MailjetConfig:
    api_key: str
    api_secret: str
    from_email: str
    from_name: str


@dataclass
class AutonomousIterationResult:
    status: str
    stop_reason: str
    workspace_root: str
    started_at: str
    completed_at: str
    elapsed_seconds: float
    limits: dict[str, object]
    steps_started: int
    steps_completed: int
    final_step: dict[str, object] | None
    step_summaries: list[dict[str, object]]
    notification: dict[str, object] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.status,
            "stop_reason": self.stop_reason,
            "workspace_root": self.workspace_root,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "elapsed_seconds": self.elapsed_seconds,
            "limits": self.limits,
            "steps_started": self.steps_started,
            "steps_completed": self.steps_completed,
            "final_step": self.final_step,
            "step_summaries": self.step_summaries,
            "notification": self.notification,
        }


SendMailjet = Callable[[dict[str, str]], None]


def parse_duration_seconds(value: str) -> int:
    """Parse positive seconds from N, Ns, Nm, or Nh syntax."""

    if not isinstance(value, str) or not value:
        raise AutonomousIterationError("duration must be a positive integer with optional s, m, or h suffix")
    match = re.fullmatch(r"([0-9]+)([smh]?)", value.strip())
    if match is None:
        raise AutonomousIterationError("duration must use N, Ns, Nm, or Nh syntax")
    amount = int(match.group(1))
    multiplier = {"": 1, "s": 1, "m": 60, "h": 3600}[match.group(2)]
    seconds = amount * multiplier
    if seconds <= 0:
        raise AutonomousIterationError("duration must be positive")
    return seconds


def run_autonomous_iteration(
    project_root: str | Path = Path("."),
    *,
    agent_command: str | None = None,
    max_steps: int | None = None,
    max_duration_seconds: int | None = None,
    notify_email: str,
    send_mailjet: SendMailjet | None = None,
) -> AutonomousIterationResult:
    """Run Autonomy Steps with execution until a limit or human-review stop condition arises."""

    root = Path(project_root).resolve()
    _validate_limits(max_steps, max_duration_seconds)
    _validate_recipient(notify_email)
    _ensure_clean_agent_workspace(root)
    mailjet = load_mailjet_config(root)
    sender = send_mailjet or (lambda message: send_mailjet_message(mailjet, message))

    started_at_dt = datetime.now(UTC)
    started_at = _format_time(started_at_dt)
    start_monotonic = monotonic()
    steps_started = 0
    steps_completed = 0
    step_summaries: list[dict[str, object]] = []
    final_step: dict[str, object] | None = None
    stop_reason = "time_limit_reached" if _time_limit_reached(start_monotonic, max_duration_seconds) else "step_limit_reached"

    while True:
        if max_steps is not None and steps_completed >= max_steps:
            stop_reason = "step_limit_reached"
            break
        if _time_limit_reached(start_monotonic, max_duration_seconds):
            stop_reason = "time_limit_reached"
            break

        steps_started += 1
        step = run_autonomy_step(root, agent_command=agent_command, execute_next_action=True)
        steps_completed += 1
        final_step = step.to_json()
        step_summaries.append(_step_summary(steps_completed, step))

        step_stop = _stop_reason_from_step(step)
        if step_stop is not None:
            stop_reason = step_stop
            break
        if max_steps is not None and steps_completed >= max_steps:
            stop_reason = "step_limit_reached"
            break
        if _time_limit_reached(start_monotonic, max_duration_seconds):
            stop_reason = "time_limit_reached"
            break

    completed_at = _format_time(datetime.now(UTC))
    result = AutonomousIterationResult(
        status="completed",
        stop_reason=stop_reason,
        workspace_root=str(root),
        started_at=started_at,
        completed_at=completed_at,
        elapsed_seconds=round(monotonic() - start_monotonic, 3),
        limits={"max_steps": max_steps, "max_duration_seconds": max_duration_seconds},
        steps_started=steps_started,
        steps_completed=steps_completed,
        final_step=final_step,
        step_summaries=step_summaries,
        notification={"status": "pending", "recipient": notify_email},
    )
    _write_result(root, result)

    try:
        sender(_mailjet_message(mailjet, notify_email, result))
    except Exception as exc:  # noqa: BLE001 - notification failure is recorded after loop completion.
        result.notification = {"status": "failed", "recipient": notify_email, "reason": str(exc)}
        _write_result(root, result)
        raise AutonomousIterationError(f"notification failed after autonomous iteration completed: {exc}") from exc

    result.notification = {"status": "sent", "recipient": notify_email}
    _write_result(root, result)
    return result


def format_autonomous_iteration_summary(result: AutonomousIterationResult) -> str:
    lines = [
        "Autonomous Iteration complete",
        f"Status: {result.status}",
        f"Stop reason: {result.stop_reason}",
        f"Steps completed: {result.steps_completed}",
        f"Elapsed seconds: {result.elapsed_seconds}",
        f"Notification: {result.notification.get('status')}",
        f"Result file: {Path(result.workspace_root) / 'agent-work' / RESULT_FILENAME}",
    ]
    return "\n".join(lines)


def load_mailjet_config(project_root: str | Path = Path(".")) -> MailjetConfig:
    import tomllib

    from ml_autoresearch.workspace import WORKSPACE_CONFIG_FILENAME

    path = Path(project_root) / WORKSPACE_CONFIG_FILENAME
    if not path.is_file():
        raise AutonomousIterationError(f"missing Workspace Configuration: {path}")
    try:
        data = tomllib.loads(path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise AutonomousIterationError(f"invalid Workspace Configuration {path}: {exc}") from exc
    mailjet = data.get("mailjet")
    if not isinstance(mailjet, dict):
        raise AutonomousIterationError(f"{path.name} must contain [mailjet]")
    values: dict[str, str] = {}
    for field_name in ("api_key", "api_secret", "from_email", "from_name"):
        value = mailjet.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise AutonomousIterationError(f"mailjet.{field_name} must be a non-empty string")
        values[field_name] = value
    return MailjetConfig(**values)


def send_mailjet_message(config: MailjetConfig, message: dict[str, str]) -> None:
    payload = {
        "Messages": [
            {
                "From": {"Email": config.from_email, "Name": config.from_name},
                "To": [{"Email": message["to_email"]}],
                "Subject": message["subject"],
                "TextPart": message["text"],
            }
        ]
    }
    body = json.dumps(payload).encode("utf-8")
    token = base64.b64encode(f"{config.api_key}:{config.api_secret}".encode("utf-8")).decode("ascii")
    request = urllib.request.Request(
        "https://api.mailjet.com/v3.1/send",
        data=body,
        headers={"Authorization": f"Basic {token}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310 - configured Mailjet HTTPS endpoint.
        if response.status >= 400:
            raise AutonomousIterationError(f"Mailjet returned HTTP {response.status}")


def _validate_limits(max_steps: int | None, max_duration_seconds: int | None) -> None:
    if max_steps is None and max_duration_seconds is None:
        raise AutonomousIterationError("at least one of max_steps or max_duration_seconds is required")
    if max_steps is not None and max_steps <= 0:
        raise AutonomousIterationError("max_steps must be positive")
    if max_duration_seconds is not None and max_duration_seconds <= 0:
        raise AutonomousIterationError("max_duration_seconds must be positive")


def _validate_recipient(email: str) -> None:
    if not isinstance(email, str) or not email.strip() or "@" not in email:
        raise AutonomousIterationError("notify_email must be a non-empty email address")


def _ensure_clean_agent_workspace(root: Path) -> None:
    artifacts = _discover_uningested_primary_handoffs(root)
    dirty = [path for paths in artifacts.values() for path in paths]
    if dirty:
        rel = ", ".join(str(path.relative_to(root)) for path in dirty)
        raise AutonomousIterationError(f"agent workspace contains un-ingested primary handoff artifacts: {rel}")


def _time_limit_reached(start_monotonic: float, max_duration_seconds: int | None) -> bool:
    return max_duration_seconds is not None and monotonic() - start_monotonic >= max_duration_seconds


def _stop_reason_from_step(step: AutonomyStepResult) -> str | None:
    if step.status == "agent_failed":
        return "agent_failed"
    if step.status == "ingestion_failed":
        return "ingestion_failed"
    if step.status == "execution_failed":
        return "execution_failed"
    ingestion = step.ingestion or {}
    if step.status == "no_handoff" or ingestion.get("status") == "no_handoff":
        return "no_handoff"
    handoff_type = ingestion.get("handoff_type")
    next_action = ingestion.get("next_action")
    if handoff_type == "capability_request":
        return "capability_request"
    if handoff_type == "campaign_report":
        return "campaign_paused" if next_action == "pause_campaign" else "human_review_requested"
    if handoff_type == "research_note" and next_action == "continue_autonomy":
        return None
    if handoff_type in {"candidate_submission", "evaluation_request"}:
        execution = step.execution or ingestion.get("next_action_result")
        if isinstance(execution, dict) and execution.get("status") == "completed" and execution.get("executed") is True:
            return None
        return "non_executable_next_action"
    if next_action == "stop_for_human":
        return "human_review_requested"
    return "non_executable_next_action"


def _step_summary(step_index: int, step: AutonomyStepResult) -> dict[str, object]:
    ingestion = step.ingestion or {}
    execution = step.execution or ingestion.get("next_action_result")
    return {
        "step_index": step_index,
        "status": step.status,
        "handoff_type": ingestion.get("handoff_type"),
        "next_action": ingestion.get("next_action"),
        "execution_status": execution.get("status") if isinstance(execution, dict) else None,
        "reason": step.reason,
    }


def _mailjet_message(config: MailjetConfig, recipient: str, result: AutonomousIterationResult) -> dict[str, str]:
    final = result.final_step or {}
    ingestion = final.get("ingestion") if isinstance(final.get("ingestion"), dict) else {}
    text = (
        "Autonomous iteration completed.\n\n"
        f"Workspace root: {result.workspace_root}\n"
        f"Status: {result.status}\n"
        f"Stop reason: {result.stop_reason}\n"
        f"Steps completed: {result.steps_completed}\n"
        f"Elapsed seconds: {result.elapsed_seconds}\n\n"
        "Final step:\n"
        f"- Status: {final.get('status')}\n"
        f"- Handoff type: {ingestion.get('handoff_type')}\n"
        f"- Next action: {ingestion.get('next_action')}\n"
        f"- Reason: {final.get('reason')}\n\n"
        f"Result file:\nagent-work/{RESULT_FILENAME}\n"
    )
    return {
        "to_email": recipient,
        "subject": f"ML Autoresearch autonomous iteration completed: {result.stop_reason}",
        "text": text,
        "from_email": config.from_email,
        "from_name": config.from_name,
    }


def _write_result(root: Path, result: AutonomousIterationResult) -> Path:
    path = root / "agent-work" / RESULT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json(), indent=2, sort_keys=True) + "\n")
    return path


def _format_time(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")
