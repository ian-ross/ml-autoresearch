"""Harness-owned Autonomy Step orchestration."""

from __future__ import annotations

import json
import shlex
import subprocess
import tomllib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ml_autoresearch.agent_boundary import prepare_agent_boundary
from ml_autoresearch.agent_handoffs import collect_agent_handoff

DEFAULT_AGENT_COMMAND = "pi --session-dir ../agent-sessions"
RESULT_FILENAME = "autonomy-step-result.json"
PROMPT_FILENAME = "prompt.txt"


class AutonomyStepError(ValueError):
    """Raised when an Autonomy Step cannot be configured safely."""


@dataclass(frozen=True)
class AutonomyStepResult:
    """Machine-readable result for one Autonomy Step."""

    status: str
    workspace_root: str
    agent_workspace: str
    prompt_path: str
    agent_command: list[str]
    agent_returncode: int
    ingestion: dict[str, object] | None
    reason: str | None = None
    execution: dict[str, object] | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "status": self.status,
            "workspace_root": self.workspace_root,
            "agent_workspace": self.agent_workspace,
            "prompt_path": self.prompt_path,
            "agent_command": self.agent_command,
            "agent_returncode": self.agent_returncode,
            "ingestion": self.ingestion,
            "execution": self.execution,
            "reason": self.reason,
        }


def load_configured_agent_command(workspace_root: Path) -> str | None:
    """Return optional Autonomy Step agent command from Workspace Configuration."""

    from ml_autoresearch.workspace import WORKSPACE_CONFIG_FILENAME

    config_path = workspace_root / WORKSPACE_CONFIG_FILENAME
    if not config_path.is_file():
        return None
    data = tomllib.loads(config_path.read_text())
    settings = data.get("autonomy_step", {})
    if not isinstance(settings, dict):
        raise AutonomyStepError("[autonomy_step] must be a table")
    value = settings.get("agent_command")
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AutonomyStepError("autonomy_step.agent_command must be a non-empty string")
    return value


def run_autonomy_step(
    project_root: str | Path = Path("."),
    agent_command: str | None = None,
    *,
    execute_next_action: bool = False,
) -> AutonomyStepResult:
    """Refresh the Agent Control Boundary, invoke the agent once, ingest one primary handoff, and optionally execute one next action."""

    root = Path(project_root).resolve()
    boundary = prepare_agent_boundary(root)
    workspace = Path(boundary["agent_workspace"])
    _remove_stale_loop_result_files(workspace)
    prompt_path = workspace / PROMPT_FILENAME
    prompt_path.write_text(render_autonomy_step_prompt(root))

    configured_command = agent_command or load_configured_agent_command(root) or DEFAULT_AGENT_COMMAND
    if configured_command == DEFAULT_AGENT_COMMAND:
        (root / "agent-sessions").mkdir(exist_ok=True)
    command = _agent_command_argv(configured_command) + ["-p", f"@{PROMPT_FILENAME}"]
    completed = subprocess.run(command, cwd=workspace, check=False)

    if completed.returncode != 0:
        result = AutonomyStepResult(
            status="agent_failed",
            workspace_root=str(root),
            agent_workspace=str(workspace),
            prompt_path=str(prompt_path),
            agent_command=command,
            agent_returncode=completed.returncode,
            ingestion=None,
            reason=f"agent command exited with status {completed.returncode}; handoff ingestion skipped",
        )
        write_result_file(workspace, result)
        return result

    ingestion = collect_agent_handoff(root)
    status = _status_from_ingestion(ingestion)
    reason = ingestion.get("reason") if isinstance(ingestion.get("reason"), str) else None
    execution = None
    if status == "ingested" and execute_next_action:
        try:
            execution = execute_ingested_next_action(root, ingestion)
        except Exception as exc:  # noqa: BLE001 - preserve ingestion record and stop for human review.
            execution = {"status": "failed", "reason": str(exc)}
            status = "execution_failed"
            reason = str(exc)
        else:
            if execution.get("executed") is True:
                ingestion["executed_next_action"] = True
                ingestion["next_action_result"] = execution

    result = AutonomyStepResult(
        status=status,
        workspace_root=str(root),
        agent_workspace=str(workspace),
        prompt_path=str(prompt_path),
        agent_command=command,
        agent_returncode=completed.returncode,
        ingestion=ingestion,
        reason=reason,
        execution=execution,
    )
    write_result_file(workspace, result)
    return result


def _remove_stale_loop_result_files(workspace: Path) -> None:
    """Remove previous loop result files before invoking the next agent step."""

    for filename in (RESULT_FILENAME, "autonomous-iteration-result.json"):
        path = workspace / filename
        if path.exists():
            path.unlink()


def render_autonomy_step_prompt(project_root: Path | None = None) -> str:
    """Render the one-step prompt written inside the Agent Workspace."""

    campaign_state = _campaign_state_prompt(project_root) if project_root is not None else ""
    return (
        "You are inside the ML Autoresearch Agent Control Boundary for exactly one Autonomy Step.\n"
        "\n"
        "Read AGENTS.md first for the boundary path map and writable handoff locations.\n"
        "Use the campaign-manager skill as the top-level workflow entry point.\n"
        "Work one step at a time: choose and complete exactly one primary research handoff outcome.\n"
        "Do not continue to a second primary outcome after producing the first one.\n"
        "\n"
        f"{campaign_state}"
        "Allowed primary handoff outcomes (choose exactly one if you hand off work):\n"
        "- one Candidate Submission under submissions/\n"
        "- one Research Note under research-notes/\n"
        "- one Capability Request under capability-requests/\n"
        "- one Evaluation Request under evaluation-requests/\n"
        "- one Campaign Report under campaign-reports/\n"
        "\n"
        "Multiple primary handoff outcomes are forbidden in one Autonomy Step. If blocked, write one Campaign Report "
        "or one Capability Request, not both. If no useful handoff is safe, stop without fabricating artifacts.\n"
        "If you write a Campaign Report, it must include these exact machine-readable headings:\n"
        "- ## Current best Result\n"
        "- ## Recent Runs\n"
        "- ## Failures\n"
        "- ## Pending Capability Requests\n"
        "- ## Budget use\n"
        "- ## Next hypothesis\n"
        "- ## Pause recommendation\n"
        "The Campaign Report Pause recommendation must include exactly `- Pause condition: none` or "
        "`- Pause condition: <approved_value>`. Do not add punctuation or prose to the Pause condition line.\n"
        "Use ml-autoresearch-agent, not ml-autoresearch, for allowed inner-boundary commands.\n"
    )


def _campaign_state_prompt(project_root: Path) -> str:
    state = _latest_campaign_pause_resume_state(project_root)
    if state is None:
        return ""
    if state["status"] == "resumed":
        return (
            "Campaign resume state:\n"
            f"- Human campaign review is complete: {state['reason']}.\n"
            "- Do not treat earlier scheduled_check_in or resolved capability-request pause recommendations as active blockers.\n"
            "- Continue from the latest Research Ledger, Research Notes, and Campaign Reports unless a new pause condition is met.\n"
            "\n"
        )
    if state["status"] == "paused":
        return (
            "Campaign pause state:\n"
            f"- Latest Campaign Report recommends pause for `{state['reason']}`.\n"
            "- Stop for human review unless a newer campaign_resumed ledger event clears this pause.\n"
            "\n"
        )
    return ""


def _latest_campaign_pause_resume_state(project_root: Path) -> dict[str, str] | None:
    ledger_path = project_root / "research-ledger.jsonl"
    if not ledger_path.is_file():
        return None
    latest_pause_index: int | None = None
    latest_pause_reason: str | None = None
    latest_resume_index: int | None = None
    latest_resume_reason: str | None = None
    for index, line in enumerate(ledger_path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_type = event.get("event_type")
        if event_type == "campaign_resumed":
            latest_resume_index = index
            latest_resume_reason = str(event.get("reason") or "human_review_complete")
        elif event_type == "campaign_paused":
            latest_pause_index = index
            latest_pause_reason = str(event.get("reason") or "unknown")
        elif event_type == "campaign_report_written":
            report_path = event.get("report_path")
            reason = _campaign_report_pause_reason(project_root, report_path) if isinstance(report_path, str) else None
            if reason and reason != "none":
                latest_pause_index = index
                latest_pause_reason = reason
    if latest_resume_index is not None and (latest_pause_index is None or latest_resume_index > latest_pause_index):
        return {"status": "resumed", "reason": latest_resume_reason or "human_review_complete"}
    if latest_pause_index is not None:
        return {"status": "paused", "reason": latest_pause_reason or "unknown"}
    return None


def _campaign_report_pause_reason(project_root: Path, report_path: str) -> str | None:
    path = Path(report_path)
    if not path.is_absolute():
        path = project_root / path
    if not path.is_file():
        return None
    prefix = "Pause condition:"
    for line in path.read_text().splitlines():
        normalized = line.strip()
        candidate = normalized[1:].strip() if normalized.startswith("-") else normalized
        if candidate.startswith(prefix):
            return _normalize_campaign_report_pause_reason(candidate[len(prefix) :])
    return None


def _normalize_campaign_report_pause_reason(raw_value: str) -> str:
    return raw_value.strip().strip("`").strip().rstrip(".").strip().strip("`")


def write_result_file(workspace: Path, result: AutonomyStepResult) -> Path:
    """Write the machine-readable Autonomy Step result file."""

    path = workspace / RESULT_FILENAME
    payload = result.to_json()
    payload["written_at"] = datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def format_autonomy_step_summary(result: AutonomyStepResult) -> str:
    """Return a human-readable summary for stdout."""

    lines = ["Autonomy Step complete", f"Status: {result.status}", f"Agent workspace: {result.agent_workspace}"]
    if result.reason:
        lines.append(f"Reason: {result.reason}")
    if result.ingestion is not None:
        ingestion_status = result.ingestion.get("status")
        lines.append(f"Handoff ingestion: {ingestion_status}")
        handoff_type = result.ingestion.get("handoff_type")
        if handoff_type:
            lines.append(f"Handoff type: {handoff_type}")
        next_action = result.ingestion.get("next_action")
        if next_action:
            lines.append(f"Next action: {next_action}")
    if result.execution is not None:
        lines.append(f"Next action execution: {result.execution.get('status')}")
    lines.append(f"Result file: {Path(result.agent_workspace) / RESULT_FILENAME}")
    return "\n".join(lines)


def find_open_executable_actions(project_root: str | Path = Path(".")) -> list[dict[str, object]]:
    """Return Harness-executable ingested handoffs that have no downstream completion event."""

    root = Path(project_root).resolve()
    ledger_path = root / "research-ledger.jsonl"
    if not ledger_path.is_file():
        return []
    events: list[dict[str, object]] = []
    for index, line in enumerate(ledger_path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            event["_ledger_index"] = index
            events.append(event)

    submitted_candidates = {
        str(event.get("candidate_id"))
        for event in events
        if event.get("event_type") == "candidate_submitted" and event.get("candidate_id")
    }
    completed_evaluations = {
        str(event.get("evaluation_request_id"))
        for event in events
        if event.get("event_type") == "evaluation_completed" and event.get("evaluation_request_id")
    }
    completed_batches = {
        str(event.get("batch_id"))
        for event in events
        if event.get("event_type") == "experiment_batch_completed" and event.get("batch_id")
    }

    open_actions: list[dict[str, object]] = []
    for event in events:
        if event.get("event_type") != "agent_handoff_ingested":
            continue
        handoff_type = event.get("handoff_type")
        canonical_path = event.get("canonical_path")
        if not isinstance(canonical_path, str) or not canonical_path:
            continue
        base = {
            "handoff_type": handoff_type,
            "canonical_path": canonical_path,
            "created_at": event.get("created_at"),
            "ledger_index": event.get("_ledger_index"),
        }
        if handoff_type == "candidate_submission":
            candidate_id = event.get("candidate_id") or event.get("artifact_id")
            if isinstance(candidate_id, str) and candidate_id and candidate_id not in submitted_candidates:
                open_actions.append({**base, "action": "run_candidate", "candidate_id": candidate_id})
        elif handoff_type == "evaluation_request":
            request_id = event.get("request_id") or event.get("artifact_id")
            if isinstance(request_id, str) and request_id and request_id not in completed_evaluations:
                open_actions.append(
                    {
                        **base,
                        "action": "run_post_run_evaluation",
                        "request_id": request_id,
                        "run_id": event.get("run_id"),
                    }
                )
        elif handoff_type == "experiment_batch_submission":
            batch_id = event.get("artifact_id")
            if isinstance(batch_id, str) and batch_id and batch_id not in completed_batches:
                open_actions.append({**base, "action": "run_experiment_batch", "batch_id": batch_id})
    return open_actions


def execute_open_actions(
    project_root: str | Path = Path("."), *, dry_run: bool = False, max_actions: int | None = None
) -> dict[str, object]:
    """Execute reconciled open Harness-owned actions in ledger order."""

    root = Path(project_root).resolve()
    open_actions = find_open_executable_actions(root)
    if dry_run:
        return {
            "status": "dry_run",
            "dry_run": True,
            "open_actions": open_actions,
            "executed_count": 0,
            "executions": [],
        }
    executions: list[dict[str, object]] = []
    limit = max_actions if max_actions is not None else len(open_actions)
    for _ in range(max(0, limit)):
        current = find_open_executable_actions(root)
        if not current:
            break
        action = current[0]
        ingestion = _ingestion_from_open_action(action)
        try:
            result = execute_ingested_next_action(root, ingestion)
        except Exception as exc:  # noqa: BLE001 - recovery command should report the failed action.
            executions.append({"action": action, "status": "failed", "reason": str(exc)})
            return {
                "status": "failed",
                "dry_run": False,
                "open_actions": current,
                "executed_count": len([item for item in executions if item.get("status") == "completed"]),
                "executions": executions,
                "reason": str(exc),
            }
        executions.append({"action": action, "status": "completed", "result": result})
    remaining = find_open_executable_actions(root)
    return {
        "status": "completed" if not remaining else "partial",
        "dry_run": False,
        "open_actions": remaining,
        "executed_count": len(executions),
        "executions": executions,
    }


def _ingestion_from_open_action(action: dict[str, object]) -> dict[str, object]:
    return {
        "status": "ingested",
        "handoff_type": action.get("handoff_type"),
        "canonical_path": action.get("canonical_path"),
        "next_action": action.get("action"),
        "executed_next_action": False,
    }


def _open_action_matches_ingestion(action: dict[str, object], ingestion: dict[str, object]) -> bool:
    return (
        action.get("handoff_type") == ingestion.get("handoff_type")
        and action.get("canonical_path") == ingestion.get("canonical_path")
        and action.get("action") == ingestion.get("next_action")
    )


def _format_open_actions(actions: list[dict[str, object]]) -> str:
    lines = ["Refusing to execute latest next action because open Harness actions exist:"]
    for index, action in enumerate(actions, start=1):
        lines.append("")
        lines.append(f"{index}. {action.get('action')}")
        lines.append(f"   handoff_type: {action.get('handoff_type')}")
        for key in ("candidate_id", "request_id", "batch_id", "run_id"):
            if action.get(key):
                lines.append(f"   {key}: {action.get(key)}")
        lines.append(f"   canonical_path: {action.get('canonical_path')}")
        lines.append(f"   created_at: {action.get('created_at')}")
    lines.append("")
    lines.append("Run `ml-autoresearch execute-open-actions` to recover in ledger order.")
    return "\n".join(lines)


def execute_outstanding_next_action(project_root: str | Path = Path(".")) -> AutonomyStepResult:
    """Execute the outstanding Harness-owned next action from the previous Autonomy Step result."""

    root = Path(project_root).resolve()
    result_path = root / "agent-work" / RESULT_FILENAME
    if not result_path.is_file():
        raise AutonomyStepError(f"previous Autonomy Step result not found: {result_path}")
    payload = json.loads(result_path.read_text())
    ingestion = payload.get("ingestion")
    if not isinstance(ingestion, dict):
        raise AutonomyStepError("previous Autonomy Step result has no ingested handoff")
    open_actions = find_open_executable_actions(root)
    if open_actions and not (len(open_actions) == 1 and _open_action_matches_ingestion(open_actions[0], ingestion)):
        raise AutonomyStepError(_format_open_actions(open_actions))

    execution = None
    status = str(payload.get("status", "ingested"))
    reason = payload.get("reason") if isinstance(payload.get("reason"), str) else None
    if ingestion.get("status") != "ingested":
        execution = {"status": "skipped", "executed": False, "reason": "previous handoff was not ingested"}
    elif ingestion.get("executed_next_action") is True and _executed_next_action_artifact_exists(root, ingestion):
        execution = {"status": "skipped", "executed": False, "reason": "next action already executed"}
    else:
        try:
            execution = execute_ingested_next_action(root, ingestion)
        except Exception as exc:  # noqa: BLE001 - preserve ingestion record and stop for human review.
            execution = {"status": "failed", "reason": str(exc)}
            status = "execution_failed"
            reason = str(exc)
        else:
            if execution.get("executed") is True:
                status = "ingested"
                reason = None
                ingestion["executed_next_action"] = True
                ingestion["next_action_result"] = execution

    result = AutonomyStepResult(
        status=status,
        workspace_root=str(root),
        agent_workspace=str(root / "agent-work"),
        prompt_path=str(root / "agent-work" / PROMPT_FILENAME),
        agent_command=_string_list(payload.get("agent_command")),
        agent_returncode=int(payload.get("agent_returncode", 0)),
        ingestion=ingestion,
        reason=reason,
        execution=execution,
    )
    write_result_file(root / "agent-work", result)
    return result


def _executed_next_action_artifact_exists(root: Path, ingestion: dict[str, object]) -> bool:
    """Return whether a recorded executed next action has its current canonical artifact."""

    if ingestion.get("handoff_type") == "candidate_submission" and ingestion.get("next_action") == "run_candidate":
        result = ingestion.get("next_action_result")
        if not isinstance(result, dict):
            return False
        run_status = result.get("run_status")
        if run_status == "accepted":
            return False
        if run_status in {"completed", "failed", "rejected", "smoke_failed"}:
            run_dir = result.get("run_dir")
            return isinstance(run_dir, str) and (Path(run_dir) / "run_metadata.json").is_file()
        return False
    if ingestion.get("handoff_type") == "experiment_batch_submission" and ingestion.get("next_action") == "run_experiment_batch":
        result = ingestion.get("next_action_result")
        if not isinstance(result, dict):
            return False
        batch_dir = result.get("batch_dir")
        return isinstance(batch_dir, str) and (Path(batch_dir) / "batch_metadata.json").is_file()
    if ingestion.get("handoff_type") == "evaluation_request" and ingestion.get("next_action") == "run_post_run_evaluation":
        request_path = _required_relative_path(root, ingestion, "canonical_path")
        from ml_autoresearch.evaluation_requests import _evaluation_id, validate_evaluation_request_file

        request = validate_evaluation_request_file(request_path)
        assert request.request_id is not None
        runs_root = _configured_runs_root(root)
        metadata_path = (
            runs_root
            / request.target_run_id
            / "outputs"
            / "evaluations"
            / _evaluation_id(request.request_id)
            / "evaluation_metadata.json"
        )
        return metadata_path.is_file()
    return True


def execute_ingested_next_action(root: Path, ingestion: dict[str, object]) -> dict[str, object]:
    """Execute at most one Harness-owned next action selected by a successful handoff ingestion."""

    from ml_autoresearch.research_loop_operations import ResearchLoopOperationError, execute_ingested_next_action as execute_next

    try:
        return execute_next(root, ingestion)
    except ResearchLoopOperationError as exc:
        raise AutonomyStepError(str(exc)) from exc


def _configured_runs_root(root: Path) -> Path:
    from ml_autoresearch.candidate_execution_config import load_candidate_execution_config

    return load_candidate_execution_config(root).runs_root


def _required_relative_path(root: Path, payload: dict[str, object], field: str) -> Path:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise AutonomyStepError(f"ingestion result missing {field}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise AutonomyStepError(f"ingestion result {field} must be a relative path inside the Research Workspace Root")
    return root / path


def _agent_command_argv(agent_command: str) -> list[str]:
    argv = shlex.split(agent_command)
    if not argv:
        raise AutonomyStepError("agent command must be non-empty")
    return argv


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _status_from_ingestion(ingestion: dict[str, object]) -> str:
    ingestion_status = ingestion.get("status")
    if ingestion_status == "ingested":
        return "ingested"
    if ingestion_status == "no_handoff":
        return "no_handoff"
    if ingestion_status == "ingestion_failed":
        return "ingestion_failed"
    return "ingestion_failed"
