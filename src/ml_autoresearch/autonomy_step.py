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

DEFAULT_AGENT_COMMAND = "pi"
RESULT_FILENAME = "autonomy-step-result.json"
PROMPT_FILENAME = "prompt.txt"


class AutonomyStepError(ValueError):
    """Raised when an Autonomy Step cannot be configured safely."""


@dataclass(frozen=True)
class AutonomyStepResult:
    """Machine-readable result for one Autonomy Step."""

    status: str
    project_root: str
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
            "project_root": self.project_root,
            "agent_workspace": self.agent_workspace,
            "prompt_path": self.prompt_path,
            "agent_command": self.agent_command,
            "agent_returncode": self.agent_returncode,
            "ingestion": self.ingestion,
            "execution": self.execution,
            "reason": self.reason,
        }


def load_configured_agent_command(project_root: Path) -> str | None:
    """Return optional Autonomy Step agent command from agent-boundary.toml."""

    config_path = project_root / "agent-boundary.toml"
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
    prompt_path = workspace / PROMPT_FILENAME
    prompt_path.write_text(render_autonomy_step_prompt())

    configured_command = agent_command or load_configured_agent_command(root) or DEFAULT_AGENT_COMMAND
    command = _agent_command_argv(configured_command) + ["-p", f"@{PROMPT_FILENAME}"]
    completed = subprocess.run(command, cwd=workspace, check=False)

    if completed.returncode != 0:
        result = AutonomyStepResult(
            status="agent_failed",
            project_root=str(root),
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
        project_root=str(root),
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


def render_autonomy_step_prompt() -> str:
    """Render the one-step prompt written inside the Agent Workspace."""

    return (
        "You are inside the ML Autoresearch Agent Control Boundary for exactly one Autonomy Step.\n"
        "\n"
        "Read AGENTS.md first for the boundary path map and writable handoff locations.\n"
        "Use the campaign-manager skill as the top-level workflow entry point.\n"
        "Work one step at a time: choose and complete exactly one primary research handoff outcome.\n"
        "Do not continue to a second primary outcome after producing the first one.\n"
        "\n"
        "Allowed primary handoff outcomes (choose exactly one if you hand off work):\n"
        "- one Candidate Submission under submissions/\n"
        "- one Research Note under research-notes/\n"
        "- one Capability Request under capability-requests/\n"
        "- one Evaluation Request under evaluation-requests/\n"
        "- one Campaign Report under campaign-reports/\n"
        "\n"
        "Multiple primary handoff outcomes are forbidden in one Autonomy Step. If blocked, write one Campaign Report "
        "or one Capability Request, not both. If no useful handoff is safe, stop without fabricating artifacts.\n"
        "Use ml-autoresearch-agent, not ml-autoresearch, for allowed inner-boundary commands.\n"
    )


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


def execute_ingested_next_action(root: Path, ingestion: dict[str, object]) -> dict[str, object]:
    """Execute at most one Harness-owned next action selected by a successful handoff ingestion."""

    handoff_type = ingestion.get("handoff_type")
    next_action = ingestion.get("next_action")
    if handoff_type == "candidate_submission" and next_action == "run_candidate":
        candidate_path = _required_relative_path(root, ingestion, "canonical_path")
        from ml_autoresearch.execution import NativeBackend
        from ml_autoresearch.runs import submit_candidate

        run = submit_candidate(
            candidate_path,
            root / "runs",
            backend=NativeBackend(),
            ledger_path=root / "research-ledger.jsonl",
            require_proposal=True,
        )
        return {
            "status": "completed",
            "executed": True,
            "action": "run_candidate",
            "run_id": run.run_id,
            "run_dir": str(run.run_dir),
            "run_status": run.status.value,
            "rejection_reason": run.rejection_reason,
            "failure_classification": run.failure_classification.value if run.failure_classification is not None else None,
        }
    if handoff_type == "evaluation_request" and next_action == "run_post_run_evaluation":
        request_path = _required_relative_path(root, ingestion, "canonical_path")
        from ml_autoresearch.evaluation_requests import run_post_run_evaluation

        evaluation = run_post_run_evaluation(
            request_path,
            runs_root=root / "runs",
            ledger_path=root / "research-ledger.jsonl",
        )
        return {
            "status": "completed",
            "executed": True,
            "action": "run_post_run_evaluation",
            "evaluation_id": evaluation["evaluation_id"],
            "evaluation": evaluation["evaluation"],
            "ledger_events": evaluation["ledger_events"],
        }
    return {"status": "skipped", "executed": False, "reason": f"no executable Harness action for {handoff_type!r}"}


def _required_relative_path(root: Path, payload: dict[str, object], field: str) -> Path:
    value = payload.get(field)
    if not isinstance(value, str) or not value:
        raise AutonomyStepError(f"ingestion result missing {field}")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise AutonomyStepError(f"ingestion result {field} must be a relative path inside the project root")
    return root / path


def _agent_command_argv(agent_command: str) -> list[str]:
    argv = shlex.split(agent_command)
    if not argv:
        raise AutonomyStepError("agent command must be non-empty")
    return argv


def _status_from_ingestion(ingestion: dict[str, object]) -> str:
    ingestion_status = ingestion.get("status")
    if ingestion_status == "ingested":
        return "ingested"
    if ingestion_status == "no_handoff":
        return "no_handoff"
    if ingestion_status == "ingestion_failed":
        return "ingestion_failed"
    return "ingestion_failed"
