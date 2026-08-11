"""Durable managed execution records for long Candidate Runs."""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

EXECUTION_RECORD_NAME = "execution.json"


def start_run_supervisor(
    run_dir: str | Path,
    *,
    command: list[str],
    log_path: str | Path,
    backend: str,
) -> dict[str, object]:
    """Launch a detached supervisor for an already-created stable Run."""

    path = Path(run_dir)
    log = Path(log_path)
    log.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path / ".execution.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        existing = read_execution_record(path)
        metadata = json.loads((path / "run_metadata.json").read_text())
        existing_supervisor = existing.get("supervisor") if isinstance(existing, dict) else None
        existing_pid = existing_supervisor.get("pid") if isinstance(existing_supervisor, dict) else None
        active_states = {"supervisor_running", "training", "container_starting", "container_running"}
        if (
            metadata.get("status") in {"accepted", "training"}
            and isinstance(existing, dict)
            and existing.get("state") in active_states
            and isinstance(existing_pid, int)
            and _pid_alive(existing_pid)
        ):
            return {**existing, "already_running": True}
        if metadata.get("status") != "accepted":
            raise RuntimeError(
                f"managed Run supervisor requires accepted status; got {metadata.get('status')!r} for {path.name}"
            )
        with log.open("ab") as log_file, Path(os.devnull).open("rb") as stdin:
            process = subprocess.Popen(
                command,
                stdin=stdin,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        record = update_execution_record(
            path,
            state="supervisor_running",
            operation="train_research_problem",
            backend=backend,
            supervisor={"pid": process.pid},
            log_path=str(log),
            command=command,
            started_at=_now_iso(),
        )
        return record


def await_supervisor_registration(
    run_dir: str | Path,
    *,
    pid: int,
    timeout_seconds: float = 5.0,
) -> None:
    """Let the launcher persist the child PID before the supervisor mutates state."""

    deadline = time.monotonic() + timeout_seconds
    record_path = Path(run_dir) / EXECUTION_RECORD_NAME
    while time.monotonic() < deadline:
        if record_path.is_file():
            try:
                record = json.loads(record_path.read_text())
            except json.JSONDecodeError:
                record = {}
            supervisor = record.get("supervisor") if isinstance(record, dict) else None
            if isinstance(supervisor, dict) and supervisor.get("pid") == pid:
                return
        time.sleep(0.01)


def update_execution_record(run_dir: str | Path, **changes: object) -> dict[str, object]:
    """Atomically create or update one Harness-owned execution record."""

    path = Path(run_dir)
    record_path = path / EXECUTION_RECORD_NAME
    if record_path.is_file():
        try:
            current = json.loads(record_path.read_text())
        except json.JSONDecodeError:
            current = {}
    else:
        current = {}
    record = {
        "schema_version": 1,
        "run_id": path.name,
        **(current if isinstance(current, dict) else {}),
        **changes,
        "updated_at": _now_iso(),
    }
    temporary = path / f".{EXECUTION_RECORD_NAME}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, record_path)
    return record


def cleanup_recorded_containers(run_dir: str | Path) -> dict[str, object] | None:
    """Remove durable Docker containers only after terminal Run finalization."""

    path = Path(run_dir)
    record = read_execution_record(path)
    if not isinstance(record, dict) or record.get("backend") != "docker":
        return record
    raw_containers = record.get("containers")
    containers = (
        [dict(item) for item in raw_containers if isinstance(item, dict)]
        if isinstance(raw_containers, list)
        else []
    )
    cleanup_errors: list[str] = []
    for container in containers:
        name = container.get("name")
        if not isinstance(name, str):
            continue
        try:
            completed = subprocess.run(
                ["docker", "rm", "-f", name],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError as exc:
            container["cleanup_state"] = "failed"
            cleanup_errors.append(f"{name}: {exc}")
            continue
        if completed.returncode == 0 or "No such container" in (completed.stderr or completed.stdout):
            container["cleanup_state"] = "removed"
        else:
            container["cleanup_state"] = "failed"
            cleanup_errors.append(f"{name}: {(completed.stderr or completed.stdout).strip()}")
    changes: dict[str, object] = {
        "containers": containers,
        "container_cleanup": "completed" if not cleanup_errors else "partial",
    }
    if cleanup_errors:
        changes["container_cleanup_errors"] = cleanup_errors[:16]
    return update_execution_record(path, **changes)


def read_execution_record(run_dir: str | Path) -> dict[str, object] | None:
    """Read one execution record and derive supervisor liveness."""

    path = Path(run_dir) / EXECUTION_RECORD_NAME
    if not path.is_file():
        return None
    try:
        record = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return {"state": "corrupt", "error": f"cannot read execution record: {exc}"}
    if not isinstance(record, dict):
        return {"state": "corrupt", "error": "execution record must contain a mapping"}
    observed = dict(record)
    supervisor = observed.get("supervisor")
    pid = supervisor.get("pid") if isinstance(supervisor, dict) else None
    if isinstance(pid, int):
        observed["supervisor_alive"] = _pid_alive(pid)
        if observed.get("state") in {"supervisor_running", "training"} and not observed["supervisor_alive"]:
            observed["observed_state"] = "supervisor_exited"
    active_container = observed.get("active_container")
    container_name = active_container.get("name") if isinstance(active_container, dict) else None
    if observed.get("backend") == "docker" and isinstance(container_name, str):
        container_state = _docker_container_state(container_name)
        observed["container_observation"] = container_state
        if container_state.get("running") is True:
            observed["observed_state"] = "container_running"
        elif container_state.get("status") == "exited":
            observed["observed_state"] = "container_exited"
    return observed


def _docker_container_state(container_name: str) -> dict[str, object]:
    try:
        completed = subprocess.run(
            ["docker", "inspect", "--format", "{{json .State}}", container_name],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as exc:
        return {"status": "unavailable", "error": str(exc)}
    if completed.returncode != 0:
        return {"status": "missing", "error": (completed.stderr or completed.stdout).strip()}
    try:
        state = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return {"status": "corrupt", "error": str(exc)}
    if not isinstance(state, dict):
        return {"status": "corrupt", "error": "Docker State must be a mapping"}
    return {
        "status": state.get("Status"),
        "running": state.get("Running"),
        "exit_code": state.get("ExitCode"),
        "started_at": state.get("StartedAt"),
        "finished_at": state.get("FinishedAt"),
        "error": state.get("Error"),
        "oom_killed": state.get("OOMKilled"),
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
