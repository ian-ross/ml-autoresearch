import json
import subprocess
import sys
from pathlib import Path

from ml_autoresearch.runs import get_best_runs, get_run_summary, list_runs


def write_run(runs_root: Path, run_id: str, status: str, dice: float | None = None, *, reason: str | None = None) -> Path:
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True)
    metadata = {
        "run_id": run_id,
        "status": status,
        "created_at": f"2026-05-02T00:00:0{len(list(runs_root.iterdir()))}Z",
        "updated_at": f"2026-05-02T00:00:1{len(list(runs_root.iterdir()))}Z",
        "candidate_source": {"path": "/tmp/candidate"},
        "rejection_reason": reason if status == "rejected" else None,
        "smoke_failure_reason": reason if status == "smoke_failed" else None,
        "training_failure_reason": reason if status == "failed" else None,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata) + "\n")
    if dice is not None:
        outputs_dir = run_dir / "outputs"
        outputs_dir.mkdir()
        (outputs_dir / "final_metrics.json").write_text(json.dumps({"val/dice": dice, "val/iou": dice / 2}) + "\n")
    return run_dir


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "ml_autoresearch.cli", *args],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def test_list_runs_summarizes_local_run_artifacts_and_reports_corrupt_runs(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "run_b", "completed", 0.7)
    write_run(runs_root, "run_a", "rejected", reason="forbidden file")
    corrupt = runs_root / "run_c"
    corrupt.mkdir()
    (corrupt / "run_metadata.json").write_text("not json")

    summaries = list_runs(runs_root)

    assert [summary["run_id"] for summary in summaries] == ["run_a", "run_b", "run_c"]
    assert summaries[0]["status"] == "rejected"
    assert summaries[0]["reason"] == "forbidden file"
    assert summaries[1]["metrics"]["val/dice"] == 0.7
    assert summaries[2]["status"] == "corrupt"
    assert "error" in summaries[2]


def test_get_run_summary_reads_one_run_without_mlflow(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "run_done", "completed", 0.82)

    summary = get_run_summary(runs_root, "run_done")

    assert summary["run_id"] == "run_done"
    assert summary["status"] == "completed"
    assert summary["metrics"]["val/dice"] == 0.82
    assert summary["run_dir"] == str(runs_root / "run_done")


def test_get_best_runs_sorts_completed_runs_by_val_dice_descending_and_skips_unrankable(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "run_low", "completed", 0.2)
    write_run(runs_root, "run_failed", "failed", reason="boom")
    write_run(runs_root, "run_high", "completed", 0.9)
    write_run(runs_root, "run_no_metric", "completed")

    best = get_best_runs(runs_root)

    assert [summary["run_id"] for summary in best] == ["run_high", "run_low"]
    assert [summary["rank_metric"] for summary in best] == [0.9, 0.2]


def test_observation_cli_json_commands_are_thin_over_local_artifacts(tmp_path: Path):
    runs_root = tmp_path / "runs"
    write_run(runs_root, "run_low", "completed", 0.2)
    write_run(runs_root, "run_high", "completed", 0.9)

    listed = run_cli("list-runs", "--runs-root", str(runs_root), "--json")
    summary = run_cli("run-summary", "--runs-root", str(runs_root), "--run-id", "run_high", "--json")
    best = run_cli("get-best-runs", "--runs-root", str(runs_root), "--json")

    assert listed.returncode == 0, listed.stderr
    assert summary.returncode == 0, summary.stderr
    assert best.returncode == 0, best.stderr
    assert [item["run_id"] for item in json.loads(listed.stdout)] == ["run_high", "run_low"]
    assert json.loads(summary.stdout)["metrics"]["val/dice"] == 0.9
    assert [item["run_id"] for item in json.loads(best.stdout)] == ["run_high", "run_low"]
