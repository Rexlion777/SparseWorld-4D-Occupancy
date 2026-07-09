import csv
import os
from pathlib import Path


def _resolve_root() -> Path:
    env = os.environ.get("CV_LIDAR_PROJECT_ROOT")
    if env:
        path = Path(env)
        if path.exists():
            return path
    return Path(__file__).resolve().parents[3]


ROOT = _resolve_root()
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training/sw91_short_train_metrics.csv"


def test_train_metrics_schema() -> None:
    rows = list(csv.DictReader(PATH.open("r", encoding="utf-8")))
    assert rows, "empty short-train metrics csv"
    required = [
        "experiment_id",
        "completed_iters",
        "planned_iters",
        "completed",
        "meets_500_iter_requirement",
        "stop_reason",
    ]
    for key in required:
        assert key in rows[0], f"missing train metric column: {key}"
    h2_rows = [row for row in rows if row["experiment_id"] in {"P1_H2_only_low_1000iter", "P2_H2_warmup_1000iter"}]
    assert h2_rows, "missing P1/P2 rows"
    assert any(
        row["completed"] == "True" or row["stop_reason"] in {"budget_exhausted", "oom", "runtime_error", "non_finite_loss"}
        for row in h2_rows
    ), "expected completed or explicit stop reason for P1/P2"
