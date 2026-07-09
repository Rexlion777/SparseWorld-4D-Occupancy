from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_route_outputs_schema() -> None:
    train_rows = read_rows(REPORTS / "routeA_scaled_h2_train_metrics.csv")
    eval_rows = read_rows(REPORTS / "routeA_scaled_h2_eval_metrics.csv")
    assert train_rows, "empty routeA train metrics"
    assert eval_rows, "empty routeA eval metrics"
    row = train_rows[0]
    for key in ["experiment_id", "target_total_iter", "completed_total_iter", "stop_reason", "completed"]:
        assert key in row and row[key] != "", f"missing {key}"
    eval_row = eval_rows[0]
    assert "checkpoint_name" in eval_row
    assert "gate_safe" in eval_row or "skipped_reason" in eval_row
