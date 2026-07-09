from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"


def test_temporal_metrics_schema() -> None:
    rows = list(csv.DictReader((REPORTS / "temporal_horizon_metrics_aggregate.csv").open(encoding="utf-8")))
    assert rows, "aggregate horizon metrics empty"
    horizons = {int(float(row["horizon_s"])) for row in rows}
    assert 0 in horizons
    assert any(h > 0 for h in horizons)
    required_cols = {
        "mean_occupied_iou",
        "mean_semantic_miou",
        "mean_false_free_rate",
        "mean_false_occupied_rate",
        "mean_pred_gt_occupied_ratio",
        "sample_count",
    }
    for col in required_cols:
        assert col in rows[0], f"missing {col}"
