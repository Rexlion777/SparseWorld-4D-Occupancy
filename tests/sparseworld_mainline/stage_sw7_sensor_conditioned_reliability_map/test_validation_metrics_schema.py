from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CORR = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/reliability_error_correlation.csv"
ABL = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/reliability_component_ablation.csv"


def test_validation_metrics_schema() -> None:
    with CORR.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert {"perturbation_id", "sample_index", "horizon_s", "component_name", "risk_error_correlation"}.issubset(rows[0])
    with ABL.open("r", encoding="utf-8", newline="") as f:
        ab_rows = list(csv.DictReader(f))
    assert ab_rows
    names = {r["component_name"] for r in ab_rows}
    assert {"semantic", "support", "contributor", "sensor", "temporal", "full_reliability"}.issubset(names)
