from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation"


def test_sw5_required_outputs_exist() -> None:
    required = [
        REPORTS / "sensor_perturbation_catalog.json",
        REPORTS / "sw5_run_manifest.json",
        REPORTS / "sw5_sample_manifest.csv",
        REPORTS / "sw5_clean_baseline_replay.json",
        REPORTS / "sw5_clean_baseline_metrics.csv",
        REPORTS / "sw5_occupancy_metrics_aggregate.csv",
        REPORTS / "sw5_query_support_metrics_aggregate.csv",
        REPORTS / "sw5_contributor_metrics_aggregate.csv",
        REPORTS / "sw5_failure_taxonomy.json",
        REPORTS / "sw5_robustness_score_ranking.json",
        REPORTS / "stage_sw5_sensor_aware_failure_propagation_report.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, f"missing outputs: {missing}"


def test_sw5_manifest_schema() -> None:
    payload = json.loads((REPORTS / "sw5_run_manifest.json").read_text(encoding="utf-8"))
    assert payload["stage"] == "SW-5"
    assert payload["effective_samples"] >= 5
    assert len(payload["effective_perturbations"]) >= 5

