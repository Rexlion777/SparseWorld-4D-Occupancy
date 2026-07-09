from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"


def test_sw2_core_outputs_exist() -> None:
    required = [
        REPORTS / "sw2_run_manifest.json",
        REPORTS / "temporal_horizon_metrics_aggregate.csv",
        REPORTS / "temporal_horizon_metrics_per_sample.csv",
        REPORTS / "class_group_horizon_metrics.csv",
        REPORTS / "query_axis_semantics_audit.json",
        REPORTS / "query_coverage_metrics_sample_subset.csv",
        REPORTS / "joint_temporal_query_diagnosis_matrix.csv",
        REPORTS / "stage_sw2_temporal_query_diagnosis_report.md",
        REPORTS / "stage_sw2_temporal_query_diagnosis_report.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, f"Missing SW-2 outputs: {missing}"


def test_sw2_has_at_least_five_effective_samples() -> None:
    manifest = json.loads((REPORTS / "sw2_run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["effective_subset_size"] >= 5
    assert manifest["forward_success_count"] >= 5
