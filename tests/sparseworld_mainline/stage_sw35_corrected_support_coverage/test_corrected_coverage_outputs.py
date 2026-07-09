from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw35_corrected_support_coverage"


def test_corrected_coverage_outputs_exist() -> None:
    required = [
        REPORTS / "corrected_query_to_pred_attribution_matrix.csv",
        REPORTS / "corrected_small_object_failure_localization.csv",
        REPORTS / "corrected_new_visible_failure_localization.csv",
        REPORTS / "corrected_joint_temporal_query_diagnosis_matrix.csv",
        REPORTS / "old_vs_corrected_support_proxy_comparison.md",
        REPORTS / "stage_sw35_corrected_support_coverage_report.md",
        REPORTS / "stage_sw35_corrected_support_coverage_report.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, f"Missing SW-3.5 outputs: {missing}"


def test_corrected_ff_coverage_is_high() -> None:
    payload = json.loads((REPORTS / "corrected_query_to_pred_attribution_matrix.json").read_text(encoding="utf-8"))
    assert payload["summary"]["mean_ff_geometric_coverage_ratio"] > 0.7
    assert payload["summary"]["mean_tp_semantic_active_coverage_ratio"] >= 0.99
