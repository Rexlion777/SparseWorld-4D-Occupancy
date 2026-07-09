from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw3_query_support_validation"


def test_sw3_core_outputs_exist() -> None:
    required = [
        REPORTS / "semantic_occ_dependency_trace.json",
        REPORTS / "support_tensor_candidate_manifest.json",
        REPORTS / "coordinate_mapping_variant_scores.csv",
        REPORTS / "query_to_pred_attribution_matrix.csv",
        REPORTS / "query_causal_controllability_tests.json",
        REPORTS / "small_object_failure_localization.csv",
        REPORTS / "new_visible_failure_localization.csv",
        REPORTS / "sw3_query_support_resolution_decision.json",
        REPORTS / "stage_sw3_query_support_validation_report.md",
        REPORTS / "stage_sw3_query_support_validation_report.json",
    ]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, f"Missing SW-3 outputs: {missing}"


def test_sw3_effective_sample_count() -> None:
    payload = json.loads((REPORTS / "sw3_run_manifest.json").read_text(encoding="utf-8"))
    assert payload["effective_sample_count"] >= 5
    assert 0 in payload["horizons"]
    assert any(h > 0 for h in payload["horizons"])
