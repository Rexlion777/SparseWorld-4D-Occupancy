from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment/sw101_alignment_audit_decision.json"


def test_decision_schema() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["decision_type"] in {
        "A1_proxy_decode_mismatch_fix_h2",
        "A2_semantic_gate_coupling_needed",
        "A3_hard_assignment_bottleneck_confirmed",
        "A4_aggregation_conflict_confirmed",
        "A5_h2_target_not_focusing_failure",
        "A6_instrumentation_incomplete",
        "A7_stop_training_package_project",
    }
    for key in [
        "summary",
        "recommended_next_route",
        "main_mismatch_type",
        "alignment_headline",
        "taxonomy_headline",
        "next_unique_action",
    ]:
        assert key in payload and payload[key], f"missing {key}"
