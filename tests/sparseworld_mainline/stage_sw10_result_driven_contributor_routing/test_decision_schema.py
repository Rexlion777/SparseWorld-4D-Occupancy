from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw10_result_driven_contributor_routing/sw10_result_driven_decision.json"


def test_decision_schema() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["selected_route"] in {"A", "B", "C", "D", "E"}
    assert payload["decision_type"] in {
        "R1_scale_h2_success",
        "R2_h2_proxy_alignment_needed",
        "R3_native_aligned_h2_promising",
        "R4_getocc_variant_promising",
        "R5_getocc_variant_unsafe",
        "R6_guard_repair_success",
        "R7_trainability_blocked",
        "R8_architecture_bottleneck_strengthened",
        "R9_package_project",
    }
    assert "best_tradeoff" in payload
    assert "next_unique_action" in payload
