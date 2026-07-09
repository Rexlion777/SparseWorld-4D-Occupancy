from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/sw12b_risk_gated_occupancy_repair_decision.json"


def test_decision_schema() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["decision_type"] in {
        "D1_branchB_promising_mainline",
        "D2_branchA_safe_candidate",
        "D3_branchA_targeted_but_unsafe_branchB_promising",
        "D4_both_promising",
        "D5_both_no_signal",
        "D6_branchB_overfits_or_density_explodes",
        "D7_teacher_quality_blocked",
    }
    assert payload["summary"]
    assert payload["next_unique_action"]
