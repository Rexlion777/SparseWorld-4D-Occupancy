from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling/sw11_risk_targeted_h2_decision.json"


def test_decision_schema() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["decision_type"] in {
        "B1_targeting_failed",
        "B2_h2_lights_up_but_no_native_assignment",
        "B3_native_assignment_improves_but_no_final_contributor",
        "B4_final_contributor_improves_but_semantic_fails",
        "B5_full_survival_improves",
        "B6_density_or_false_positive_risk",
        "B7_instrumentation_blocked",
    }
    assert "summary" in payload and payload["summary"]
    assert "next_unique_action" in payload and payload["next_unique_action"]
