from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing/sw12a_soft_neighbor_routing_decision.json"


def test_decision_schema() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["decision_type"] in {
        "C1_no_routing_signal",
        "C2_oracle_only_signal",
        "C3_targeted_but_unsafe",
        "C4_safe_soft_neighbor_candidate",
        "C5_gate_bottleneck",
        "C6_scatter_aggregation_bottleneck",
        "C7_semantic_competition_bottleneck",
        "C8_instrumentation_blocked",
    }
    assert payload["summary"]
    assert payload["next_unique_action"]
