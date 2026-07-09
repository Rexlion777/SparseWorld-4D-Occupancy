from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"


def test_outputs_exist() -> None:
    required = [
        REPORTS / "sw10_time_budget_manifest.json",
        REPORTS / "sw91_result_digest_for_sw10.json",
        REPORTS / "sw10_route_selection.json",
        REPORTS / "sw10_result_driven_decision.json",
        REPORTS / "stage_sw10_result_driven_contributor_routing_report.md",
    ]
    for path in required:
        assert path.exists(), f"missing {path}"
