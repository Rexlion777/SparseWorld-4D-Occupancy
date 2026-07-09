from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12a_soft_neighbor_getocc_routing"


def test_outputs_exist() -> None:
    required = [
        REPORTS / "sw12a_time_budget_manifest.json",
        REPORTS / "sw11_digest_for_sw12a.json",
        REPORTS / "sw12a_getocc_variant_manifest.json",
        REPORTS / "sw12a_getocc_code_diff_summary.md",
        REPORTS / "sw12a_replay_manifest.csv",
        REPORTS / "sw12a_variant_sweep_metrics.csv",
        REPORTS / "sw12a_pareto_candidates.csv",
        REPORTS / "sw12a_survival_chain_before_after.csv",
        REPORTS / "sw12a_soft_neighbor_routing_decision.json",
        REPORTS / "sw12b_recommended_route_plan.json",
        REPORTS / "stage_sw12a_soft_neighbor_getocc_routing_report.md",
    ]
    for path in required:
        assert path.exists(), f"missing {path}"
