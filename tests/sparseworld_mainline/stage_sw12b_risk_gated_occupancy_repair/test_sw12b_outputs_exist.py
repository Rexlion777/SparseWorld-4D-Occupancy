from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair"


def test_outputs_exist() -> None:
    required = [
        REPORTS / "sw12b_execution_manifest.json",
        REPORTS / "sw12a_digest_for_sw12b.json",
        REPORTS / "sw12b_eval_protocol.json",
        REPORTS / "branchA_routing_replay_metrics.csv",
        REPORTS / "branchA_routing_candidates.csv",
        REPORTS / "branchB_teacher_cache_manifest.csv",
        REPORTS / "branchB_pair_builder_manifest.csv",
        REPORTS / "branchB_gradient_check.csv",
        REPORTS / "branchB_20iter_smoke_metrics.csv",
        REPORTS / "sw12b_risk_gated_occupancy_repair_decision.json",
        REPORTS / "stage_sw12b_risk_gated_occupancy_repair_report.md",
    ]
    for path in required:
        assert path.exists(), f"missing {path}"
