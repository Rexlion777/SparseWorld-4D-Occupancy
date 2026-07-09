from pathlib import Path

BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair")


def test_outputs_exist() -> None:
    required = [
        "sw12c_execution_manifest.json",
        "sw12c_progress_state.json",
        "sw12b_digest_for_sw12c.json",
        "branchA_A2_100iter_train_metrics.csv",
        "branchA_A2_100iter_eval.csv",
        "branchA_A2_100iter_decision.json",
        "branchB_residual_mask_audit.csv",
        "branchB_residual_pair_distribution.csv",
        "branchB_residual_gradient_check.csv",
        "branchB_residual_20iter_smoke_metrics.csv",
        "branchB_residual_smoke_decision.json",
        "sw12c_decision.json",
        "stage_sw12c_safe_routing_and_residual_repair_report.md",
    ]
    for name in required:
        path = BASE / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
