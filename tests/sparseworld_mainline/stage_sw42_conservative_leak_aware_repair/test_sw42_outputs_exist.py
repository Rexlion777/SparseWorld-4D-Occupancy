from pathlib import Path

BASE = Path("/mnt/d/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw42_conservative_leak_aware_repair")


def test_sw42_outputs_exist():
    required = [
        "baseline_k7_replay_equivalence.json",
        "sw42_pareto_acceptance_criteria.json",
        "conservative_soft_splat_metrics_5sample.csv",
        "leak_aware_repair_metrics_5sample.csv",
        "false_positive_guard_metrics.csv",
        "small_object_local_repair_metrics.csv",
        "new_visible_conservative_repair_metrics.csv",
        "sw42_conservative_repair_decision.json",
        "stage_sw42_conservative_leak_aware_repair_report.json",
        "stage_sw42_conservative_leak_aware_repair_report.md",
    ]
    for name in required:
        assert (BASE / name).exists(), name
    assert (BASE / "combined_conservative_candidate_metrics_20sample.csv").exists() or (BASE / "sw42_20sample_pareto_validation.csv").exists()
