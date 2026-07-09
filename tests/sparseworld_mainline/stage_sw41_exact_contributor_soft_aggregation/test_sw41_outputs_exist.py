from pathlib import Path


PROJECT_ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation"


def test_sw41_required_outputs_exist():
    required = [
        "baseline_replay_equivalence.json",
        "exact_contributor_waterfall.csv",
        "nearest_support_offset_by_error_type.csv",
        "neighbor_leakage_analysis.csv",
        "get_occ_hard_mode_equivalence.json",
        "soft_splat_ablation_metrics.csv",
        "adaptive_gate_ablation_metrics.csv",
        "class_aware_aggregation_ablation_metrics.csv",
        "combined_repair_candidate_metrics.csv",
        "best_candidate_20sample_validation.csv",
        "sw41_repair_candidate_decision.json",
        "stage_sw41_exact_contributor_soft_aggregation_report.json",
        "stage_sw41_exact_contributor_soft_aggregation_report.md",
    ]
    missing = [name for name in required if not (REPORTS_DIR / name).exists()]
    assert not missing, f"missing SW-4.1 outputs: {missing}"
