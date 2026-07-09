from pathlib import Path
BASE = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13b_counterfactual_density_constrained_feature_memory')

def test_outputs_exist() -> None:
    required = [
        'sw13b_inherited_sw13a_summary.json',
        'sw13b_execution_manifest.json',
        'sw13b_paired_replay_dump_manifest.csv',
        'sw13b_delta_filter_variant_manifest.json',
        'sw13b_selection_rule.md',
        'sw13b_non_oracle_candidate_selection.json',
        'sw13b_replay_manifest.csv',
        'sw13b_delta_filter_metrics.csv',
        'sw13b_delta_filter_aggregate_metrics.csv',
        'sw13b_metric_summary.csv',
        'sw13b_counterfactual_density_constrained_decision.json',
        'stage_sw13b_counterfactual_density_constrained_feature_memory_report.md',
    ]
    for name in required:
        path = BASE / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
