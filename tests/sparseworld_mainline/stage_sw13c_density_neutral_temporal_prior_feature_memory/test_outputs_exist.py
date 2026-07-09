from pathlib import Path
BASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory')
def test_outputs_exist():
    for n in ['sw13c_inherited_sw13a_sw13b_summary.json','sw13c_raw_repair_manifest.csv','sw13c_temporal_prior_manifest.json','sw13c_protected_zone_variant_manifest.json','sw13c_pruning_pool_definition.md','sw13c_density_neutral_metrics.csv','sw13c_metric_summary.csv','sw13c_density_neutral_temporal_prior_decision.json','stage_sw13c_density_neutral_temporal_prior_feature_memory_report.md']:
        p=BASE/n; assert p.exists() and p.stat().st_size>0, n
