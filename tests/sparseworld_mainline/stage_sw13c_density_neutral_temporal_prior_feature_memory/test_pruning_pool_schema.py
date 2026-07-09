from pathlib import Path
def test_pruning_pool_schema():
    text=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_pruning_pool_definition.md').read_text().lower(); assert 'gt is not used' in text
