import csv
from pathlib import Path
def test_preservation_metric():
    rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_metric_summary.csv').open())); assert rows
    assert 'protected_zone_preservation_ratio' in rows[0] and 'pruning_from_protected_ratio' in rows[0]
