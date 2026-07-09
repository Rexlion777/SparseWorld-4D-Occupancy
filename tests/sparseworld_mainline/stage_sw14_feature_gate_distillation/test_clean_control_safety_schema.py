import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_clean_control_safety.csv').open()))
def test_clean_control_safety_schema():
    assert rows
    need={'case','clean_metric_drift','clean_density_drift','clean_false_positive_delta','alpha_mean_on_clean','alpha_mean_on_non_degraded_cameras'}
    assert need.issubset(rows[0].keys())
