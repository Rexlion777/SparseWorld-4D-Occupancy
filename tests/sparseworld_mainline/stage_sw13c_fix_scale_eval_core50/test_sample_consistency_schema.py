import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_sample_consistency.csv').open()))
def test_sample_consistency_schema():
    assert rows
    need={'front_false_free_improved','future_false_free_improved','density_safe','front_local_safe','false_positive_safe','wrong_class_safe','joint_success'}
    assert need.issubset(rows[0].keys())
