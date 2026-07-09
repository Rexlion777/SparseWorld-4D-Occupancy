import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100/sw13c_fix_frontcap_eval_core100_sample_consistency.csv').open()))
def test_sample_consistency_schema():
    assert rows
    need={'candidate_name','front_false_free_improved_rate','future_false_free_improved_rate','density_safe_rate','front_local_safe_rate','joint_success_rate'}
    assert need.issubset(rows[0].keys())
