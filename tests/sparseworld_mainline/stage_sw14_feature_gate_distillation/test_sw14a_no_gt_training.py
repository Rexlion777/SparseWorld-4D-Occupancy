import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14a_training_log.csv').open()))
def test_sw14a_no_gt_training():
    assert rows
    assert all(r['uses_gt_loss'] == 'False' for r in rows)
