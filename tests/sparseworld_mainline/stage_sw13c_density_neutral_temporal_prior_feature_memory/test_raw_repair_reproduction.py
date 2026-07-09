import csv
from pathlib import Path
P=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_density_neutral_temporal_prior_feature_memory/sw13c_raw_repair_reproduction_metrics.csv')
def test_raw_repair_reproduction():
    rows=list(csv.DictReader(P.open())); assert rows
    h6=[r for r in rows if r['horizon_s']=='6']
    assert any(r['perturbation_id']=='A1_drop_cam_front' and float(r['pred_gt_density_delta'])>0.1 for r in h6)
    assert any(r['perturbation_id']=='A10_drop_front_triplet' and float(r['pred_gt_density_delta'])>0.1 for r in h6)
