import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_density_audit.csv').open()))
def test_density_audit_schema():
    assert rows
    need={'front_occ_native','front_occ_before_cap','front_occ_after_cap','front_local_density_proxy_before','front_local_density_proxy_after','front_pred_gt_density_delta_before','front_pred_gt_density_delta_after','front_local_density_risk_before','front_local_density_risk_after'}
    assert need.issubset(rows[0].keys())
