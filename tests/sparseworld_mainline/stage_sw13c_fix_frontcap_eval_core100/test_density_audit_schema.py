import csv
from pathlib import Path
rows=list(csv.DictReader(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core100/sw13c_fix_frontcap_eval_core100_density_audit.csv').open()))
def test_density_audit_schema():
    assert rows
    need={'candidate_name','front_local_density_proxy_after_mean','front_local_safe_rate','front_pruning_from_protected_ratio','frontcap_target_met_rate','front_local_density_risk_after'}
    assert need.issubset(rows[0].keys())
