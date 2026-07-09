from pathlib import Path
def test_front_local_density_audit_exists():
    p=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_no_gt_density_budget/sw13c_fix_front_local_density_audit.csv')
    assert p.exists() and p.stat().st_size>0
