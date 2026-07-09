from pathlib import Path
REPORT = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/stage_sw14b_metric_gt_adapter_report.md').read_text().lower()
def test_no_false_claims():
    assert 'official benchmark' not in REPORT
    assert 'sota' not in REPORT
