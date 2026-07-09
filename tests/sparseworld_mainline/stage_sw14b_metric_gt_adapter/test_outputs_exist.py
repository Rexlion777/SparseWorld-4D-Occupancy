from pathlib import Path
ROOT = Path('/home/rexlion/ComputerVision/cv_lidar_transition')
REPORTS = ROOT / 'reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter'
def test_outputs_exist():
    required = [
        'sw14b_inherited_sw14a_exact_parity_closure.json',
        'sw14b_split_manifest.json',
        'sw14b_adapter_architecture.json',
        'sw14b_postprocess_chain_audit.json',
        'sw14b_final_decision.json',
        'stage_sw14b_metric_gt_adapter_report.md',
    ]
    for name in required:
        assert (REPORTS / name).exists(), name
