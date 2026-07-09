from pathlib import Path
BASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50')
def test_outputs_exist():
    names=['sw13c_fix_scale_frozen_candidate_manifest.json','sw13c_fix_scale_replay_manifest.csv','sw13c_fix_scale_metrics.csv','sw13c_fix_scale_metric_summary.csv','sw13c_fix_scale_front_local_density_audit.csv','sw13c_fix_scale_sample_consistency.csv','sw13c_fix_scale_eval_core50_decision.json','stage_sw13c_fix_scale_eval_core50_report.md']
    for name in names:
        p=BASE/name
        assert p.exists() and p.stat().st_size>0, name
