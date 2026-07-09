from pathlib import Path
BASE=Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50')
def test_outputs_exist():
    names=['sw13c_fix_frontcap_inherited_scale_audit.json','sw13c_fix_frontcap_proxy_definition_audit.json','sw13c_fix_frontcap_frozen_candidate_manifest.json','sw13c_fix_frontcap_metrics.csv','sw13c_fix_frontcap_metric_summary.csv','sw13c_fix_frontcap_density_audit.csv','sw13c_fix_frontcap_candidate_selection.json','sw13c_fix_frontcap_eval_core50_decision.json','stage_sw13c_fix_frontcap_eval_core50_report.md']
    for name in names:
        p=BASE/name
        assert p.exists() and p.stat().st_size>0, name
