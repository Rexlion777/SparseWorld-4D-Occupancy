import json
from pathlib import Path
path = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_density_clean_rescue/sw14b_density_clean_rescue_final_decision.json')
def test_rescue_decision_schema():
    payload = json.loads(path.read_text())
    required = {'decision','clean_drift_attribution','density_attribution','alpha_scale_sweep','density_aware_clamp','conservative_retrain','frozen_eval_debug_verification','best_rescue_config','whether_can_proceed_to_eval_core100','whether_can_use_in_resume','safe_wording'}
    assert required.issubset(payload.keys())
