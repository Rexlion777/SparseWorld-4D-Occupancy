import json
from pathlib import Path
def test_frozen_candidate_manifest():
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_scale_eval_core50/sw13c_fix_scale_frozen_candidate_manifest.json').read_text())
    assert obj['selection_frozen'] is True
    assert obj['no_reselection'] is True
