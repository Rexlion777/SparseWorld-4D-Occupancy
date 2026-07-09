import json
from pathlib import Path
def test_frozen_candidate_manifest():
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core50/sw13c_fix_frontcap_frozen_candidate_manifest.json').read_text())
    assert obj['frozen_candidates'] is True
