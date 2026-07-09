import json
from pathlib import Path
def test_protocol_fingerprint():
    obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13c_fix_frontcap_eval_core500/sw13c_fix_frontcap_eval_core500_protocol_fingerprint.json').read_text())
    assert obj['protocol_fingerprint_match'] is True
