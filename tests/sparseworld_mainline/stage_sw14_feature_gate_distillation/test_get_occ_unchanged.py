import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_feature_path_audit.json').read_text())
def test_get_occ_unchanged():
    assert obj['no_get_occ_modification'] is True
    assert len(obj['get_occ_sha256']) == 64
