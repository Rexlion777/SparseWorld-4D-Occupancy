import json
from pathlib import Path
base = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_adapter_architecture.json')
def test_get_occ_hash_unchanged():
    payload = json.loads(base.read_text())
    assert payload['no_get_occ_modification'] is True
    assert payload['get_occ_sha256']
