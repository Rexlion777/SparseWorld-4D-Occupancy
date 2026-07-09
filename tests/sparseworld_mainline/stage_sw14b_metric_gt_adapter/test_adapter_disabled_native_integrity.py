import json
from pathlib import Path
PATH = Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14b_metric_gt_adapter/sw14b_adapter_architecture.json')
def test_adapter_disabled_native_integrity():
    payload = json.loads(PATH.read_text())
    assert payload['adapter_disabled_path_unchanged'] is True
