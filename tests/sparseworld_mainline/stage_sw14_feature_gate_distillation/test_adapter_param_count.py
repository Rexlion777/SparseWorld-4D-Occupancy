import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_feature_path_audit.json').read_text())
def test_adapter_param_count():
    assert obj['trainable_parameter_count'] > 0
    assert obj['trainable_parameter_count'] < 500000
