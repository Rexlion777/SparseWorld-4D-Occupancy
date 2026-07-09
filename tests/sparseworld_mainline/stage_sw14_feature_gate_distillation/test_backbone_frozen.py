import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_feature_path_audit.json').read_text())
def test_backbone_frozen():
    assert obj['all_model_params_frozen'] is True
    assert obj['backbone_trainable'] == 0
