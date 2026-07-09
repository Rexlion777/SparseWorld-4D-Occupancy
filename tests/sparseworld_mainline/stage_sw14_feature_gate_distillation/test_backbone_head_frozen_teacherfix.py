import json
from pathlib import Path
obj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_checkpoint_manifest.json').read_text())
def test_backbone_head_frozen_teacherfix():
    if obj.get('executed'):
        assert obj['backbone_frozen'] is True
        assert obj['head_frozen'] is True
