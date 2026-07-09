import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")


def test_backbone_head_frozen():
    obj = json.loads((BASE / "sw14c_round2b_training_summary.json").read_text())
    frozen = obj["backbone_head_frozen"]
    assert frozen["backbone_frozen"] is True
    assert frozen["head_frozen"] is True
    assert int(frozen["model_trainable_param_count"]) == 0
    assert int(frozen["head_trainable_param_count"]) == 0
