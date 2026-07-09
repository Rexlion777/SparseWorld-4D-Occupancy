import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")


def test_gamma_selection_val_only():
    obj = json.loads((BASE / "sw14c_round2b_gamma_selection.json").read_text())
    assert obj["uses_val_only_for_selection"] is True
    assert obj["val_samples"] == [100, 149]
    assert obj["uses_eval_debug_for_selection"] is False
    assert obj["uses_core100_or_core500_for_selection"] is False
