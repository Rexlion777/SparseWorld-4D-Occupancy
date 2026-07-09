import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")


def test_no_core100_used():
    for name in ["sw14c_round2b_inherited_state.json", "sw14c_round2b_gamma_selection.json", "sw14c_round2b_final_decision.json"]:
        obj = json.loads((BASE / name).read_text())
        assert obj.get("uses_core100_or_core500", obj.get("uses_core100_or_core500_for_selection", False)) is False
    assert not (BASE / "sw14c_round2b_eval_core100_metrics.csv").exists()
    assert not (BASE / "sw14c_round2b_eval_core500_metrics.csv").exists()
