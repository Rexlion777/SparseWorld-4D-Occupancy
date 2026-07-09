import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")


def test_no_eval_debug_used():
    for name in ["sw14c_round2b_inherited_state.json", "sw14c_round2b_gamma_selection.json", "sw14c_round2b_final_decision.json"]:
        obj = json.loads((BASE / name).read_text())
        assert obj.get("uses_eval_debug", obj.get("uses_eval_debug_for_selection", False)) is False
    assert not (BASE / "sw14c_round2b_eval_debug_metrics.csv").exists()
