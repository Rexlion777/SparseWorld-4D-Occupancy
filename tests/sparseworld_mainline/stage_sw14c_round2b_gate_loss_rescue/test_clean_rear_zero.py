import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")


def test_clean_rear_zero():
    health = json.loads((BASE / "sw14c_round2b_residual_health_summary.json").read_text())
    assert float(health["rear_feature_delta_max"]) == 0.0
    assert float(health["clean_residual_max_abs"]) == 0.0
    assert float(health["rear_gate_mean"]) == 0.0
    assert float(health["clean_gate_mean"]) == 0.0
    decision = json.loads((BASE / "sw14c_round2b_final_decision.json").read_text())
    assert decision["clean_rear_safety"] is True
