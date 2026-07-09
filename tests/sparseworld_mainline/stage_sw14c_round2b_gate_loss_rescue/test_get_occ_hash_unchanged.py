import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")


def test_get_occ_hash_unchanged():
    inherited = json.loads((BASE / "sw14c_round2b_inherited_state.json").read_text())
    decision = json.loads((BASE / "sw14c_round2b_final_decision.json").read_text())
    assert inherited["get_occ_sha256_before"] == decision["get_occ_sha256_after"]
