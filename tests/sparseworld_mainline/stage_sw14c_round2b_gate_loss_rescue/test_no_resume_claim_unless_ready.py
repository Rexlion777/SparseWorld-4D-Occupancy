import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_round2b_gate_loss_rescue")


def test_no_resume_claim_unless_ready():
    obj = json.loads((BASE / "sw14c_round2b_final_decision.json").read_text())
    if obj["decision"] not in {"SW14C_R2B_3_SAFE_SMALL_GAIN_READY_DEBUG", "SW14C_R2B_4_SAFE_STRONG_GAIN_READY_DEBUG"}:
        assert obj["whether_resume_allowed"] is False
        assert obj["whether_round3_allowed"] is False
