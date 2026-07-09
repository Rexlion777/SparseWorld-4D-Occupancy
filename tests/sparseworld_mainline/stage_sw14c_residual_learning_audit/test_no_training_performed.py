import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_learning_audit")


def test_no_training_performed() -> None:
    ckpt = json.loads((BASE / "sw14c_checkpoint_sanity_audit.json").read_text(encoding="utf-8"))
    decision = json.loads((BASE / "sw14c_residual_learning_audit_decision.json").read_text(encoding="utf-8"))
    loss = json.loads((BASE / "sw14c_loss_dominance_summary.json").read_text(encoding="utf-8"))
    assert ckpt["training_performed_in_audit"] is False
    assert ckpt["checkpoint_modified_in_audit"] is False
    assert decision["training_performed_in_audit"] is False
    assert decision["checkpoint_modified_in_audit"] is False
    assert loss["training_performed_in_audit"] is False
