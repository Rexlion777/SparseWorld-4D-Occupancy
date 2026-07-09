from pathlib import Path

MASK_DEF = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/branchB_residual_mask_definition.md")
LOSS_DEF = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12c_safe_routing_and_residual_repair/branchB_residual_loss_design.md")


def test_no_full_teacher_consistency() -> None:
    mask_text = MASK_DEF.read_text(encoding="utf-8").lower()
    loss_text = LOSS_DEF.read_text(encoding="utf-8").lower()
    assert "no reliable_teacher & risk_mask union" in mask_text
    assert "no full teacher consistency" in loss_text
    assert "all reliable teacher voxels" not in loss_text
    assert "all selected voxels" not in loss_text
