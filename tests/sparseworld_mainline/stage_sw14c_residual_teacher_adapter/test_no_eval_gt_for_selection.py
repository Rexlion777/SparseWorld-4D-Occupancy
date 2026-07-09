import json
from pathlib import Path


BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_teacher_adapter")


def test_no_eval_gt_for_selection() -> None:
    split = json.loads((BASE / "sw14c_split_manifest.json").read_text(encoding="utf-8"))
    gamma = json.loads((BASE / "sw14c_gamma_selection.json").read_text(encoding="utf-8"))
    assert split["eval_debug"]["tuning_allowed"] is False
    assert split["eval_debug"]["uses_gt_for_training"] is False
    assert gamma["uses_eval_debug_for_selection"] is False
    assert gamma["uses_eval_core100_or_core500_for_selection"] is False
