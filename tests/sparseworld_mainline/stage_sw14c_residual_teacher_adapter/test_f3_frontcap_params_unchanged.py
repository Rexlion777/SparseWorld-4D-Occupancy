import json
from pathlib import Path


PROTOCOL = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_teacher_adapter/sw14c_postprocess_protocol.json")


def test_f3_frontcap_params_unchanged() -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    assert payload["f3"]["reuse_sw13c_fix_function"] is True
    assert payload["f3"]["expansion_ratio"] == 0.12
    assert payload["f3"]["protected_variant"] == "PZ_fix_3_strong_core"
    assert payload["f3"]["uses_gt_budget"] is False
    assert payload["frontcap"]["variant"] == "FC1_1p3"
    assert payload["frontcap"]["cap_ratio"] == 1.3
    assert payload["frontcap"]["uses_gt_frontcap"] is False
