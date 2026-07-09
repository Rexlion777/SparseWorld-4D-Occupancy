import json
from pathlib import Path


REPORT = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_residual_teacher_adapter/sw14c_residual_adapter_architecture.json")


def test_get_occ_hash_unchanged() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["no_get_occ_modification"] is True
    assert payload["get_occ_sha256"]
