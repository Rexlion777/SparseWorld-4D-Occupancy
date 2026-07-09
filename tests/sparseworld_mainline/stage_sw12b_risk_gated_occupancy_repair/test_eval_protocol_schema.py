from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw12b_risk_gated_occupancy_repair/sw12b_eval_protocol.json"


def test_eval_protocol_schema() -> None:
    payload = json.loads(PATH.read_text(encoding="utf-8"))
    assert payload["subset_name"] == "eval_core_20"
    assert set(payload["perturbations"]) >= {"A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9"}
    assert set(payload["horizons"]) >= {0, 2, 4, 6}
    assert "safety_gates" in payload
