from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"


def test_query_axis_semantics_content() -> None:
    payload = json.loads((REPORTS / "query_axis_semantics_audit.json").read_text(encoding="utf-8"))
    cls_shape = payload["runtime_tensor_shapes"]["cls_score"]
    refine_shape = payload["runtime_tensor_shapes"]["refine_pts"]
    assert cls_shape == [1, 720, 48, 17]
    assert refine_shape == [1, 720, 48, 3]
    assert payload["cls_score_axis_semantics"]["dim1"]["confidence"] == "confirmed"
    assert payload["refine_pts_axis_semantics"]["coordinate_space"]["confidence"] == "confirmed"
