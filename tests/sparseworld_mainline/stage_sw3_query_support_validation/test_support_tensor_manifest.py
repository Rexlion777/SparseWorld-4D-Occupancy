from __future__ import annotations

import json
from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw3_query_support_validation"


def test_support_tensor_manifest_content() -> None:
    payload = json.loads((REPORTS / "support_tensor_candidate_manifest.json").read_text(encoding="utf-8"))
    assert payload["candidate_count"] > 0
    assert "refine_pts_current" in payload["candidate_names"]
    assert "cls_score_current" in payload["candidate_names"]


def test_dependency_trace_marks_refine_pts_used() -> None:
    payload = json.loads((REPORTS / "semantic_occ_dependency_trace.json").read_text(encoding="utf-8"))
    assert payload["refine_pts_used_by_semantic_occ"]["status"] == "definitely_used_by_semantic_occ"
    assert payload["cls_score_used_by_semantic_occ"]["status"] == "definitely_used_by_semantic_occ"
