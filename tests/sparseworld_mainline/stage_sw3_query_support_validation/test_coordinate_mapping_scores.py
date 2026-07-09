from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw3_query_support_validation"


def test_coordinate_mapping_scores_exist_and_ranked() -> None:
    rows = list(csv.DictReader((REPORTS / "coordinate_mapping_variant_scores.csv").open(encoding="utf-8")))
    assert rows, "coordinate mapping score table is empty"
    top = max(rows, key=lambda row: float(row["mapping_score"]))
    assert top["tensor_name"]
    assert float(top["pred_occupied_overlap"]) >= 0.0
    assert float(top["valid_ratio"]) >= 0.0


def test_resolution_case_is_valid() -> None:
    payload = json.loads((REPORTS / "sw3_query_support_resolution_decision.json").read_text(encoding="utf-8"))
    assert payload["resolution_case"] in {"A", "B", "C", "D", "E", "F", "G"}
