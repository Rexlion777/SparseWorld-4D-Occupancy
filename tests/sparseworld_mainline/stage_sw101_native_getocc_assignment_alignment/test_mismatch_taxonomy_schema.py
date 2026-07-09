from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment/h2_native_mismatch_taxonomy.csv"


def test_mismatch_taxonomy_schema() -> None:
    with PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "empty mismatch taxonomy csv"
    row = rows[0]
    assert row["mismatch_type"] in {
        "M1_coordinate_decode_mismatch",
        "M2_voxel_rounding_mismatch",
        "M3_valid_mask_or_range_filter",
        "M4_gate_or_score_filter",
        "M5_horizon_source_mismatch",
        "M6_neighbor_leakage",
        "M7_semantic_confidence_mismatch",
        "M8_aggregation_conflict",
        "M9_unknown",
    }
    for key in ["checkpoint_name", "perturbation_id", "sample_index", "horizon_s", "sector_name", "class_group", "note"]:
        assert key in row and row[key] != "", f"missing {key}"
