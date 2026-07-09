import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis/motion_blur_geometry_semantic_contributor_decomposition.csv"


def test_motion_blur_decomposition_schema():
    with REPORT.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    required = {
        "sample_index",
        "horizon_s",
        "support_count_delta",
        "support_coverage_delta_r2",
        "support_centroid_shift",
        "nearest_clean_support_match_distance",
        "support_matched_ratio",
        "cls_entropy_delta",
        "top1_confidence_delta",
        "class_margin_delta",
        "gate_pass_delta",
        "exact_contributor_delta",
        "false_occupied_delta",
    }
    assert required.issubset(row.keys())
