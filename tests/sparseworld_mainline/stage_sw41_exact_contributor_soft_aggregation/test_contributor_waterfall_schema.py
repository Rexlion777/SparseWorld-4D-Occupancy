import csv
from pathlib import Path


PROJECT_ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation"


def test_sw41_contributor_waterfall_schema():
    path = REPORTS_DIR / "exact_contributor_waterfall.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows, "exact_contributor_waterfall.csv is empty"
    required = {
        "sample_index",
        "horizon_s",
        "region_name",
        "support_r2_ratio",
        "support_r3_ratio",
        "support_r5_ratio",
        "valid_range_support_ratio",
        "gate_pass_support_ratio",
        "exact_bev_match_ratio",
        "exact_voxel_assignment_ratio",
        "final_contributor_ratio",
        "final_occupied_ratio",
        "final_correct_semantic_ratio",
    }
    assert required.issubset(rows[0].keys())
