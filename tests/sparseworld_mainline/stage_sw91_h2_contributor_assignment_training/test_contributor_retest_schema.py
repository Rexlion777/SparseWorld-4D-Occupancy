import csv
import os
from pathlib import Path


def _resolve_root() -> Path:
    env = os.environ.get("CV_LIDAR_PROJECT_ROOT")
    if env:
        path = Path(env)
        if path.exists():
            return path
    return Path(__file__).resolve().parents[3]


ROOT = _resolve_root()
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training/sw91_contributor_diagnostic_retest.csv"


def test_contributor_retest_schema() -> None:
    rows = list(csv.DictReader(PATH.open("r", encoding="utf-8")))
    assert rows, "empty contributor retest csv"
    if "skipped_reason" in rows[0]:
        assert rows[0]["skipped_reason"], "skipped contributor row must include skipped_reason"
        return
    required = [
        "checkpoint_name",
        "radius_support_coverage",
        "exact_voxel_assignment_ratio",
        "final_contributor_ratio",
        "neighbor_leakage_ratio",
        "front_sector_contributor_ratio",
        "small_object_contributor_ratio",
        "new_visible_contributor_ratio",
        "covered_false_free_contributor_count",
        "tp_contributor_count",
    ]
    for key in required:
        assert key in rows[0], f"missing contributor column: {key}"
