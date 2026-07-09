import csv
from pathlib import Path


BASE = Path(__file__).resolve().parents[3] / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw42_conservative_leak_aware_repair"


def test_candidate_metrics_schema():
    files = [
        "conservative_soft_splat_metrics_5sample.csv",
        "leak_aware_repair_metrics_5sample.csv",
        "false_positive_guard_metrics.csv",
        "small_object_local_repair_metrics.csv",
        "new_visible_conservative_repair_metrics.csv",
    ]
    required_cols = {
        "variant_name",
        "occupied_iou",
        "semantic_miou",
        "false_free_rate",
        "false_occupied_rate",
        "pred_gt_occupied_ratio",
        "small_object_false_free",
        "new_visible_recall",
    }
    for name in files:
        with (BASE / name).open("r", encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f))
        assert required_cols.issubset(row.keys()), name
