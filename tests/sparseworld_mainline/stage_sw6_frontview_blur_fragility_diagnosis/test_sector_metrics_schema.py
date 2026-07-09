import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis/sector_metrics_aggregate.csv"


def test_sector_metrics_schema():
    with REPORT.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    required = {
        "perturbation_id",
        "sector_name",
        "horizon_s",
        "sample_count",
        "mean_false_free_rate",
        "mean_false_occupied_rate",
        "mean_occupied_iou",
        "mean_pred_gt_occupied_ratio",
        "mean_new_visible_recall",
        "mean_dynamic_false_free",
        "mean_small_object_false_free",
        "mean_gate_pass_ratio",
        "mean_exact_contributor_ratio",
        "mean_final_contributor_ratio",
        "mean_support_density",
    }
    assert required.issubset(row.keys())
