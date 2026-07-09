import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis/frontview_chain_decomposition_clean_a1_a10.csv"


def test_frontview_chain_schema():
    with REPORT.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
    required = {
        "perturbation_id",
        "sample_index",
        "horizon_s",
        "false_free_rate",
        "false_occupied_rate",
        "occupied_iou",
        "pred_gt_occupied_ratio",
        "new_visible_recall",
        "dynamic_false_free",
        "small_object_false_free",
        "support_query_count",
        "support_mean_best_score",
        "support_mean_entropy",
        "gate_query_pass_ratio",
        "tp_contributor_ratio",
        "ff_contributor_ratio",
    }
    assert required.issubset(row.keys())
