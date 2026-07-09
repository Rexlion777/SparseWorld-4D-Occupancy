import csv
from pathlib import Path


PROJECT_ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw41_exact_contributor_soft_aggregation"


def _check_csv(name: str):
    rows = list(csv.DictReader((REPORTS_DIR / name).open(encoding="utf-8")))
    assert rows, f"{name} is empty"
    required = {
        "variant_name",
        "sample_index",
        "horizon_s",
        "occupied_iou",
        "semantic_miou",
        "false_free_rate",
        "false_occupied_rate",
        "pred_occupied_count",
        "pred_gt_occupied_ratio",
        "small_object_false_free",
        "new_visible_recall",
    }
    assert required.issubset(rows[0].keys()), f"{name} missing required columns"


def test_sw41_soft_gate_class_csv_schemas():
    for name in [
        "soft_splat_ablation_metrics.csv",
        "adaptive_gate_ablation_metrics.csv",
        "class_aware_aggregation_ablation_metrics.csv",
        "combined_repair_candidate_metrics.csv",
        "best_candidate_20sample_validation.csv",
    ]:
        _check_csv(name)
