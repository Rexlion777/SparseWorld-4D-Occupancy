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
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training/sw91_fixed_subset_eval.csv"


def test_eval_metrics_schema() -> None:
    rows = list(csv.DictReader(PATH.open("r", encoding="utf-8")))
    assert rows, "empty eval csv"
    if "skipped_reason" in rows[0]:
        assert rows[0]["skipped_reason"], "skipped eval row must include skipped_reason"
        return
    required = [
        "checkpoint_name",
        "subset_name",
        "perturbation_id",
        "mean_occupied_iou_delta",
        "mean_semantic_miou_delta",
        "mean_false_free_rate_delta",
        "mean_false_occupied_rate_delta",
        "mean_pred_gt_occupied_ratio_delta",
        "mean_small_object_false_free_delta",
        "mean_new_visible_recall_delta",
        "mean_front_sector_false_free_delta",
        "mean_dynamic_false_free_delta",
        "mean_static_false_free_delta",
        "gate_safe",
        "gate_targeted_hit",
    ]
    for key in required:
        assert key in rows[0], f"missing eval metric column: {key}"
