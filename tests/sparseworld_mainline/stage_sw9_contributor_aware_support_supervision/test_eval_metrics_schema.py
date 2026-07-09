import csv
import os
from pathlib import Path


def _resolve_root() -> Path:
    env = os.environ.get("CV_LIDAR_PROJECT_ROOT")
    if env:
        p = Path(env)
        if p.exists():
            return p
    return Path(__file__).resolve().parents[3]


ROOT = _resolve_root()
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision/sw9_fixed_subset_eval.csv"
TRAIN_PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision/sw9_short_train_metrics.csv"


def test_eval_metrics_schema():
    train_rows = list(csv.DictReader(TRAIN_PATH.open("r", encoding="utf-8")))
    assert train_rows, "empty short-train summary csv"
    assert any(row["experiment_id"] in {"P1_H1_lambda001", "P3_H1_H3_lambda001"} for row in train_rows), "missing P1/P3 train rows"

    rows = list(csv.DictReader(PATH.open("r", encoding="utf-8")))
    assert rows, "empty fixed-subset eval csv"
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
        "gate_safe",
        "gate_targeted_hit",
    ]
    for key in required:
        assert key in rows[0], f"missing eval column: {key}"
