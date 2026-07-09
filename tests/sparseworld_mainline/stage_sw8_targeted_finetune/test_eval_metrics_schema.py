import csv
from pathlib import Path


ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw8_targeted_finetune/e0_baseline_resume_control_eval.csv"


def test_eval_metrics_schema():
    rows = list(csv.DictReader(PATH.open("r", encoding="utf-8")))
    assert rows, "empty baseline eval csv"
    required = [
        "occupied_iou",
        "semantic_miou",
        "false_free_rate",
        "false_occupied_rate",
        "pred_gt_occupied_ratio",
        "small_object_false_free",
        "new_visible_recall",
        "front_sector_false_free",
    ]
    for key in required:
        assert key in rows[0], f"missing column: {key}"
