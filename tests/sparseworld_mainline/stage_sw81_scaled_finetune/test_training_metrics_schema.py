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
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/e0_control_eval_by_iter.csv"


def test_training_metrics_schema():
    rows = list(csv.DictReader(PATH.open("r", encoding="utf-8")))
    assert rows, "empty e0 eval-by-iter csv"
    required = [
        "subset_name",
        "perturbation_id",
        "mean_occupied_iou_delta",
        "mean_semantic_miou_delta",
        "mean_false_occupied_rate_delta",
        "lr_scale",
        "train_iters",
    ]
    for key in required:
        assert key in rows[0], f"missing column: {key}"
    assert "experiment_id" in rows[0] or "checkpoint_name" in rows[0], "missing experiment identifier column"
