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
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training/h2_lambda_gradient_sweep.csv"


def test_h2_lambda_sweep_schema() -> None:
    rows = list(csv.DictReader(PATH.open("r", encoding="utf-8")))
    assert len(rows) >= 5, "expected five H2 sweep rows"
    required = [
        "experiment_id",
        "total_loss",
        "original_loss",
        "h2_assign_loss",
        "h2_leak_loss",
        "grad_finite",
        "grad_norm_total",
        "grad_norm_decoder",
        "grad_norm_forecast_path",
        "peak_gpu_memory_mb",
        "backward_time_sec",
        "has_nan",
        "has_inf",
    ]
    for key in required:
        assert key in rows[0], f"missing sweep column: {key}"
