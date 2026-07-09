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
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision/sw9_gradient_check.csv"


def test_gradient_check_schema():
    rows = list(csv.DictReader(PATH.open("r", encoding="utf-8")))
    assert len(rows) >= 5, "expected G0..G4 rows in gradient check csv"
    required = [
        "experiment_id",
        "total_loss",
        "original_loss",
        "h1_pos_loss",
        "h1_neg_loss",
        "h2_assign_loss",
        "h2_leak_loss",
        "grad_finite",
        "grad_norm_total",
        "grad_norm_decoder_layers",
        "has_nan",
        "has_inf",
        "peak_gpu_memory_mb",
        "backward_time_sec",
        "support_path_nonzero_grad",
    ]
    for key in required:
        assert key in rows[0], f"missing gradient column: {key}"
    non_control = [row for row in rows if row["experiment_id"] != "G0_original_control"]
    assert any(row["grad_finite"] == "True" for row in non_control), "no finite gradient SW-9 row"
    assert any(row["support_path_nonzero_grad"] == "True" for row in non_control), "no non-zero support-path gradient"
    assert all(row["has_nan"] == "False" for row in rows), "NaN found in gradient check"
    assert all(row["has_inf"] == "False" for row in rows), "Inf found in gradient check"

