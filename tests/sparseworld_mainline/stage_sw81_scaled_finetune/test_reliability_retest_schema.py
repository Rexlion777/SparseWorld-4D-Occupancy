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
PATH = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/sw7_reliability_retest_scaled.csv"


def test_reliability_retest_schema():
    rows = list(csv.DictReader(PATH.open("r", encoding="utf-8")))
    assert rows, "empty reliability retest csv"
    required = [
        "checkpoint_name",
        "perturbation_id",
        "sample_index",
        "horizon_s",
        "mean_reliability",
        "high_risk_voxel_ratio",
        "front_sector_reliability",
    ]
    for key in required:
        assert key in rows[0], f"missing column: {key}"
