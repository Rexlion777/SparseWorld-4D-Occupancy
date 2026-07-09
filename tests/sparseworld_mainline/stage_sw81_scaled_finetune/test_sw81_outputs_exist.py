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
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune"


def test_sw81_outputs_exist():
    required = [
        REPORTS / "lr_stability_audit.json",
        REPORTS / "eval_subset_manifest.csv",
        REPORTS / "e0_control_eval_by_iter.csv",
        REPORTS / "sw7_reliability_retest_scaled.csv",
        REPORTS / "sw81_scaled_finetune_decision.json",
        REPORTS / "stage_sw81_scaled_finetune_report.md",
    ]
    for path in required:
        assert path.exists(), f"missing required output: {path}"
