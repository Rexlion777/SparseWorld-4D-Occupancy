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
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"


def test_sw91_outputs_exist() -> None:
    required = [
        REPORTS / "sw91_time_budget_manifest.json",
        REPORTS / "sw9_result_reinterpretation_for_sw91.json",
        REPORTS / "h2_lambda_gradient_sweep.csv",
        REPORTS / "sw91_config_manifest.json",
        REPORTS / "sw91_short_train_metrics.csv",
        REPORTS / "sw91_fixed_subset_eval.csv",
        REPORTS / "sw91_h2_contributor_assignment_decision.json",
        REPORTS / "stage_sw91_h2_contributor_assignment_training_report.md",
    ]
    for path in required:
        assert path.exists(), f"missing required output: {path}"
