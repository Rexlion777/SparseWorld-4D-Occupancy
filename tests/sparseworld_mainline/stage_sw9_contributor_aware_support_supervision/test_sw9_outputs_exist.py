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
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision"


def test_sw9_outputs_exist():
    required = [
        REPORTS / "sw81_result_digest_for_sw9.json",
        REPORTS / "sw9_tensor_access_audit.json",
        REPORTS / "h1_soft_support_coverage_config.json",
        REPORTS / "sw9_config_manifest.json",
        REPORTS / "sw9_gradient_check.csv",
        REPORTS / "sw9_fixed_subset_eval.csv",
        REPORTS / "sw9_contributor_supervision_decision.json",
        REPORTS / "stage_sw9_contributor_aware_support_supervision_report.md",
    ]
    for path in required:
        assert path.exists(), f"missing required output: {path}"

