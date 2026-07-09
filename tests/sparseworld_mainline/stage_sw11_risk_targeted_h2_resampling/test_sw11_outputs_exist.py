from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw11_risk_targeted_h2_resampling"


def test_outputs_exist() -> None:
    required = [
        REPORTS / "sw11_time_budget_manifest.json",
        REPORTS / "sw101_digest_for_sw11.json",
        REPORTS / "sw11_failure_target_pool_manifest.csv",
        REPORTS / "sw11_sampler_config_manifest.json",
        REPORTS / "sw11_gradient_check.csv",
        REPORTS / "sw11_20iter_smoke_train_metrics.csv",
        REPORTS / "sw11_alignment_survival_retest.csv",
        REPORTS / "sw11_mismatch_taxonomy_after_resampling.csv",
        REPORTS / "sw11_risk_targeted_h2_decision.json",
        REPORTS / "sw12_recommended_route_plan.json",
        REPORTS / "stage_sw11_risk_targeted_h2_resampling_report.md",
    ]
    for path in required:
        assert path.exists(), f"missing {path}"
