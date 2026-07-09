import json
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


def test_loss_config_schema():
    h1_cfg = json.loads((REPORTS / "h1_soft_support_coverage_config.json").read_text(encoding="utf-8"))
    manifest = json.loads((REPORTS / "sw9_config_manifest.json").read_text(encoding="utf-8"))
    for key in [
        "sigma_voxel",
        "tau_pos",
        "lambda_h1",
        "lambda_neg",
        "tau_neg",
        "small_object_names",
        "target_horizons",
        "max_gt_voxels",
        "max_support_points",
        "max_negative_voxels",
    ]:
        assert key in h1_cfg, f"missing H1 config key: {key}"
    for key in [
        "base_config",
        "resume_checkpoint",
        "lr_scale",
        "config_ids",
        "lambda_h1",
        "lambda_neg",
        "lambda_h2",
        "lambda_h3_beta",
        "max_gt_voxels",
        "max_support_points",
        "max_negative_voxels",
    ]:
        assert key in manifest, f"missing config manifest key: {key}"
    assert "h6_approx_plus3s" in h1_cfg["target_horizons"]

