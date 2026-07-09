from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FIGS = ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_swvis1_paper_style_visualization"
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization"


def test_core_outputs_exist() -> None:
    rollout = list(FIGS.glob("paper_style_gt_sparseworld_rollout_sample*.png"))
    composite = list(FIGS.glob("paper_style_observation_future_composite_sample*.png"))
    gifs = list(FIGS.glob("paper_style_future_rollout_sample*.gif"))
    assert len(rollout) >= 3
    assert len(composite) >= 1
    assert len(gifs) >= 1
    assert (REPORTS / "paper_style_visualization_config.json").exists()
    assert (REPORTS / "visualization_quality_check.json").exists()

