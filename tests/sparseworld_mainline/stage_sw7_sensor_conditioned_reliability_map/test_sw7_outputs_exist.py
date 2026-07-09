from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
VIDEOS = ROOT / "videos/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"


def test_sw7_outputs_exist() -> None:
    required = [
        REPORTS / "sw7_run_manifest.json",
        REPORTS / "sw7_reliability_config.json",
        REPORTS / "voxel_reliability_metrics_aggregate.csv",
        REPORTS / "sector_reliability_summary.csv",
        REPORTS / "class_group_reliability_summary.csv",
        REPORTS / "score_alpha_visualization_manifest.json",
        REPORTS / "risk_map_visualization_manifest.json",
        REPORTS / "dashboard_manifest.json",
        REPORTS / "high_risk_region_ranking.json",
        REPORTS / "planning_facing_risk_interface_schema.json",
        REPORTS / "reliability_error_correlation.csv",
        REPORTS / "reliability_component_ablation.csv",
        REPORTS / "sw8_recommended_training_plan.json",
        REPORTS / "stage_sw7_sensor_conditioned_reliability_map_report.json",
        REPORTS / "stage_sw7_sensor_conditioned_reliability_map_report.md",
    ]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, f"missing outputs: {missing}"
    expected_videos = [
        VIDEOS / "a1_front_dropout_reliability_dashboard.mp4",
        VIDEOS / "a10_front_triplet_reliability_dashboard.mp4",
        VIDEOS / "c4_motion_blur_reliability_dashboard.mp4",
        VIDEOS / "a7_rear_dropout_control_dashboard.mp4",
    ]
    assert all(p.exists() for p in expected_videos)
