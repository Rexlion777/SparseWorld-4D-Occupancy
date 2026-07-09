from pathlib import Path
BASE = Path("/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay")

def test_outputs_exist() -> None:
    required = [
        "sw13a_execution_manifest.json",
        "sw13a_progress_state.json",
        "sw13a_feature_path_audit.json",
        "sw13a_feature_memory_cache_manifest.csv",
        "sw13a_feature_repair_variant_manifest.json",
        "sw13a_replay_manifest.csv",
        "sw13a_feature_memory_replay_metrics.csv",
        "sw13a_feature_memory_aggregate_metrics.csv",
        "sw13a_feature_memory_replay_decision.json",
        "stage_sw13a_sensor_fault_feature_memory_replay_report.md",
    ]
    for name in required:
        path = BASE / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name
