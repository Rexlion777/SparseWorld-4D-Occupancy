from __future__ import annotations

from pathlib import Path


ROOT = Path("/home/rexlion/ComputerVision/cv_lidar_transition")
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment"


def test_outputs_exist() -> None:
    required = [
        REPORTS / "sw101_time_budget_manifest.json",
        REPORTS / "sw91_sw10_digest_for_sw101.json",
        REPORTS / "native_getocc_path_audit.json",
        REPORTS / "alignment_dump_manifest.csv",
        REPORTS / "h2_native_assignment_alignment_metrics.csv",
        REPORTS / "h2_native_mismatch_taxonomy.csv",
        REPORTS / "h2_to_semantic_occ_waterfall.csv",
        REPORTS / "checkpoint_alignment_evolution.csv",
        REPORTS / "native_aligned_h2_feasibility_design.json",
        REPORTS / "sw101_alignment_audit_decision.json",
        REPORTS / "stage_sw101_native_getocc_assignment_alignment_report.md",
    ]
    for path in required:
        assert path.exists(), f"missing {path}"
