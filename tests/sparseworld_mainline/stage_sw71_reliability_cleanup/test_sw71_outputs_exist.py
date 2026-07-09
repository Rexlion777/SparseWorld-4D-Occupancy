from pathlib import Path


def repo_root() -> Path:
    candidates = [Path("D:/ComputerVision/cv_lidar_transition"), Path("/mnt/d/ComputerVision/cv_lidar_transition")]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


ROOT = repo_root()
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw71_reliability_cleanup"
FIGURES = ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw71_reliability_cleanup"


def test_sw71_outputs_exist():
    required = [
        FIGURES / "reliability_component_breakdown.png",
        FIGURES / "risk_component_breakdown.png",
        FIGURES / "high_risk_topk_precision_recall.png",
        FIGURES / "sw7_reliability_summary_ppt_ready.png",
        REPORTS / "component_breakdown_values.csv",
        REPORTS / "high_risk_topk_validation.csv",
        REPORTS / "sw7_one_page_project_card.md",
        REPORTS / "selected_assets_manifest.json",
        REPORTS / "stage_sw71_reliability_cleanup_report.json",
        REPORTS / "stage_sw71_reliability_cleanup_report.md",
    ]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, f"missing outputs: {missing}"
