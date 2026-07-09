from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORTS = ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"
FIGURES = ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"


def test_sw6_outputs_exist():
    required_reports = [
        "coordinate_sector_audit.json",
        "sw6_perturbation_subset_manifest.json",
        "sw6_run_manifest.json",
        "sw6_sample_manifest.csv",
        "sector_metrics_aggregate.csv",
        "frontview_chain_decomposition_clean_a1_a10.csv",
        "motion_blur_geometry_semantic_contributor_decomposition.csv",
        "group_fragility_metrics.csv",
        "clean_perturbed_support_matching.csv",
        "causal_restore_probe_metrics.csv",
        "causal_restore_probe_metrics.json",
        "sw6_refined_failure_taxonomy.csv",
        "sw6_refined_failure_taxonomy.json",
        "model_level_repair_proposal_ranking.csv",
        "sw7_recommended_next_action.json",
        "sw6_case_gallery_manifest.json",
        "stage_sw6_frontview_blur_fragility_diagnosis_report.md",
        "stage_sw6_frontview_blur_fragility_diagnosis_report.json",
    ]
    required_figures = [
        "front_sector_false_free_heatmap.png",
        "sector_support_density_delta_heatmap.png",
        "sector_contributor_delta_heatmap.png",
        "frontview_failure_chain_waterfall.png",
        "frontview_temporal_amplification_curves.png",
        "motion_blur_geometry_vs_semantic_panel.png",
        "dynamic_small_newvisible_fragility_panel.png",
        "causal_restore_probe_matrix.png",
        "repair_proposal_ranking_matrix.png",
    ]
    missing = [name for name in required_reports if not (REPORTS / name).exists()]
    missing += [name for name in required_figures if not (FIGURES / name).exists()]
    assert not missing, f"missing SW-6 outputs: {missing}"
