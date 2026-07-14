# Supported SW13A/R8 Result Gallery

[中文首页](../README.md) | [English homepage](../README_EN.md) | [Result contract](RESULTS.md)

This gallery contains **20 figures generated from 1,480 committed machine-readable records**. Only frozen, supported SW13A/R8 evidence is included. Ongoing learned repair, negative branches, and the active Robot workspace are not used to create these figures.

## Template 1: aligned before/after video

| R0 degraded native | R8 causal feature repair |
|---|---|
| ![Template 1 before R8](../assets/template1/r8_comparison/template1_a10_native_before_r8.gif) | ![Template 1 after R8](../assets/template1/r8_comparison/template1_a10_r8_after_r8.gif) |
| [1280×1180 MP4](../assets/template1/r8_comparison/template1_a10_native_before_r8.mp4) | [1280×1180 MP4](../assets/template1/r8_comparison/template1_a10_r8_after_r8.mp4) |

The comparison is based on one frozen A10 sample. `native_semantic` supplies the before anchors and raw `teacher_raw_semantic` from SW13A/R8 supplies the after anchors. The animation interpolates only for display between the real 0/2/4/6 s outputs.

## 1. Headline recovery and future horizons

| Supported recovery summary | False-free recovery by horizon |
|---|---|
| ![Supported recovery summary](../assets/portfolio/01_supported_recovery_summary.png) | ![A10 false-free horizons](../assets/portfolio/02_a10_false_free_horizons.png) |
| **Occupied IoU** | **Semantic mIoU** |
| ![A10 occupied IoU](../assets/portfolio/03_a10_occupied_iou.png) | ![A10 semantic mIoU](../assets/portfolio/04_a10_semantic_miou.png) |

## 2. Where recovery occurs

| Front-sector recovery | Dynamic-object recovery |
|---|---|
| ![Front-sector recovery](../assets/portfolio/05_front_sector_recovery.png) | ![Dynamic-object recovery](../assets/portfolio/06_dynamic_object_recovery.png) |
| **Small-object recovery** | **Sample-level IoU gains** |
| ![Small-object recovery](../assets/portfolio/07_small_object_recovery.png) | ![Sample IoU improvement](../assets/portfolio/08_sample_iou_improvement.png) |

## 3. Paired-sample stability

| Sample-level false-free gains | Sample-level semantic gains |
|---|---|
| ![Sample false-free reduction](../assets/portfolio/09_sample_false_free_reduction.png) | ![Sample semantic improvement](../assets/portfolio/10_sample_semantic_improvement.png) |

The sample plots expose distributions rather than only aggregate means, making it possible to inspect consistency and outliers across prediction horizons.

## 4. Ablations and operating point

| Recovery-density Pareto | Memory-age ablation |
|---|---|
| ![Recovery-density Pareto](../assets/portfolio/11_recovery_density_pareto.png) | ![Memory-age ablation](../assets/portfolio/12_memory_age_ablation.png) |
| **Motion-blur blend ablation** | **Fault-scope summary** |
| ![Motion-blur blend recovery](../assets/portfolio/13_motion_blur_blend_recovery.png) | ![Fault-scope summary](../assets/portfolio/14_fault_scope_summary.png) |

## 5. Risk controls

| Occupancy-density guardrail | Recovery-risk balance |
|---|---|
| ![Occupancy-density guardrail](../assets/portfolio/15_occupancy_density_guardrail.png) | ![Recovery-risk balance](../assets/portfolio/16_recovery_risk_balance.png) |

The density figures are intentionally shown next to false-free recovery: lowering misses is not considered useful if it is achieved through uncontrolled occupancy expansion.

## 6. System and evidence contracts

| SparseWorld tensor contract | R8 causal feature-memory pipeline |
|---|---|
| ![SparseWorld tensor contract](../assets/portfolio/17_sparseworld_tensor_contract.png) | ![R8 causal pipeline](../assets/portfolio/18_r8_causal_pipeline.png) |
| **Experiment atlas** | **Public evidence coverage** |
| ![Experiment atlas](../assets/portfolio/19_experiment_atlas.png) | ![Public evidence coverage](../assets/portfolio/20_public_evidence_coverage.png) |

## Reproduce the gallery

```bash
python scripts/portfolio/generate_supported_result_gallery.py
python scripts/portfolio/generate_r8_template1_comparison.py \
  --anchors evidence/r8_template1_sample000 \
  --output-dir assets/template1/r8_comparison
```

The generator reads the committed tables under [`evidence/`](../evidence/README.md). It does not require the private research repository, nuScenes images, model checkpoints, or the active Robot workspace.
