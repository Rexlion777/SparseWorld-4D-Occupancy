# Source Code Map

The repository contains more than 56,000 lines of project experiment code. This map is the recommended reading order; stage numbers reflect the real research sequence rather than a reconstructed portfolio narrative.

## Fast reviewer path

| Goal | Source directory | What to inspect |
|---|---|---|
| Understand model/data bring-up | `stage_sw1_bringup/` | configuration, nuScenes sample contract, inference smoke checks |
| Trace temporal queries | `stage_sw2_temporal_query_diagnosis/` | query state, horizon mapping, temporal feature diagnostics |
| Validate sparse support | `stage_sw3_query_support_validation/` | query-to-voxel coverage and support attribution |
| Reproduce sensor failures | `stage_sw5_sensor_aware_failure_propagation/` | camera perturbations, causal manifests, failure metrics |
| Inspect blur fragility | `stage_sw6_frontview_blur_fragility_diagnosis/` | controlled degradation and front-sector attribution |
| Build reliability maps | `stage_sw7_sensor_conditioned_reliability_map/` | camera/sector/class/horizon risk decomposition |
| Review R8 implementation | `stage_sw13a_sensor_fault_feature_memory_replay/` | feature cache, failure mask, selective replay, metric audit |
| Review scale validation | `stage_sw13c_fix_frontcap_eval_core100/` and `core500/` | frozen protocol, incremental shards, density/front-cap checks |
| Review query-memory research | `stage_mcqm_motion_compensated_query_memory/` | ego-motion compensation, query-to-FPN reconstruction, contribution tests |
| Regenerate figures | `stage_swvis1_paper_style_visualization/` and `stage_swvis4_query_support_interpolation/` | Template 1, future rollout, query-support interpolation |

## Full experiment progression

```text
SW1–SW4    bring-up → geometry → query support → semantic activation
SW5–SW7    fault injection → propagation diagnosis → reliability map
SW8–SW12   targeted fine-tuning → contributor routing → safe repair
SW13       causal feature memory → density constraints → R8 validation
SW14       learned gates/residuals/reranking (mixed or negative outcomes)
MCQM       motion-compensated query memory and Query-to-FPN research
SWVIS      paper-style 4D visualization and support interpolation
```

## Supported result path

The best-supported chain is:

1. `stage_sw5_sensor_aware_failure_propagation` defines deterministic sensor faults.
2. `stage_sw7_sensor_conditioned_reliability_map` localizes their consequences.
3. `stage_sw13a_sensor_fault_feature_memory_replay` implements causal same-camera FPN memory.
4. `stage_sw13c_fix_frontcap_eval_core100` and `core500` freeze and scale the evaluation protocol.
5. `stage_swvis1_paper_style_visualization` turns frozen outputs into Template 1 evidence.

## Why negative branches remain public

The SW14 and MCQM paths are intentionally retained. They show that the project did not select results after the fact: feature-space reconstruction could improve while occupancy contribution degraded, learned gates could violate density or clean-scene constraints, and objective conflict could block a residual branch. These branches document the diagnosis and stop decisions instead of being presented as successes.

## Portability note

The compact package under `src/sparseworld_reliability/` is dependency-light and covered by CI. Historical full-pipeline runners mirror the original CUDA/MMCV experiment environment and may contain environment-specific default paths; override them with the runner arguments or `CV_LIDAR_PROJECT_ROOT`. Dataset, checkpoints, caches, and generated artifacts are intentionally not committed.
