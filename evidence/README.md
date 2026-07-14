# Public evidence

The portfolio charts are regenerated from the following frozen, machine-readable evidence:

- `headline_supported_results.csv`: supported headline values already published in `docs/RESULTS.md`;
- `raw/sw13a_feature_memory_aggregate_metrics.csv`: 68 aggregate rows from the frozen SW13A evaluation;
- `raw/sw13a_feature_memory_replay_metrics.csv`: 1,360 per-sample rows from the same evaluation;
- `raw/sw13a_recovery_density_tradeoff.csv`: 48 supported recovery/risk comparison rows.
- `r8_template1_sample000/`: rendered 0/2/4/6 s R0-native and raw-R8 anchors for the aligned Template 1 video comparison.

The files were copied from the read-only experiment output under
`reports/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/`.
No current Robot source code, checkpoints, cached tensors, or unsupported learned-repair results are included. The Template 1 directory contains only rendered PNG anchors and their provenance note.

These are reliability-study results under the repository's frozen protocol, not official nuScenes leaderboard scores.
