# R8 Results and Evaluation Contract

## Result summary

All changes below are relative to the same frozen SparseWorld checkpoint under the corresponding degraded-input baseline.

| Case | Primary metric | R8 result |
|---|---|---:|
| A10: front triplet missing | false-negative count | **-23.2%** |
| A10 at 4 s future horizon | false-negative count | **-10.5%** |
| A1: front camera missing | false-negative count | **-17.4%** |
| C4: motion blur | false-negative count | **-7.1%** |
| Repaired occupancy density | increase vs degraded prediction | **+2.3% to +6.0%** |

The density interval is reported next to FN recovery because a method can trivially reduce misses by marking too much free space as occupied. R8 is evaluated as a recovery/risk pair, not by recovery alone.

## Supported result gallery

The public portfolio includes a [20-figure supported-result atlas](SUPPORTED_RESULT_GALLERY.md) generated from 1,480 frozen SW13A/R8 records. It covers horizon-, sector-, class-, and sample-level recovery, memory/blur ablations, density guardrails, and the causal system contract. The committed CSVs and deterministic generator make every public chart reproducible without copying the active Robot workspace.

## Frozen comparison protocol

- same checkpoint and model code for native degraded and repaired runs;
- same sample IDs, perturbation severity, calibration, post-processing, and metrics;
- current clean frames and future frames are unavailable to the repair path;
- ground truth is used only after inference for evaluation;
- only failed-camera features are eligible for replacement;
- clean and non-target camera parity are checked;
- results are split by perturbation and horizon before aggregation.

## Metric definition

Relative FN reduction is:

```text
(FN_degraded - FN_repaired) / FN_degraded
```

The percentages are reliability-study results and must not be interpreted as official nuScenes test-server or leaderboard scores.

## Engineering conclusion

R8 is the strongest interpretable baseline in the current study: it provides substantial recovery under camera loss while keeping the intervention causal and spatially restricted. Later residual/transport experiments were retained only when they passed the same occupancy-task criteria. Improving feature reconstruction alone was not considered sufficient evidence of a better perception system.
