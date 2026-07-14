# Experiment Atlas

This document treats the project as an engineering investigation. A route is marked successful only when its occupancy-task evidence supports the original hypothesis. Negative results are retained because they narrow the design space and demonstrate reproducible failure attribution.

## Status vocabulary

- **Supported** — passed the declared task-level checks on the frozen protocol.
- **Partial** — mechanism worked or a subset improved, but generalization/safety was not stable.
- **Rejected** — failed the declared acceptance criteria; not promoted as an improvement.
- **Infrastructure** — diagnostic or tooling work that enabled later experiments.

## SW1-SW4: model bring-up and semantic support

### SW1 — Native forward and post-processing

- **Question:** Can the upstream model, checkpoint, dataset item, and occupancy output be reproduced end to end?
- **Work:** single-sample forward, output normalization, semantic/empty-label handling, and BEV export.
- **Outcome:** **Infrastructure.** Established the trusted native baseline.

### SW2-SW3 — Temporal query and support audit

- **Question:** Do current/future queries, coordinate transforms, and support tensors represent what the configuration claims?
- **Work:** query-axis checks, temporal metric schemas, causal probes, and coordinate mapping audits.
- **Outcome:** **Infrastructure.** Converted implicit tensor semantics into testable contracts.

### SW4 — Occupancy contributor instrumentation

- **Question:** Which sparse contributors actually activate covered false-free and missed-occupancy regions?
- **Work:** instrumented `get_occ`, captured dense support, and performed oracle-only diagnostic intervention.
- **Outcome:** **Partial.** Contributor evidence was useful; oracle diagnostics were never promoted to an inference method.

## SW5-SW7: sensor degradation and reliability diagnosis

### SW5 — Sensor-aware failure propagation

- **Question:** How do camera dropout, blur, low light, and rear-view loss propagate into occupancy errors?
- **Work:** deterministic perturbation engine, severity manifests, sector/horizon metrics, and paired native comparisons.
- **Outcome:** **Infrastructure.** Built the degradation suite used throughout the project.

### SW6 — Front-view blur fragility

- **Question:** Which stages and future horizons are most sensitive to forward-camera motion blur?
- **Work:** front-view chain tracing, failure taxonomy, sector metrics, and causal restoration probes.
- **Outcome:** **Supported diagnostic conclusion.** Blur damage is not spatially or temporally uniform; aggregate IoU hides important forward-sector failures.

### SW7 / SW7.1 — Reliability maps and cleanup

- **Question:** Can sensor-conditioned risk be ranked without relabeling error metrics as confidence?
- **Work:** risk maps, validation manifests, top-k audits, and terminology cleanup.
- **Outcome:** **Infrastructure.** Produced auditable risk localization and removed ambiguous reliability claims.

## SW8-SW12: training, routing, and repair search

### SW8 / SW8.1 — Targeted fine-tuning

- **Question:** Does targeted training improve degraded scenes while preserving clean behavior?
- **Work:** fault-targeted fine-tuning, throughput profiling on RTX 5070 Ti, scaled evaluation, and clean/degraded parity checks.
- **Outcome:** **Partial.** Training was operational, but gains depended on scale and protocol; no blanket robustness claim was accepted.

### SW9-SW11 — Contributor-aware supervision and resampling

- **Question:** Can supervision focus learning on high-risk contributors and failure regions?
- **Work:** contributor assignment, gradient checks, risk-targeted resampling, and survival retests.
- **Outcome:** **Partial / Rejected as final repair.** Improved understanding of data and gradient routing, but did not establish a stable production candidate.

### SW12A-SW12C — Soft routing and residual repair

- **Question:** Can conservative routing or residual correction improve occupied recall without expanding false occupancy?
- **Work:** soft-neighbor variants, strict masks, teacher-cache contracts, density gates, and Pareto selection.
- **Outcome:** **Mostly rejected.** Several local gains failed safety/generalization thresholds; the tests became valuable guardrails for later stages.

## SW13: causal temporal feature memory

### SW13A — Feature-memory replay

- **Question:** Can causal historical image features repair sensor faults before the unchanged occupancy head?
- **Work:** same-camera FPN cache, `t-1` replacement, EMA/blend variants, source-age manifests, and no-future/no-GT tests.
- **Outcome:** **Supported.** Established the temporal feature-memory mechanism.

### R8 — Front-triplet camera-group repair

- **Question:** Does repairing exactly the failed front camera group outperform broader or less targeted replay?
- **Work:** replace `CAM_FRONT`, `CAM_FRONT_LEFT`, and `CAM_FRONT_RIGHT` at all four FPN levels from causal same-camera history.
- **Outcome:** **Strongest supported result.** See [RESULTS.md](RESULTS.md).

### SW13B-SW13C — Density-constrained temporal prior

- **Question:** Can R8 recovery be retained while explicitly controlling occupancy expansion?
- **Work:** counterfactual density analysis, temporal priors, protected zones, frozen front caps, and larger-core evaluations.
- **Outcome:** **Partial.** Improved risk control and protocol quality; R8 remained the more defensible mainline contribution.

## SW14: learned repair beyond R8

### Feature-gate distillation and residual adapters

- **Question:** Can a small trainable module improve on the hard R8 replacement while keeping the baseline frozen?
- **Work:** zero-initialized residuals, strict front-triplet masks, frozen backbone/FPN/head, teacher parity, and clean/rear-camera invariants.
- **Outcome:** **Rejected as a stable upgrade.** Some variants reduced feature reconstruction error, but occupancy-task gains did not generalize consistently.

### Candidate reranking and completion routes

- **Question:** Can sparse contributor ranking or local completion rescue missed occupied voxels safely?
- **Work:** candidate rerankers, local attention, neighborhood/morphology features, BEV completion, and early-stop selection.
- **Outcome:** **Rejected / diagnostic only.** Weak-add precision and selection constraints prevented promotion to the mainline.

## MCQM and Query-to-FPN research

### Motion-Compensated Query Memory (MCQM)

- **Question:** Can sparse query memory be ego-motion compensated, projected, and reused across time?
- **Work:** coordinate transforms, camera projection, confidence weighting, bilinear scatter, and query-memory contracts.
- **Outcome:** **Mechanism supported.** Motion compensation and projection were made executable and testable.

### Full Query-to-FPN reconstruction

- **Question:** Can transported sparse queries reconstruct missing dense FPN features better than R8?
- **Work:** four-level projector, decoder, residual paths, seed/window factorials, and query-removal ablations.
- **Outcome:** **Rejected as an R8 replacement.** The dense reconstruction path was query-sensitive, but learned query contribution often hurt occupancy performance.

### Gradient alignment and PCGrad

- **Question:** Is conflict between feature reconstruction and occupancy objectives causing the failure?
- **Work:** task/feature gradient cosine and norm audits, projected-gradient training, and repeated evaluation windows.
- **Outcome:** **Diagnostic success, model result still unsupported.** Objective conflict was measurable; correcting it did not yet yield a stable task-level improvement.

## Visualization track

- **Question:** Can spatial and temporal behavior be inspected beyond scalar metrics?
- **Work:** semantic BEV, first-person views, future-only video, support interpolation, exact-keyframe checks, and stabilized temporal rendering.
- **Outcome:** **Infrastructure.** Made failures and recovery behavior reviewable frame by frame.

## Decision discipline

Three rules governed promotion:

1. **Task metrics beat proxy metrics.** Better feature cosine/L1 does not equal better occupancy.
2. **Safety metrics accompany recall.** FN recovery is reported with false occupancy and density drift.
3. **Causal deployability is mandatory.** Future frames, current-clean oracles, and GT-driven routing are diagnostic-only and cannot enter an inference claim.
