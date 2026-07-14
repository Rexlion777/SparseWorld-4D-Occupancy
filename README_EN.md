# SparseWorld Reliability Lab

[简体中文](README.md) | **English**

**A system-algorithm project for robust 4D occupancy under real sensor degradation.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](#quick-start)
[![Core tests](https://github.com/Rexlion777/SparseWorld-4D-Occupancy/actions/workflows/core-tests.yml/badge.svg)](https://github.com/Rexlion777/SparseWorld-4D-Occupancy/actions/workflows/core-tests.yml)
[![Task](https://img.shields.io/badge/Task-4D%20Occupancy-7B61FF)](#system-at-a-glance)
[![Dataset](https://img.shields.io/badge/Dataset-nuScenes-00A6D6)](#evaluation-protocol)
[![Safety](https://img.shields.io/badge/Protocol-Causal%20%7C%20No%20GT%20Repair-1F9D55)](#causal-and-safety-contract)

> The project turns a research-grade SparseWorld model into an auditable perception system: it injects camera faults, traces failure propagation, restores missing camera features from causal temporal memory, and measures both recovery and false-occupancy risk.

![R8 degradation recovery summary](assets/r8_result_card.svg)

## Headline result: R8 camera-group memory repair

R8 caches the last valid same-camera FPN features and replaces only the failed front-camera group before the occupancy head. On the frozen degradation evaluation suite, it achieved:

| Sensor degradation | Reliability result vs. degraded baseline |
|---|---:|
| Front triplet missing | **23.2% relative FN reduction** |
| Front triplet missing, 4 s future horizon | **10.5% relative FN reduction** |
| Single front camera missing | **17.4% relative FN reduction** |
| Motion blur | **7.1% relative FN reduction** |
| Occupancy-density expansion | **controlled to +2.3% to +6.0%** |

These are fault-recovery results, not nuScenes leaderboard claims. Metric definitions, causal constraints, and the frozen comparison protocol are documented in [RESULTS.md](docs/RESULTS.md).

### R8 visual evidence

The qualitative panels use the same frozen predictions as the metric audit. Red regions expose false-free errors; the repaired column shows what causal same-camera memory restores without reading the current clean frame or ground truth.

![R8 BEV repair comparison](assets/r8/r8_bev_repair_comparison.png)

<details>
<summary><strong>Expanded 100-sample front-cap diagnostic</strong></summary>

![R8 front-cap diagnostic](assets/r8/r8_frontcap_core100_diagnostic.png)

![R8 recovery-density trade-off](assets/r8/sw13a_recovery_density_tradeoff.png)

</details>

## System at a glance

- **30 images per sample:** 5 temporal frames × 6 surround cameras.
- **Sparse-to-dense reasoning:** 1,040 queries represent a 200 × 200 × 16 occupancy volume (**640,000 voxels**).
- **4D output:** semantic occupancy at 0 s, 2 s, 4 s, and 6 s.
- **Reliability suite:** camera loss, front-triplet loss, rear-camera loss, motion blur, and low-light perturbations.
- **System diagnostics:** sector-, class-, and horizon-level FN/FP, occupancy density, temporal stability, and latency.

```mermaid
flowchart LR
    A["5 frames × 6 cameras"] --> B["Image backbone + FPN"]
    B --> C["Sparse queries"]
    C --> D["4D occupancy head"]
    D --> E["0 / 2 / 4 / 6 s occupancy"]
    F["Sensor fault injector"] -. "camera loss / blur / low light" .-> A
    G["Causal same-camera memory"] --> H["R8 selective feature repair"]
    B --> H
    H --> D
    E --> I["FN / FP / density / stability audit"]
```

## What I built

1. **End-to-end evaluation path** — nuScenes temporal data organization, calibration loading, SparseWorld inference/fine-tuning, semantic occupancy metrics, and BEV visualization.
2. **Sensor degradation engine** — deterministic perturbations with manifests that preserve camera, frame, and severity provenance.
3. **Causal feature memory** — same-camera historical FPN cache, selective repair masks, and zero-oracle runtime behavior.
4. **Failure attribution** — region-, class-, and future-horizon analysis instead of a single aggregate score.
5. **Engineering decision loop** — preregistered thresholds, parity checks, ablations, and explicit rejection of variants that improve feature reconstruction but not occupancy quality.

## Template 1: four complementary 4D views

“Template 1” is the project-specific visualization family used to inspect the same prediction from four angles. These figures are qualitative diagnostics, not manually edited predictions or leaderboard evidence.

| 1. Surround observation + future occupancy | 2. Strict BEV projection |
|---|---|
| ![Template 1 overview](assets/template1/01_observation_future_overview.png) | ![Template 1 strict BEV](assets/template1/02_strict_bev.png) |
| **3. Front/ego-centric view** | **4. GT vs. SparseWorld temporal rollout** |
| ![Template 1 front view](assets/template1/03_front_view.png) | ![Template 1 GT versus prediction](assets/template1/04_gt_vs_prediction_rollout.png) |

<details>
<summary><strong>Animated future rollout</strong></summary>

![Template 1 future rollout](assets/template1/template1_future_rollout.gif)

</details>

## Read-only research snapshot: MCQM × R8

The active Robot workspace was inspected **read-only** on 2026-07-14. Its source code is intentionally not copied into this portfolio branch. The ongoing line asks whether motion-compensated sparse queries can improve R8 without sacrificing its causal, camera-selective safety contract.

```mermaid
flowchart LR
    A["t-1 query memory"] --> B["ego-motion compensation"]
    B --> C["camera projection + confidence-weighted splat"]
    C --> D["four-level FPN reconstruction / local residual"]
    E["R8: t-1 same-camera FPN replacement"] --> F["transport + residual candidate"]
    D --> F
    F --> G["frozen occupancy head"]
    G --> H["IoU / false-free / horizon audit"]
    I["feature loss"] --> J["gradient-alignment audit / PCGrad"]
    G --> J
```

The audit found three important engineering conclusions:

1. **The corrected four-FPN contract is valid:** native bypass, healthy-camera passthrough, history passthrough, and zero-initialization parity all pass.
2. **Feature reconstruction is not equivalent to task improvement:** transport/residual variants reduce feature L1 error, but their Occupancy gains do not generalize across evaluation windows.
3. **The conflict is measurable:** feature and task gradients have negative mean cosine (`-0.487` for centered/local residual and `-0.256` for transport residual), with a 100% negative fraction in the audited window. Level-wise PCGrad removes opposing components, but the preregistered task thresholds still do not pass.

Therefore, **R8 remains the supported mainline**. Full Q2F, dense residual, transport, retrieval attention, and PCGrad remain research evidence rather than claimed improvements. See the bilingual [MCQM × R8 read-only audit](docs/ROBOT_READONLY_AUDIT.md) for equations, tensor contracts, factorial design, results, and the next solution direction.

### Proposed next-generation chain: feature firewall → task-aligned retrieval

The following is a **progressive candidate architecture**, not a claimed result. It preserves the original R8 path while assigning each information source only the role it can reliably perform.

```mermaid
flowchart TB
    subgraph C0["0. Coordinate contract"]
        CT["Current clean temporal bundle"] --> CF["Current teacher slot: t"]
        CT --> HF["Shared-augmentation history slot: t-1"]
        IDA["Assert identical resize / crop / flip"] --> HF
        IDA --> CF
    end

    subgraph C1["1. Feature firewall"]
        DI["Current degraded images"] --> NF["Backbone + four-level native FPN"]
        HF --> R8["R8 same-camera replacement"]
        NF --> R8
        MQ["Memory Query: geometry + motion only"] --> FLOW["Motion-compensated local flow / visibility"]
        R8 --> TRANS["Query-guided transport of real R8 values"]
        FLOW --> TRANS
        TRANS --> CEN["Counterfactual-centered local residual<br/>T(query)-T(zero)"]
        CEN --> SAFE["Fault-sanitized FPN"]
    end

    subgraph C2["2. Query-level temporal retrieval"]
        SAFE --> OPUS["OPUS Decoder"]
        OPUS --> QB["Fault-sanitized current Query"]
        MQ --> COORD["Aligned reference points / sampling coordinates"]
        HF --> KV["Small multi-level historical evidence as K/V"]
        COORD --> KV
        QB --> ATTN["Local historical Dense Feature Retrieval Attention"]
        KV --> ATTN
        ATTN --> QOUT["Current Query + zero-init retrieval delta"]
    end

    subgraph C3["3. Task-aligned correction"]
        QOUT --> HEAD["Frozen Occupancy Head"]
        HEAD --> ZB["R8-safe Occupancy logits"]
        QOUT --> LR["Bounded Query / logit residual"]
        ZB --> ZOUT["Task-aligned output logits"]
        LR --> ZOUT
        ZOUT --> OCC["0 / 2 / 4 / 6 s Occupancy"]
        CF -. "training-only clean teacher" .-> DISTILL["Logit / decoder-state distillation"]
        DISTILL -.-> ZOUT
    end

    R8 -. "supported baseline" .-> SAFE
    classDef supported fill:#d8f3dc,stroke:#2d6a4f,stroke-width:2px,color:#081c15;
    classDef validating fill:#fff3bf,stroke:#e67700,stroke-width:2px,color:#5f3b00;
    classDef proposed fill:#dbeafe,stroke:#2563eb,stroke-width:2px,color:#172554;
    classDef trainonly fill:#f3e8ff,stroke:#7e22ce,stroke-width:1px,color:#3b0764;
    class R8 supported;
    class FLOW,TRANS,CEN,SAFE,ATTN,QOUT validating;
    class LR,ZOUT,DISTILL proposed;
    class CF,CT,IDA trainonly;
```

The structural contracts are:

$$
F_t^{safe}=F_t^{transport}+M_Q\odot\alpha\,H\!\left(F_t^{transport},T(S(Q))-T(S(0))\right),
\quad \alpha=\alpha_{max}\tanh(a),\ a_0=0,
$$

$$
Q_t^{out}=Q_t^{base}+W_o\operatorname{Attn}\!\left(Q_t^{base},K(F_{t-1}^{clean}),V(F_{t-1}^{clean})\right),
\quad W_{o,0}=0,
$$

$$
z_t^{out}=z_t^{R8}+\beta\,\Delta z(Q_t^{out}),\quad \beta_0=0.
$$

This creates five no-regret checks at initialization: exact R8 parity, zero Query ⇒ zero residual, support exterior ⇒ zero modification, healthy cameras unchanged, and retrieval disabled ⇒ the Feature route unchanged.

**Innovation focus:** R8 carries real visual content; Memory Query predicts geometry and motion; current Query requests only local historical evidence; the logit residual corrects the final task. This avoids asking a sparse Query to synthesize an arbitrary 256-channel dense feature map and prevents duplicate replay of the full `t-1` FPN.

## Experiment atlas: beyond R8

R8 is the strongest headline result, but the project is a broader systems investigation rather than a single trick. The full research sequence includes:

| Track | Question | Outcome |
|---|---|---|
| Bring-up & geometry audit | Are temporal ordering, calibration, query semantics, and occupancy labels correct? | **Infrastructure established** |
| Sensor-fault propagation | Where do camera loss, blur, and low light first damage the 4D prediction chain? | **Failure taxonomy established** |
| Reliability maps | Can risk be localized by camera sector, class, distance, and future horizon? | **Diagnostic tool established** |
| Targeted fine-tuning | Does direct retraining recover fault robustness without clean-scene regression? | **Mixed; scale- and protocol-sensitive** |
| Contributor routing | Which sparse queries and support voxels actually affect failed regions? | **Useful causal evidence; limited repair gain** |
| R8 feature memory | Can causal same-camera history repair missing front views? | **Strongest supported recovery** |
| Learned gates / residuals | Can a small learned adapter improve on R8? | **Feature error improved; occupancy gain not stable** |
| MCQM / Query-to-FPN | Can motion-compensated sparse query memory reconstruct dense FPN features? | **Mechanism validated; task contribution often harmful** |
| Gradient alignment / PCGrad | Is objective conflict blocking the residual path? | **Conflict exposed; no stable replacement for R8 yet** |
| Visualization | Can 4D occupancy and support evolution be audited frame by frame? | **BEV / first-person / temporal assets produced** |

The detailed [EXPERIMENTS.md](docs/EXPERIMENTS.md) records hypotheses, evidence, and stop/continue decisions—including negative results—without pretending every experiment was a win.

## Causal and safety contract

The public core module makes the main invariants executable:

- only `t-1` or older features may be used;
- current clean images are never used to repair a degraded frame;
- ground truth is never used in repair or routing;
- only declared failed cameras may be overwritten;
- non-target cameras remain bitwise unchanged;
- clean input is a no-op.

See [`src/sparseworld_reliability/feature_memory.py`](src/sparseworld_reliability/feature_memory.py) and its focused tests.

## Repository map

```text
SparseWorld-4D-Occupancy/
├── src/sparseworld_reliability/  # clean, dependency-light reliability core
├── tests/                        # causal, shape, and metric contracts
├── scripts/lidar_system_algorithm/
│   └── sparseworld_mainline/      # 30+ real experiment stages, 56k+ lines
├── docs/
│   ├── ARCHITECTURE.md           # model and data flow
│   ├── EXPERIMENTS.md            # complete hypothesis/evidence/decision map
│   ├── RESULTS.md                # R8 protocol and result interpretation
│   ├── REPRODUCIBILITY.md        # environment and full-pipeline guidance
│   └── CODE_MAP.md               # source-level reading guide
├── assets/                       # Template 1 and R8 evidence gallery
└── external/SparseWorld          # upstream research code as a submodule
```

The source tree deliberately keeps substantial experiment code rather than only presenting a polished toy module. It covers model bring-up, geometry and query-support audits, sensor-fault propagation, targeted fine-tuning, contributor routing, R8 causal feature memory, density-constrained repair, learned residual/gating attempts, MCQM query memory, and visualization generation. Repeated caches, checkpoints, generated test output, and active unfinished Robot work are excluded.

Start with [CODE_MAP.md](docs/CODE_MAP.md): it separates the supported path from negative-result branches and explains which files are worth reading first.

## Quick start

The reliability core is intentionally independent of MMCV/MMDetection3D:

```bash
git clone --recurse-submodules https://github.com/Rexlion777/SparseWorld-4D-Occupancy.git
cd SparseWorld-4D-Occupancy
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Running the complete nuScenes/SparseWorld pipeline additionally requires the upstream CUDA/MMCV environment, dataset, and checkpoint. See [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Evaluation protocol

The R8 numbers use the same model checkpoint, sample set, perturbation definitions, post-processing, and metric implementation for native degraded and repaired inference. Hyper-parameter selection and final evaluation are separated; failed variants remain failed rather than being relabeled as improvements.

## Upstream attribution

This project extends [MSunDYY/SparseWorld](https://github.com/MSunDYY/SparseWorld), which is included as a Git submodule. The reliability pipeline, perturbation analysis, causal feature-memory repair, evaluation contracts, and result interpretation in this repository are project-specific additions. Please follow the upstream repository's license and dataset terms.
