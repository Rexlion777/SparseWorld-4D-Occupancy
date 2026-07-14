# Architecture

## 1. Baseline path

SparseWorld consumes five temporal frames from six surround cameras. The image backbone and FPN produce four multi-scale feature levels. Sparse current and future queries interact with image features and are decoded into semantic occupancy at four horizons: 0 s, 2 s, 4 s, and 6 s.

| Item | Shape / count |
|---|---:|
| Input images | 5 × 6 = 30 |
| Sparse queries | 1,040 |
| Current / future queries | 720 / 320 |
| Dense occupancy grid | 200 × 200 × 16 = 640,000 voxels |
| Semantic labels | 17 foreground classes + empty |

## 2. Reliability path

The system sits around, rather than inside, the upstream model:

1. Load the temporal multi-camera sample and calibration metadata.
2. Apply a deterministic sensor perturbation and write a provenance manifest.
3. Run the native degraded model and capture post-neck FPN features.
4. Load only causal, same-camera historical features.
5. Apply the declared repair policy to failed cameras and no others.
6. Re-run the unchanged occupancy head.
7. Compare native degraded and repaired outputs by region, class, and horizon.

## 3. R8 design

R8 targets front-triplet camera loss. It replaces `CAM_FRONT`, `CAM_FRONT_LEFT`, and `CAM_FRONT_RIGHT` at all four post-neck FPN levels using the corresponding `t-1` feature tensors. Rear-camera features remain current and unchanged.

This is deliberately simpler than a learned fusion block. Its value is that the source, target, time offset, and overwrite mask are all inspectable. It establishes a strong, causal recovery baseline before more complex learned memory is justified.

## 4. Failure analysis

Aggregate IoU alone can hide unsafe changes. The evaluation therefore separates:

- false-negative recovery in the forward sector;
- false occupancy introduced into previously free space;
- dynamic and small-object behavior;
- near/far and camera-sector behavior;
- error propagation over 0/2/4/6 s;
- occupancy-density drift;
- model and post-processing latency.
