# Reproducibility

## Lightweight audit

The extracted R8 policy and metric definitions run without the upstream perception stack:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

## Full SparseWorld pipeline

The full experiment additionally requires:

- Linux with a CUDA-capable GPU;
- the CUDA/PyTorch/MMCV/MMDetection3D versions required by upstream SparseWorld;
- nuScenes data organized according to the upstream repository;
- the upstream SparseWorld checkpoint;
- initialization of `external/SparseWorld` via `git submodule update --init --recursive`.

Data, checkpoints, caches, generated videos, and per-run artifacts are intentionally excluded from Git. The repository stores code, protocol, and compact result summaries only.

## Suggested verification order

1. Run the lightweight unit tests.
2. Verify upstream clean inference on one nuScenes sample.
3. Verify calibration and temporal ordering on the same sample.
4. Run native degraded inference and record its manifest.
5. Run R8 with the identical sample and perturbation manifest.
6. Confirm clean/non-target parity and absence of future/GT inputs.
7. Evaluate FN, false occupancy, density drift, and horizon breakdown together.
