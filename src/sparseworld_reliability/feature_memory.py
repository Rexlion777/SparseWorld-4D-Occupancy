"""Causal camera-feature repair used by the R8 reliability path.

The full SparseWorld experiment works on four post-neck FPN levels with tensors
shaped ``[batch, camera, channel, height, width]``.  This module keeps the core
policy explicit and independently testable without importing MMCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


CAMERA_ORDER = (
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
)


@dataclass(frozen=True)
class RepairPolicy:
    """A causal feature-repair policy.

    ``history_offset`` must be positive: offset 1 means t-1.  The policy never
    receives ground truth or a current clean frame, making those data leaks
    impossible at this interface.
    """

    failed_cameras: tuple[str, ...]
    history_offset: int = 1

    def __post_init__(self) -> None:
        unknown = set(self.failed_cameras) - set(CAMERA_ORDER)
        if unknown:
            raise ValueError(f"unknown cameras: {sorted(unknown)}")
        if self.history_offset < 1:
            raise ValueError("history_offset must be >= 1 for causal repair")


def _validate_level_pair(current: np.ndarray, memory: np.ndarray, level: int) -> None:
    if current.shape != memory.shape:
        raise ValueError(
            f"level {level}: current/memory shape mismatch "
            f"{current.shape} != {memory.shape}"
        )
    if current.ndim != 5:
        raise ValueError(f"level {level}: expected [B,N,C,H,W], got ndim={current.ndim}")
    if current.shape[1] != len(CAMERA_ORDER):
        raise ValueError(
            f"level {level}: expected {len(CAMERA_ORDER)} cameras, got {current.shape[1]}"
        )


def repair_failed_cameras(
    current_levels: Sequence[np.ndarray],
    memory_by_offset: Mapping[int, Sequence[np.ndarray]],
    policy: RepairPolicy,
) -> tuple[list[np.ndarray], dict[str, object]]:
    """Replace only failed camera features with causal same-camera memory.

    Args:
        current_levels: Post-FPN tensors for the degraded current frame.
        memory_by_offset: Historical tensors keyed by positive time offset.
        policy: Declares exactly which cameras may be replaced.

    Returns:
        Repaired copies and an audit dictionary. Inputs are never mutated.
    """

    if policy.history_offset not in memory_by_offset:
        raise KeyError(f"missing history offset t-{policy.history_offset}")

    memory_levels = memory_by_offset[policy.history_offset]
    if len(current_levels) != len(memory_levels):
        raise ValueError("current and memory must contain the same number of FPN levels")

    repaired = [np.array(level, copy=True) for level in current_levels]
    camera_indices = [CAMERA_ORDER.index(name) for name in policy.failed_cameras]
    changed_elements = 0

    for level_index, (current, memory) in enumerate(zip(current_levels, memory_levels)):
        _validate_level_pair(current, memory, level_index)
        for camera_index in camera_indices:
            before = repaired[level_index][:, camera_index].copy()
            repaired[level_index][:, camera_index] = memory[:, camera_index]
            changed_elements += int(np.count_nonzero(before != memory[:, camera_index]))

    audit = {
        "causal": True,
        "source_offset": -policy.history_offset,
        "repaired_cameras": list(policy.failed_cameras),
        "fpn_levels": len(repaired),
        "changed_elements": changed_elements,
        "uses_ground_truth": False,
        "uses_current_clean_frame": False,
    }
    return repaired, audit
