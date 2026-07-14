"""Deterministic sensor degradation with an auditable provenance manifest."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from .feature_memory import CAMERA_ORDER


@dataclass(frozen=True)
class DegradationManifest:
    degradation_id: str
    affected_cameras: tuple[str, ...]
    affected_frames: tuple[int, ...]
    replacement_value: float
    input_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def drop_cameras(
    images: np.ndarray,
    *,
    cameras: Sequence[str],
    frame_indices: Sequence[int] | None = None,
    replacement_value: float = 0.0,
) -> tuple[np.ndarray, DegradationManifest]:
    """Drop selected cameras from `[B,T,N,C,H,W]` images without mutation."""

    if images.ndim != 6:
        raise ValueError(f"expected [B,T,N,C,H,W], got shape={images.shape}")
    if images.shape[2] != len(CAMERA_ORDER):
        raise ValueError(f"expected {len(CAMERA_ORDER)} cameras, got {images.shape[2]}")

    camera_names = tuple(cameras)
    unknown = set(camera_names) - set(CAMERA_ORDER)
    if unknown:
        raise ValueError(f"unknown cameras: {sorted(unknown)}")

    frames = tuple(range(images.shape[1])) if frame_indices is None else tuple(frame_indices)
    if any(frame < 0 or frame >= images.shape[1] for frame in frames):
        raise IndexError(f"frame index outside [0, {images.shape[1]})")

    output = np.array(images, copy=True)
    camera_indices = [CAMERA_ORDER.index(name) for name in camera_names]
    for frame in frames:
        for camera in camera_indices:
            output[:, frame, camera] = replacement_value

    degradation_id = "drop_" + "_".join(name.removeprefix("CAM_").lower() for name in camera_names)
    manifest = DegradationManifest(
        degradation_id=degradation_id,
        affected_cameras=camera_names,
        affected_frames=frames,
        replacement_value=float(replacement_value),
        input_shape=tuple(images.shape),
    )
    return output, manifest
