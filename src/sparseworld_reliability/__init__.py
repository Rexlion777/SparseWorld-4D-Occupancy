"""Dependency-light reliability components extracted from the SparseWorld study."""

from .degradation import DegradationManifest, drop_cameras
from .feature_memory import RepairPolicy, repair_failed_cameras
from .metrics import occupancy_metrics, relative_error_reduction

__all__ = [
    "DegradationManifest",
    "RepairPolicy",
    "drop_cameras",
    "occupancy_metrics",
    "repair_failed_cameras",
    "relative_error_reduction",
]
