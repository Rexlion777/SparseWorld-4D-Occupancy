import numpy as np
import pytest

from sparseworld_reliability import RepairPolicy, repair_failed_cameras
from sparseworld_reliability.feature_memory import CAMERA_ORDER


def make_levels(fill: float) -> list[np.ndarray]:
    return [
        np.full((1, 6, 4, 8, 12), fill, dtype=np.float32),
        np.full((1, 6, 4, 4, 6), fill, dtype=np.float32),
    ]


def test_r8_replaces_only_front_triplet() -> None:
    current = make_levels(0.0)
    memory = make_levels(1.0)
    policy = RepairPolicy(
        failed_cameras=("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"),
        history_offset=1,
    )

    repaired, audit = repair_failed_cameras(current, {1: memory}, policy)

    repaired_names = set(policy.failed_cameras)
    for level in repaired:
        for camera_index, camera_name in enumerate(CAMERA_ORDER):
            expected = 1.0 if camera_name in repaired_names else 0.0
            assert np.all(level[:, camera_index] == expected)
    assert audit["causal"] is True
    assert audit["source_offset"] == -1
    assert audit["uses_ground_truth"] is False
    assert audit["uses_current_clean_frame"] is False


def test_inputs_are_not_mutated() -> None:
    current = make_levels(0.0)
    originals = [level.copy() for level in current]
    repair_failed_cameras(
        current,
        {1: make_levels(1.0)},
        RepairPolicy(("CAM_FRONT",)),
    )
    assert all(np.array_equal(now, before) for now, before in zip(current, originals))


def test_noncausal_offset_is_rejected() -> None:
    with pytest.raises(ValueError, match="causal"):
        RepairPolicy(("CAM_FRONT",), history_offset=0)
