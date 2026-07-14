import numpy as np

from sparseworld_reliability import drop_cameras
from sparseworld_reliability.feature_memory import CAMERA_ORDER


def test_front_triplet_dropout_is_selective_and_auditable() -> None:
    images = np.ones((1, 5, 6, 3, 4, 4), dtype=np.float32)
    front_triplet = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")
    degraded, manifest = drop_cameras(images, cameras=front_triplet)

    for index, name in enumerate(CAMERA_ORDER):
        expected = 0.0 if name in front_triplet else 1.0
        assert np.all(degraded[:, :, index] == expected)
    assert np.all(images == 1.0)
    assert manifest.affected_frames == (0, 1, 2, 3, 4)
    assert manifest.affected_cameras == front_triplet
