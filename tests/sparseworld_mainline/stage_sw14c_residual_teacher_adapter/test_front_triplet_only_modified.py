import torch

from sw14c_residual_adapter_modules import FRONT_TRIPLET_INDICES, build_sw13_r8_base_repair


def test_front_triplet_only_modified_by_r8_base() -> None:
    current = [torch.randn(1, 6, 4, 2, 2)]
    memory = [current[0] + 7.0]
    repaired = build_sw13_r8_base_repair(current, memory, "A10_drop_front_triplet")[0]
    for camera_idx in range(6):
        expected = memory[0][:, camera_idx] if camera_idx in FRONT_TRIPLET_INDICES else current[0][:, camera_idx]
        assert torch.equal(repaired[:, camera_idx], expected)
