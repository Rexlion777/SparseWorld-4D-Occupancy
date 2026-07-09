import torch

from sw14c_residual_adapter_modules import REAR_CAMERA_INDICES, SpatialResidualAdapter, build_sw13_r8_base_repair, front_triplet_degradation_mask


def test_rear_cameras_unchanged_after_residual_adapter() -> None:
    current = [torch.randn(1, 6, 8, 3, 3) for _ in range(4)]
    memory = [level + 2.0 for level in current]
    base = build_sw13_r8_base_repair(current, memory, "A10_drop_front_triplet")
    adapter = SpatialResidualAdapter([8, 8, 8, 8], hidden_channels=4, camera_embed_dim=2)
    mask = front_triplet_degradation_mask("A10_drop_front_triplet", batch_size=1)
    final, _debug = adapter.apply_to_levels(current, memory, base, mask, gamma=0.30)
    for level_idx in range(4):
        assert torch.equal(final[level_idx][:, REAR_CAMERA_INDICES], current[level_idx][:, REAR_CAMERA_INDICES])
