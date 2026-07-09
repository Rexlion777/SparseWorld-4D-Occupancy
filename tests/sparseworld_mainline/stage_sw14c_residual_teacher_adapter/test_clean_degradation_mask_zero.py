import torch

from sw14c_residual_adapter_modules import SpatialResidualAdapter, build_sw13_r8_base_repair, front_triplet_degradation_mask


def test_clean_degradation_mask_zero_and_noop() -> None:
    current = [torch.randn(1, 6, 8, 3, 3) for _ in range(4)]
    memory = [level + 2.0 for level in current]
    base = build_sw13_r8_base_repair(current, memory, "A0_clean")
    adapter = SpatialResidualAdapter([8, 8, 8, 8], hidden_channels=4, camera_embed_dim=2)
    mask = front_triplet_degradation_mask("A0_clean", batch_size=1)
    assert mask.sum().item() == 0.0
    final, debug = adapter.apply_to_levels(current, memory, base, mask, gamma=0.50)
    for level_idx in range(4):
        assert torch.equal(final[level_idx], current[level_idx])
    assert all(value.abs().sum().item() == 0.0 for key, value in debug.items() if key.startswith("residual_gate"))
