import torch

from sw14c_residual_adapter_modules import SpatialResidualAdapter, build_sw13_r8_base_repair, front_triplet_degradation_mask


def test_residual_init_equals_sw13_r8_base() -> None:
    torch.manual_seed(3)
    current = [torch.randn(1, 6, 8, 4, 4) for _ in range(4)]
    memory = [level + 1.0 for level in current]
    base = build_sw13_r8_base_repair(current, memory, "A10_drop_front_triplet")
    adapter = SpatialResidualAdapter([8, 8, 8, 8], hidden_channels=4, camera_embed_dim=2)
    mask = front_triplet_degradation_mask("A10_drop_front_triplet", batch_size=1)
    final, debug = adapter.apply_to_levels(current, memory, base, mask, gamma=0.50)
    for final_level, base_level in zip(final, base):
        assert torch.equal(final_level, base_level)
    assert all(value.abs().sum().item() == 0.0 for key, value in debug.items() if key.startswith("residual_delta"))
