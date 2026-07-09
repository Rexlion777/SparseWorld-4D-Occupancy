import torch

from sw14c_residual_adapter_modules import FRONT_TRIPLET_INDICES, build_sw13_r8_base_repair


def test_r8_base_repair_front_triplet_replaced_all_levels() -> None:
    current = [torch.randn(1, 6, 4, 3, 3) + level for level in range(4)]
    memory = [level_tensor + 10.0 for level_tensor in current]
    repaired = build_sw13_r8_base_repair(current, memory, "A10_drop_front_triplet")
    for level_idx in range(4):
        assert torch.equal(repaired[level_idx][:, FRONT_TRIPLET_INDICES], memory[level_idx][:, FRONT_TRIPLET_INDICES])


def test_r8_base_repair_clean_noop() -> None:
    current = [torch.randn(1, 6, 4, 3, 3) for _ in range(4)]
    memory = [level_tensor + 10.0 for level_tensor in current]
    repaired = build_sw13_r8_base_repair(current, memory, "A0_clean")
    for level_idx in range(4):
        assert torch.equal(repaired[level_idx], current[level_idx])
