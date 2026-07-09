from torch import nn

from sw14c_residual_adapter_modules import SpatialResidualAdapter, freeze_module


def test_only_adapter_trainable_when_model_is_frozen() -> None:
    frozen_model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    freeze_module(frozen_model)
    adapter = SpatialResidualAdapter([8, 8, 8, 8], hidden_channels=4, camera_embed_dim=2)
    assert all(not param.requires_grad for param in frozen_model.parameters())
    assert any(param.requires_grad for param in adapter.parameters())
