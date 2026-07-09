import torch
from torch import nn

from sw14c_residual_adapter_modules import freeze_module


def test_backbone_head_can_be_frozen() -> None:
    backbone = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    head = nn.Linear(4, 2)
    freeze_module(backbone)
    freeze_module(head)
    assert all(not param.requires_grad for param in backbone.parameters())
    assert all(not param.requires_grad for param in head.parameters())
    with torch.no_grad():
        _ = head(backbone(torch.randn(2, 4)))
