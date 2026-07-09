from __future__ import annotations

from typing import Any

import torch

from sw14c_error_mask_utils import build_oracle_proxy_masks


def build_proxy_mask_no_gt(teacher_h: dict[str, Any], horizon_s: int, sectors: dict[str, torch.Tensor]) -> torch.Tensor:
    return build_oracle_proxy_masks(teacher_h, horizon_s, sectors)["proxy_error_mask_no_gt"]
