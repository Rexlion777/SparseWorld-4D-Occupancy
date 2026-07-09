from __future__ import annotations

from conftest import load_json


def test_backbone_head_frozen() -> None:
    summary = load_json("sw14c_oracle_mask_training_summary.json")
    frozen = summary["backbone_head_frozen"]
    assert frozen["backbone_frozen"] is True
    assert frozen["head_frozen"] is True
    assert int(frozen["model_trainable_param_count"]) == 0
    assert int(frozen["head_trainable_param_count"]) == 0
