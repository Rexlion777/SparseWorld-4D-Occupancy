from __future__ import annotations

from conftest import load_json


def test_transformer_no_gt_in_inference_features() -> None:
    manifest = load_json("sw14hi_baseline_freeze_manifest.json")["inference_feature_manifest"]
    gt_cols = set(manifest["gt_columns_not_used_at_inference"])
    assert manifest["inference_features_contain_gt"] is False
    assert not (set(manifest["h_query_features"]) & gt_cols)
