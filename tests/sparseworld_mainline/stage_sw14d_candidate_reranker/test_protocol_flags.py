from __future__ import annotations

from conftest import load_json


def test_protocol_flags() -> None:
    inherited = load_json("sw14d_candidate_reranker_inherited_state.json")
    final = load_json("sw14d_candidate_reranker_final_decision.json")
    assert inherited["no_sparseworld_backbone_training"] is True
    assert inherited["no_occupancy_head_training"] is True
    assert inherited["no_get_occ_or_f3_frontcap_modification"] is True
    assert inherited["no_eval_debug_or_core"] is True
    assert final["uses_eval_debug"] is False
    assert final["uses_core100_or_core500"] is False
    assert final["deployable_claim"] is False
