from __future__ import annotations

from conftest import load_csv, load_json


def test_roi_generation_constraints() -> None:
    rows = load_csv("sw14i_roi_stats.csv")
    assert {r["split"] for r in rows} == {"train", "val"}
    for row in rows:
        assert int(row["roi_count"]) > 0
        assert int(row["weak_add_in_main_roi"]) == 0
    target = load_json("sw14i_generation_target_stats.json")
    assert target["decision"] in {
        "I_TARGET_1_READY",
        "I_TARGET_2_MISSING_TOO_SPARSE",
        "I_TARGET_3_ROI_TOO_NOISY",
        "I_TARGET_4_TARGET_BUG",
    }
