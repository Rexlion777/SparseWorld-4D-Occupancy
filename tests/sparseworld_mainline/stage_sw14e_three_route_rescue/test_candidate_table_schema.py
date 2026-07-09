from __future__ import annotations

from conftest import load_json, load_table


def test_candidate_table_schema() -> None:
    table = load_table("val")
    tensors = table["tensors"]
    required = {
        "sample_id",
        "horizon_id",
        "voxel_x",
        "voxel_y",
        "voxel_z",
        "sector_id",
        "distance_bin",
        "front_region",
        "future_h4h6",
        "raw_occ",
        "raw_confidence",
        "raw_margin",
        "teacher_final_occ",
        "native_final_occ",
        "was_pruned_by_F3",
        "was_pruned_by_FrontCap",
        "GT_occ",
        "teacher_FN",
        "teacher_FP",
        "strong_add_candidate",
        "medium_add_candidate",
        "weak_add_candidate",
        "suppress_candidate",
        "keep_candidate",
    }
    assert required.issubset(tensors)
    n = int(table["meta"]["row_count"])
    assert n > 0
    assert all(int(v.numel()) == n for v in tensors.values())
    stats = load_json("sw14d_b_candidate_stats.json")
    assert stats["decision"].startswith("D_B_CAND_")
    assert stats["val_rows"] == n
