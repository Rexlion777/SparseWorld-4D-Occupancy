from __future__ import annotations

from conftest import load_json


def test_sw14g_insertion_audit_schema() -> None:
    audit = load_json("sw14g_insertion_point_audit.json")
    assert audit["decision"] in {
        "G_INSERT_1_QUERY_POINT_FOUND",
        "G_INSERT_2_ONLY_CAMERA_FEATURE_POINT",
        "G_INSERT_3_ONLY_SCORE_POINT",
        "G_INSERT_4_NO_SAFE_INSERTION_POINT",
    }
    points = audit["candidate_insertion_points"]
    assert len(points) >= 3
    required = {
        "point",
        "tensor_name",
        "shape",
        "dtype",
        "requires_grad",
        "camera_dimension_present",
        "query_voxel_alignment",
        "close_to_final_occupancy",
        "noop_init_possible",
        "engineering_cost",
        "risk",
    }
    assert required.issubset(points[0])
