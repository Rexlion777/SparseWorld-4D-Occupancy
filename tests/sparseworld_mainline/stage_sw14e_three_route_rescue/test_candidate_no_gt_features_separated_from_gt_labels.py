from __future__ import annotations

from conftest import load_table


def test_candidate_no_gt_features_separated_from_gt_labels() -> None:
    meta = load_table("train")["meta"]
    features = set(meta["no_gt_feature_columns"])
    labels = set(meta["gt_label_columns"])
    assert features
    assert labels == {"GT_occ", "teacher_FN", "teacher_FP", "teacher_correct_occ", "teacher_correct_free"}
    assert features.isdisjoint(labels)
    assert "GT_occ" not in features
    assert "teacher_FN" not in features
    assert "teacher_FP" not in features
