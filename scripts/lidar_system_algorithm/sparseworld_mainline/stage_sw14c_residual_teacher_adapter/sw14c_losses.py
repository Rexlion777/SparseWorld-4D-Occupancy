from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SW14CLossConfig:
    lambda_teacher_fn_recover: float = 3.0
    lambda_teacher_fp_suppress: float = 2.0
    lambda_teacher_correct_preserve: float = 1.0
    lambda_density: float = 3.0
    lambda_fp: float = 3.0
    lambda_residual_l1: float = 0.5
    lambda_residual_smooth: float = 0.1
    lambda_gate_sparse: float = 0.2
    lambda_clean: float = 2.0
    lambda_rear_consistency: float = 2.0
    target_density_delta_rule: str = "min(SW13_teacher_density_delta + 0.02, 0.08)"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LOSS_TERMS = [
    {
        "term": "L_teacher_FN_recover",
        "region": "GT occupied and SW13 teacher final predicts free/empty",
        "purpose": "recover errors SW13 still misses",
        "uses_eval_gt_for_training": False,
        "uses_train_gt_for_training": True,
    },
    {
        "term": "L_teacher_FP_suppress",
        "region": "GT free and SW13 teacher final predicts occupied",
        "purpose": "suppress stale/ghost occupied regions from the hard t-1 teacher",
        "uses_eval_gt_for_training": False,
        "uses_train_gt_for_training": True,
    },
    {
        "term": "L_teacher_correct_preserve",
        "region": "teacher-correct voxels",
        "purpose": "preserve the already strong R8 base result",
        "uses_eval_gt_for_training": False,
        "uses_train_gt_for_training": True,
    },
    {
        "term": "L_density_hinge",
        "region": "raw student proxy",
        "purpose": "avoid gaining recall by unsafe density growth",
        "uses_eval_gt_for_training": False,
        "uses_train_gt_for_training": True,
    },
    {
        "term": "L_residual_regularization",
        "region": "front triplet degraded camera features",
        "purpose": "keep residual small, smooth, and sparse",
        "uses_eval_gt_for_training": False,
        "uses_train_gt_for_training": False,
    },
    {
        "term": "L_clean_rear_consistency",
        "region": "A0 clean and rear cameras",
        "purpose": "enforce clean/rear no-op behavior",
        "uses_eval_gt_for_training": False,
        "uses_train_gt_for_training": False,
    },
]
