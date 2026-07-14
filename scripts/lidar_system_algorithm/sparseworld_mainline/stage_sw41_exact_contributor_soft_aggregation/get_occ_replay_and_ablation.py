"""Exact get_occ replay and diagnostic ablations for Stage SW-4.1.

This module reconstructs SparseWorld OPUSHead.get_occ() into modular steps and
adds non-training diagnostic ablation variants:
- soft splatting
- adaptive gate relax
- class-aware aggregation

Safe-claim boundary:
- diagnostic repair ablation only
- no training
- no official benchmark
- oracle variants must remain separate from deployable candidates
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch_scatter


EMPTY_IDX = 17
SMALL_OBJECT_IDS = [1, 2, 6, 7, 8]
DYNAMIC_IDS = [2, 3, 4, 5, 6, 7, 9, 10]


@dataclass
class GateConfig:
    name: str = "G0_baseline_gate"
    margin_thr: float | None = None
    margin_band: float = 0.08
    entropy_thr: float | None = None
    entropy_band: float = 0.08
    near_thr_band: float | None = None
    future_band: float | None = None
    low_density_count: int | None = None
    low_density_band: float | None = None
    small_object_band: float | None = None


@dataclass
class AssignConfig:
    name: str = "V0_hard_baseline"
    mode: str = "hard"
    radius_xy: int = 0
    radius_z: int = 0
    sigma_xy: float = 1.0
    sigma_z: float = 1.0
    topk: int = 0
    distance_weighted: bool = False
    small_object_only: bool = False


@dataclass
class AggregateConfig:
    name: str = "A0_hard_max"
    mode: str = "max"
    small_object_scale: float = 1.0
    dynamic_scale: float = 1.0
    preserve_small_topk: int = 0
    balanced_topk: int = 0


def tensor_to_cpu(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: tensor_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [tensor_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return [tensor_to_cpu(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _apply_class_specific_distance_rescale(
    cls_scores_sigmoid: torch.Tensor,
    raw_refine_pts: torch.Tensor,
    thre1: float | None,
    thre2: float | None,
) -> torch.Tensor:
    cls_scores_sigmoid = cls_scores_sigmoid.clone()
    if thre1 is not None:
        mask = cls_scores_sigmoid.argmax(-1) == 15
        dis = torch.norm(raw_refine_pts - torch.mean(raw_refine_pts, dim=2, keepdim=True), dim=-1)
        cls_scores_sigmoid[mask] = cls_scores_sigmoid[mask] * torch.clamp(thre1 / dis[mask], max=1)[:, None]
    if thre2 is not None:
        mask = cls_scores_sigmoid.argmax(-1) == 16
        dis = torch.norm(raw_refine_pts - torch.mean(raw_refine_pts, dim=2, keepdim=True), dim=-1)
        cls_scores_sigmoid[mask] = cls_scores_sigmoid[mask] * torch.clamp(thre2 / dis[mask], max=1)[:, None]
    return cls_scores_sigmoid


def decode_support_points(head: Any, pred_dict: dict[str, torch.Tensor], thre1: float | None = 0.1, thre2: float | None = 0.1) -> dict[str, torch.Tensor]:
    from mmdet3d.models.sparsedetectors.bbox.utils import decode_points

    raw_logits = pred_dict["cls_scores"]
    raw_refine_pts = pred_dict["refine_pts"]
    cls_scores_sigmoid = _apply_class_specific_distance_rescale(raw_logits.sigmoid(), raw_refine_pts, thre1, thre2)
    decoded_points = decode_points(raw_refine_pts[0], head.pc_range)
    cls_scores = cls_scores_sigmoid[0]
    centers = decoded_points.mean(dim=1, keepdim=True)
    ctr_dists = torch.norm(decoded_points - centers, dim=-1)
    top2_vals, top2_idx = torch.topk(cls_scores, k=min(2, cls_scores.shape[-1]), dim=-1)
    best_score = top2_vals[..., 0]
    best_class = top2_idx[..., 0]
    second_score = top2_vals[..., 1] if top2_vals.shape[-1] > 1 else torch.zeros_like(best_score)
    margin = best_score - second_score
    entropy = -(cls_scores.clamp(min=1e-8) * torch.log(cls_scores.clamp(min=1e-8))).sum(dim=-1)
    pre_voxel_index = ((decoded_points.reshape(-1, 3) - head.pc_range[:3]) // head.voxel_size).long()
    voxel_num = head.voxel_num
    valid_range_mask = torch.logical_and(pre_voxel_index >= 0, pre_voxel_index < voxel_num).all(-1)
    return {
        "raw_logits": raw_logits[0],
        "raw_refine_pts": raw_refine_pts[0],
        "cls_scores": cls_scores,
        "decoded_points": decoded_points,
        "centers": centers,
        "ctr_dists": ctr_dists,
        "best_score": best_score,
        "best_class": best_class,
        "second_score": second_score,
        "margin": margin,
        "entropy": entropy,
        "pre_voxel_index_flat": pre_voxel_index,
        "valid_range_mask_flat": valid_range_mask,
    }


def _local_density_counts(pre_voxel_index_flat: torch.Tensor, valid_range_mask_flat: torch.Tensor) -> torch.Tensor:
    counts = torch.zeros(pre_voxel_index_flat.shape[0], device=pre_voxel_index_flat.device, dtype=torch.long)
    if not bool(valid_range_mask_flat.any().item()):
        return counts
    vox = pre_voxel_index_flat[valid_range_mask_flat]
    unique_vox, inv, pts_num = torch.unique(vox, return_inverse=True, return_counts=True, dim=0)
    counts_valid = pts_num[inv]
    counts[valid_range_mask_flat] = counts_valid
    return counts


def compute_gate_mask(
    head: Any,
    point_info: dict[str, torch.Tensor],
    gate_cfg: GateConfig,
    horizon_s: int,
) -> dict[str, torch.Tensor]:
    score_thr = torch.as_tensor(head.test_cfg.get("score_thr", 0.1), device=point_info["cls_scores"].device, dtype=point_info["cls_scores"].dtype)
    point_thr = score_thr[point_info["best_class"]]
    mask_dist = point_info["ctr_dists"] < head.test_cfg.get("ctr_dist_thr", 3.0)
    base_mask_score = point_info["best_score"] > point_thr
    relaxed_mask = torch.zeros_like(base_mask_score, dtype=torch.bool)
    if gate_cfg.margin_thr is not None:
        relaxed_mask |= (point_info["margin"] < gate_cfg.margin_thr) & (point_info["best_score"] > (point_thr - gate_cfg.margin_band))
    if gate_cfg.entropy_thr is not None:
        relaxed_mask |= (point_info["entropy"] > gate_cfg.entropy_thr) & (point_info["best_score"] > (point_thr - gate_cfg.entropy_band))
    if gate_cfg.near_thr_band is not None:
        relaxed_mask |= (point_info["best_score"] > (point_thr - gate_cfg.near_thr_band)) & (~base_mask_score)
    if gate_cfg.future_band is not None and horizon_s >= 1:
        relaxed_mask |= point_info["best_score"] > (point_thr - gate_cfg.future_band)
    local_density = _local_density_counts(point_info["pre_voxel_index_flat"], point_info["valid_range_mask_flat"]).reshape_as(base_mask_score)
    if gate_cfg.low_density_count is not None and gate_cfg.low_density_band is not None:
        relaxed_mask |= (local_density <= gate_cfg.low_density_count) & (point_info["best_score"] > (point_thr - gate_cfg.low_density_band))
    if gate_cfg.small_object_band is not None:
        topk = torch.topk(point_info["cls_scores"], k=min(3, point_info["cls_scores"].shape[-1]), dim=-1).indices
        small_mask = torch.zeros_like(base_mask_score, dtype=torch.bool)
        for cid in SMALL_OBJECT_IDS:
            small_mask |= (topk == cid).any(dim=-1)
        relaxed_mask |= small_mask & (point_info["best_score"] > (point_thr - gate_cfg.small_object_band))
    mask_score = base_mask_score | relaxed_mask
    gate_mask = mask_dist & mask_score
    return {
        "mask_dist": mask_dist,
        "point_thr": point_thr,
        "base_mask_score": base_mask_score,
        "mask_score": mask_score,
        "relaxed_mask": relaxed_mask,
        "gate_mask": gate_mask,
        "local_density": local_density,
    }


def _flatten_points(point_info: dict[str, torch.Tensor], gate_info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    points_flat = point_info["decoded_points"].reshape(-1, 3)
    scores_flat = point_info["cls_scores"].reshape(-1, point_info["cls_scores"].shape[-1])
    gate_mask_flat = gate_info["gate_mask"].reshape(-1)
    best_class_flat = point_info["best_class"].reshape(-1)
    best_score_flat = point_info["best_score"].reshape(-1)
    margin_flat = point_info["margin"].reshape(-1)
    entropy_flat = point_info["entropy"].reshape(-1)
    valid_range_mask_flat = point_info["valid_range_mask_flat"]
    pre_voxel_index_flat = point_info["pre_voxel_index_flat"]
    local_density_flat = gate_info["local_density"].reshape(-1)
    flat_index = torch.arange(points_flat.shape[0], device=points_flat.device)
    return {
        "points_flat": points_flat,
        "scores_flat": scores_flat,
        "gate_mask_flat": gate_mask_flat,
        "best_class_flat": best_class_flat,
        "best_score_flat": best_score_flat,
        "margin_flat": margin_flat,
        "entropy_flat": entropy_flat,
        "valid_range_mask_flat": valid_range_mask_flat,
        "pre_voxel_index_flat": pre_voxel_index_flat,
        "local_density_flat": local_density_flat,
        "flat_index": flat_index,
    }


def _voxel_center_distance(cont_xyz: torch.Tensor, target_idx: torch.Tensor) -> torch.Tensor:
    return torch.norm(cont_xyz - (target_idx.float() + 0.5), dim=-1)


def assign_points_to_voxels_hard(head: Any, flat: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mask = flat["gate_mask_flat"] & flat["valid_range_mask_flat"]
    voxel_indices = flat["pre_voxel_index_flat"][mask]
    points_metric = flat["points_flat"][mask]
    scores = flat["scores_flat"][mask]
    flat_index = flat["flat_index"][mask]
    best_class = flat["best_class_flat"][mask]
    best_score = flat["best_score_flat"][mask]
    margin = flat["margin_flat"][mask]
    entropy = flat["entropy_flat"][mask]
    local_density = flat["local_density_flat"][mask]
    if voxel_indices.numel():
        cont_xyz = (points_metric - head.pc_range[:3]) / head.voxel_size
        weights = torch.ones(voxel_indices.shape[0], device=voxel_indices.device, dtype=points_metric.dtype)
        point_dist = _voxel_center_distance(cont_xyz, voxel_indices)
    else:
        weights = scores.new_zeros((0,))
        point_dist = scores.new_zeros((0,))
    return {
        "target_voxel_indices": voxel_indices,
        "point_metric": points_metric,
        "point_scores": scores,
        "point_flat_index": flat_index,
        "point_best_class": best_class,
        "point_best_score": best_score,
        "point_margin": margin,
        "point_entropy": entropy,
        "point_local_density": local_density,
        "point_weight": weights,
        "point_to_target_dist": point_dist,
    }


def _neighbor_offsets(radius_xy: int, radius_z: int, device: torch.device) -> torch.Tensor:
    offsets = []
    for dx in range(-radius_xy, radius_xy + 1):
        for dy in range(-radius_xy, radius_xy + 1):
            for dz in range(-radius_z, radius_z + 1):
                offsets.append([dx, dy, dz])
    return torch.tensor(offsets, device=device, dtype=torch.long)


def _expand_soft_targets(
    head: Any,
    points_metric: torch.Tensor,
    scores: torch.Tensor,
    point_flat_index: torch.Tensor,
    best_class: torch.Tensor,
    best_score: torch.Tensor,
    margin: torch.Tensor,
    entropy: torch.Tensor,
    local_density: torch.Tensor,
    radius_xy: int,
    radius_z: int,
    sigma_xy: float,
    sigma_z: float,
    distance_weighted: bool,
    small_object_only: bool,
) -> dict[str, torch.Tensor]:
    if points_metric.numel() == 0:
        empty = points_metric.new_zeros((0, 3), dtype=torch.long)
        return {
            "target_voxel_indices": empty,
            "point_metric": points_metric,
            "point_scores": scores,
            "point_flat_index": point_flat_index,
            "point_best_class": best_class,
            "point_best_score": best_score,
            "point_margin": margin,
            "point_entropy": entropy,
            "point_local_density": local_density,
            "point_weight": points_metric.new_zeros((0,)),
            "point_to_target_dist": points_metric.new_zeros((0,)),
        }
    cont_xyz = (points_metric - head.pc_range[:3]) / head.voxel_size
    base_vox = torch.floor(cont_xyz).long()
    offsets = _neighbor_offsets(radius_xy, radius_z, points_metric.device)
    if small_object_only:
        topk = torch.topk(scores, k=min(3, scores.shape[-1]), dim=-1).indices
        small_mask = torch.zeros(scores.shape[0], device=scores.device, dtype=torch.bool)
        for cid in SMALL_OBJECT_IDS:
            small_mask |= (topk == cid).any(dim=-1)
    else:
        small_mask = torch.ones(scores.shape[0], device=scores.device, dtype=torch.bool)
    expanded_targets = []
    expanded_points = []
    expanded_scores = []
    expanded_flat_idx = []
    expanded_best_class = []
    expanded_best_score = []
    expanded_margin = []
    expanded_entropy = []
    expanded_local_density = []
    expanded_weight = []
    expanded_dist = []
    voxel_num = head.voxel_num.long()
    for offset in offsets:
        target = base_vox + offset[None, :]
        valid = torch.logical_and(target >= 0, target < voxel_num).all(dim=-1) & small_mask
        if not bool(valid.any().item()):
            continue
        target_valid = target[valid]
        cont_valid = cont_xyz[valid]
        dist_xy = torch.norm(cont_valid[:, :2] - (target_valid[:, :2].float() + 0.5), dim=-1)
        dist_z = torch.abs(cont_valid[:, 2] - (target_valid[:, 2].float() + 0.5))
        if distance_weighted or radius_xy > 0 or radius_z > 0:
            weight = torch.exp(-(dist_xy**2) / max(1e-6, sigma_xy**2)) * torch.exp(-(dist_z**2) / max(1e-6, sigma_z**2))
        else:
            weight = torch.ones_like(dist_xy)
        expanded_targets.append(target_valid)
        expanded_points.append(points_metric[valid])
        expanded_scores.append(scores[valid])
        expanded_flat_idx.append(point_flat_index[valid])
        expanded_best_class.append(best_class[valid])
        expanded_best_score.append(best_score[valid])
        expanded_margin.append(margin[valid])
        expanded_entropy.append(entropy[valid])
        expanded_local_density.append(local_density[valid])
        expanded_weight.append(weight)
        expanded_dist.append(torch.sqrt(dist_xy**2 + dist_z**2))
    if not expanded_targets:
        return assign_points_to_voxels_hard(head, {
            "gate_mask_flat": torch.zeros(head.num_query * 1, device=points_metric.device, dtype=torch.bool),  # unreachable placeholder
            "valid_range_mask_flat": torch.zeros(head.num_query * 1, device=points_metric.device, dtype=torch.bool),
            "pre_voxel_index_flat": torch.zeros((head.num_query * 1, 3), device=points_metric.device, dtype=torch.long),
            "points_flat": points_metric,
            "scores_flat": scores,
            "flat_index": point_flat_index,
            "best_class_flat": best_class,
            "best_score_flat": best_score,
            "margin_flat": margin,
            "entropy_flat": entropy,
            "local_density_flat": local_density,
        })
    return {
        "target_voxel_indices": torch.cat(expanded_targets, dim=0),
        "point_metric": torch.cat(expanded_points, dim=0),
        "point_scores": torch.cat(expanded_scores, dim=0),
        "point_flat_index": torch.cat(expanded_flat_idx, dim=0),
        "point_best_class": torch.cat(expanded_best_class, dim=0),
        "point_best_score": torch.cat(expanded_best_score, dim=0),
        "point_margin": torch.cat(expanded_margin, dim=0),
        "point_entropy": torch.cat(expanded_entropy, dim=0),
        "point_local_density": torch.cat(expanded_local_density, dim=0),
        "point_weight": torch.cat(expanded_weight, dim=0),
        "point_to_target_dist": torch.cat(expanded_dist, dim=0),
    }


def assign_points_to_voxels_soft(head: Any, flat: dict[str, torch.Tensor], assign_cfg: AssignConfig) -> dict[str, torch.Tensor]:
    base = assign_points_to_voxels_hard(head, flat)
    if assign_cfg.mode == "hard":
        return base
    if assign_cfg.mode == "bev_soft":
        return _expand_soft_targets(
            head,
            base["point_metric"],
            base["point_scores"],
            base["point_flat_index"],
            base["point_best_class"],
            base["point_best_score"],
            base["point_margin"],
            base["point_entropy"],
            base["point_local_density"],
            radius_xy=assign_cfg.radius_xy,
            radius_z=0,
            sigma_xy=assign_cfg.sigma_xy,
            sigma_z=1.0,
            distance_weighted=True,
            small_object_only=assign_cfg.small_object_only,
        )
    if assign_cfg.mode == "xyz_soft":
        return _expand_soft_targets(
            head,
            base["point_metric"],
            base["point_scores"],
            base["point_flat_index"],
            base["point_best_class"],
            base["point_best_score"],
            base["point_margin"],
            base["point_entropy"],
            base["point_local_density"],
            radius_xy=assign_cfg.radius_xy,
            radius_z=assign_cfg.radius_z,
            sigma_xy=assign_cfg.sigma_xy,
            sigma_z=assign_cfg.sigma_z,
            distance_weighted=True,
            small_object_only=assign_cfg.small_object_only,
        )
    if assign_cfg.mode == "z_soft_only":
        return _expand_soft_targets(
            head,
            base["point_metric"],
            base["point_scores"],
            base["point_flat_index"],
            base["point_best_class"],
            base["point_best_score"],
            base["point_margin"],
            base["point_entropy"],
            base["point_local_density"],
            radius_xy=0,
            radius_z=assign_cfg.radius_z,
            sigma_xy=1.0,
            sigma_z=assign_cfg.sigma_z,
            distance_weighted=True,
            small_object_only=assign_cfg.small_object_only,
        )
    if assign_cfg.mode == "topk_neighbor":
        return _expand_soft_targets(
            head,
            base["point_metric"],
            base["point_scores"],
            base["point_flat_index"],
            base["point_best_class"],
            base["point_best_score"],
            base["point_margin"],
            base["point_entropy"],
            base["point_local_density"],
            radius_xy=1,
            radius_z=1,
            sigma_xy=assign_cfg.sigma_xy,
            sigma_z=assign_cfg.sigma_z,
            distance_weighted=True,
            small_object_only=assign_cfg.small_object_only,
        )
    if assign_cfg.mode == "distance_weighted_neighbor":
        return _expand_soft_targets(
            head,
            base["point_metric"],
            base["point_scores"],
            base["point_flat_index"],
            base["point_best_class"],
            base["point_best_score"],
            base["point_margin"],
            base["point_entropy"],
            base["point_local_density"],
            radius_xy=assign_cfg.radius_xy,
            radius_z=assign_cfg.radius_z,
            sigma_xy=assign_cfg.sigma_xy,
            sigma_z=assign_cfg.sigma_z,
            distance_weighted=True,
            small_object_only=assign_cfg.small_object_only,
        )
    raise ValueError(f"unknown assign mode: {assign_cfg.mode}")


def _unique_targets_with_inv(target_voxel_indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if target_voxel_indices.numel() == 0:
        return target_voxel_indices.new_zeros((0, 3)), target_voxel_indices.new_zeros((0,), dtype=torch.long)
    unique_voxels, inv = torch.unique(target_voxel_indices, return_inverse=True, dim=0)
    return unique_voxels, inv


def _aggregate_topk_mean(
    unique_voxels: torch.Tensor,
    inv: torch.Tensor,
    scores: torch.Tensor,
    weights: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    out = scores.new_zeros((unique_voxels.shape[0], scores.shape[-1]))
    for idx in range(unique_voxels.shape[0]):
        mask = inv == idx
        local_scores = scores[mask]
        local_weights = weights[mask][:, None]
        if local_scores.numel() == 0:
            continue
        weighted = local_scores * local_weights
        k = min(topk, weighted.shape[0])
        topk_vals = torch.topk(weighted, k=k, dim=0).values
        out[idx] = topk_vals.mean(dim=0)
    return out


def _aggregate_weighted_mean(unique_voxels: torch.Tensor, inv: torch.Tensor, scores: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if unique_voxels.numel() == 0:
        return scores.new_zeros((0, scores.shape[-1]))
    weighted_scores = scores * weights[:, None]
    out_sum = torch_scatter.scatter_add(weighted_scores, inv, dim=0, dim_size=unique_voxels.shape[0])
    out_w = torch_scatter.scatter_add(weights[:, None], inv, dim=0, dim_size=unique_voxels.shape[0]).clamp_min(1e-6)
    return out_sum / out_w


def aggregate_voxel_scores_hard(
    head: Any,
    assignment: dict[str, torch.Tensor],
    aggregate_cfg: AggregateConfig,
) -> dict[str, torch.Tensor]:
    unique_voxels, inv = _unique_targets_with_inv(assignment["target_voxel_indices"])
    scores = assignment["point_scores"]
    weights = assignment["point_weight"]
    if scores.numel() == 0:
        agg_scores = scores.new_zeros((0, head.num_classes))
    else:
        scores_mod = scores.clone()
        if aggregate_cfg.small_object_scale != 1.0:
            scores_mod[:, SMALL_OBJECT_IDS] = scores_mod[:, SMALL_OBJECT_IDS] * aggregate_cfg.small_object_scale
        if aggregate_cfg.dynamic_scale != 1.0:
            scores_mod[:, DYNAMIC_IDS] = scores_mod[:, DYNAMIC_IDS] * aggregate_cfg.dynamic_scale

        if aggregate_cfg.mode == "max":
            weighted = scores_mod * weights[:, None]
            agg_scores = torch_scatter.scatter_max(weighted, inv, dim=0, dim_size=unique_voxels.shape[0])[0]
        elif aggregate_cfg.mode == "weighted_mean":
            agg_scores = _aggregate_weighted_mean(unique_voxels, inv, scores_mod, weights)
        elif aggregate_cfg.mode == "topkmean":
            agg_scores = _aggregate_topk_mean(unique_voxels, inv, scores_mod, weights, topk=max(1, aggregate_cfg.balanced_topk))
        elif aggregate_cfg.mode == "small_preserve":
            weighted = scores_mod * weights[:, None]
            base = torch_scatter.scatter_max(weighted, inv, dim=0, dim_size=unique_voxels.shape[0])[0]
            preserve = _aggregate_topk_mean(unique_voxels, inv, scores_mod[:, SMALL_OBJECT_IDS], weights, topk=max(1, aggregate_cfg.preserve_small_topk))
            base[:, SMALL_OBJECT_IDS] = torch.maximum(base[:, SMALL_OBJECT_IDS], preserve)
            agg_scores = base
        elif aggregate_cfg.mode == "class_balanced":
            agg_scores = _aggregate_topk_mean(unique_voxels, inv, scores_mod, weights, topk=max(1, aggregate_cfg.balanced_topk))
        else:
            raise ValueError(f"unknown aggregate mode: {aggregate_cfg.mode}")

    voxel_num = head.voxel_num
    occ = agg_scores.new_zeros((int(voxel_num[0].item()), int(voxel_num[1].item()), int(voxel_num[2].item()), head.num_classes))
    if unique_voxels.numel():
        occ[unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2]] = agg_scores
    occ_chw = occ.permute(3, 0, 1, 2).unsqueeze(0)
    return {
        "unique_voxels": unique_voxels,
        "inv": inv,
        "agg_scores_sparse": agg_scores,
        "dense_occ_before_padding": occ_chw,
    }


def produce_semantic_occ(head: Any, agg: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    score_thr = torch.as_tensor(head.test_cfg.get("score_thr", 0.1), device=agg["dense_occ_before_padding"].device, dtype=agg["dense_occ_before_padding"].dtype)
    occ_chw = agg["dense_occ_before_padding"]
    if head.test_cfg.get("padding", True):
        dilated_occ = F.max_pool3d(occ_chw, 3, stride=1, padding=1)
        eroded_occ = -F.max_pool3d(-dilated_occ, 3, stride=1, padding=1)
        max_score_occ, index_occ = occ_chw.max(dim=1)
        original_mask = (max_score_occ > score_thr[index_occ]).expand_as(eroded_occ)
        eroded_occ[original_mask] = occ_chw[original_mask]
    else:
        eroded_occ = occ_chw
    eroded_occ_dense = eroded_occ.squeeze(0).permute(1, 2, 3, 0)
    active_voxels = torch.nonzero((eroded_occ_dense > score_thr).any(dim=-1), as_tuple=False)
    voxel_num = head.voxel_num.long()
    occ_pred = torch.ones(voxel_num.tolist(), device=eroded_occ_dense.device, dtype=torch.long) * EMPTY_IDX
    if active_voxels.numel():
        occ_pred[active_voxels[:, 0], active_voxels[:, 1], active_voxels[:, 2]] = eroded_occ_dense[active_voxels[:, 0], active_voxels[:, 1], active_voxels[:, 2]].argmax(dim=-1)
    contributor_count_dense = torch.zeros_like(occ_pred, dtype=torch.int32)
    if agg["unique_voxels"].numel():
        pts_per_unique = torch_scatter.scatter_add(torch.ones_like(agg["inv"], dtype=torch.int32), agg["inv"], dim=0, dim_size=agg["unique_voxels"].shape[0])
        contributor_count_dense[agg["unique_voxels"][:, 0], agg["unique_voxels"][:, 1], agg["unique_voxels"][:, 2]] = pts_per_unique
    return {
        "dense_occ_after_padding": eroded_occ_dense,
        "active_voxels": active_voxels,
        "occ_pred": occ_pred,
        "contributor_count_dense": contributor_count_dense,
    }


def replay_get_occ_variant(
    head: Any,
    pred_dict: dict[str, torch.Tensor],
    horizon_s: int,
    gate_cfg: GateConfig | None = None,
    assign_cfg: AssignConfig | None = None,
    aggregate_cfg: AggregateConfig | None = None,
) -> dict[str, Any]:
    gate_cfg = gate_cfg or GateConfig()
    assign_cfg = assign_cfg or AssignConfig()
    aggregate_cfg = aggregate_cfg or AggregateConfig()
    point_info = decode_support_points(head, pred_dict)
    gate_info = compute_gate_mask(head, point_info, gate_cfg, horizon_s)
    flat = _flatten_points(point_info, gate_info)
    assignment = assign_points_to_voxels_soft(head, flat, assign_cfg)
    agg = aggregate_voxel_scores_hard(head, assignment, aggregate_cfg)
    occ = produce_semantic_occ(head, agg)
    return {
        "point_info": point_info,
        "gate_info": gate_info,
        "assignment": assignment,
        "aggregation": agg,
        "output": occ,
        "variant_meta": {
            "gate": asdict(gate_cfg),
            "assign": asdict(assign_cfg),
            "aggregate": asdict(aggregate_cfg),
        },
    }


def compare_occ_exact(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
    same_shape = tuple(a.shape) == tuple(b.shape)
    if not same_shape:
        return {"same_shape": False, "exact_equal": False, "max_abs_diff": None}
    diff = (a.float() - b.float()).abs()
    return {
        "same_shape": True,
        "exact_equal": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs_diff": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def get_default_variant_library() -> dict[str, tuple[GateConfig, AssignConfig, AggregateConfig]]:
    lib: dict[str, tuple[GateConfig, AssignConfig, AggregateConfig]] = {}
    lib["V0_hard_baseline"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))
    lib["V1_bev_soft_splat_r1"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V1_bev_soft_splat_r1", mode="bev_soft", radius_xy=1, sigma_xy=1.0), AggregateConfig(name="A1_weighted_max", mode="max"))
    lib["V2_bev_soft_splat_r2"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V2_bev_soft_splat_r2", mode="bev_soft", radius_xy=2, sigma_xy=1.5), AggregateConfig(name="A1_weighted_max", mode="max"))
    lib["V3_xyz_soft_splat_3x3x3"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V3_xyz_soft_splat_3x3x3", mode="xyz_soft", radius_xy=1, radius_z=1, sigma_xy=1.0, sigma_z=1.0), AggregateConfig(name="A1_weighted_max", mode="max"))
    lib["V4_z_soft_only"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V4_z_soft_only", mode="z_soft_only", radius_z=1, sigma_z=1.0), AggregateConfig(name="A1_weighted_max", mode="max"))
    lib["V5_topk_nearest_support_aggregate_k1"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V5_topk_nearest_support_aggregate_k1", mode="topk_neighbor", radius_xy=1, radius_z=1, sigma_xy=1.0, sigma_z=1.0, topk=1), AggregateConfig(name="A2_topkmean1", mode="topkmean", balanced_topk=1))
    lib["V5_topk_nearest_support_aggregate_k3"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V5_topk_nearest_support_aggregate_k3", mode="topk_neighbor", radius_xy=1, radius_z=1, sigma_xy=1.0, sigma_z=1.0, topk=3), AggregateConfig(name="A2_topkmean3", mode="topkmean", balanced_topk=3))
    lib["V5_topk_nearest_support_aggregate_k5"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V5_topk_nearest_support_aggregate_k5", mode="topk_neighbor", radius_xy=1, radius_z=1, sigma_xy=1.0, sigma_z=1.0, topk=5), AggregateConfig(name="A2_topkmean5", mode="topkmean", balanced_topk=5))
    lib["V6_distance_weighted_neighbor_aggregate"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V6_distance_weighted_neighbor_aggregate", mode="distance_weighted_neighbor", radius_xy=1, radius_z=1, sigma_xy=1.0, sigma_z=1.0, distance_weighted=True), AggregateConfig(name="A3_weighted_mean", mode="weighted_mean"))

    lib["G0_baseline_gate"] = lib["V0_hard_baseline"]
    lib["G1_margin_005"] = (GateConfig(name="G1_margin_005", margin_thr=0.05, margin_band=0.05), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))
    lib["G1_margin_010"] = (GateConfig(name="G1_margin_010", margin_thr=0.10, margin_band=0.08), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))
    lib["G1_margin_020"] = (GateConfig(name="G1_margin_020", margin_thr=0.20, margin_band=0.10), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))
    lib["G1_margin_040"] = (GateConfig(name="G1_margin_040", margin_thr=0.40, margin_band=0.12), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))
    lib["G2_entropy_20"] = (GateConfig(name="G2_entropy_20", entropy_thr=2.0, entropy_band=0.08), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))
    lib["G2_entropy_25"] = (GateConfig(name="G2_entropy_25", entropy_thr=2.5, entropy_band=0.08), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))
    lib["G3_near_thr_005"] = (GateConfig(name="G3_near_thr_005", near_thr_band=0.05), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))
    lib["G3_near_thr_010"] = (GateConfig(name="G3_near_thr_010", near_thr_band=0.10), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))
    lib["G3_near_thr_020"] = (GateConfig(name="G3_near_thr_020", near_thr_band=0.20), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))
    lib["G4_future_relax_005"] = (GateConfig(name="G4_future_relax_005", future_band=0.05), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))
    lib["G5_low_density_relax"] = (GateConfig(name="G5_low_density_relax", low_density_count=1, low_density_band=0.10), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))
    lib["G6_small_object_relax"] = (GateConfig(name="G6_small_object_relax", small_object_band=0.10), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="A0_hard_max", mode="max"))

    lib["C0_baseline_aggregation"] = lib["V0_hard_baseline"]
    lib["C1_small_object_topk_preserve"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="C1_small_object_topk_preserve", mode="small_preserve", preserve_small_topk=3))
    lib["C1_small_object_topk_preserve_5"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="C1_small_object_topk_preserve_5", mode="small_preserve", preserve_small_topk=5))
    lib["C2_small_object_temperature_scaling_12"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="C2_small_object_temperature_scaling_12", mode="max", small_object_scale=1.2))
    lib["C2_small_object_temperature_scaling_15"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="C2_small_object_temperature_scaling_15", mode="max", small_object_scale=1.5))
    lib["C2_small_object_temperature_scaling_20"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="C2_small_object_temperature_scaling_20", mode="max", small_object_scale=2.0))
    lib["C3_dynamic_class_reweight_12"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="C3_dynamic_class_reweight_12", mode="max", dynamic_scale=1.2))
    lib["C4_class_balanced_aggregate"] = (GateConfig(name="G0_baseline_gate"), AssignConfig(name="V0_hard_baseline", mode="hard"), AggregateConfig(name="C4_class_balanced_aggregate", mode="class_balanced", balanced_topk=2))
    lib["C5_small_object_local_soft_splat"] = (GateConfig(name="G6_small_object_relax", small_object_band=0.08), AssignConfig(name="C5_small_object_local_soft_splat", mode="bev_soft", radius_xy=1, sigma_xy=1.0, small_object_only=True), AggregateConfig(name="A1_weighted_max", mode="max"))
    return lib


def save_replay_debug(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor_to_cpu(payload), path)
