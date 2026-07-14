"""Instrument SparseWorld get_occ() for SW-4.0 covered false-free diagnosis.

This script provides a debug-equivalent wrapper around OPUSHead.get_occ():
- capture decoded points, gate masks, voxel assignments, aggregation inputs
- optionally keep dense pre/post-padding occupancy scores
- verify wrapper/output equivalence against the original get_occ()

Safe-claim boundary:
- diagnostic instrumentation only
- no training
- no model improvement claim
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch_scatter


PROJECT_ROOT = Path("/mnt/d/ComputerVision/cv_lidar_transition")
DEFAULT_REPO_ROOT = PROJECT_ROOT / "external/SparseWorld"
DEFAULT_CONFIG = DEFAULT_REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"
DEFAULT_CHECKPOINT = DEFAULT_REPO_ROOT / "ckpts/epoch_56.pth"
DEFAULT_SW2_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation"
DEFAULT_LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation"
DEFAULT_ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def load_module_from_path(name: str, path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build_sparseworld_model(repo_root: Path, config_path: Path, checkpoint_path: Path, use_fp16: bool = False):
    sys.path.insert(0, str(repo_root))
    from mmcv import Config
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config

    cfg = Config.fromfile(str(config_path))
    cfg = compat_cfg(cfg)
    cfg = patch_config(cfg)
    setup_multi_processes(cfg)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True
    cfg.gpu_ids = [0]
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if "4D" in cfg.model.type:
        cfg.model.align_after_view_transfromation = True
    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if use_fp16:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model = model.cuda()
    model.eval()
    return cfg, model, checkpoint


def get_pts_bbox_head(model: Any):
    if hasattr(model, "pts_bbox_head"):
        return model.pts_bbox_head
    raise AttributeError("model has no pts_bbox_head")


def extract_pred_dict_for_horizon(query_artifact: dict[str, Any], horizon_s: int) -> dict[str, torch.Tensor]:
    fb = query_artifact["forward_backbone_outputs"]
    if horizon_s == 0:
        return {
            "cls_scores": fb["cls_score"].cuda(non_blocking=False),
            "refine_pts": fb["refine_pts"].cuda(non_blocking=False),
        }
    return {
        "cls_scores": fb["forecast_semantics_list"][horizon_s - 1].cuda(non_blocking=False),
        "refine_pts": fb["forecast_points_list"][horizon_s - 1].cuda(non_blocking=False),
    }


def _apply_class_specific_distance_rescale(cls_scores_sigmoid: torch.Tensor, raw_refine_pts: torch.Tensor, thre1: float | None, thre2: float | None) -> torch.Tensor:
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


def get_occ_debug(
    head: Any,
    pred_dicts: dict[str, torch.Tensor],
    expand_range: bool = False,
    thre1: float | None = 0.1,
    thre2: float | None = 0.1,
    capture_dense: bool = False,
    gate_force_flat_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Debug-equivalent implementation of OPUSHead.get_occ()."""
    from mmdet3d.models.sparsedetectors.bbox.utils import decode_points

    if expand_range:
        pc_range = head.pc_range_new
        voxel_num = head.voxel_num_new
    else:
        pc_range = head.pc_range
        voxel_num = head.voxel_num

    raw_cls_logits = pred_dicts["cls_scores"]
    raw_refine_pts = pred_dicts["refine_pts"]
    cls_scores_sigmoid = _apply_class_specific_distance_rescale(raw_cls_logits.sigmoid(), raw_refine_pts, thre1, thre2)

    batch_size = raw_refine_pts.shape[0]
    ctr_dist_thr = head.test_cfg.get("ctr_dist_thr", 3.0)
    score_thr = head.test_cfg.get("score_thr", 0.1)
    score_thr_tensor = torch.as_tensor(score_thr, device=cls_scores_sigmoid.device, dtype=cls_scores_sigmoid.dtype)
    result_list: list[torch.Tensor] = []
    debug_list: list[dict[str, Any]] = []

    for batch_idx in range(batch_size):
        raw_refine_pts_i = raw_refine_pts[batch_idx]
        cls_scores_i = cls_scores_sigmoid[batch_idx]
        decoded_points = decode_points(raw_refine_pts_i, head.pc_range)

        centers = decoded_points.mean(dim=1, keepdim=True)
        ctr_dists = torch.norm(decoded_points - centers, dim=-1)
        mask_dist = ctr_dists < ctr_dist_thr

        max_score, arg_cls = cls_scores_i.max(dim=-1)
        point_thr = score_thr_tensor[arg_cls]
        mask_score = max_score > point_thr
        gate_mask = mask_dist & mask_score

        if gate_force_flat_mask is not None:
            forced = gate_force_flat_mask.to(device=gate_mask.device, dtype=torch.bool).reshape_as(gate_mask)
            gate_mask = gate_mask | forced

        q_count, refine_count = decoded_points.shape[:2]
        flat_indices = torch.arange(q_count * refine_count, device=decoded_points.device).reshape(q_count, refine_count)

        decoded_points_flat = decoded_points.reshape(-1, 3)
        cls_scores_flat = cls_scores_i.reshape(-1, cls_scores_i.shape[-1])
        gate_mask_flat = gate_mask.reshape(-1)
        valid_flat_indices = flat_indices.reshape(-1)

        pre_gate_voxel_index = ((decoded_points_flat - pc_range[:3]) // head.voxel_size).long()
        valid_range_mask = torch.logical_and(pre_gate_voxel_index >= 0, pre_gate_voxel_index < voxel_num).all(-1)

        geometric_voxels = pre_gate_voxel_index[valid_range_mask]
        geometric_points_metric = decoded_points_flat[valid_range_mask]
        geometric_scores = cls_scores_flat[valid_range_mask]
        geometric_flat_indices = valid_flat_indices[valid_range_mask]

        gated_points_metric = decoded_points_flat[gate_mask_flat]
        gated_scores = cls_scores_flat[gate_mask_flat]
        gated_pre_voxel_index = pre_gate_voxel_index[gate_mask_flat]
        gated_flat_indices = valid_flat_indices[gate_mask_flat]
        gated_in_range_mask = torch.logical_and(gated_pre_voxel_index >= 0, gated_pre_voxel_index < voxel_num).all(-1)

        voxel_indices = gated_pre_voxel_index[gated_in_range_mask]
        gated_points_metric = gated_points_metric[gated_in_range_mask]
        gated_scores = gated_scores[gated_in_range_mask]
        gated_flat_indices = gated_flat_indices[gated_in_range_mask]

        if voxel_indices.numel() == 0:
            unique_voxels = voxel_indices.new_zeros((0, 3))
            unq_inv = voxel_indices.new_zeros((0,), dtype=torch.long)
            pts_num = voxel_indices.new_zeros((0,), dtype=torch.long)
            agg_scores = gated_scores.new_zeros((0, head.num_classes))
        else:
            unique_voxels, unq_inv, pts_num = torch.unique(voxel_indices, return_inverse=True, return_counts=True, dim=0)
            agg_scores = torch_scatter.scatter_max(gated_scores, unq_inv, dim=0)[0]

        occ = gated_scores.new_zeros((int(voxel_num[0].item()), int(voxel_num[1].item()), int(voxel_num[2].item()), head.num_classes))
        if unique_voxels.numel():
            occ[unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2]] = agg_scores
        occ_chw = occ.permute(3, 0, 1, 2).unsqueeze(0)
        if head.test_cfg.get("padding", True):
            dilated_occ = F.max_pool3d(occ_chw, 3, stride=1, padding=1)
            eroded_occ = -F.max_pool3d(-dilated_occ, 3, stride=1, padding=1)
            max_score_occ, index_occ = occ_chw.max(dim=1)
            original_mask = (max_score_occ > score_thr_tensor[index_occ]).expand_as(eroded_occ)
            eroded_occ[original_mask] = occ_chw[original_mask]
        else:
            eroded_occ = occ_chw
        eroded_occ_dense = eroded_occ.squeeze(0).permute(1, 2, 3, 0)
        active_voxels = torch.nonzero((eroded_occ_dense > score_thr_tensor).any(dim=-1), as_tuple=False)
        occ_pred = torch.ones(voxel_num.tolist(), device=eroded_occ_dense.device, dtype=torch.long) * 17
        if active_voxels.numel():
            occ_pred[active_voxels[:, 0], active_voxels[:, 1], active_voxels[:, 2]] = eroded_occ_dense[active_voxels[:, 0], active_voxels[:, 1], active_voxels[:, 2]].argmax(dim=-1)

        geometric_mask = torch.zeros(tuple(int(x.item()) for x in voxel_num), dtype=torch.bool, device=eroded_occ_dense.device)
        semantic_active_mask = torch.zeros_like(geometric_mask)
        if geometric_voxels.numel():
            geometric_mask[geometric_voxels[:, 0], geometric_voxels[:, 1], geometric_voxels[:, 2]] = True
        if unique_voxels.numel():
            semantic_active_mask[unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2]] = True

        if unique_voxels.numel():
            contributor_count_dense = torch.zeros_like(geometric_mask, dtype=torch.int32)
            contributor_count_dense[unique_voxels[:, 0], unique_voxels[:, 1], unique_voxels[:, 2]] = pts_num.int()
        else:
            contributor_count_dense = torch.zeros_like(geometric_mask, dtype=torch.int32)

        debug = {
            "batch_index": batch_idx,
            "pc_range": pc_range.detach().cpu().tolist(),
            "voxel_num": voxel_num.detach().cpu().tolist(),
            "voxel_size": head.voxel_size.detach().cpu().tolist(),
            "score_thr": score_thr_tensor.detach().cpu().tolist(),
            "ctr_dist_thr": float(ctr_dist_thr),
            "raw_refine_pts": raw_refine_pts_i if capture_dense else None,
            "cls_scores_sigmoid": cls_scores_i if capture_dense else None,
            "decoded_points_metric": decoded_points if capture_dense else None,
            "centers_metric": centers if capture_dense else None,
            "ctr_dists": ctr_dists if capture_dense else None,
            "mask_dist": mask_dist,
            "max_score": max_score,
            "arg_cls": arg_cls,
            "point_thr": point_thr,
            "mask_score": mask_score,
            "gate_mask": gate_mask,
            "flat_indices": flat_indices,
            "decoded_points_flat_metric": decoded_points_flat,
            "pre_gate_voxel_index": pre_gate_voxel_index,
            "valid_range_mask": valid_range_mask,
            "geometric_points_metric": geometric_points_metric,
            "geometric_voxels": geometric_voxels,
            "geometric_scores": geometric_scores,
            "geometric_flat_indices": geometric_flat_indices,
            "gated_points_metric": gated_points_metric,
            "gated_scores": gated_scores,
            "gated_flat_indices": gated_flat_indices,
            "unique_voxels": unique_voxels,
            "point_to_unique_inv": unq_inv,
            "points_per_unique_voxel": pts_num,
            "agg_scores_sparse": agg_scores,
            "geometric_mask": geometric_mask,
            "semantic_active_mask": semantic_active_mask,
            "contributor_count_dense": contributor_count_dense,
            "dense_occ_before_padding": occ_chw.squeeze(0).permute(1, 2, 3, 0) if capture_dense else None,
            "dense_occ_after_padding": eroded_occ_dense if capture_dense else None,
            "active_voxels": active_voxels,
            "active_scores": eroded_occ_dense[active_voxels[:, 0], active_voxels[:, 1], active_voxels[:, 2]] if active_voxels.numel() else eroded_occ_dense.new_zeros((0, head.num_classes)),
            "occ_pred": occ_pred,
        }
        result_list.append(occ_pred)
        debug_list.append(debug)
    return torch.stack(result_list, dim=0), debug_list


def compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, Any]:
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


def run_sample0_equivalence(
    repo_root: Path,
    config_path: Path,
    checkpoint_path: Path,
    reports_dir: Path,
    artifacts_dir: Path,
    sample_index: int = 0,
) -> dict[str, Any]:
    import csv

    sw2_rows = list(csv.DictReader((DEFAULT_SW2_REPORTS / "sw2_sample_manifest.csv").open(encoding="utf-8")))
    row = [r for r in sw2_rows if r["forward_status"] == "success" and int(r["sample_index"]) == sample_index][0]
    query_artifact = torch.load(row["query_output_path"], map_location="cpu", weights_only=False)

    _cfg, model, _ckpt = build_sparseworld_model(repo_root, config_path, checkpoint_path, use_fp16=False)
    head = get_pts_bbox_head(model)
    eq_rows: list[dict[str, Any]] = []
    manifest = {
        "repo_root": str(repo_root),
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "head_class": type(head).__name__,
        "get_occ_source_file": str((repo_root / "mmdet3d/models/sparsedetectors/opus_head.py").resolve()),
        "captured_nodes": [
            "decoded_points_metric",
            "mask_dist",
            "mask_score",
            "gate_mask",
            "valid_range_mask",
            "unique_voxels",
            "agg_scores_sparse",
            "dense_occ_before_padding",
            "dense_occ_after_padding",
            "occ_pred",
        ],
    }
    for horizon_s in [0, 6]:
        pred_dict = extract_pred_dict_for_horizon(query_artifact, horizon_s)
        with torch.no_grad():
            original = head.get_occ(pred_dict)
            debug_pred, debug_list = get_occ_debug(head, pred_dict, capture_dense=True)
        cmp = compare_tensors(original, debug_pred)
        eq_rows.append(
            {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                **cmp,
            }
        )
        artifact_path = artifacts_dir / f"get_occ_debug_sample{sample_index}_t{horizon_s}.pt"
        ensure_parent(artifact_path)
        torch.save(tensor_to_cpu(debug_list[0]), artifact_path)
    eq_payload = {
        "sample_index": sample_index,
        "checks": eq_rows,
        "all_exact_equal": all(row["exact_equal"] for row in eq_rows),
    }
    write_json(reports_dir / "get_occ_instrumentation_manifest.json", manifest)
    write_json(reports_dir / "get_occ_output_equivalence_check.json", eq_payload)
    return {"manifest": manifest, "equivalence": eq_payload}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Instrument SparseWorld get_occ()")
    parser.add_argument("--repo-root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--reports-dir", default=str(DEFAULT_REPORTS_DIR))
    parser.add_argument("--artifacts-dir", default=str(DEFAULT_ARTIFACTS_DIR))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_sample0_equivalence(
        repo_root=Path(args.repo_root).resolve(),
        config_path=Path(args.config).resolve(),
        checkpoint_path=Path(args.checkpoint).resolve(),
        reports_dir=Path(args.reports_dir).resolve(),
        artifacts_dir=Path(args.artifacts_dir).resolve(),
        sample_index=int(args.sample_index),
    )
    print(json.dumps(payload["equivalence"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
