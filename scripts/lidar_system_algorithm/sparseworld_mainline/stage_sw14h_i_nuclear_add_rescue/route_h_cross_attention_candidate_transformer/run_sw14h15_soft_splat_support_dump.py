from __future__ import annotations

import argparse
import copy
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[5])))
SW13A_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw13a_sensor_fault_feature_memory_replay/run_sw13a_main.py"
SW14E_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14e_three_route_rescue/run_sw14e_three_route_rescue.py"
H10_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue/route_h_cross_attention_candidate_transformer/run_sw14h10_compact_sparse_ranker.py"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"

CORE_HORIZONS = [0, 2, 4, 6]
PERTURBATION_ID = "A10_drop_front_triplet"
VARIANT_LABEL = "R8_camera_group_repair_front_triplet"
SMALL_CLASS_IDS = {1, 2, 6, 7, 8}
DYNAMIC_CLASS_IDS = {2, 3, 4, 5, 6, 7, 9, 10}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw13a = load_module("sw14h15_sw13a", SW13A_SCRIPT)
sw14e = load_module("sw14h15_sw14e", SW14E_SCRIPT)
h10 = load_module("sw14h15_h10", H10_SCRIPT)


RADIUS_CONFIGS = [
    ("r1xy0z", 1, 0, 1.0, 1.0),
    ("r1xy1z", 1, 1, 1.0, 0.8),
    ("r2xy1z", 2, 1, 1.3, 0.8),
]
BASE_FEATURES = [
    "soft_count",
    "soft_weight_sum",
    "soft_top1_max",
    "soft_top1_wmean",
    "soft_margin_max",
    "soft_margin_wmean",
    "soft_over_thr_max",
    "soft_over_thr_wmean",
    "soft_gate_count",
    "soft_score_pass_count",
    "soft_dist_pass_count",
    "soft_min_dist",
    "soft_small_count",
    "soft_dynamic_count",
]
H15_FEATURE_NAMES = [f"h15_{prefix}_{name}" for prefix, *_ in RADIUS_CONFIGS for name in BASE_FEATURES]


def normalize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): normalize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [normalize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return normalize(obj.detach().cpu().item())
        return obj.detach().cpu().tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([normalize(r) for r in rows])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14H15 point-level soft-splat support feature dump")
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--sample-start", type=int, default=100)
    parser.add_argument("--sample-end", type=int, default=149)
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def add_mask(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return tensors["strong_add_candidate"].bool() | tensors["medium_add_candidate"].bool()


def table_for_split(split: str) -> dict[str, Any]:
    return sw14e.load_tensor_table(sw14e.table_path(split))


def candidate_rows_for_group(table: dict[str, Any], sample_id: int, horizon_s: int) -> torch.Tensor:
    tensors = table["tensors"]
    mask = (tensors["sample_id"].long() == sample_id) & (tensors["horizon_id"].long() == horizon_s) & add_mask(tensors)
    return torch.nonzero(mask, as_tuple=False).flatten()


def linear_index(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    return x.long() * int(shape[1] * shape[2]) + y.long() * int(shape[2]) + z.long()


def flatten_voxels(voxels: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    return linear_index(voxels[:, 0], voxels[:, 1], voxels[:, 2], shape)


def scatter_sum(lin: torch.Tensor, values: torch.Tensor, num_voxels: int) -> torch.Tensor:
    out = torch.zeros((num_voxels,), dtype=torch.float32)
    if lin.numel():
        out.scatter_add_(0, lin.long(), values.float().cpu())
    return out


def scatter_count(lin: torch.Tensor, num_voxels: int) -> torch.Tensor:
    if lin.numel() == 0:
        return torch.zeros((num_voxels,), dtype=torch.float32)
    return torch.bincount(lin.long().cpu(), minlength=num_voxels).float()


def scatter_max(lin: torch.Tensor, values: torch.Tensor, num_voxels: int) -> torch.Tensor:
    out = torch.full((num_voxels,), -torch.inf, dtype=torch.float32)
    if lin.numel():
        out.scatter_reduce_(0, lin.long().cpu(), values.float().cpu(), reduce="amax", include_self=True)
    return out.nan_to_num(0.0, neginf=0.0)


def scatter_min(lin: torch.Tensor, values: torch.Tensor, num_voxels: int) -> torch.Tensor:
    out = torch.full((num_voxels,), torch.inf, dtype=torch.float32)
    if lin.numel():
        out.scatter_reduce_(0, lin.long().cpu(), values.float().cpu(), reduce="amin", include_self=True)
    return out.nan_to_num(0.0, posinf=0.0)


def neighbor_offsets(radius_xy: int, radius_z: int, device: torch.device) -> torch.Tensor:
    offsets = []
    for dx in range(-radius_xy, radius_xy + 1):
        for dy in range(-radius_xy, radius_xy + 1):
            for dz in range(-radius_z, radius_z + 1):
                offsets.append((dx, dy, dz))
    return torch.tensor(offsets, dtype=torch.long, device=device)


def soft_splat_for_radius(
    debug: dict[str, Any],
    cand_lin: torch.Tensor,
    shape: tuple[int, int, int],
    radius_xy: int,
    radius_z: int,
    sigma_xy: float,
    sigma_z: float,
) -> dict[str, torch.Tensor]:
    num_voxels = int(shape[0] * shape[1] * shape[2])
    device = debug["geometric_voxels"].device
    geo_voxels = debug["geometric_voxels"].long()
    geo_scores = debug["geometric_scores"].float()
    if geo_voxels.numel() == 0:
        return {name: torch.zeros((len(cand_lin),), dtype=torch.float16) for name in BASE_FEATURES}
    geo_points = debug["geometric_points_metric"].float()
    pc_range = torch.tensor(debug["pc_range"], device=device, dtype=torch.float32)
    voxel_size = torch.tensor(debug["voxel_size"], device=device, dtype=torch.float32)
    cont_xyz = (geo_points - pc_range[:3]) / voxel_size
    top2 = torch.topk(geo_scores, k=min(2, geo_scores.shape[-1]), dim=-1)
    top1 = top2.values[:, 0]
    margin = top2.values[:, 0] - (top2.values[:, 1] if top2.values.shape[-1] > 1 else 0.0)
    top_cls = top2.indices[:, 0].long()
    geo_flat = debug["geometric_flat_indices"].long()
    flat_point_thr = debug["point_thr"].reshape(-1).float()
    flat_mask_score = debug["mask_score"].reshape(-1).bool()
    flat_mask_dist = debug["mask_dist"].reshape(-1).bool()
    flat_gate = debug["gate_mask"].reshape(-1).bool()
    over_thr = top1 - flat_point_thr[geo_flat]
    score_pass = flat_mask_score[geo_flat].float()
    dist_pass = flat_mask_dist[geo_flat].float()
    gate_pass = flat_gate[geo_flat].float()
    small = torch.zeros_like(top1)
    dynamic = torch.zeros_like(top1)
    for cls_id in SMALL_CLASS_IDS:
        small = torch.maximum(small, (top_cls == cls_id).float())
    for cls_id in DYNAMIC_CLASS_IDS:
        dynamic = torch.maximum(dynamic, (top_cls == cls_id).float())

    target_lins: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    dists: list[torch.Tensor] = []
    vals = {
        "top1": [],
        "margin": [],
        "over_thr": [],
        "gate": [],
        "score_pass": [],
        "dist_pass": [],
        "small": [],
        "dynamic": [],
    }
    voxel_num = torch.tensor(shape, device=device, dtype=torch.long)
    for offset in neighbor_offsets(radius_xy, radius_z, device):
        target = geo_voxels + offset[None, :]
        valid = torch.logical_and(target >= 0, target < voxel_num[None, :]).all(dim=-1)
        if not bool(valid.any().item()):
            continue
        target_v = target[valid]
        cont_v = cont_xyz[valid]
        dist_xy = torch.norm(cont_v[:, :2] - (target_v[:, :2].float() + 0.5), dim=-1)
        dist_z = torch.abs(cont_v[:, 2] - (target_v[:, 2].float() + 0.5))
        weight = torch.exp(-(dist_xy.square()) / (2.0 * sigma_xy * sigma_xy)) * torch.exp(-(dist_z.square()) / (2.0 * sigma_z * sigma_z))
        target_lins.append(flatten_voxels(target_v, shape).cpu())
        weights.append(weight.detach().cpu())
        dists.append(torch.sqrt(dist_xy.square() + dist_z.square()).detach().cpu())
        vals["top1"].append(top1[valid].detach().cpu())
        vals["margin"].append(margin[valid].detach().cpu())
        vals["over_thr"].append(over_thr[valid].detach().cpu())
        vals["gate"].append(gate_pass[valid].detach().cpu())
        vals["score_pass"].append(score_pass[valid].detach().cpu())
        vals["dist_pass"].append(dist_pass[valid].detach().cpu())
        vals["small"].append(small[valid].detach().cpu())
        vals["dynamic"].append(dynamic[valid].detach().cpu())
    if not target_lins:
        return {name: torch.zeros((len(cand_lin),), dtype=torch.float16) for name in BASE_FEATURES}
    lin = torch.cat(target_lins, dim=0)
    weight_all = torch.cat(weights, dim=0).float()
    dist_all = torch.cat(dists, dim=0).float()
    top1_all = torch.cat(vals["top1"], dim=0).float()
    margin_all = torch.cat(vals["margin"], dim=0).float()
    over_all = torch.cat(vals["over_thr"], dim=0).float()
    gate_all = torch.cat(vals["gate"], dim=0).float()
    score_pass_all = torch.cat(vals["score_pass"], dim=0).float()
    dist_pass_all = torch.cat(vals["dist_pass"], dim=0).float()
    small_all = torch.cat(vals["small"], dim=0).float()
    dynamic_all = torch.cat(vals["dynamic"], dim=0).float()
    weight_sum = scatter_sum(lin, weight_all, num_voxels).clamp_min(1.0e-6)
    dense = {
        "soft_count": scatter_count(lin, num_voxels),
        "soft_weight_sum": weight_sum,
        "soft_top1_max": scatter_max(lin, top1_all, num_voxels),
        "soft_top1_wmean": scatter_sum(lin, weight_all * top1_all, num_voxels) / weight_sum,
        "soft_margin_max": scatter_max(lin, margin_all, num_voxels),
        "soft_margin_wmean": scatter_sum(lin, weight_all * margin_all, num_voxels) / weight_sum,
        "soft_over_thr_max": scatter_max(lin, over_all, num_voxels),
        "soft_over_thr_wmean": scatter_sum(lin, weight_all * over_all, num_voxels) / weight_sum,
        "soft_gate_count": scatter_sum(lin, gate_all, num_voxels),
        "soft_score_pass_count": scatter_sum(lin, score_pass_all, num_voxels),
        "soft_dist_pass_count": scatter_sum(lin, dist_pass_all, num_voxels),
        "soft_min_dist": scatter_min(lin, dist_all, num_voxels),
        "soft_small_count": scatter_sum(lin, small_all, num_voxels),
        "soft_dynamic_count": scatter_sum(lin, dynamic_all, num_voxels),
    }
    return {name: dense[name][cand_lin].float().half() for name in BASE_FEATURES}


def gather_soft_splat_features(table: dict[str, Any], rows: torch.Tensor, debug: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors = table["tensors"]
    x = tensors["voxel_x"][rows].long()
    y = tensors["voxel_y"][rows].long()
    z = tensors["voxel_z"][rows].long()
    shape = tuple(int(v) for v in debug["voxel_num"])
    cand_lin = linear_index(x, y, z, shape)
    out: dict[str, torch.Tensor] = {
        "row_index": rows.cpu().long(),
        "sample_id": tensors["sample_id"][rows].cpu().short(),
        "horizon_id": tensors["horizon_id"][rows].cpu().short(),
        "voxel_x": x.cpu().short(),
        "voxel_y": y.cpu().short(),
        "voxel_z": z.cpu().short(),
        "GT_occ": tensors["GT_occ"][rows].cpu().byte(),
        "teacher_FN": tensors["teacher_FN"][rows].cpu().byte(),
    }
    for prefix, radius_xy, radius_z, sigma_xy, sigma_z in RADIUS_CONFIGS:
        part = soft_splat_for_radius(debug, cand_lin, shape, radius_xy, radius_z, sigma_xy, sigma_z)
        for name, value in part.items():
            out[f"h15_{prefix}_{name}"] = value
    return out


def append_feature_parts(dst: dict[str, list[torch.Tensor]], part: dict[str, torch.Tensor]) -> None:
    for key, value in part.items():
        dst.setdefault(key, []).append(value)


def concat_parts(parts: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat(value, dim=0) for key, value in parts.items()}


def norm(values: torch.Tensor) -> torch.Tensor:
    vals = values.float().nan_to_num(0.0)
    return ((vals - vals.min()) / (vals.max() - vals.min() + 1.0e-6)).clamp(0.0, 1.0) if vals.numel() and vals.max() > vals.min() else torch.zeros_like(vals)


def precision_rows(features: dict[str, torch.Tensor], split: str) -> list[dict[str, Any]]:
    label = features["GT_occ"].bool()
    rows: list[dict[str, Any]] = []
    score_map: dict[str, torch.Tensor] = {}
    for name in H15_FEATURE_NAMES:
        if name in features:
            score_map[name] = norm(features[name].float())
    try:
        compact = h10.build_compact_split(split)
        if torch.equal(compact["rows"].long(), features["row_index"].long()):
            h7b = compact["features"]["h7b_lean"].float()
            best_feature = max(score_map.items(), key=lambda kv: float(label[torch.topk(kv[1], k=min(100000, len(label))).indices].float().mean().item()))[1]
            score_map["h15_best_feature_h7b_blend_25"] = 0.25 * best_feature + 0.75 * h7b
            score_map["h15_best_feature_h7b_blend_50"] = 0.50 * best_feature + 0.50 * h7b
            score_map["h7b_lean"] = h7b
    except Exception as exc:
        print(f"[h15] compact blend skipped: {exc}", flush=True)
    for name, score in score_map.items():
        order = torch.argsort(score.float(), descending=True)
        for k in [1000, 5000, 10000, 25000, 50000, 100000, 150000, 250000]:
            if len(order) < k:
                continue
            top = order[:k]
            rows.append(
                {
                    "split": split,
                    "score_name": name,
                    "topk": k,
                    "precision": float(label[top].float().mean().item()),
                    "positive_count": int(label[top].sum().item()),
                }
            )
    return rows


def run() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    local_sw13a_artifacts = ARTIFACTS_DIR / "sw14h15_sw13a_local"
    local_sw13a_artifacts.mkdir(parents=True, exist_ok=True)
    sw13a.ARTIFACTS_DIR = local_sw13a_artifacts
    (sw13a.ARTIFACTS_DIR / "feature_memory_cache").mkdir(parents=True, exist_ok=True)
    write_json(
        REPORTS_DIR / "sw14h15_soft_splat_support_dump_config.json",
        {
            "split": args.split,
            "sample_start": args.sample_start,
            "sample_end": args.sample_end,
            "max_samples": args.max_samples,
            "radius_configs": RADIUS_CONFIGS,
            "feature_names": H15_FEATURE_NAMES,
            "perturbation_id": PERTURBATION_ID,
            "variant_label": VARIANT_LABEL,
            "source": "SW13A R8 feature repair + get_occ_debug point-level soft splat",
            "modifies_get_occ": False,
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
            "gt_used_as_inference_feature": False,
        },
    )
    table = table_for_split(args.split)
    print("[h15] building SparseWorld runtime", flush=True)
    _cfg, dataset, model = sw13a.build_runtime()
    model.eval()
    variant = next(v for v in sw13a.make_variants() if v.label == VARIANT_LABEL)
    spec = sw13a.sw81.sw5_engine.build_catalog()[PERTURBATION_ID]
    feature_parts: dict[str, list[torch.Tensor]] = {}
    manifest: list[dict[str, Any]] = []
    sample_ids = list(range(args.sample_start, args.sample_end + 1))[: args.max_samples]
    try:
        for sample_id in sample_ids:
            print(f"[h15] sample={sample_id} clean memory", flush=True)
            raw_sample, batch_clean = sw13a.sw2.extract_sample_batch(dataset, sample_id, sw13a.collate_fn)
            sample_unwrapped = sw13a.sw2.unwrap(raw_sample)
            moved_clean = sw13a.sw2.move_to_cuda(batch_clean)
            _cache_path, _rows, cache = sw13a.extract_clean_memory_for_sample(model, moved_clean, sample_unwrapped, sample_id)
            batch_deg = copy.deepcopy(batch_clean)
            batch_deg = sw13a.sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec)[0]
            print(f"[h15] sample={sample_id} R8 forward", flush=True)
            _raw_result_cpu, per_h, runtime_debug = sw13a.run_variant_forward(model, sample_unwrapped, batch_deg, cache, variant, PERTURBATION_ID)
            head = sw13a.sw4_inst.get_pts_bbox_head(model)
            for horizon_s in CORE_HORIZONS:
                rows = candidate_rows_for_group(table, sample_id, horizon_s)
                if len(rows) == 0:
                    continue
                print(f"[h15] sample={sample_id} h={horizon_s} soft rows={len(rows)}", flush=True)
                _occ_pred, debug_list = sw13a.sw4_inst.get_occ_debug(head, per_h[horizon_s]["pred_dict"], capture_dense=True)
                part = gather_soft_splat_features(table, rows, debug_list[0])
                append_feature_parts(feature_parts, part)
                label = part["GT_occ"].bool()
                manifest.append(
                    {
                        "split": args.split,
                        "sample_id": sample_id,
                        "horizon_id": horizon_s,
                        "candidate_rows": int(len(rows)),
                        "positive_rows": int(label.sum().item()),
                        "positive_rate": float(label.float().mean().item()) if len(label) else 0.0,
                        "repair_applied": bool(runtime_debug.get("frame_debug", [{}])[0].get("repair_applied", False)),
                    }
                )
    finally:
        try:
            del model
            torch.cuda.empty_cache()
        except Exception:
            pass
    features = concat_parts(feature_parts) if feature_parts else {}
    out_path = ARTIFACTS_DIR / f"sw14h15_soft_splat_support_dump_{args.split}_{args.sample_start}_{args.sample_end}.pt"
    torch.save(
        {
            "features": features,
            "manifest": manifest,
            "split": args.split,
            "sample_ids": sample_ids,
            "feature_names": H15_FEATURE_NAMES,
            "gt_used_as_inference_feature": False,
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
        },
        out_path,
    )
    write_csv(REPORTS_DIR / f"sw14h15_soft_splat_support_dump_manifest_{args.split}.csv", manifest)
    topk = precision_rows(features, args.split) if features else []
    write_csv(REPORTS_DIR / f"sw14h15_soft_splat_support_topk_precision_{args.split}.csv", topk)
    rows100 = [r for r in topk if int(r["topk"]) == 100000]
    best = max(rows100, key=lambda r: float(r["precision"])) if rows100 else {}
    summary = {
        "decision": "SW14H15_DUMP_READY" if features else "SW14H15_DUMP_NO_ROWS",
        "artifact": out_path,
        "split": args.split,
        "sample_ids": sample_ids,
        "candidate_rows": int(sum(int(r["candidate_rows"]) for r in manifest)),
        "positive_rows": int(sum(int(r["positive_rows"]) for r in manifest)),
        "best_top100k": best,
        "target_precision_at_100k": 0.9,
        "target_reached": bool(float(best.get("precision", 0.0)) >= 0.9),
        "modifies_get_occ": False,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gt_used_as_inference_feature": False,
        "whether_resume_claim_allowed": False,
        "sw13_remains_main_result": True,
    }
    write_json(REPORTS_DIR / f"sw14h15_soft_splat_support_summary_{args.split}.json", summary)
    print(json.dumps(normalize(summary), indent=2), flush=True)


if __name__ == "__main__":
    run()
