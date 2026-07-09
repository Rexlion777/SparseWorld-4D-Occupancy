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
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14h_i_nuclear_add_rescue"

CORE_HORIZONS = [0, 2, 4, 6]
PERTURBATION_ID = "A10_drop_front_triplet"
VARIANT_LABEL = "R8_camera_group_repair_front_triplet"

H9_FEATURE_NAMES = [
    "h9_geo_count",
    "h9_geo_query_count",
    "h9_geo_score_pass_count",
    "h9_geo_dist_pass_count",
    "h9_geo_gate_pass_count",
    "h9_geo_top1_max",
    "h9_geo_top1_mean",
    "h9_geo_margin_max",
    "h9_geo_margin_mean",
    "h9_geo_score_over_thr_max",
    "h9_geo_score_over_thr_mean",
    "h9_geo_dist_margin_max",
    "h9_geo_dist_margin_mean",
    "h9_gate_top1_max",
    "h9_gate_top1_mean",
    "h9_gate_margin_max",
    "h9_gate_margin_mean",
    "h9_gate_score_over_thr_max",
    "h9_gate_dist_margin_max",
    "h9_gate_fraction",
    "h9_score_pass_fraction",
    "h9_dist_pass_fraction",
    "h9_non_gate_geo_count",
]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw13a = load_module("sw14h9_sw13a", SW13A_SCRIPT)
sw14e = load_module("sw14h9_sw14e", SW14E_SCRIPT)


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
    parser = argparse.ArgumentParser(description="SW14H9 sparse get_occ contributor feature dump")
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


def scatter_count(lin: torch.Tensor, num_voxels: int) -> torch.Tensor:
    if lin.numel() == 0:
        return torch.zeros((num_voxels,), dtype=torch.float32)
    return torch.bincount(lin.long(), minlength=num_voxels).float()


def scatter_sum(lin: torch.Tensor, values: torch.Tensor, num_voxels: int) -> torch.Tensor:
    out = torch.zeros((num_voxels,), dtype=torch.float32)
    if lin.numel():
        out.scatter_add_(0, lin.long(), values.float())
    return out


def scatter_max(lin: torch.Tensor, values: torch.Tensor, num_voxels: int) -> torch.Tensor:
    out = torch.full((num_voxels,), -torch.inf, dtype=torch.float32)
    if lin.numel():
        out.scatter_reduce_(0, lin.long(), values.float(), reduce="amax", include_self=True)
    return out.nan_to_num(0.0, neginf=0.0)


def gather_at_rows(dense_flat: torch.Tensor, cand_lin: torch.Tensor) -> torch.Tensor:
    return dense_flat[cand_lin].float().cpu().half()


def unique_query_count_by_voxel(lin: torch.Tensor, flat_indices: torch.Tensor, refine_count: int, num_voxels: int) -> torch.Tensor:
    if lin.numel() == 0:
        return torch.zeros((num_voxels,), dtype=torch.float32)
    query_id = flat_indices.long() // int(refine_count)
    combo = lin.long() * (int(query_id.max().item()) + 1) + query_id
    uniq = torch.unique(combo)
    voxel = uniq // (int(query_id.max().item()) + 1)
    return torch.bincount(voxel, minlength=num_voxels).float()


def gather_sparse_contributor_features(table: dict[str, Any], rows: torch.Tensor, debug: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors = table["tensors"]
    x = tensors["voxel_x"][rows].long()
    y = tensors["voxel_y"][rows].long()
    z = tensors["voxel_z"][rows].long()
    shape = tuple(int(v) for v in debug["voxel_num"])
    num_voxels = int(shape[0] * shape[1] * shape[2])
    cand_lin = linear_index(x, y, z, shape)

    geometric_voxels = debug["geometric_voxels"].detach().cpu().long()
    geometric_scores = debug["geometric_scores"].detach().cpu().float()
    geometric_flat = debug["geometric_flat_indices"].detach().cpu().long()
    q_count, refine_count = debug["flat_indices"].shape
    flat_ctr = debug["ctr_dists"].detach().cpu().float().reshape(-1)
    flat_mask_dist = debug["mask_dist"].detach().cpu().bool().reshape(-1)
    flat_mask_score = debug["mask_score"].detach().cpu().bool().reshape(-1)
    flat_gate = debug["gate_mask"].detach().cpu().bool().reshape(-1)
    flat_point_thr = debug["point_thr"].detach().cpu().float().reshape(-1)

    geo_lin = flatten_voxels(geometric_voxels, shape) if geometric_voxels.numel() else torch.zeros((0,), dtype=torch.long)
    if geometric_scores.numel():
        top2 = torch.topk(geometric_scores, k=min(2, geometric_scores.shape[-1]), dim=-1).values
        geo_top1 = top2[:, 0]
        geo_margin = top2[:, 0] - (top2[:, 1] if top2.shape[-1] > 1 else 0.0)
        geo_thr = flat_point_thr[geometric_flat]
        geo_score_over = geo_top1 - geo_thr
        geo_dist_margin = float(debug["ctr_dist_thr"]) - flat_ctr[geometric_flat]
        geo_score_pass = flat_mask_score[geometric_flat].float()
        geo_dist_pass = flat_mask_dist[geometric_flat].float()
        geo_gate_pass = flat_gate[geometric_flat].float()
    else:
        geo_top1 = geo_margin = geo_score_over = geo_dist_margin = geo_score_pass = geo_dist_pass = geo_gate_pass = torch.zeros((0,), dtype=torch.float32)

    geo_count = scatter_count(geo_lin, num_voxels)
    geo_query_count = unique_query_count_by_voxel(geo_lin, geometric_flat, int(refine_count), num_voxels)
    geo_top1_sum = scatter_sum(geo_lin, geo_top1, num_voxels)
    geo_margin_sum = scatter_sum(geo_lin, geo_margin, num_voxels)
    geo_score_over_sum = scatter_sum(geo_lin, geo_score_over, num_voxels)
    geo_dist_margin_sum = scatter_sum(geo_lin, geo_dist_margin, num_voxels)
    geo_score_pass_count = scatter_sum(geo_lin, geo_score_pass, num_voxels)
    geo_dist_pass_count = scatter_sum(geo_lin, geo_dist_pass, num_voxels)
    geo_gate_pass_count = scatter_sum(geo_lin, geo_gate_pass, num_voxels)
    denom = geo_count.clamp_min(1.0)

    gate_mask = geo_gate_pass.bool()
    gate_lin = geo_lin[gate_mask]
    gate_top1 = geo_top1[gate_mask]
    gate_margin = geo_margin[gate_mask]
    gate_score_over = geo_score_over[gate_mask]
    gate_dist_margin = geo_dist_margin[gate_mask]
    gate_count = scatter_count(gate_lin, num_voxels)
    gate_denom = gate_count.clamp_min(1.0)

    dense_map = {
        "h9_geo_count": geo_count,
        "h9_geo_query_count": geo_query_count,
        "h9_geo_score_pass_count": geo_score_pass_count,
        "h9_geo_dist_pass_count": geo_dist_pass_count,
        "h9_geo_gate_pass_count": geo_gate_pass_count,
        "h9_geo_top1_max": scatter_max(geo_lin, geo_top1, num_voxels),
        "h9_geo_top1_mean": geo_top1_sum / denom,
        "h9_geo_margin_max": scatter_max(geo_lin, geo_margin, num_voxels),
        "h9_geo_margin_mean": geo_margin_sum / denom,
        "h9_geo_score_over_thr_max": scatter_max(geo_lin, geo_score_over, num_voxels),
        "h9_geo_score_over_thr_mean": geo_score_over_sum / denom,
        "h9_geo_dist_margin_max": scatter_max(geo_lin, geo_dist_margin, num_voxels),
        "h9_geo_dist_margin_mean": geo_dist_margin_sum / denom,
        "h9_gate_top1_max": scatter_max(gate_lin, gate_top1, num_voxels),
        "h9_gate_top1_mean": scatter_sum(gate_lin, gate_top1, num_voxels) / gate_denom,
        "h9_gate_margin_max": scatter_max(gate_lin, gate_margin, num_voxels),
        "h9_gate_margin_mean": scatter_sum(gate_lin, gate_margin, num_voxels) / gate_denom,
        "h9_gate_score_over_thr_max": scatter_max(gate_lin, gate_score_over, num_voxels),
        "h9_gate_dist_margin_max": scatter_max(gate_lin, gate_dist_margin, num_voxels),
    }
    dense_map["h9_gate_fraction"] = (geo_gate_pass_count / denom).clamp(0.0, 1.0)
    dense_map["h9_score_pass_fraction"] = (geo_score_pass_count / denom).clamp(0.0, 1.0)
    dense_map["h9_dist_pass_fraction"] = (geo_dist_pass_count / denom).clamp(0.0, 1.0)
    dense_map["h9_non_gate_geo_count"] = (geo_count - geo_gate_pass_count).clamp_min(0.0)

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
    for name in H9_FEATURE_NAMES:
        out[name] = gather_at_rows(dense_map[name], cand_lin)
    return out


def append_feature_parts(dst: dict[str, list[torch.Tensor]], part: dict[str, torch.Tensor]) -> None:
    for key, value in part.items():
        dst.setdefault(key, []).append(value)


def concat_parts(parts: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat(value, dim=0) for key, value in parts.items()}


def run() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    local_sw13a_artifacts = ARTIFACTS_DIR / "sw14h9_sw13a_local"
    local_sw13a_artifacts.mkdir(parents=True, exist_ok=True)
    sw13a.ARTIFACTS_DIR = local_sw13a_artifacts
    (sw13a.ARTIFACTS_DIR / "feature_memory_cache").mkdir(parents=True, exist_ok=True)
    write_json(
        REPORTS_DIR / "sw14h9_sparse_contributor_dump_config.json",
        {
            "split": args.split,
            "sample_start": args.sample_start,
            "sample_end": args.sample_end,
            "max_samples": args.max_samples,
            "feature_names": H9_FEATURE_NAMES,
            "source": "SW13A R8 feature repair + sw4 get_occ_debug sparse contributor fields",
            "modifies_get_occ": False,
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
            "gt_used_as_inference_feature": False,
        },
    )
    table = table_for_split(args.split)
    print("[h9] building SparseWorld runtime", flush=True)
    _cfg, dataset, model = sw13a.build_runtime()
    model.eval()
    variant = next(v for v in sw13a.make_variants() if v.label == VARIANT_LABEL)
    spec = sw13a.sw81.sw5_engine.build_catalog()[PERTURBATION_ID]
    feature_parts: dict[str, list[torch.Tensor]] = {}
    manifest: list[dict[str, Any]] = []
    sample_ids = list(range(args.sample_start, args.sample_end + 1))[: args.max_samples]
    try:
        for sample_id in sample_ids:
            print(f"[h9] sample={sample_id} clean memory", flush=True)
            raw_sample, batch_clean = sw13a.sw2.extract_sample_batch(dataset, sample_id, sw13a.collate_fn)
            sample_unwrapped = sw13a.sw2.unwrap(raw_sample)
            moved_clean = sw13a.sw2.move_to_cuda(batch_clean)
            _cache_path, _rows, cache = sw13a.extract_clean_memory_for_sample(model, moved_clean, sample_unwrapped, sample_id)
            batch_deg = copy.deepcopy(batch_clean)
            batch_deg = sw13a.sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec)[0]
            print(f"[h9] sample={sample_id} R8 forward", flush=True)
            _raw_result_cpu, per_h, runtime_debug = sw13a.run_variant_forward(model, sample_unwrapped, batch_deg, cache, variant, PERTURBATION_ID)
            head = sw13a.sw4_inst.get_pts_bbox_head(model)
            for horizon_s in CORE_HORIZONS:
                rows = candidate_rows_for_group(table, sample_id, horizon_s)
                if len(rows) == 0:
                    continue
                print(f"[h9] sample={sample_id} h={horizon_s} sparse rows={len(rows)}", flush=True)
                _occ_pred, debug_list = sw13a.sw4_inst.get_occ_debug(head, per_h[horizon_s]["pred_dict"], capture_dense=True)
                part = gather_sparse_contributor_features(table, rows, debug_list[0])
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
                        "mean_geo_count": float(part["h9_geo_count"].float().mean().item()) if len(label) else 0.0,
                        "mean_gate_fraction": float(part["h9_gate_fraction"].float().mean().item()) if len(label) else 0.0,
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
    out_path = ARTIFACTS_DIR / f"sw14h9_sparse_contributor_dump_{args.split}_{args.sample_start}_{args.sample_end}.pt"
    torch.save(
        {
            "features": features,
            "manifest": manifest,
            "split": args.split,
            "sample_ids": sample_ids,
            "feature_names": H9_FEATURE_NAMES,
            "gt_used_as_inference_feature": False,
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
        },
        out_path,
    )
    write_csv(REPORTS_DIR / f"sw14h9_sparse_contributor_dump_manifest_{args.split}.csv", manifest)
    total_rows = int(sum(int(r["candidate_rows"]) for r in manifest))
    total_pos = int(sum(int(r["positive_rows"]) for r in manifest))
    summary = {
        "decision": "SW14H9_DUMP_READY" if total_rows > 0 else "SW14H9_DUMP_NO_ROWS",
        "artifact": out_path,
        "split": args.split,
        "sample_ids": sample_ids,
        "candidate_rows": total_rows,
        "positive_rows": total_pos,
        "positive_rate": float(total_pos / max(1, total_rows)),
        "feature_columns": sorted(features.keys()) if features else [],
        "modifies_get_occ": False,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gt_used_as_inference_feature": False,
    }
    write_json(REPORTS_DIR / f"sw14h9_sparse_contributor_dump_summary_{args.split}.json", summary)
    print(json.dumps(normalize(summary), indent=2), flush=True)


if __name__ == "__main__":
    run()
