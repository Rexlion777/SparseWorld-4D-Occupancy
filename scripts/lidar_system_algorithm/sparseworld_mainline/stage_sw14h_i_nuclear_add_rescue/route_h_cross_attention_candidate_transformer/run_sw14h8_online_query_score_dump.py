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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw13a = load_module("sw14h8_sw13a", SW13A_SCRIPT)
sw14e = load_module("sw14h8_sw14e", SW14E_SCRIPT)


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
    p = argparse.ArgumentParser(description="SW14H8 online query/gate score dump for add rescue")
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--sample-start", type=int, default=100)
    p.add_argument("--sample-end", type=int, default=100)
    p.add_argument("--max-samples", type=int, default=1)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def add_mask(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return tensors["strong_add_candidate"].bool() | tensors["medium_add_candidate"].bool()


def table_for_split(split: str) -> dict[str, Any]:
    return sw14e.load_tensor_table(sw14e.table_path(split))


def candidate_rows_for_group(table: dict[str, Any], sample_id: int, horizon_s: int) -> torch.Tensor:
    tensors = table["tensors"]
    mask = (tensors["sample_id"].long() == sample_id) & (tensors["horizon_id"].long() == horizon_s) & add_mask(tensors)
    return torch.nonzero(mask, as_tuple=False).flatten()


def top_margin(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    top2 = torch.topk(scores.float(), k=min(2, scores.shape[-1]), dim=-1).values
    top1 = top2[..., 0]
    margin = top1 - (top2[..., 1] if top2.shape[-1] > 1 else 0.0)
    return top1, margin


def gather_debug_features(table: dict[str, Any], rows: torch.Tensor, debug: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors = table["tensors"]
    x = tensors["voxel_x"][rows].long()
    y = tensors["voxel_y"][rows].long()
    z = tensors["voxel_z"][rows].long()
    dense = debug["dense_occ_after_padding"].detach().cpu().float()
    top1, margin = top_margin(dense[x, y, z])
    zeros = torch.zeros_like(top1)
    geometric = debug.get("geometric_mask")
    semantic_active = debug["semantic_active_mask"].detach().cpu().bool()[x, y, z].float()
    geometric_active = geometric.detach().cpu().bool()[x, y, z].float() if isinstance(geometric, torch.Tensor) else zeros
    contributor = debug["contributor_count_dense"].detach().cpu().float()[x, y, z]
    gate_dense = debug.get("gate_pass_dense")
    extra_dense = debug.get("extra_routed_contributor_count_dense")
    gate_pass = gate_dense.detach().cpu().float()[x, y, z] if isinstance(gate_dense, torch.Tensor) else zeros
    extra = extra_dense.detach().cpu().float()[x, y, z] if isinstance(extra_dense, torch.Tensor) else zeros
    occ_pred = debug["occ_pred"].detach().cpu().long()[x, y, z]
    return {
        "row_index": rows.cpu().long(),
        "sample_id": tensors["sample_id"][rows].cpu().short(),
        "horizon_id": tensors["horizon_id"][rows].cpu().short(),
        "voxel_x": x.cpu().short(),
        "voxel_y": y.cpu().short(),
        "voxel_z": z.cpu().short(),
        "h8_dense_top1_score": top1.cpu().half(),
        "h8_dense_top1_margin": margin.cpu().half(),
        "h8_geometric_active": geometric_active.cpu().half(),
        "h8_semantic_active": semantic_active.cpu().half(),
        "h8_contributor_count": contributor.cpu().half(),
        "h8_gate_pass_count": gate_pass.cpu().half(),
        "h8_extra_routed_count": extra.cpu().half(),
        "h8_occ_pred_nonempty": (occ_pred != 17).float().cpu().half(),
        "GT_occ": tensors["GT_occ"][rows].cpu().byte(),
        "teacher_FN": tensors["teacher_FN"][rows].cpu().byte(),
    }


def append_feature_parts(dst: dict[str, list[torch.Tensor]], part: dict[str, torch.Tensor]) -> None:
    for key, value in part.items():
        dst.setdefault(key, []).append(value)


def concat_parts(parts: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat(value, dim=0) for key, value in parts.items()}


def run() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    local_sw13a_artifacts = ARTIFACTS_DIR / "sw14h8_sw13a_local"
    local_sw13a_artifacts.mkdir(parents=True, exist_ok=True)
    sw13a.ARTIFACTS_DIR = local_sw13a_artifacts
    (sw13a.ARTIFACTS_DIR / "feature_memory_cache").mkdir(parents=True, exist_ok=True)
    write_json(
        REPORTS_DIR / "sw14h8_online_query_score_dump_config.json",
        {
            "split": args.split,
            "sample_start": args.sample_start,
            "sample_end": args.sample_end,
            "max_samples": args.max_samples,
            "perturbation_id": PERTURBATION_ID,
            "variant_label": VARIANT_LABEL,
            "source": "SW13A R8 feature repair + sw4 get_occ_debug capture_dense",
            "modifies_get_occ": False,
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
            "gt_used_as_inference_feature": False,
        },
    )
    table = table_for_split(args.split)
    print("[h8] building SparseWorld runtime", flush=True)
    _cfg, dataset, model = sw13a.build_runtime()
    model.eval()
    variant = next(v for v in sw13a.make_variants() if v.label == VARIANT_LABEL)
    spec = sw13a.sw81.sw5_engine.build_catalog()[PERTURBATION_ID]
    feature_parts: dict[str, list[torch.Tensor]] = {}
    manifest: list[dict[str, Any]] = []
    sample_ids = list(range(args.sample_start, args.sample_end + 1))[: args.max_samples]
    try:
        for sample_id in sample_ids:
            print(f"[h8] sample={sample_id} clean memory", flush=True)
            raw_sample, batch_clean = sw13a.sw2.extract_sample_batch(dataset, sample_id, sw13a.collate_fn)
            sample_unwrapped = sw13a.sw2.unwrap(raw_sample)
            moved_clean = sw13a.sw2.move_to_cuda(batch_clean)
            _cache_path, _rows, cache = sw13a.extract_clean_memory_for_sample(model, moved_clean, sample_unwrapped, sample_id)
            batch_deg = copy.deepcopy(batch_clean)
            batch_deg = sw13a.sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec)[0]
            print(f"[h8] sample={sample_id} R8 forward", flush=True)
            _raw_result_cpu, per_h, runtime_debug = sw13a.run_variant_forward(model, sample_unwrapped, batch_deg, cache, variant, PERTURBATION_ID)
            head = sw13a.sw4_inst.get_pts_bbox_head(model)
            for horizon_s in CORE_HORIZONS:
                rows = candidate_rows_for_group(table, sample_id, horizon_s)
                if len(rows) == 0:
                    continue
                print(f"[h8] sample={sample_id} h={horizon_s} get_occ_debug rows={len(rows)}", flush=True)
                _occ_pred, debug_list = sw13a.sw4_inst.get_occ_debug(head, per_h[horizon_s]["pred_dict"], capture_dense=True)
                part = gather_debug_features(table, rows, debug_list[0])
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
                        "semantic_active_rate": float(part["h8_semantic_active"].float().mean().item()) if len(label) else 0.0,
                        "mean_gate_pass_count": float(part["h8_gate_pass_count"].float().mean().item()) if len(label) else 0.0,
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
    out_path = ARTIFACTS_DIR / f"sw14h8_online_query_score_dump_{args.split}_{args.sample_start}_{args.sample_end}.pt"
    torch.save(
        {
            "features": features,
            "manifest": manifest,
            "split": args.split,
            "sample_ids": sample_ids,
            "gt_used_as_inference_feature": False,
            "uses_eval_debug": False,
            "uses_core100_or_core500": False,
        },
        out_path,
    )
    write_csv(REPORTS_DIR / "sw14h8_online_query_score_dump_manifest.csv", manifest)
    total_rows = int(sum(int(r["candidate_rows"]) for r in manifest))
    total_pos = int(sum(int(r["positive_rows"]) for r in manifest))
    summary = {
        "decision": "SW14H8_1_SMOKE_READY" if total_rows > 0 else "SW14H8_0_NO_ROWS",
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
        "next": "Scale this dump to train 0..99 and val 100..149, then train H8 query-score ranker toward add@100K 0.9.",
    }
    write_json(REPORTS_DIR / "sw14h8_online_query_score_dump_smoke_summary.json", summary)
    print(json.dumps(normalize(summary), indent=2), flush=True)


if __name__ == "__main__":
    run()
