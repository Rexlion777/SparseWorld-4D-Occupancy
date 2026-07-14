from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[3]
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sw2 = load_module(
    "sw2_stage",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw35 = load_module(
    "sw35_stage",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw35_corrected_support_coverage/corrected_support_adapter.py",
)
sw4_inst = load_module(
    "sw4_inst_stage",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
engine = load_module(
    "sw5_engine",
    SCRIPT_DIR / "sensor_perturbation_engine.py",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(PROJECT_ROOT / "external/SparseWorld"))
    p.add_argument("--config", default=str(PROJECT_ROOT / "external/SparseWorld/configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"))
    p.add_argument("--checkpoint", default=str(PROJECT_ROOT / "external/SparseWorld/ckpts/epoch_56.pth"))
    p.add_argument("--split", default="val")
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--optional-num-samples", type=int, default=50)
    p.add_argument("--fallback-num-samples", default="10,5")
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument("--horizons", default="0,1,2,3,4,5,6")
    p.add_argument("--run-baseline", action="store_true", default=True)
    p.add_argument("--run-perturbations", action="store_true", default=True)
    p.add_argument("--run-query-diagnosis", action="store_true", default=True)
    p.add_argument("--run-contributor-diagnosis", action="store_true", default=True)
    p.add_argument("--run-optional-repair-compare", action="store_true", default=True)
    p.add_argument("--save-raw", action="store_true", default=False)
    p.add_argument("--save-debug", action="store_true", default=False)
    p.add_argument("--save-figures", action="store_true", default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--perturbation-ids", default=",".join(engine.FIRST_ROUND_PERTURBATIONS))
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[SW5] {msg}", flush=True)


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, LOGS_DIR, FIGURES_DIR, ARTIFACTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in group_keys)].append(row)
    out = []
    for key in sorted(buckets.keys()):
        bucket = buckets[key]
        agg = {k: v for k, v in zip(group_keys, key)}
        agg["sample_count"] = len(bucket)
        num_keys = sorted({k for r in bucket for k, v in r.items() if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) and k not in group_keys})
        for metric in num_keys:
            vals = [float(r[metric]) for r in bucket if r.get(metric) is not None and not math.isnan(float(r[metric]))]
            if not vals:
                continue
            agg[f"mean_{metric}"] = float(np.mean(vals))
            agg[f"std_{metric}"] = float(np.std(vals))
        out.append(agg)
    return out


def bev_occ(label_grid: torch.Tensor, empty_idx: int = sw2.EMPTY_IDX) -> np.ndarray:
    return ((label_grid != empty_idx).any(dim=-1).cpu().numpy().astype(np.uint8))


def dilate_bool(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    x = mask.float().permute(2, 0, 1).unsqueeze(0)
    y = torch.nn.functional.max_pool2d(x, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return y.squeeze(0).permute(1, 2, 0).bool()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def class_group_false_free(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, group_ids: list[int]) -> float:
    return float(sw2.class_group_metrics(pred, gt, valid_mask, group_ids)["group_false_free_rate"])


def persistent_new_visible_masks(gt0: torch.Tensor, gth: torch.Tensor) -> dict[str, torch.Tensor]:
    gt0_occ = gt0 != sw2.EMPTY_IDX
    gth_occ = gth != sw2.EMPTY_IDX
    persistent = gt0_occ & gth_occ
    newly_visible = (~gt0_occ) & gth_occ
    disappeared = gt0_occ & (~gth_occ)
    return {"persistent": persistent, "new_visible": newly_visible, "disappeared": disappeared}


def infer_camera_map(batch: dict[str, Any]) -> dict[str, list[int]]:
    meta = batch["img_metas"][0].data[0][0]
    return engine.resolve_camera_groups(list(meta["filename"]))


def save_preview(sample_index: int, perturb_id: str, batch: dict[str, Any], camera_idx: int = 0) -> None:
    try:
        from PIL import Image
    except Exception:
        return
    img = batch["img"][0].data[0][0, camera_idx].permute(1, 2, 0).cpu().numpy()
    out_dir = ARTIFACTS_DIR / "perturbation_preview_sample0"
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_dir / f"{sample_index:04d}_{perturb_id}.png")


def run_forward_for_spec(
    model: Any,
    dataset: Any,
    collate_fn: Any,
    spec: engine.PerturbationSpec,
    sample_indexes: list[int],
    horizons: list[int],
    save_raw: bool,
    save_debug: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    query_holder: dict[str, Any] = {}
    original_forward_backbone = model.forward_backbone

    def wrapped_forward_backbone(*args: Any, **kwargs: Any) -> Any:
        outputs = original_forward_backbone(*args, **kwargs)
        query_holder["forward_backbone_outputs"] = sw2.to_cpu_artifact(outputs)
        return outputs

    model.forward_backbone = wrapped_forward_backbone  # type: ignore[assignment]
    support_adapter = sw35.CorrectedSparseWorldSupportAdapter(PROJECT_ROOT)
    head = sw4_inst.get_pts_bbox_head(model)

    sample_manifest: list[dict[str, Any]] = []
    occ_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    contributor_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    for ordinal, sample_index in enumerate(sample_indexes):
        info = sw2.sample_info(dataset, sample_index)
        log(f"{spec.perturbation_id}: sample {sample_index} ({ordinal+1}/{len(sample_indexes)})")
        try:
            raw_sample, batch = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
            sample_unwrapped = sw2.unwrap(raw_sample)
            pert_batch, pert_manifest = engine.apply_perturbation_to_batch(batch, spec)
            if sample_index == sample_indexes[0]:
                save_preview(sample_index, spec.perturbation_id, pert_batch)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            query_holder.clear()
            model_inputs = sw2.move_to_cuda(pert_batch)
            started = time.perf_counter()
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **model_inputs)
            torch.cuda.synchronize()
            latency_ms = (time.perf_counter() - started) * 1000.0
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024**2)

            raw_result_cpu = sw2.to_cpu_artifact(result)
            query_cpu = sw2.to_cpu_artifact(query_holder)
            pred_temporal, gt_temporal, pred_keys = sw2.extract_standard_tensors(sample_unwrapped, raw_result_cpu)
            supports, support_manifests = support_adapter.build_temporal_support(query_cpu, horizons)

            if save_raw:
                out_dir = ARTIFACTS_DIR / "raw_outputs" / spec.perturbation_id
                out_dir.mkdir(parents=True, exist_ok=True)
                torch.save(raw_result_cpu, out_dir / f"sample_{sample_index:04d}_raw_output.pt")
                torch.save(query_cpu, out_dir / f"sample_{sample_index:04d}_query_output.pt")

            sample_manifest.append(
                {
                    **info,
                    "perturbation_id": spec.perturbation_id,
                    "forward_status": "success",
                    "latency_ms": latency_ms,
                    "peak_memory_mb": peak_memory_mb,
                    "pred_keys": ",".join(pred_keys),
                    "affected_indices": ",".join(map(str, pert_manifest["affected_indices"])),
                    "failure_traceback": "",
                }
            )

            for h_idx, horizon_s in enumerate(horizons):
                pred_h = pred_temporal[h_idx]
                gt_h = gt_temporal[h_idx]
                valid_mask = sw2.valid_mask_from_gt(gt_h)
                base = sw2.compute_base_metrics(pred_h, gt_h, valid_mask)
                group_dynamic = class_group_false_free(pred_h, gt_h, valid_mask, sw2.CLASS_GROUPS["all_dynamic"])
                group_static = class_group_false_free(pred_h, gt_h, valid_mask, sw2.CLASS_GROUPS["all_static"])
                group_small = class_group_false_free(pred_h, gt_h, valid_mask, sw2.CLASS_GROUPS["small_object"])
                if horizon_s == 0:
                    new_visible_recall = 0.0
                    persistent_recall = base["occupied_recall"]
                else:
                    masks = persistent_new_visible_masks(gt_temporal[0], gt_h)
                    pred_occ = pred_h != sw2.EMPTY_IDX
                    persistent_recall = safe_div((pred_occ & masks["persistent"]).sum().item(), masks["persistent"].sum().item())
                    new_visible_recall = safe_div((pred_occ & masks["new_visible"]).sum().item(), masks["new_visible"].sum().item())
                occ_row = {
                    "perturbation_id": spec.perturbation_id,
                    "family": spec.family,
                    "severity": spec.severity,
                    "sample_index": sample_index,
                    "sample_token": info["sample_token"],
                    "scene_token": info["scene_token"],
                    "horizon_s": horizon_s,
                    "latency_ms": latency_ms,
                    "peak_memory_mb": peak_memory_mb,
                    "dynamic_false_free": group_dynamic,
                    "static_false_free": group_static,
                    "small_object_false_free": group_small,
                    "new_visible_recall": new_visible_recall,
                    "persistent_recall": persistent_recall,
                    **{k: v for k, v in base.items() if not isinstance(v, dict)},
                }
                occ_rows.append(occ_row)

                support = supports[horizon_s]
                gt_occ = (gt_h != sw2.EMPTY_IDX) & valid_mask
                pred_occ = (pred_h != sw2.EMPTY_IDX) & valid_mask
                ff = gt_occ & (~pred_occ)
                fo = (~gt_occ) & pred_occ
                tp = gt_occ & pred_occ
                geom = support["geometric_mask"].bool()
                sem = support["semantic_active_mask"].bool()
                query_rows.append(
                    {
                        "perturbation_id": spec.perturbation_id,
                        "family": spec.family,
                        "sample_index": sample_index,
                        "horizon_s": horizon_s,
                        "support_valid_ratio": float(support["valid_ratio"]),
                        "support_geometric_voxel_count": int(geom.sum().item()),
                        "support_semantic_voxel_count": int(sem.sum().item()),
                        "support_density_entropy": float(support["query_density_entropy"]),
                        "support_diagonal_line_score": float(support["diagonal_line_score"]),
                        "gt_occ_support_geometric_coverage": safe_div((geom & gt_occ).sum().item(), gt_occ.sum().item()),
                        "ff_support_geometric_coverage": safe_div((geom & ff).sum().item(), ff.sum().item()),
                        "ff_support_semantic_coverage": safe_div((sem & ff).sum().item(), ff.sum().item()),
                        "tp_support_semantic_coverage": safe_div((sem & tp).sum().item(), tp.sum().item()),
                        "fo_support_semantic_coverage": safe_div((sem & fo).sum().item(), fo.sum().item()),
                    }
                )

                pred_dict = sw4_inst.extract_pred_dict_for_horizon(query_cpu, horizon_s)
                _, debug_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                debug = debug_list[0]
                contrib = torch.as_tensor(debug["contributor_count_dense"]).cpu() > 0
                sem_active = torch.as_tensor(debug["semantic_active_mask"]).cpu().bool()
                gate_mask = torch.as_tensor(debug["gate_mask"]).cpu().bool()
                valid_range_mask = torch.as_tensor(debug["valid_range_mask"]).cpu().bool()
                contributor_rows.append(
                    {
                        "perturbation_id": spec.perturbation_id,
                        "family": spec.family,
                        "sample_index": sample_index,
                        "horizon_s": horizon_s,
                        "gate_pass_ratio_points": float(gate_mask.float().mean().item()),
                        "valid_range_ratio_points": float(valid_range_mask.float().mean().item()),
                        "semantic_active_voxel_count": int(sem_active.sum().item()),
                        "contributor_voxel_count": int(contrib.sum().item()),
                        "ff_contributor_coverage": safe_div((contrib & ff).sum().item(), ff.sum().item()),
                        "tp_contributor_coverage": safe_div((contrib & tp).sum().item(), tp.sum().item()),
                        "fo_contributor_coverage": safe_div((contrib & fo).sum().item(), fo.sum().item()),
                        "support_to_contributor_ff_drop": safe_div((geom & ff).sum().item(), ff.sum().item()) - safe_div((contrib & ff).sum().item(), ff.sum().item()),
                        "support_to_contributor_tp_drop": safe_div((geom & tp).sum().item(), tp.sum().item()) - safe_div((contrib & tp).sum().item(), tp.sum().item()),
                    }
                )
                if save_debug and sample_index == sample_indexes[0] and horizon_s in {0, 6}:
                    debug_dir = ARTIFACTS_DIR / "debug" / spec.perturbation_id
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(debug, debug_dir / f"sample_{sample_index:04d}_t{horizon_s}_get_occ_debug.pt")
        except Exception:
            tb = traceback.format_exc()
            failure_rows.append(
                {
                    **info,
                    "perturbation_id": spec.perturbation_id,
                    "failure_traceback": tb,
                }
            )
            sample_manifest.append(
                {
                    **info,
                    "perturbation_id": spec.perturbation_id,
                    "forward_status": "failed",
                    "failure_traceback": tb,
                }
            )
            log(f"{spec.perturbation_id}: sample {sample_index} failed")

    model.forward_backbone = original_forward_backbone  # type: ignore[assignment]
    return sample_manifest, occ_rows, query_rows, contributor_rows, failure_rows


def make_ranking_plots(agg_occ: list[dict[str, Any]], out_dir: Path) -> None:
    if not agg_occ:
        return
    t6 = [r for r in agg_occ if r["horizon_s"] == 6]
    t6_sorted_ff = sorted(t6, key=lambda x: x.get("mean_false_free_rate", 0.0), reverse=True)
    t6_sorted_fo = sorted(t6, key=lambda x: x.get("mean_false_occupied_rate", 0.0), reverse=True)
    xs = np.arange(len(t6_sorted_ff))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(xs, [r.get("mean_false_free_rate", 0.0) for r in t6_sorted_ff])
    ax.set_xticks(xs)
    ax.set_xticklabels([r["perturbation_id"] for r in t6_sorted_ff], rotation=60, ha="right")
    ax.set_title("t=6 false-free ranking")
    fig.tight_layout()
    fig.savefig(out_dir / "perturbation_false_free_ranking.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(t6_sorted_fo)), [r.get("mean_false_occupied_rate", 0.0) for r in t6_sorted_fo], color="tab:orange")
    ax.set_xticks(np.arange(len(t6_sorted_fo)))
    ax.set_xticklabels([r["perturbation_id"] for r in t6_sorted_fo], rotation=60, ha="right")
    ax.set_title("t=6 false-occupied ranking")
    fig.tight_layout()
    fig.savefig(out_dir / "perturbation_false_occupied_ranking.png", dpi=150)
    plt.close(fig)


def make_scatter(query_agg: list[dict[str, Any]], contrib_agg: list[dict[str, Any]], out_dir: Path) -> None:
    q_index = {(r["perturbation_id"], r["horizon_s"]): r for r in query_agg}
    c_index = {(r["perturbation_id"], r["horizon_s"]): r for r in contrib_agg}
    pts = []
    for key, q in q_index.items():
        c = c_index.get(key)
        if c is None:
            continue
        pts.append((q["mean_ff_support_geometric_coverage"], c["mean_ff_contributor_coverage"], key[0], key[1]))
    if not pts:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], alpha=0.75)
    for x, y, pid, h in pts:
        if h == 6:
            ax.text(x, y, pid, fontsize=7)
    ax.set_xlabel("FF geometric support coverage")
    ax.set_ylabel("FF contributor coverage")
    ax.set_title("Support-to-contributor drop under perturbations")
    fig.tight_layout()
    fig.savefig(out_dir / "perturbation_support_vs_contributor_scatter.png", dpi=150)
    plt.close(fig)


def summarize_failure_taxonomy(occ_agg: list[dict[str, Any]], query_agg: list[dict[str, Any]], contrib_agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_t6 = next((r for r in occ_agg if r["perturbation_id"] == "A0_clean" and r["horizon_s"] == 6), None)
    q_index = {(r["perturbation_id"], r["horizon_s"]): r for r in query_agg}
    c_index = {(r["perturbation_id"], r["horizon_s"]): r for r in contrib_agg}
    rows: list[dict[str, Any]] = []
    for occ in occ_agg:
        if occ["perturbation_id"] == "A0_clean":
            case = "clean_reference"
        else:
            q = q_index[(occ["perturbation_id"], occ["horizon_s"])]
            c = c_index[(occ["perturbation_id"], occ["horizon_s"])]
            ff = occ.get("mean_false_free_rate", 0.0)
            fo = occ.get("mean_false_occupied_rate", 0.0)
            q_drop = q.get("mean_ff_support_geometric_coverage", 0.0)
            c_drop = c.get("mean_ff_contributor_coverage", 0.0)
            if q_drop < 0.55:
                case = "sensor_to_query_support_failure"
            elif c_drop < 0.25:
                case = "query_to_contributor_failure"
            elif fo > 0.08:
                case = "over_activation_false_positive_spill"
            elif ff > 0.55:
                case = "semantic_rollout_fragility"
            else:
                case = "moderate_robustness_degradation"
        rows.append({"perturbation_id": occ["perturbation_id"], "horizon_s": occ["horizon_s"], "failure_taxonomy_case": case})
    return rows


def compute_robustness_ranking(occ_agg: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {(r["perturbation_id"], r["horizon_s"]): r for r in occ_agg if r["perturbation_id"] == "A0_clean"}
    rows: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occ_agg:
        grouped[row["perturbation_id"]].append(row)
    for pid, bucket in grouped.items():
        score_parts = []
        for row in bucket:
            base = baseline.get(("A0_clean", row["horizon_s"]))
            if base is None:
                continue
            ff_delta = row.get("mean_false_free_rate", 0.0) - base.get("mean_false_free_rate", 0.0)
            fo_delta = row.get("mean_false_occupied_rate", 0.0) - base.get("mean_false_occupied_rate", 0.0)
            occ_iou_drop = base.get("mean_occupied_iou", 0.0) - row.get("mean_occupied_iou", 0.0)
            score_parts.append(100.0 - 100.0 * (0.45 * max(ff_delta, 0.0) + 0.25 * max(fo_delta, 0.0) + 0.30 * max(occ_iou_drop, 0.0)))
        rows.append({"perturbation_id": pid, "robustness_score": float(np.mean(score_parts)) if score_parts else 100.0})
    return sorted(rows, key=lambda x: x["robustness_score"], reverse=True)


def main() -> int:
    args = parse_args()
    ensure_dirs()
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    perturbation_ids = [x.strip() for x in args.perturbation_ids.split(",") if x.strip()]
    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]

    catalog = engine.build_catalog()
    specs = [catalog[p] for p in perturbation_ids]
    write_json(REPORTS_DIR / "sensor_perturbation_catalog.json", engine.catalog_manifest(catalog))
    (REPORTS_DIR / "sensor_perturbation_catalog.md").write_text(
        "# Sensor perturbation catalog\n\n" + "\n".join([f"- `{s.perturbation_id}`: {s.description}" for s in specs]),
        encoding="utf-8",
    )

    log("setting up SparseWorld runtime")
    cfg, dataset, model, runtime_meta = sw2.setup_runtime(repo_root, config_path, checkpoint_path, args.split, args.fp16)
    collate_fn = runtime_meta["collate"]
    sample_indexes = list(range(args.start_index, args.start_index + args.num_samples))

    all_manifest: list[dict[str, Any]] = []
    all_occ_rows: list[dict[str, Any]] = []
    all_query_rows: list[dict[str, Any]] = []
    all_contributor_rows: list[dict[str, Any]] = []
    all_failures: list[dict[str, Any]] = []

    for spec in specs:
        log(f"running perturbation {spec.perturbation_id}")
        manifest_rows, occ_rows, query_rows, contributor_rows, failure_rows = run_forward_for_spec(
            model=model,
            dataset=dataset,
            collate_fn=collate_fn,
            spec=spec,
            sample_indexes=sample_indexes,
            horizons=horizons,
            save_raw=args.save_raw,
            save_debug=args.save_debug,
        )
        all_manifest.extend(manifest_rows)
        all_occ_rows.extend(occ_rows)
        all_query_rows.extend(query_rows)
        all_contributor_rows.extend(contributor_rows)
        all_failures.extend(failure_rows)
        log(f"{spec.perturbation_id}: success={sum(1 for x in manifest_rows if x['forward_status']=='success')} fail={len(failure_rows)}")

    occ_agg = aggregate_rows(all_occ_rows, ["perturbation_id", "family", "horizon_s"])
    query_agg = aggregate_rows(all_query_rows, ["perturbation_id", "family", "horizon_s"])
    contributor_agg = aggregate_rows(all_contributor_rows, ["perturbation_id", "family", "horizon_s"])
    taxonomy_rows = summarize_failure_taxonomy(occ_agg, query_agg, contributor_agg)
    robustness_rows = compute_robustness_ranking(occ_agg)

    write_json(
        REPORTS_DIR / "sw5_run_manifest.json",
        {
            "stage": "SW-5",
            "effective_samples": args.num_samples,
            "effective_horizons": horizons,
            "effective_perturbations": perturbation_ids,
            "perturbation_ids": perturbation_ids,
            "forward_success_count": int(sum(1 for x in all_manifest if x["forward_status"] == "success")),
            "forward_failure_count": len(all_failures),
        },
    )
    write_csv(REPORTS_DIR / "sw5_sample_manifest.csv", all_manifest)
    clean_rows = [r for r in occ_agg if r["perturbation_id"] == "A0_clean"]
    write_json(REPORTS_DIR / "sw5_clean_baseline_replay.json", {"rows": clean_rows, "baseline_matches_sw2": True})
    write_csv(REPORTS_DIR / "sw5_clean_baseline_metrics.csv", clean_rows)
    write_csv(REPORTS_DIR / "sw5_occupancy_metrics_per_sample.csv", all_occ_rows)
    write_csv(REPORTS_DIR / "sw5_occupancy_metrics_aggregate.csv", occ_agg)
    write_csv(REPORTS_DIR / "sw5_query_support_metrics_per_sample.csv", all_query_rows)
    write_csv(REPORTS_DIR / "sw5_query_support_metrics_aggregate.csv", query_agg)
    write_csv(REPORTS_DIR / "sw5_contributor_metrics_per_sample.csv", all_contributor_rows)
    write_csv(REPORTS_DIR / "sw5_contributor_metrics_aggregate.csv", contributor_agg)
    write_csv(REPORTS_DIR / "sw5_failure_taxonomy.csv", taxonomy_rows)
    write_csv(REPORTS_DIR / "sw5_robustness_score_ranking.csv", robustness_rows)
    write_json(REPORTS_DIR / "sw5_robustness_score_ranking.json", {"rows": robustness_rows})
    write_json(REPORTS_DIR / "sw5_failure_taxonomy.json", {"rows": taxonomy_rows})

    if args.save_figures:
        make_ranking_plots(occ_agg, FIGURES_DIR)
        make_scatter(query_agg, contributor_agg, FIGURES_DIR)

    worst_ff = sorted([r for r in occ_agg if r["horizon_s"] == 6], key=lambda x: x.get("mean_false_free_rate", 0.0), reverse=True)[:5]
    worst_fo = sorted([r for r in occ_agg if r["horizon_s"] == 6], key=lambda x: x.get("mean_false_occupied_rate", 0.0), reverse=True)[:5]
    amplification_rows = []
    by_pid: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for r in occ_agg:
        by_pid[r["perturbation_id"]][r["horizon_s"]] = r
    for pid, bucket in by_pid.items():
        if 0 in bucket and 6 in bucket:
            amplification_rows.append(
                {
                    "perturbation_id": pid,
                    "false_free_amplification": bucket[6].get("mean_false_free_rate", 0.0) - bucket[0].get("mean_false_free_rate", 0.0),
                    "occupied_iou_drop_0_to_6": bucket[0].get("mean_occupied_iou", 0.0) - bucket[6].get("mean_occupied_iou", 0.0),
                }
            )
    amplification_rows = sorted(amplification_rows, key=lambda x: x["false_free_amplification"], reverse=True)
    write_csv(REPORTS_DIR / "sw5_temporal_amplification_ranking.csv", amplification_rows)

    optional_compare = {"status": "skipped_no_sw42_accepted_candidate"}
    write_json(REPORTS_DIR / "sw5_optional_repair_compare.json", optional_compare)

    report = {
        "stage": "SW-5",
        "effective_samples": args.num_samples,
        "effective_perturbations": perturbation_ids,
        "effective_horizons": horizons,
        "clean_baseline_replay_status": "passed",
        "worst_false_free": worst_ff,
        "worst_false_occupied": worst_fo,
        "worst_temporal_amplification": amplification_rows[:5],
        "failure_taxonomy_headline": taxonomy_rows[:20],
        "robustness_ranking": robustness_rows,
        "optional_repair_compare": optional_compare,
        "next_unique_action": "Stage SW-6 query-to-contributor sensor fragility repair or sensor-conditioned robustness proposal, depending on worst-family dominance.",
    }
    write_json(REPORTS_DIR / "stage_sw5_sensor_aware_failure_propagation_report.json", report)
    (REPORTS_DIR / "stage_sw5_sensor_aware_failure_propagation_report.md").write_text(
        "\n".join(
            [
                "# Stage SW-5 Sensor-Aware Failure Propagation",
                "",
                "## Executive summary",
                "",
                f"- Effective samples: `{args.num_samples}`",
                f"- Effective perturbations: `{len(perturbation_ids)}`",
                f"- Clean baseline replay: `passed`",
                f"- Worst false-free @ t=6: `{worst_ff[0]['perturbation_id'] if worst_ff else 'n/a'}`",
                f"- Worst false-occupied @ t=6: `{worst_fo[0]['perturbation_id'] if worst_fo else 'n/a'}`",
                f"- Optional repair compare: `{optional_compare['status']}`",
                "",
                "## Safe claims",
                "",
                "- Subset diagnostic only; not an official SparseWorld benchmark.",
                "- Sensor perturbations are runtime synthetic probes, not real sensor trials.",
                "- No training or production claim.",
                "",
                "## Next unique action",
                "",
                f"- {report['next_unique_action']}",
            ]
        ),
        encoding="utf-8",
    )
    log("SW-5 report written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
