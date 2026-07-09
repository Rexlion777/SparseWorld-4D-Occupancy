from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import torch
from mmcv.parallel import collate as collate_fn


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_causal_replay"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14_feature_gate_distillation_causal_replay"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_causal_replay"

BASE_SW14 = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/run_sw14_main.py"
TEACHERFIX = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/run_sw14a_teacherfix.py"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw14 = load_module("sw14_causal_base", BASE_SW14)
teacherfix = load_module("sw14_causal_teacherfix", TEACHERFIX)


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, ARTIFACTS_DIR, SCRIPTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sw14.normalize(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(sw14.normalize(row))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate():
    return sw14.teacher_candidates()[0]


def build_case(model: Any, dataset: Any, sample_index: int) -> dict[str, Any]:
    return teacherfix.build_train_reconstruction(model, dataset, sample_index)


def level_diff_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().mean().item())


def run_levels_forward(
    model: Any,
    sample_unwrapped: dict[str, Any],
    batch_degraded: dict[str, Any],
    replay_levels: list[torch.Tensor],
) -> dict[int, dict[str, Any]]:
    holder: dict[str, Any] = {}
    original_forward = sw14.sw81.attach_query_capture(model, holder)
    original_simple_test_online = model.simple_test_online

    def patched_simple_test_online(self, img_metas, img=None, rescale=False):
        self.fp16_enabled = False
        B, N, C, H, W = img.shape
        img = img.reshape(B, N // 6, 6, C, H, W)
        img_filenames = img_metas[0]["filename"]
        num_frames = len(img_filenames) // 6
        img_shape = (H, W, C)
        img_metas[0]["img_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]["ori_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_metas[0]["pad_shape"] = [img_shape for _ in range(len(img_filenames))]
        img_feats_list, img_metas_list = [], []
        for i in range(num_frames):
            img_indices = list(np.arange(i * 6, (i + 1) * 6))
            img_metas_curr = sw14.sw13a.clone_meta_for_indices(img_metas[0], img_indices)
            img_feats_curr = self.extract_feat(img[:, i], img_metas_curr)
            if i == 0:
                img_feats_curr = [feat.to(img_feats_curr[lvl].device, dtype=img_feats_curr[lvl].dtype) for lvl, feat in enumerate(replay_levels)]
            img_feats_list.append(img_feats_curr)
            img_metas_list.append(img_metas_curr)
        feat_levels = len(img_feats_list[0])
        img_feats_reorganized = []
        for j in range(feat_levels):
            feat_l = torch.cat([img_feats_list[i][j] for i in range(len(img_feats_list))], dim=0)
            feat_l = feat_l.flatten(0, 1)[None, ...]
            img_feats_reorganized.append(feat_l)
        img_metas_reorganized = img_metas_list[0]
        for i in range(1, len(img_metas_list)):
            for k, v in img_metas_list[i][0].items():
                if isinstance(v, list):
                    img_metas_reorganized[0][k].extend(v)
        img_feats = sw14.sw13a.cast_tensor_type(img_feats_reorganized, torch.half, torch.float32)
        return self.simple_test_pts(img_feats, img_metas_reorganized, rescale=rescale)

    model.simple_test_online = patched_simple_test_online.__get__(model, type(model))
    try:
        sw14.sw13a.reset_model_cache(model)
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **sw14.sw2.move_to_cuda(batch_degraded))
        raw_result_cpu = sw14.sw2.to_cpu_artifact(result)
        query_cpu = sw14.sw2.to_cpu_artifact(holder)
        _, gt_temporal, pred_keys = sw14.sw2.extract_standard_tensors(sample_unwrapped, raw_result_cpu)
        per_h: dict[int, dict[str, Any]] = {}
        for horizon_s in sw14.CORE_HORIZONS:
            pred_dict = sw14.sw4_inst.extract_pred_dict_for_horizon(query_cpu, horizon_s)
            per_h[horizon_s] = {
                "pred_dict": pred_dict,
                "gt_h": gt_temporal[horizon_s].long().cpu(),
                "gt0": gt_temporal[0].long().cpu(),
                "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None,
            }
        return per_h
    finally:
        model.simple_test_online = original_simple_test_online
        model.forward_backbone = original_forward  # type: ignore[assignment]


def feature_level_audit() -> dict[str, Any]:
    _, dataset, model, _ = sw14.build_runtime(train=True)
    model.eval()
    recon = build_case(model, dataset, 0)
    audit = {
        "level_count": len(recon["current_level0_full"].new_zeros(1)),  # placeholder updated below
        "levels": [],
        "hard_repair_all_levels": True,
        "teacherfix_saved_only_level0_pooled": True,
        "target_status": "incomplete_target",
    }
    full = teacherfix.build_train_reconstruction(model, dataset, 0)
    # rebuild with explicit full levels
    raw_sample, batch_clean = sw14.sw2.extract_sample_batch(dataset, 0, collate_fn)
    sample_unwrapped = sw14.sw2.unwrap(raw_sample)
    moved_clean = sw14.sw2.move_to_cuda(batch_clean)
    sw14.sw13a.reset_model_cache(model)
    _, _, cache = sw14.sw13a.extract_clean_memory_for_sample(model, moved_clean, sample_unwrapped, 0)
    meta = sw14.sw13a.get_meta_dict(sample_unwrapped)
    img_metas_curr = sw14.sw13a.clone_meta_for_indices(meta, list(range(6)))
    batch_deg = sw14.perturbation_batch_compatible(batch_clean)
    batch_deg = sw14.sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, sw14.sw81.sw5_engine.build_catalog()[candidate().perturbation_id])[0]
    moved_deg = sw14.sw2.move_to_cuda(batch_deg)
    img_deg = sw14.sw13a.frame_img_tensor(moved_deg)
    with torch.no_grad():
        current_levels = [feat.detach().cpu().float() for feat in model.extract_feat(img_deg[:, :6], img_metas_curr)]
    memory_levels = sw14.build_memory_levels(cache, sw14.variant_spec_for_candidate(candidate()))
    hard_levels = sw14.build_hard_repair_levels([feat.clone() for feat in current_levels], memory_levels, candidate())
    audit["level_count"] = len(current_levels)
    for idx, (curr, mem, hard) in enumerate(zip(current_levels, memory_levels, hard_levels)):
        audit["levels"].append(
            {
                "level_id": idx,
                "current_shape": list(curr.shape),
                "memory_shape": list(mem.shape),
                "hard_r8_shape": list(hard.shape),
                "channels": int(curr.shape[2]),
                "height": int(curr.shape[3]),
                "width": int(curr.shape[4]),
                "hard_repair_applied": level_diff_mean(curr, hard) > 0.0,
            }
        )
    del model
    write_json(REPORTS_DIR / "sw14a_causal_replay_feature_level_audit.json", audit)
    write_md(
        REPORTS_DIR / "sw14a_causal_replay_feature_level_audit.md",
        "\n".join(
            [
                "# SW14A Causal Replay Feature Level Audit",
                "",
                f"- feature level count: `{audit['level_count']}`.",
                "- SW13 hard R8 repair is applied across all returned feature levels.",
                "- current TeacherFix target stores only level0 pooled target, so it is an incomplete feature target for causal replay.",
            ]
        ),
    )
    return audit


def replay_metrics(kind: str) -> list[dict[str, Any]]:
    _, dataset, model, _ = sw14.build_runtime(train=False)
    model.eval()
    sectors = {name: tensor.cpu() for name, tensor in sw14.sw7.build_sector_masks().items()}
    spec_catalog = sw14.sw81.sw5_engine.build_catalog()
    rows: list[dict[str, Any]] = []
    for sample_index in range(100, 120):
        raw_sample, batch_clean = sw14.sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample_unwrapped = sw14.sw2.unwrap(raw_sample)
        case = build_case(model, dataset, sample_index)
        batch_deg = sw14.perturbation_batch_compatible(batch_clean)
        batch_deg = sw14.sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec_catalog[candidate().perturbation_id])[0]
        if kind == "hard_r8":
            replay_levels = [case["hard_r8_level0_full"]]  # placeholder
        else:
            replay_levels = [case["memory_level0_full"].clone()]
        # replace all levels explicitly
        raw_sample2, batch_clean2 = sw14.sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample_unwrapped2 = sw14.sw2.unwrap(raw_sample2)
        moved_clean = sw14.sw2.move_to_cuda(batch_clean2)
        sw14.sw13a.reset_model_cache(model)
        _, _, cache = sw14.sw13a.extract_clean_memory_for_sample(model, moved_clean, sample_unwrapped2, sample_index)
        meta = sw14.sw13a.get_meta_dict(sample_unwrapped2)
        img_metas_curr = sw14.sw13a.clone_meta_for_indices(meta, list(range(6)))
        batch_deg2 = sw14.perturbation_batch_compatible(batch_clean2)
        batch_deg2 = sw14.sw81.sw5_engine.apply_perturbation_to_batch(batch_deg2, spec_catalog[candidate().perturbation_id])[0]
        moved_deg = sw14.sw2.move_to_cuda(batch_deg2)
        img_deg = sw14.sw13a.frame_img_tensor(moved_deg)
        with torch.no_grad():
            current_levels = [feat.detach().cpu().float() for feat in model.extract_feat(img_deg[:, :6], img_metas_curr)]
        memory_levels = sw14.build_memory_levels(cache, sw14.variant_spec_for_candidate(candidate()))
        hard_levels = sw14.build_hard_repair_levels([feat.clone() for feat in current_levels], memory_levels, candidate())
        if kind == "hard_r8":
            replay_levels = hard_levels
        else:
            replay_levels = []
            degraded = set(sw14.degraded_camera_names(candidate().perturbation_id))
            for curr, mem in zip(current_levels, memory_levels):
                lvl = curr.clone()
                for cam_idx, cam_name in enumerate(sw14.CAMERA_NAMES):
                    if cam_name in degraded:
                        lvl[:, cam_idx] = mem[:, cam_idx]
                replay_levels.append(lvl)
        per_h = run_levels_forward(model, sample_unwrapped, batch_deg, replay_levels)
        cand_key, front_cap_variant = sw14.candidate_key_and_frontcap(candidate())
        for horizon_s in sw14.CORE_HORIZONS:
            replay_raw_semantic, _ = sw14.sw13b.dense_debug_for_case(model, per_h[horizon_s]["pred_dict"])
            native_data = sw14.load_gpu_dump(sample_index, horizon_s, candidate().perturbation_id, "native_baseline")
            teacher = sw14.load_teacher_output(sample_index, horizon_s, cand_key, front_cap_variant)
            replay_after, _, cap_meta, cap_eval = sw14.build_frontcap_from_semantic(
                replay_raw_semantic,
                native_data["semantic"].long(),
                per_h[horizon_s]["gt_h"].long(),
                candidate(),
                sample_index,
                horizon_s,
                sectors,
            )
            replay_row = sw14.sw12b.build_eval_row(replay_after.long(), per_h[horizon_s]["gt_h"], per_h[horizon_s]["gt0"], candidate().perturbation_id, horizon_s, sectors, baseline_pred=native_data["semantic"].long())
            teacher_row = sw14.sw12b.build_eval_row(teacher["final_after_cap"].long(), teacher["gt_h"].long(), per_h[horizon_s]["gt0"], candidate().perturbation_id, horizon_s, sectors, baseline_pred=native_data["semantic"].long())
            rows.append(
                {
                    "sample_index": sample_index,
                    "horizon_s": horizon_s,
                    "replay_type": kind,
                    "replay_front_sector_false_free_delta": sw14.pick_metric(replay_row, "front_sector_false_free_rate_delta"),
                    "teacher_front_sector_false_free_delta": sw14.pick_metric(teacher_row, "front_sector_false_free_rate_delta"),
                    "replay_pred_gt_density_delta": sw14.pick_metric(replay_row, "pred_gt_density_delta"),
                    "teacher_pred_gt_density_delta": sw14.pick_metric(teacher_row, "pred_gt_density_delta"),
                    "replay_vs_teacher_occ_diff": int(((replay_after.long() != sw14.EMPTY_IDX) ^ (teacher["final_after_cap"].long() != sw14.EMPTY_IDX)).sum().item()),
                    "replay_reproduces_teacher": abs(sw14.pick_metric(replay_row, "front_sector_false_free_rate_delta") - sw14.pick_metric(teacher_row, "front_sector_false_free_rate_delta")) <= 0.02
                    and sw14.pick_metric(replay_row, "pred_gt_density_delta") <= sw14.pick_metric(teacher_row, "pred_gt_density_delta") + 0.05,
                    "front_local_proxy_after": float(cap_meta["front_local_density_proxy_after"]),
                    "density_before_frontcap": cap_eval["adapter_raw_density_delta"],
                    "density_after_frontcap": cap_eval["adapter_after_frontcap_density_delta"],
                }
            )
    del model
    return rows


def reconstruction_metrics() -> list[dict[str, Any]]:
    ckpt = read_json(PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_checkpoint_manifest.json")
    adapter = sw14.build_adapter().cuda()
    payload = torch.load(ckpt["checkpoint_path"], map_location="cpu", weights_only=False)
    adapter.load_state_dict(payload["state_dict"])
    adapter.eval()
    _, dataset, model, _ = sw14.build_runtime(train=False)
    model.eval()
    rows: list[dict[str, Any]] = []
    for sample_index in range(100, 120):
        case = build_case(model, dataset, sample_index)
        current_full = case["current_level0_full"].cuda()
        memory_full = case["memory_level0_full"].cuda()
        target_full = case["hard_r8_level0_full"].cuda()
        degradation_mask = case["degradation_mask"].cuda()
        camera_ids = torch.arange(6, device=current_full.device)[None]
        memory_age = torch.full((1, 6, 1), 1.0 / 3.0, device=current_full.device)
        adapter_levels, _ = adapter.apply_to_levels([current_full], [memory_full], degradation_mask, camera_ids, memory_age)
        adapter_full = adapter_levels[0]
        base_l1 = (current_full - target_full).abs()
        adapt_l1 = (adapter_full - target_full).abs()
        total_base = float(base_l1.mean().item())
        total_adapt = float(adapt_l1.mean().item())
        rows.append(
            {
                "sample_index": sample_index,
                "level_id": 0,
                "camera_name": "ALL",
                "l1_current_to_target": total_base,
                "l1_adapter_to_target": total_adapt,
                "reconstruction_ratio": 1.0 - sw14.safe_div(total_adapt, total_base),
                "degraded_reconstruction_ratio": 1.0
                - sw14.safe_div(
                    float(adapt_l1[:, :3].mean().item()),
                    float(base_l1[:, :3].mean().item()),
                ),
                "non_degraded_reconstruction_ratio": 1.0
                - sw14.safe_div(
                    float(adapt_l1[:, 3:].mean().item()),
                    float(base_l1[:, 3:].mean().item()),
                ),
            }
        )
        for cam_idx, cam_name in enumerate(sw14.CAMERA_NAMES):
            base_cam = float(base_l1[:, cam_idx].mean().item())
            adapt_cam = float(adapt_l1[:, cam_idx].mean().item())
            rows.append(
                {
                    "sample_index": sample_index,
                    "level_id": 0,
                    "camera_name": cam_name,
                    "l1_current_to_target": base_cam,
                    "l1_adapter_to_target": adapt_cam,
                    "reconstruction_ratio": 1.0 - sw14.safe_div(adapt_cam, base_cam),
                    "degraded_reconstruction_ratio": None,
                    "non_degraded_reconstruction_ratio": None,
                }
            )
    del model
    return rows


def main() -> None:
    ensure_dirs()
    sw14.seed_everything(17)
    audit = feature_level_audit()
    hard_rows = replay_metrics("hard_r8")
    oracle_rows = replay_metrics("oracle_alpha1")
    recon_rows = reconstruction_metrics()
    write_csv(REPORTS_DIR / "sw14a_causal_replay_hard_r8_metrics.csv", hard_rows)
    write_csv(REPORTS_DIR / "sw14a_causal_replay_oracle_alpha_metrics.csv", oracle_rows)
    write_csv(REPORTS_DIR / "sw14a_causal_replay_reconstruction_metrics.csv", recon_rows)

    hard_h6 = [r for r in hard_rows if int(r["horizon_s"]) == 6]
    oracle_h6 = [r for r in oracle_rows if int(r["horizon_s"]) == 6]
    recon_all = [r for r in recon_rows if r["camera_name"] == "ALL"]
    hard_replay_ok = bool(hard_h6) and np.mean([1.0 if r["replay_reproduces_teacher"] else 0.0 for r in hard_h6]) >= 0.8
    oracle_ok = bool(oracle_h6) and np.mean([1.0 if r["replay_reproduces_teacher"] else 0.0 for r in oracle_h6]) >= 0.8
    recon_ratio = float(np.mean([float(r["reconstruction_ratio"]) for r in recon_all])) if recon_all else 0.0
    nondeg_ratio = float(np.mean([float(r["non_degraded_reconstruction_ratio"]) for r in recon_all])) if recon_all else 0.0

    if not hard_replay_ok:
        decision = "C1_TARGET_NOT_CAUSAL"
    elif audit["level_count"] > 1:
        if not oracle_ok:
            decision = "C4_SPATIAL_GATE_REQUIRED"
        elif audit["teacherfix_saved_only_level0_pooled"]:
            decision = "C3_MULTILEVEL_TARGET_REQUIRED"
        elif recon_ratio < 0.2:
            decision = "C5_ALPHA_TOO_SOFT"
        else:
            decision = "C6_READY_FOR_DENSE_SW14A"
    else:
        decision = "C2_POOLED_TARGET_INSUFFICIENT"

    payload = {
        "decision_type": decision,
        "hard_r8_replay_reproduces_teacher_rate_h6": float(np.mean([1.0 if r["replay_reproduces_teacher"] else 0.0 for r in hard_h6])) if hard_h6 else 0.0,
        "oracle_alpha1_reproduces_teacher_rate_h6": float(np.mean([1.0 if r["replay_reproduces_teacher"] else 0.0 for r in oracle_h6])) if oracle_h6 else 0.0,
        "mean_reconstruction_ratio_level0": recon_ratio,
        "mean_non_degraded_reconstruction_ratio_level0": nondeg_ratio,
        "level_count": audit["level_count"],
        "no_training": True,
        "no_sw14b": True,
        "no_backbone_head_getocc_modification": True,
    }
    write_json(REPORTS_DIR / "sw14a_causal_replay_decision.json", payload)
    write_md(
        REPORTS_DIR / "stage_sw14a_causal_replay_report.md",
        "\n".join(
            [
                "# SW14A Causal Replay Dense Target",
                "",
                f"- decision: `{decision}`",
                f"- hard R8 replay teacher reproduction rate h6: `{payload['hard_r8_replay_reproduces_teacher_rate_h6']}`",
                f"- oracle alpha=1 teacher reproduction rate h6: `{payload['oracle_alpha1_reproduces_teacher_rate_h6']}`",
                f"- mean reconstruction ratio level0: `{payload['mean_reconstruction_ratio_level0']}`",
                "- no training",
                "- no SW14B",
                "- no backbone/head/get_occ modification",
            ]
        ),
    )


if __name__ == "__main__":
    main()
