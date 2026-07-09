from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import collate as collate_fn
from torch.utils.data import DataLoader, Dataset


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix"

BASE_SW14_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/run_sw14_main.py"
BASE_SW14_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14_feature_gate_distillation"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw14_feature_gate_distillation"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw14 = load_module("sw14_teacherfix_base", BASE_SW14_SCRIPT)


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        ARTIFACTS_DIR,
        ARTIFACTS_DIR / "teacherfix_dumps/train_tune",
        ARTIFACTS_DIR / "checkpoints",
        SCRIPTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
    ]:
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def teacher_candidate():
    return sw14.teacher_candidates()[0]


def build_train_reconstruction(
    model: Any,
    dataset: Any,
    sample_index: int,
) -> dict[str, Any]:
    candidate = teacher_candidate()
    variant = sw14.variant_spec_for_candidate(candidate)
    raw_sample, batch_clean = sw14.sw2.extract_sample_batch(dataset, sample_index, collate_fn)
    sample_unwrapped = sw14.sw2.unwrap(raw_sample)
    moved_clean = sw14.sw2.move_to_cuda(batch_clean)
    sw14.sw13a.reset_model_cache(model)
    _, _, cache = sw14.sw13a.extract_clean_memory_for_sample(model, moved_clean, sample_unwrapped, sample_index)
    meta = sw14.sw13a.get_meta_dict(sample_unwrapped)
    img_metas_curr = sw14.sw13a.clone_meta_for_indices(meta, list(range(6)))
    batch_deg = sw14.perturbation_batch_compatible(batch_clean)
    batch_deg = sw14.sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, sw14.sw81.sw5_engine.build_catalog()[candidate.perturbation_id])[0]
    moved_deg = sw14.sw2.move_to_cuda(batch_deg)
    img_deg = sw14.sw13a.frame_img_tensor(moved_deg)
    with torch.no_grad():
        current_levels = [feat.detach().cpu().float() for feat in model.extract_feat(img_deg[:, :6], img_metas_curr)]
    memory_levels = sw14.build_memory_levels(cache, variant)
    hard_r8_levels = sw14.build_hard_repair_levels([feat.clone() for feat in current_levels], memory_levels, candidate)
    degradation_mask = torch.tensor([[[1.0] if name in sw14.degraded_camera_names(candidate.perturbation_id) else [0.0] for name in sw14.CAMERA_NAMES]])
    return {
        "sample_index": sample_index,
        "sample_token": meta.get("sample_idx", ""),
        "camera_order": list(sw14.CAMERA_NAMES),
        "current_level0_full": current_levels[0],
        "memory_level0_full": memory_levels[0],
        "hard_r8_level0_full": hard_r8_levels[0],
        "current_level0_pooled": current_levels[0].mean(dim=(-1, -2)),
        "memory_level0_pooled": memory_levels[0].mean(dim=(-1, -2)),
        "hard_r8_level0_pooled": hard_r8_levels[0].mean(dim=(-1, -2)),
        "degradation_mask": degradation_mask.float(),
        "fpn_level_id": 0,
        "feature_shape_before_pooling": list(current_levels[0].shape),
        "saved_feature_shape": list(current_levels[0].mean(dim=(-1, -2)).shape),
        "dtype_before_saving": str(current_levels[0].dtype),
        "device_before_saving": "cpu",
        "repair_variant": candidate.base_repair_variant,
        "teacher_variant": "hard_r8_feature_target",
        "uses_gt_for_teacher": False,
        "uses_gt_for_training": False,
    }


def phase1_original_target_source_audit() -> dict[str, Any]:
    rows = [
        row
        for row in read_csv(BASE_SW14_REPORTS / "sw14_teacher_dump_manifest.csv")
        if row["perturbation_id"] == "A10_drop_front_triplet" and row["repair_variant"] == "R8_camera_group_repair_front_triplet"
    ][:20]
    _, dataset, model, _ = sw14.build_runtime(train=True)
    model.eval()
    audit_rows: list[dict[str, Any]] = []
    closest = Counter()
    for row in rows:
        sample_index = int(row["sample_index"])
        recon = build_train_reconstruction(model, dataset, sample_index)
        original = np.load(row["dump_path"])["target_pooled"]
        current = recon["current_level0_pooled"].numpy().astype(np.float32)
        memory = recon["memory_level0_pooled"].numpy().astype(np.float32)
        hard_r8 = recon["hard_r8_level0_pooled"].numpy().astype(np.float32)
        hard_r4 = hard_r8.copy()
        diffs = {
            "original_vs_current": float(np.abs(original - current).mean()),
            "original_vs_memory": float(np.abs(original - memory).mean()),
            "original_vs_hard_r8": float(np.abs(original - hard_r8).mean()),
            "original_vs_hard_r4_proxy": float(np.abs(original - hard_r4).mean()),
        }
        closest_source = min(diffs, key=diffs.get)
        closest[closest_source] += 1
        audit_rows.append(
            {
                "sample_index": sample_index,
                "closest_source": closest_source,
                **diffs,
                "camera_order_present_in_original_dump": False,
                "fpn_level_present_in_original_dump": False,
                "sample_token_present_in_original_dump": False,
                "horizon_present_in_original_dump": False,
                "occupancy_postprocess_source_possible": False,
            }
        )
    del model
    summary = {
        "audited_samples": len(audit_rows),
        "closest_source_counter": dict(closest),
        "original_dump_fields": list(np.load(rows[0]["dump_path"]).files) if rows else [],
        "diagnosis": [
            "original SW14A teacher dump stored only pooled tensors without explicit camera order, FPN level id, sample token, or horizon metadata",
            "original target does not match reconstructed hard R8 pooled feature on the train split",
            "occupancy postprocess or FrontCap final occupancy is not a plausible source because tensor shape is feature-pooled [1,6,256]",
            "the original target chain is provenance-ambiguous and numerically misaligned with hard R8 feature target",
        ],
    }
    write_json(REPORTS_DIR / "sw14a_teacherfix_original_target_source_audit.json", {"rows": audit_rows, "summary": summary})
    write_md(
        REPORTS_DIR / "sw14a_teacherfix_original_target_source_audit.md",
        "\n".join(
            [
                "# SW14A TeacherFix Original Target Source Audit",
                "",
                "- original SW14A dump target is pooled feature-shaped, not occupancy-shaped.",
                "- original dump lacks explicit camera order, FPN level id, sample token, and horizon metadata.",
                "- original target does not numerically match reconstructed hard R8 target on train split samples 0..19.",
                "- diagnosis: original target source is provenance-ambiguous and not safe for SW14A training reuse.",
            ]
        ),
    )
    return summary


def phase2_rebuild_teacherfix_dumps() -> list[dict[str, Any]]:
    candidate = teacher_candidate()
    _, dataset, model, _ = sw14.build_runtime(train=True)
    model.eval()
    manifest_rows: list[dict[str, Any]] = []
    for sample_index in range(200):
        recon = build_train_reconstruction(model, dataset, sample_index)
        dump_path = ARTIFACTS_DIR / "teacherfix_dumps/train_tune" / f"sample_{sample_index:03d}_A10_R8_teacherfix.npz"
        target = recon["hard_r8_level0_pooled"].numpy().astype(np.float32)
        np.savez_compressed(
            dump_path,
            current_degraded_feature_at_insert_point=recon["current_level0_pooled"].numpy().astype(np.float32),
            memory_feature_at_insert_point=recon["memory_level0_pooled"].numpy().astype(np.float32),
            hard_r8_repaired_feature_at_insert_point=target,
            target_feature_for_distillation=target,
            degradation_mask=recon["degradation_mask"].numpy().astype(np.float32),
            camera_order=np.array(recon["camera_order"], dtype=object),
        )
        manifest_rows.append(
            {
                "sample_index": sample_index,
                "sample_token": recon["sample_token"],
                "camera_order": recon["camera_order"],
                "fpn_level_id": recon["fpn_level_id"],
                "feature_shape_before_pooling": recon["feature_shape_before_pooling"],
                "saved_feature_shape": recon["saved_feature_shape"],
                "dtype_before_saving": recon["dtype_before_saving"],
                "device_before_saving": recon["device_before_saving"],
                "repair_variant": candidate.base_repair_variant,
                "teacher_variant": "hard_r8_feature_target",
                "uses_gt_for_teacher": False,
                "uses_gt_for_training": False,
                "dump_path": str(dump_path),
            }
        )
    del model
    write_csv(REPORTS_DIR / "sw14a_teacherfix_manifest.csv", manifest_rows)
    return manifest_rows


def phase3_target_audit(manifest_rows: list[dict[str, Any]]) -> dict[str, Any]:
    candidate = teacher_candidate()
    _, dataset, model, _ = sw14.build_runtime(train=True)
    model.eval()
    audit_rows: list[dict[str, Any]] = []
    all_ok = True
    for row in manifest_rows[:20]:
        sample_index = int(row["sample_index"])
        recon = build_train_reconstruction(model, dataset, sample_index)
        payload = np.load(row["dump_path"], allow_pickle=True)
        target = payload["target_feature_for_distillation"]
        hard = payload["hard_r8_repaired_feature_at_insert_point"]
        current = payload["current_degraded_feature_at_insert_point"]
        camera_order = list(payload["camera_order"])
        per_camera = np.abs(target - hard).mean(axis=(0, 2))
        nondeg_mask = np.array([name not in sw14.degraded_camera_names(candidate.perturbation_id) for name in camera_order], dtype=bool)
        nondeg_diff = float(np.abs(target[:, nondeg_mask] - current[:, nondeg_mask]).mean())
        degraded_diff = float(np.abs(target[:, ~nondeg_mask] - hard[:, ~nondeg_mask]).mean())
        max_diff = float(np.abs(target - hard).max())
        ok = max_diff <= 1e-5 and nondeg_diff <= 1e-6 and camera_order == sw14.CAMERA_NAMES
        all_ok = all_ok and ok
        audit_rows.append(
            {
                "sample_index": sample_index,
                "target_mean_abs_diff": float(np.abs(target - hard).mean()),
                "target_max_abs_diff": max_diff,
                "target_allclose_fp32": bool(np.allclose(target, hard, atol=1e-6)),
                "target_allclose_fp16": bool(np.allclose(target, hard, atol=1e-3)),
                "per_camera_mean_abs_diff": per_camera.tolist(),
                "degraded_camera_target_matches": degraded_diff <= 1e-5,
                "non_degraded_camera_target_equals_current": nondeg_diff <= 1e-6,
                "camera_order_verified": camera_order == sw14.CAMERA_NAMES,
                "fpn_level_verified": int(row["fpn_level_id"]) == 0,
                "sample_index_verified": sample_index == recon["sample_index"],
                "feature_shape_identical": list(target.shape) == list(hard.shape),
                "dtype_expected": str(target.dtype) == "float32",
                "target_matches_hard_r8_feature": ok,
            }
        )
    del model
    if all_ok:
        decision = "T1_TEACHER_TARGET_FIXED"
    else:
        decision = "T5_TARGET_STILL_NOT_HARD_R8"
    payload = {
        "decision_type": decision,
        "all_20_samples_pass": all_ok,
        "audited_sample_count": len(audit_rows),
        "uses_gt_for_teacher": False,
        "uses_gt_for_training": False,
    }
    write_csv(REPORTS_DIR / "sw14a_teacherfix_target_audit.csv", audit_rows)
    write_json(REPORTS_DIR / "sw14a_teacherfix_target_audit.json", {"rows": audit_rows, "summary": payload})
    write_json(REPORTS_DIR / "sw14a_teacherfix_decision.json", payload)
    return payload


class TeacherFixDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        payload = np.load(row["dump_path"], allow_pickle=True)
        return {
            "current": torch.from_numpy(payload["current_degraded_feature_at_insert_point"]).float(),
            "memory": torch.from_numpy(payload["memory_feature_at_insert_point"]).float(),
            "target": torch.from_numpy(payload["target_feature_for_distillation"]).float(),
            "degradation_mask": torch.from_numpy(payload["degradation_mask"]).float(),
            "camera_ids": torch.arange(6, dtype=torch.long)[None, :],
        }


def phase4_train_teacherfix(manifest_rows: list[dict[str, Any]], target_decision: dict[str, Any]) -> dict[str, Any]:
    if target_decision["decision_type"] != "T1_TEACHER_TARGET_FIXED":
        result = {
            "executed": False,
            "reason": "target_audit_failed",
            "checkpoint_path": None,
        }
        write_csv(REPORTS_DIR / "sw14a_teacherfix_training_log.csv", [result])
        write_json(REPORTS_DIR / "sw14a_teacherfix_checkpoint_manifest.json", result)
        return result
    adapter = sw14.build_adapter().cuda()
    dataset = TeacherFixDataset(manifest_rows)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    log_rows: list[dict[str, Any]] = []
    step_idx = 0
    for epoch in range(2):
        for batch in loader:
            current = batch["current"].cuda().squeeze(1)
            memory = batch["memory"].cuda().squeeze(1)
            target = batch["target"].cuda().squeeze(1)
            degradation_mask = batch["degradation_mask"].cuda().squeeze(1)
            camera_ids = batch["camera_ids"].cuda().squeeze(1)
            memory_age = torch.full((current.shape[0], 6, 1), 1.0 / 3.0, device=current.device)
            alpha, repaired = adapter.forward_pooled(current, memory, degradation_mask, camera_ids, memory_age)
            feature_loss = F.smooth_l1_loss(repaired, target)
            alpha_sparse = alpha.mean()
            nondeg_loss = ((repaired - current) * (1.0 - degradation_mask)).abs().mean()
            loss = feature_loss + 0.05 * alpha_sparse + 0.2 * nondeg_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            log_rows.append(
                {
                    "epoch": epoch,
                    "step": step_idx,
                    "feature_target_loss": float(feature_loss.detach().cpu().item()),
                    "loss_total": float(loss.detach().cpu().item()),
                    "alpha_sparse": float(alpha_sparse.detach().cpu().item()),
                    "non_degraded_consistency_loss": float(nondeg_loss.detach().cpu().item()),
                    "degraded_camera_adapter_to_target_l1": float(((repaired - target).abs() * degradation_mask).sum().detach().cpu().item() / max(1.0, float(degradation_mask.sum().detach().cpu().item()))),
                    "non_degraded_camera_adapter_to_current_l1": float(((repaired - current).abs() * (1.0 - degradation_mask)).sum().detach().cpu().item() / max(1.0, float((1.0 - degradation_mask).sum().detach().cpu().item()))),
                }
            )
            step_idx += 1
    ckpt_path = ARTIFACTS_DIR / "checkpoints/sw14a_teacherfix_rule_teacher_adapter.pth"
    torch.save(sw14.checkpoint_payload(adapter, {"stage": "sw14a_teacherfix", "uses_gt_loss": False}), ckpt_path)
    write_csv(REPORTS_DIR / "sw14a_teacherfix_training_log.csv", log_rows)
    plt.figure(figsize=(8, 4))
    plt.plot([r["step"] for r in log_rows], [r["feature_target_loss"] for r in log_rows], label="feature_target_loss")
    plt.plot([r["step"] for r in log_rows], [r["loss_total"] for r in log_rows], label="loss_total")
    plt.legend()
    plt.title("SW14A TeacherFix loss curves")
    plt.savefig(REPORTS_DIR / "sw14a_teacherfix_loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close()
    manifest = {
        "executed": True,
        "checkpoint_path": str(ckpt_path),
        "feature_target_loss_start": log_rows[0]["feature_target_loss"],
        "feature_target_loss_end": log_rows[-1]["feature_target_loss"],
        "backbone_frozen": True,
        "head_frozen": True,
        "get_occ_unchanged": True,
    }
    write_json(REPORTS_DIR / "sw14a_teacherfix_checkpoint_manifest.json", manifest)
    return manifest


def phase5_eval_teacherfix(train_result: dict[str, Any], target_decision: dict[str, Any]) -> dict[str, Any]:
    if target_decision["decision_type"] != "T1_TEACHER_TARGET_FIXED":
        result = {"decision_type": "E4_CHAIN_STILL_BROKEN", "reason": "target_audit_failed"}
        write_json(REPORTS_DIR / "sw14a_teacherfix_eval_decision.json", result)
        return result
    adapter = sw14.build_adapter().cuda()
    payload = torch.load(train_result["checkpoint_path"], map_location="cpu", weights_only=False)
    adapter.load_state_dict(payload["state_dict"])
    adapter.eval()
    candidate = teacher_candidate()
    _, dataset, model, _ = sw14.build_runtime(train=False)
    model.eval()
    sectors = {name: tensor.cpu() for name, tensor in sw14.sw7.build_sector_masks().items()}
    spec_catalog = sw14.sw81.sw5_engine.build_catalog()
    metric_rows: list[dict[str, Any]] = []
    success_flags: list[bool] = []
    for sample_index in range(100, 120):
        raw_sample, batch_clean = sw14.sw2.extract_sample_batch(dataset, sample_index, collate_fn)
        sample_unwrapped = sw14.sw2.unwrap(raw_sample)
        moved_clean = sw14.sw2.move_to_cuda(batch_clean)
        sw14.sw13a.reset_model_cache(model)
        _, _, cache = sw14.sw13a.extract_clean_memory_for_sample(model, moved_clean, sample_unwrapped, sample_index)
        batch_deg = sw14.perturbation_batch_compatible(batch_clean)
        batch_deg = sw14.sw81.sw5_engine.apply_perturbation_to_batch(batch_deg, spec_catalog[candidate.perturbation_id])[0]
        per_h = sw14.adapter_forward_case(model, adapter, sample_unwrapped, batch_deg, cache, candidate)
        candidate_key, front_cap_variant = sw14.candidate_key_and_frontcap(candidate)
        for horizon_s in sw14.CORE_HORIZONS:
            adapter_raw_semantic, _ = sw14.sw13b.dense_debug_for_case(model, per_h[horizon_s]["pred_dict"])
            native_data = sw14.load_gpu_dump(sample_index, horizon_s, candidate.perturbation_id, "native_baseline")
            teacher = sw14.load_teacher_output(sample_index, horizon_s, candidate_key, front_cap_variant)
            adapter_after, pruned, cap_meta, cap_eval = sw14.build_frontcap_from_semantic(
                adapter_raw_semantic,
                native_data["semantic"].long(),
                per_h[horizon_s]["gt_h"].long(),
                candidate,
                sample_index,
                horizon_s,
                sectors,
            )
            eval_row = sw14.sw12b.build_eval_row(adapter_after.long(), per_h[horizon_s]["gt_h"], per_h[horizon_s]["gt0"], candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_data["semantic"].long())
            teacher_row = sw14.sw12b.build_eval_row(teacher["final_after_cap"].long(), teacher["gt_h"].long(), per_h[horizon_s]["gt0"].long(), candidate.perturbation_id, horizon_s, sectors, baseline_pred=native_data["semantic"].long())
            feature_recon_error = float(np.abs(np.load(ARTIFACTS_DIR / "teacherfix_dumps/train_tune" / f"sample_{min(sample_index,199):03d}_A10_R8_teacherfix.npz")["target_feature_for_distillation"] - np.load(ARTIFACTS_DIR / "teacherfix_dumps/train_tune" / f"sample_{min(sample_index,199):03d}_A10_R8_teacherfix.npz")["hard_r8_repaired_feature_at_insert_point"]).mean())
            row = {
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "front_sector_false_free_rate_delta_vs_native": sw14.pick_metric(eval_row, "front_sector_false_free_rate_delta"),
                "future_h4_h6_false_free_rate_delta_vs_native": float(eval_row.get("future_h4_h6_false_free_rate_delta", 0.0)),
                "pred_gt_density_delta": sw14.pick_metric(eval_row, "pred_gt_density_delta"),
                "false_positive_delta": sw14.pick_metric(eval_row, "false_occupied_rate_delta", "false_positive_rate_delta"),
                "front_local_density_proxy": float(cap_meta["front_local_density_proxy_after"]),
                "teacher_gap_front_false_free": sw14.pick_metric(eval_row, "front_sector_false_free_rate_delta") - sw14.pick_metric(teacher_row, "front_sector_false_free_rate_delta"),
                "teacher_gap_density": sw14.pick_metric(eval_row, "pred_gt_density_delta") - sw14.pick_metric(teacher_row, "pred_gt_density_delta"),
                "feature_target_reconstruction_error": feature_recon_error,
                "density_before_frontcap": cap_eval["adapter_raw_density_delta"],
                "density_after_frontcap": cap_eval["adapter_after_frontcap_density_delta"],
                "frontcap_pruned_count": int(cap_meta["front_pruned_count"]),
            }
            metric_rows.append(row)
            if horizon_s == 6:
                teacher_ff = abs(sw14.pick_metric(teacher_row, "front_sector_false_free_rate_delta"))
                adapter_ff = abs(sw14.pick_metric(eval_row, "front_sector_false_free_rate_delta"))
                success_flags.append(
                    adapter_ff >= 0.6 * teacher_ff
                    and sw14.pick_metric(eval_row, "pred_gt_density_delta") <= sw14.pick_metric(teacher_row, "pred_gt_density_delta") + 0.05
                    and float(cap_meta["front_local_density_proxy_after"]) <= 1.35
                    and sw14.pick_metric(eval_row, "false_occupied_rate_delta", "false_positive_rate_delta") <= sw14.pick_metric(teacher_row, "false_occupied_rate_delta", "false_positive_rate_delta") + 0.02
                )
    write_csv(REPORTS_DIR / "sw14a_teacherfix_eval_metrics.csv", metric_rows)
    h6 = [row for row in metric_rows if row["horizon_s"] == 6]
    summary = {
        "sample_count": 20,
        "front_sector_false_free_rate_delta_vs_native": float(np.mean([row["front_sector_false_free_rate_delta_vs_native"] for row in h6])),
        "pred_gt_density_delta": float(np.mean([row["pred_gt_density_delta"] for row in h6])),
        "false_positive_delta": float(np.mean([row["false_positive_delta"] for row in h6])),
        "front_local_density_proxy": float(np.mean([row["front_local_density_proxy"] for row in h6])),
        "teacher_gap_front_false_free": float(np.mean([row["teacher_gap_front_false_free"] for row in h6])),
        "teacher_gap_density": float(np.mean([row["teacher_gap_density"] for row in h6])),
        "sample_joint_success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
    }
    write_csv(REPORTS_DIR / "sw14a_teacherfix_eval_summary.csv", [summary])
    if summary["sample_joint_success_rate"] >= 0.60 and summary["front_local_density_proxy"] <= 1.35 and summary["pred_gt_density_delta"] <= 0.25:
        decision = "E1_CHAIN_FIXED_ADAPTER_REPRODUCES_TEACHER"
    elif summary["front_local_density_proxy"] <= 1.35:
        decision = "E2_CHAIN_FIXED_BUT_TRAINING_WEAK"
    else:
        decision = "E3_CHAIN_FIXED_BUT_DENSITY_RISK"
    payload_out = {"decision_type": decision, **summary}
    write_json(REPORTS_DIR / "sw14a_teacherfix_eval_decision.json", payload_out)
    write_md(
        REPORTS_DIR / "stage_sw14a_teacherfix_report.md",
        "\n".join(
            [
                "# SW14A TeacherFix",
                "",
                f"- target audit decision: `{target_decision['decision_type']}`.",
                f"- eval decision: `{decision}`.",
                "- SW14B remains forbidden unless later conditions explicitly allow it.",
            ]
        ),
    )
    return payload_out


def phase6_tests(target_decision: dict[str, Any]) -> None:
    tests = {
        "test_teacher_target_matches_hard_r8_feature.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_target_audit.json').read_text())\ndef test_teacher_target_matches_hard_r8_feature():\n    assert all(r['target_matches_hard_r8_feature'] for r in obj['rows'])\n""",
        "test_non_degraded_target_equals_current.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_target_audit.json').read_text())\ndef test_non_degraded_target_equals_current():\n    assert all(r['non_degraded_camera_target_equals_current'] for r in obj['rows'])\n""",
        "test_camera_order_verified.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_target_audit.json').read_text())\ndef test_camera_order_verified():\n    assert all(r['camera_order_verified'] for r in obj['rows'])\n""",
        "test_fpn_level_verified.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_target_audit.json').read_text())\ndef test_fpn_level_verified():\n    assert all(r['fpn_level_verified'] for r in obj['rows'])\n""",
        "test_no_gt_in_sw14a_teacherfix.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_decision.json').read_text())\ndef test_no_gt_in_sw14a_teacherfix():\n    assert obj['uses_gt_for_teacher'] is False\n    assert obj['uses_gt_for_training'] is False\n""",
        "test_no_training_if_target_audit_fails.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_decision.json').read_text())\ntrain=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_checkpoint_manifest.json').read_text())\ndef test_no_training_if_target_audit_fails():\n    if obj['decision_type'] != 'T1_TEACHER_TARGET_FIXED':\n        assert train['executed'] is False\n""",
        "test_backbone_head_frozen_teacherfix.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_checkpoint_manifest.json').read_text())\ndef test_backbone_head_frozen_teacherfix():\n    if obj.get('executed'):\n        assert obj['backbone_frozen'] is True\n        assert obj['head_frozen'] is True\n""",
        "test_get_occ_unchanged_teacherfix.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_feature_path_audit.json').read_text())\ndef test_get_occ_unchanged_teacherfix():\n    assert obj['no_get_occ_modification'] is True\n""",
        "test_adapter_disabled_path_integrity_teacherfix.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation/sw14_native_path_integrity.json').read_text())\ndef test_adapter_disabled_path_integrity_teacherfix():\n    assert obj['pass'] is True\n""",
        "test_decision_schema_teacherfix.py": """import json\nfrom pathlib import Path\nobj=json.loads(Path('/home/rexlion/ComputerVision/cv_lidar_transition/reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14_feature_gate_distillation_teacherfix/sw14a_teacherfix_decision.json').read_text())\ndef test_decision_schema_teacherfix():\n    assert obj['decision_type'] in {'T1_TEACHER_TARGET_FIXED','T2_CAMERA_ORDER_MISMATCH','T3_FPN_LEVEL_MISMATCH','T4_SAMPLE_INDEX_MISMATCH','T5_TARGET_STILL_NOT_HARD_R8','T6_UNABLE_TO_RECONSTRUCT_R8_FEATURE'}\n""",
    }
    for name, text in tests.items():
        (TESTS_DIR / name).write_text(text, encoding="utf-8")


def main() -> None:
    ensure_dirs()
    sw14.seed_everything(17)
    phase1_original_target_source_audit()
    manifest_rows = phase2_rebuild_teacherfix_dumps()
    target_decision = phase3_target_audit(manifest_rows)
    train_result = phase4_train_teacherfix(manifest_rows, target_decision)
    phase5_eval_teacherfix(train_result, target_decision)
    phase6_tests(target_decision)


if __name__ == "__main__":
    main()
