from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import queue
import random
import re
import sys
import time
import traceback
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", "/mnt/d/ComputerVision/cv_lidar_transition"))
REPO_ROOT = PROJECT_ROOT / "external/SparseWorld"
CONFIG_PATH = REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"
CHECKPOINT_PATH = REPO_ROOT / "ckpts/epoch_56.pth"

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw8_targeted_finetune"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw8_targeted_finetune"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw8_targeted_finetune"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw8_targeted_finetune"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw8_targeted_finetune"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw8_targeted_finetune"

SW7_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map"
SW6_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw6_frontview_blur_fragility_diagnosis"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sw2 = load_module(
    "sw8_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw35 = load_module(
    "sw8_sw35",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw35_corrected_support_coverage/corrected_support_adapter.py",
)
sw4_inst = load_module(
    "sw8_sw4_inst",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
sw5_engine = load_module(
    "sw8_sw5_engine",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw5_sensor_aware_failure_propagation/sensor_perturbation_engine.py",
)
sw7 = load_module(
    "sw8_sw7",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/run_sparseworld_sw7_main.py",
)


CORE_PERTURBATIONS = ["A0_clean", "A1_drop_cam_front", "A10_drop_front_triplet", "C4_motion_blur_9", "A7_drop_all_rear"]
DEFAULT_SAMPLE_INDICES = list(range(10))
DEFAULT_HORIZONS = list(range(7))
SMALL_OBJECT_CLASS_IDS = [1, 2, 5, 6, 7, 8]
WINDOWS_PROJECT_ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-root", default=str(REPO_ROOT))
    p.add_argument("--config", default=str(CONFIG_PATH))
    p.add_argument("--checkpoint", default=str(CHECKPOINT_PATH))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--smoke-iters", default="50,100")
    p.add_argument("--base-train-iters", type=int, default=100)
    p.add_argument("--combined-train-iters", type=int, default=120)
    p.add_argument("--samples-per-gpu", type=int, default=1)
    p.add_argument("--workers-per-gpu", type=int, default=0)
    p.add_argument("--num-eval-samples", type=int, default=10)
    p.add_argument("--eval-sample-indices", default="")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--save-debug", action="store_true", default=True)
    p.add_argument("--save-figures", action="store_true", default=True)
    p.add_argument("--run-combined", action="store_true", default=True)
    p.add_argument("--fast", action="store_true", default=False)
    return p.parse_args()


def ensure_dirs() -> None:
    for p in [REPORTS_DIR, LOGS_DIR, SCRIPTS_DIR, ARTIFACTS_DIR, FIGURES_DIR, TESTS_DIR, ARTIFACTS_DIR / "checkpoints", ARTIFACTS_DIR / "reliability_retest"]:
        p.mkdir(parents=True, exist_ok=True)


def to_windows_path(path: str | Path) -> str:
    text = str(path)
    if text.startswith("/mnt/") and len(text) > 6 and text[6] == "/":
        drive = text[5].upper()
        rest = text[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return text


def normalize_export(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_export(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_export(v) for v in obj]
    if isinstance(obj, tuple):
        return [normalize_export(v) for v in obj]
    if isinstance(obj, Path):
        return to_windows_path(obj)
    if isinstance(obj, str):
        return to_windows_path(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_export(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_export(row))


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class StageLogger:
    def __init__(self) -> None:
        self.status_path = LOGS_DIR / "sw8_status.json"
        self.progress_path = LOGS_DIR / "sw8_progress.jsonl"
        self.state: dict[str, Any] = {"started_at": time.time(), "current_stage": None, "stages": {}}
        self.progress_path.write_text("", encoding="utf-8")
        self.flush()

    def flush(self) -> None:
        self.status_path.write_text(json.dumps(normalize_export(self.state), indent=2, ensure_ascii=False), encoding="utf-8")

    def event(self, stage: str, event: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(normalize_export({"ts": time.time(), "stage": stage, "event": event, **payload}), ensure_ascii=False) + "\n")
        self.flush()

    def start(self, stage: str, **payload: Any) -> None:
        self.state["current_stage"] = stage
        self.state["stages"][stage] = {"status": "running", "start_ts": time.time(), **payload}
        self.event(stage, "start", payload)

    def progress(self, stage: str, current: int, total: int, **payload: Any) -> None:
        self.state["stages"].setdefault(stage, {})
        self.state["stages"][stage].update({"status": "running", "progress_current": current, "progress_total": total, **payload})
        self.event(stage, "progress", {"current": current, "total": total, **payload})

    def done(self, stage: str, **payload: Any) -> None:
        now = time.time()
        item = self.state["stages"].setdefault(stage, {})
        start_ts = float(item.get("start_ts", now))
        item.update({"status": "done", "end_ts": now, "duration_sec": now - start_ts, **payload})
        if self.state.get("current_stage") == stage:
            self.state["current_stage"] = None
        self.event(stage, "done", payload)

    def fail(self, stage: str, error: str) -> None:
        now = time.time()
        item = self.state["stages"].setdefault(stage, {})
        start_ts = float(item.get("start_ts", now))
        item.update({"status": "failed", "end_ts": now, "duration_sec": now - start_ts, "error": error})
        if self.state.get("current_stage") == stage:
            self.state["current_stage"] = None
        self.event(stage, "failed", {"error": error})


def log(msg: str) -> None:
    print(f"[SW8] {msg}", flush=True)
    with (LOGS_DIR / "phase1_training_entry_audit.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) != 0.0 else 0.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_sparseworld_runtime(train: bool, cfg_overrides: dict[str, Any] | None = None) -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(REPO_ROOT))
    os.chdir(REPO_ROOT)
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config

    cfg = Config.fromfile(str(CONFIG_PATH))
    cfg = compat_cfg(cfg)
    cfg = patch_config(cfg)
    setup_multi_processes(cfg)
    cfg.gpu_ids = [0]
    cfg.model.pretrained = None
    if cfg_overrides:
        for key, value in cfg_overrides.items():
            cfg.merge_from_dict({key: value})
    if train:
        split_cfg = cfg.data.train
        split_cfg.test_mode = False
    else:
        split_cfg = cfg.data.val
        split_cfg.test_mode = True
    dataset = build_dataset(split_cfg)
    model = build_model(cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    checkpoint = load_checkpoint(model, str(CHECKPOINT_PATH), map_location="cpu")
    model = model.cuda()
    if hasattr(model, "set_epoch"):
        meta_epoch = 56
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("meta"), dict):
            meta_epoch = int(checkpoint["meta"].get("epoch", meta_epoch))
        model.set_epoch(meta_epoch)
    if train:
        model.train()
    else:
        model.eval()
    return cfg, dataset, model, checkpoint


def build_dataloader_for_cfg(dataset: Any, samples_per_gpu: int, workers_per_gpu: int, shuffle: bool, seed: int):
    try:
        from mmdet3d.datasets import build_dataloader
        return build_dataloader(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=workers_per_gpu, num_gpus=1, dist=False, shuffle=shuffle, seed=seed)
    except Exception:
        from mmdet.datasets import build_dataloader
        return build_dataloader(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=workers_per_gpu, num_gpus=1, dist=False, shuffle=shuffle, seed=seed)


def reset_online_cache(model: Any) -> None:
    if hasattr(model, "memory") and isinstance(model.memory, dict):
        model.memory.clear()
    if hasattr(model, "queue"):
        model.queue = queue.Queue()


def move_train_batch_to_cuda(obj: Any) -> Any:
    try:
        from mmcv.parallel import DataContainer
    except Exception:
        DataContainer = None
    if DataContainer is not None and isinstance(obj, DataContainer):
        return move_train_batch_to_cuda(obj.data)
    if isinstance(obj, dict):
        return {k: move_train_batch_to_cuda(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) == 1:
            only = obj[0]
            if isinstance(only, list) and (len(only) == 0 or isinstance(only[0], dict)):
                return only
            return move_train_batch_to_cuda(only)
        return [move_train_batch_to_cuda(v) for v in obj]
    if isinstance(obj, tuple):
        if len(obj) == 1:
            only = obj[0]
            if isinstance(only, list) and (len(only) == 0 or isinstance(only[0], dict)):
                return only
            return move_train_batch_to_cuda(only)
        return [move_train_batch_to_cuda(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.cuda(non_blocking=False)
    return obj


def parse_losses(losses: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    log_vars: dict[str, float] = {}
    total = None
    for name, value in losses.items():
        if isinstance(value, torch.Tensor):
            loss = value.mean()
        elif isinstance(value, list):
            loss = sum(v.mean() for v in value)
        else:
            continue
        log_vars[name] = float(loss.detach().cpu().item())
        if "loss" in name:
            total = loss if total is None else total + loss
    if total is None:
        total = torch.zeros((), device="cuda", dtype=torch.float32)
    log_vars["loss_total"] = float(total.detach().cpu().item())
    return total, log_vars


def current_base_cls_weights() -> list[float]:
    cfg_text = json.loads((SW7_REPORTS / "sw8_recommended_training_plan.json").read_text(encoding="utf-8")) if (SW7_REPORTS / "sw8_recommended_training_plan.json").exists() else {}
    _ = cfg_text
    return [3, 2, 3, 2, 2, 3, 3, 2, 3, 2, 2, 1, 2, 1, 1, 1, 1]


def build_experiment_defs(base_iters: int, combined_iters: int) -> list[dict[str, Any]]:
    return [
        {"id": "E0_baseline_resume_control", "train_iters": base_iters, "small_object_factor": 1.0, "future_scales": {}, "train_aug": []},
        {"id": "E1_small_object_reweight_light", "train_iters": base_iters, "small_object_factor": 1.5, "future_scales": {}, "train_aug": []},
        {"id": "E4_future_horizon_reweight", "train_iters": base_iters, "small_object_factor": 1.0, "future_scales": {4: 1.15, 5: 1.30, 6: 1.45}, "train_aug": []},
        {"id": "E7_front_dropout_aug_light", "train_iters": base_iters, "small_object_factor": 1.0, "future_scales": {}, "train_aug": [{"prob": 0.15, "spec_choices": ["A1_drop_cam_front", "A10_drop_front_triplet"]}]},
        {"id": "E8_motion_blur_aug_light", "train_iters": base_iters, "small_object_factor": 1.0, "future_scales": {}, "train_aug": [{"prob": 0.15, "spec_choices": ["C4_motion_blur_9"]}]},
        {"id": "E10_small_object_plus_future", "train_iters": combined_iters, "small_object_factor": 1.5, "future_scales": {4: 1.15, 5: 1.30, 6: 1.45}, "train_aug": []},
    ]


def apply_experiment_config_overrides(exp_cfg: dict[str, Any], samples_per_gpu: int, workers_per_gpu: int) -> dict[str, Any]:
    cls_weights = current_base_cls_weights()
    factor = float(exp_cfg.get("small_object_factor", 1.0))
    if abs(factor - 1.0) > 1e-8:
        for cid in SMALL_OBJECT_CLASS_IDS:
            if 0 <= cid < len(cls_weights):
                cls_weights[cid] *= factor
    return {
        "data.samples_per_gpu": samples_per_gpu,
        "data.workers_per_gpu": workers_per_gpu,
        "model.train_cfg.pts.cls_weights": cls_weights,
    }


def maybe_apply_train_augmentation(batch: dict[str, Any], exp_cfg: dict[str, Any], step_seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    aug_defs = exp_cfg.get("train_aug", [])
    if not aug_defs:
        return batch, {"applied": False}
    wrapped_batch = deepcopy(batch)
    if "img" in wrapped_batch and not isinstance(wrapped_batch["img"], list):
        wrapped_batch["img"] = [wrapped_batch["img"]]
    if "img_metas" in wrapped_batch and not isinstance(wrapped_batch["img_metas"], list):
        wrapped_batch["img_metas"] = [wrapped_batch["img_metas"]]
    out_batch = wrapped_batch
    manifests: list[dict[str, Any]] = []
    catalog = sw5_engine.build_catalog()
    rng = random.Random(step_seed)
    applied = False
    for aug_idx, aug in enumerate(aug_defs):
        if rng.random() >= float(aug["prob"]):
            continue
        spec_name = rng.choice(list(aug["spec_choices"]))
        spec = catalog[spec_name]
        out_batch, manifest = sw5_engine.apply_perturbation_to_batch(out_batch, spec)
        manifests.append({"aug_index": aug_idx, **manifest})
        applied = True
    if isinstance(out_batch.get("img"), list):
        out_batch["img"] = out_batch["img"][0]
    if isinstance(out_batch.get("img_metas"), list):
        out_batch["img_metas"] = out_batch["img_metas"][0]
    return out_batch, {"applied": applied, "manifests": manifests}


def apply_future_loss_reweight(losses: dict[str, Any], exp_cfg: dict[str, Any]) -> dict[str, Any]:
    future_scales = exp_cfg.get("future_scales", {})
    if not future_scales:
        return losses
    out = dict(losses)
    for key, value in list(out.items()):
        m = re.match(r"fu(\d+)\.", key)
        if not m:
            continue
        horizon = int(m.group(1))
        scale = float(future_scales.get(horizon, 1.0))
        if abs(scale - 1.0) > 1e-8:
            out[key] = value * scale
    return out


def save_checkpoint(path: Path, model: Any, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = model.state_dict()
    torch.save({"meta": meta, "state_dict": state_dict}, path)


def small_object_false_free(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor) -> float:
    group_mask = torch.zeros_like(gt, dtype=torch.bool)
    for cid in SMALL_OBJECT_CLASS_IDS:
        group_mask |= gt == int(cid)
    group_mask &= valid_mask
    pred_occ = (pred != sw2.EMPTY_IDX) & valid_mask
    ff = group_mask & (~pred_occ)
    return safe_div(ff.sum().item(), group_mask.sum().item())


def front_sector_false_free(pred: torch.Tensor, gt: torch.Tensor, valid_mask: torch.Tensor, front_mask: torch.Tensor) -> float:
    gt_occ = (gt != sw2.EMPTY_IDX) & valid_mask & front_mask
    pred_occ = (pred != sw2.EMPTY_IDX) & valid_mask & front_mask
    ff = gt_occ & (~pred_occ)
    return safe_div(ff.sum().item(), gt_occ.sum().item())


def compute_eval_row(checkpoint_name: str, perturbation_id: str, sample_index: int, horizon_s: int, pred_h: torch.Tensor, gt_h: torch.Tensor, gt0: torch.Tensor, sector_masks: dict[str, torch.Tensor]) -> dict[str, Any]:
    valid_mask = sw2.valid_mask_from_gt(gt_h)
    base = sw2.compute_base_metrics(pred_h, gt_h, valid_mask)
    region = sw2.region_proxy_metrics(pred_h, gt0, gt_h)
    front_ff = front_sector_false_free(pred_h, gt_h, valid_mask, sector_masks["front"])
    pred_occ = (pred_h != sw2.EMPTY_IDX) & valid_mask & sector_masks["front"]
    gt_occ = (gt_h != sw2.EMPTY_IDX) & valid_mask & sector_masks["front"]
    inter = (pred_occ & gt_occ).sum().item()
    union = (pred_occ | gt_occ).sum().item()
    row = {
        "checkpoint_name": checkpoint_name,
        "perturbation_id": perturbation_id,
        "sample_index": sample_index,
        "horizon_s": horizon_s,
        **base,
        **region,
        "small_object_false_free": small_object_false_free(pred_h, gt_h, valid_mask),
        "new_visible_recall": float(region["new_visible_proxy_recall"]),
        "future_false_free": float(region["new_visible_false_free"]),
        "front_sector_false_free": front_ff,
        "front_sector_occupied_iou": safe_div(inter, union),
    }
    return row


def attach_query_capture(model: Any, holder: dict[str, Any]) -> Any:
    original_forward_backbone = model.forward_backbone

    def wrapped_forward_backbone(*a: Any, **kw: Any) -> Any:
        outputs = original_forward_backbone(*a, **kw)
        holder["forward_backbone_outputs"] = sw2.to_cpu_artifact(outputs)
        return outputs

    model.forward_backbone = wrapped_forward_backbone  # type: ignore[assignment]
    return original_forward_backbone


def smoke_and_train(exp_cfg: dict[str, Any], samples_per_gpu: int, workers_per_gpu: int, seed: int, stage_logger: StageLogger, checkpoint_override: Path | None = None) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    stage = f"train_{exp_cfg['id']}"
    stage_logger.start(stage, train_iters=exp_cfg["train_iters"])
    try:
        cfg_overrides = apply_experiment_config_overrides(exp_cfg, samples_per_gpu, workers_per_gpu)
        cfg, dataset, model, checkpoint = build_sparseworld_runtime(train=True, cfg_overrides=cfg_overrides)
        if checkpoint_override is not None:
            state = torch.load(checkpoint_override, map_location="cpu", weights_only=False)
            model.load_state_dict(state["state_dict"], strict=False)
        dataloader = build_dataloader_for_cfg(dataset, samples_per_gpu=samples_per_gpu, workers_per_gpu=workers_per_gpu, shuffle=True, seed=seed)
        from mmcv.runner import build_optimizer

        optimizer = build_optimizer(model, cfg.optimizer)
        iter_rows: list[dict[str, Any]] = []
        data_iter = iter(dataloader)
        start_epoch = 56
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("meta"), dict):
            start_epoch = int(checkpoint["meta"].get("epoch", start_epoch))
        if hasattr(model, "set_epoch"):
            model.set_epoch(start_epoch)
        for iter_idx in range(1, int(exp_cfg["train_iters"]) + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)
            aug_batch, aug_manifest = maybe_apply_train_augmentation(batch, exp_cfg, seed * 1000 + iter_idx)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            model_inputs = move_train_batch_to_cuda(aug_batch)
            losses = model(return_loss=True, **model_inputs)
            losses = apply_future_loss_reweight(losses, exp_cfg)
            total_loss, log_vars = parse_losses(losses)
            finite = bool(torch.isfinite(total_loss).item())
            if not finite:
                raise RuntimeError(f"non-finite loss at iter {iter_idx}: {log_vars}")
            total_loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0).detach().cpu().item())
            optimizer.step()
            torch.cuda.synchronize()
            iter_sec = time.perf_counter() - started
            peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            row = {
                "experiment_id": exp_cfg["id"],
                "iter": iter_idx,
                "iter_time_sec": iter_sec,
                "peak_memory_mb": peak_mem_mb,
                "grad_norm": grad_norm,
                "aug_applied": bool(aug_manifest.get("applied", False)),
                **log_vars,
            }
            iter_rows.append(row)
            if (iter_idx % 10 == 0) or (iter_idx == int(exp_cfg["train_iters"])):
                stage_logger.progress(stage, iter_idx, int(exp_cfg["train_iters"]), last_loss=log_vars["loss_total"])
        ckpt_path = ARTIFACTS_DIR / "checkpoints" / f"{exp_cfg['id']}_iter{int(exp_cfg['train_iters']):04d}.pth"
        meta = {
            "experiment_id": exp_cfg["id"],
            "epoch": start_epoch,
            "iter": int(exp_cfg["train_iters"]),
            "train_iters": int(exp_cfg["train_iters"]),
            "effective_samples_per_gpu": samples_per_gpu,
            "effective_workers_per_gpu": workers_per_gpu,
            "small_object_factor": exp_cfg.get("small_object_factor", 1.0),
            "future_scales": exp_cfg.get("future_scales", {}),
            "train_aug": exp_cfg.get("train_aug", []),
        }
        save_checkpoint(ckpt_path, model, meta)
        tail = iter_rows[-min(10, len(iter_rows)) :]
        head = iter_rows[: min(10, len(iter_rows))]
        summary = {
            "checkpoint_path": str(ckpt_path),
            "iter_count": len(iter_rows),
            "peak_gpu_memory_mb": float(max(r["peak_memory_mb"] for r in iter_rows)),
            "mean_iter_time_sec": float(np.mean([r["iter_time_sec"] for r in iter_rows])),
            "loss_finite": all(math.isfinite(float(r["loss_total"])) for r in iter_rows),
            "loss_head_mean": float(np.mean([r["loss_total"] for r in head])) if head else None,
            "loss_tail_mean": float(np.mean([r["loss_total"] for r in tail])) if tail else None,
            "loss_trend_delta": float(np.mean([r["loss_total"] for r in tail]) - np.mean([r["loss_total"] for r in head])) if head and tail else None,
            "save_ok": ckpt_path.exists(),
        }
        stage_logger.done(stage, checkpoint_path=str(ckpt_path))
        return ckpt_path, iter_rows, summary
    except Exception as exc:
        stage_logger.fail(stage, repr(exc))
        raise


def evaluate_checkpoint(
    checkpoint_path: Path,
    checkpoint_name: str,
    sample_indices: list[int],
    horizons: list[int],
    capture_reliability: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cfg, dataset, model, _ = build_sparseworld_runtime(train=False, cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0})
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()
    from mmcv.parallel import collate as collate_fn
    head = sw4_inst.get_pts_bbox_head(model)
    catalog = sw5_engine.build_catalog()
    support_adapter = sw35.CorrectedSparseWorldSupportAdapter(PROJECT_ROOT)
    sector_masks = sw7.build_sector_masks()
    rel_cfg = sw7.build_reliability_config()
    rows: list[dict[str, Any]] = []
    rel_rows: list[dict[str, Any]] = []
    query_holder: dict[str, Any] = {}
    original_forward_backbone = attach_query_capture(model, query_holder)
    debug_examples: dict[str, Any] = {}
    try:
        for perturbation_id in CORE_PERTURBATIONS:
            spec = catalog[perturbation_id]
            for sample_index in sample_indices:
                raw_sample, batch = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
                sample_unwrapped = sw2.unwrap(raw_sample)
                if perturbation_id != "A0_clean":
                    batch, _ = sw5_engine.apply_perturbation_to_batch(batch, spec)
                query_holder.clear()
                reset_online_cache(model)
                model_inputs = sw2.move_to_cuda(batch)
                with torch.no_grad():
                    result = model(return_loss=False, rescale=True, **model_inputs)
                raw_result_cpu = sw2.to_cpu_artifact(result)
                query_cpu = sw2.to_cpu_artifact(query_holder)
                pred_temporal, gt_temporal, _ = sw2.extract_standard_tensors(sample_unwrapped, raw_result_cpu)
                gt0 = gt_temporal[0]
                supports = support_adapter.build_temporal_support(query_cpu, horizons)[0] if capture_reliability else {}
                for horizon_s in horizons:
                    pred_h = pred_temporal[horizon_s].long().cpu()
                    gt_h = gt_temporal[horizon_s].long().cpu()
                    row = compute_eval_row(checkpoint_name, perturbation_id, sample_index, horizon_s, pred_h, gt_h, gt0, sector_masks)
                    rows.append(row)
                    if capture_reliability:
                        pred_dict = sw4_inst.extract_pred_dict_for_horizon(query_cpu, horizon_s)
                        _, dbg_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                        debug = dbg_list[0]
                        maps = sw7.compute_reliability_maps(
                            perturbation_id=perturbation_id,
                            horizon_s=horizon_s,
                            pred=pred_h,
                            gt=gt_h,
                            gt0=gt0,
                            support=supports[horizon_s],
                            debug=debug,
                            config=rel_cfg,
                            sectors=sector_masks,
                        )
                        rel_row = {
                            "checkpoint_name": checkpoint_name,
                            "perturbation_id": perturbation_id,
                            "sample_index": sample_index,
                            "horizon_s": horizon_s,
                            **maps["metrics"],
                        }
                        rel_rows.append(rel_row)
                        if perturbation_id == "A10_drop_front_triplet" and sample_index == sample_indices[0] and horizon_s == 6:
                            debug_examples["best_case"] = {
                                "pred": pred_h,
                                "maps": maps,
                            }
    finally:
        model.forward_backbone = original_forward_backbone  # type: ignore[assignment]
    return rows, rel_rows, debug_examples


def aggregate_eval(rows: list[dict[str, Any]], group_keys: list[str], value_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[k] for k in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key, items in buckets.items():
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["count"] = len(items)
        for vk in value_keys:
            vals = [float(it[vk]) for it in items if it.get(vk) is not None]
            rec[f"mean_{vk}"] = float(np.mean(vals)) if vals else None
        out.append(rec)
    return out


def compare_against_reference(candidate_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], checkpoint_name: str) -> dict[str, Any]:
    ref_map = {(r["perturbation_id"], r["sample_index"], r["horizon_s"]): r for r in reference_rows}
    deltas: list[dict[str, Any]] = []
    metric_names = [
        "occupied_iou",
        "semantic_miou",
        "false_free_rate",
        "false_occupied_rate",
        "pred_gt_occupied_ratio",
        "small_object_false_free",
        "new_visible_recall",
        "front_sector_false_free",
    ]
    for row in candidate_rows:
        key = (row["perturbation_id"], row["sample_index"], row["horizon_s"])
        if key not in ref_map:
            continue
        ref = ref_map[key]
        rec = {"checkpoint_name": checkpoint_name, "perturbation_id": row["perturbation_id"], "sample_index": row["sample_index"], "horizon_s": row["horizon_s"]}
        for m in metric_names:
            rec[f"{m}_delta"] = float(row[m]) - float(ref[m])
        deltas.append(rec)
    agg = aggregate_eval(deltas, ["checkpoint_name", "perturbation_id"], [f"{m}_delta" for m in metric_names])
    return {"deltas": deltas, "aggregate": agg}


def choose_best_experiment(
    candidates: list[tuple[str, list[dict[str, Any]], dict[str, Any]]],
    target: str,
) -> dict[str, Any] | None:
    if not candidates:
        return None
    scored: list[tuple[float, dict[str, Any]]] = []
    for exp_id, rows, cmp in candidates:
        agg_map = {r["perturbation_id"]: r for r in cmp["aggregate"]}
        clean = agg_map.get("A0_clean", {})
        a1 = agg_map.get("A1_drop_cam_front", {})
        a10 = agg_map.get("A10_drop_front_triplet", {})
        c4 = agg_map.get("C4_motion_blur_9", {})
        if target == "small_object":
            score = -(
                float(a1.get("mean_small_object_false_free_delta", 0.0))
                + float(a10.get("mean_small_object_false_free_delta", 0.0))
            ) / 2.0
            penalty = max(0.0, float(clean.get("mean_semantic_miou_delta", 0.0)) * -2.0) + max(0.0, float(a10.get("mean_false_occupied_rate_delta", 0.0)) * 20.0)
        elif target == "new_visible":
            score = (
                float(a1.get("mean_new_visible_recall_delta", 0.0))
                + float(a10.get("mean_new_visible_recall_delta", 0.0))
                + float(c4.get("mean_new_visible_recall_delta", 0.0))
            ) / 3.0
            penalty = max(0.0, float(c4.get("mean_false_occupied_rate_delta", 0.0)) * 15.0)
        elif target == "sensor_aug":
            score = -(
                float(a1.get("mean_front_sector_false_free_delta", 0.0))
                + float(a10.get("mean_front_sector_false_free_delta", 0.0))
            ) / 2.0
            penalty = max(0.0, float(clean.get("mean_occupied_iou_delta", 0.0)) * -10.0) + max(0.0, float(c4.get("mean_false_occupied_rate_delta", 0.0)) * 20.0)
        else:
            score = 0.0
            penalty = 0.0
        scored.append((score - penalty, {"experiment_id": exp_id, "rows": rows, "compare": cmp}))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def render_reliability_before_after(fig_name: str, baseline_maps: dict[str, Any], candidate_maps: dict[str, Any], out_path: Path) -> None:
    tmp_dir = ARTIFACTS_DIR / "tmp_sw8_panels"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    base_a = tmp_dir / f"{fig_name}_baseline_alpha.png"
    cand_a = tmp_dir / f"{fig_name}_candidate_alpha.png"
    base_r = tmp_dir / f"{fig_name}_baseline_risk.png"
    cand_r = tmp_dir / f"{fig_name}_candidate_risk.png"
    sw7.render_alpha_bev(baseline_maps["pred"], baseline_maps["maps"]["score_channels"]["final_reliability"], "baseline reliability alpha", base_a)
    sw7.render_alpha_bev(candidate_maps["pred"], candidate_maps["maps"]["score_channels"]["final_reliability"], "candidate reliability alpha", cand_a)
    sw7.render_risk_overlay_bev(baseline_maps["pred"], baseline_maps["maps"]["risk"], baseline_maps["maps"]["debug_masks"]["false_free"], baseline_maps["maps"]["debug_masks"]["false_positive"], "baseline risk", base_r)
    sw7.render_risk_overlay_bev(candidate_maps["pred"], candidate_maps["maps"]["risk"], candidate_maps["maps"]["debug_masks"]["false_free"], candidate_maps["maps"]["debug_masks"]["false_positive"], "candidate risk", cand_r)
    images = [Image.open(p).convert("RGB").resize((720, 720)) for p in [base_a, cand_a, base_r, cand_r]]
    canvas = Image.new("RGB", (1440, 1440), "white")
    labels = ["Baseline alpha", "Candidate alpha", "Baseline risk", "Candidate risk"]
    for idx, img in enumerate(images):
        r, c = divmod(idx, 2)
        x = c * 720
        y = r * 720
        canvas.paste(img, (x, y))
    canvas.save(out_path)


def topk_metrics_from_reliability_rows(retest_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for case in retest_cases:
        risk = case["maps"]["risk"].reshape(-1)
        ff = case["maps"]["debug_masks"]["false_free"].reshape(-1)
        fo = case["maps"]["debug_masks"]["false_positive"].reshape(-1)
        total = risk.numel()
        for frac in [0.01, 0.05, 0.10]:
            k = max(1, int(total * frac))
            idx = torch.topk(risk, k=k).indices
            sel = torch.zeros_like(risk, dtype=torch.bool)
            sel[idx] = True
            out.append({
                "checkpoint_name": case["checkpoint_name"],
                "perturbation_id": case["perturbation_id"],
                "sample_index": case["sample_index"],
                "horizon_s": case["horizon_s"],
                "topk_frac": frac,
                "false_free_recall": safe_div((sel & ff).sum().item(), ff.sum().item()),
                "false_free_precision": safe_div((sel & ff).sum().item(), sel.sum().item()),
                "false_positive_recall": safe_div((sel & fo).sum().item(), fo.sum().item()),
                "false_positive_precision": safe_div((sel & fo).sum().item(), sel.sum().item()),
            })
    return out


def plot_training_curve(rows: list[dict[str, Any]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    xs = [r["iter"] for r in rows]
    ys = [r["loss_total"] for r in rows]
    ax.plot(xs, ys, marker="o", markersize=2)
    ax.set_xlabel("iter")
    ax.set_ylabel("loss_total")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_delta_bars(agg_rows: list[dict[str, Any]], metric_key: str, title: str, ylabel: str, out_path: Path) -> None:
    labels = [str(r.get("checkpoint_name") or r.get("experiment_id") or "unknown") for r in agg_rows]
    vals = [float(r.get(metric_key, 0.0) or 0.0) for r in agg_rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(np.arange(len(labels)), vals, color="#3a7bd5")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize_training_entry(checkpoint_path: Path, iter_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg = aggregate_eval(eval_rows, ["checkpoint_name", "perturbation_id"], ["occupied_iou", "semantic_miou", "false_free_rate", "false_occupied_rate"])
    return {
        "smoke_checkpoint_path": str(checkpoint_path),
        "iter_count": len(iter_rows),
        "loss_finite": all(math.isfinite(float(r["loss_total"])) for r in iter_rows),
        "peak_gpu_memory_mb": float(max(r["peak_memory_mb"] for r in iter_rows)) if iter_rows else None,
        "mean_iter_time_sec": float(np.mean([r["iter_time_sec"] for r in iter_rows])) if iter_rows else None,
        "checkpoint_reload_eval_ok": bool(eval_rows),
        "clean_subset_eval_summary": [r for r in agg if r["perturbation_id"] == "A0_clean"],
    }


def make_report(
    decision: dict[str, Any],
    training_entry_audit: dict[str, Any],
    experiment_summaries: list[dict[str, Any]],
    reliability_retest_summary: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    lines = [
        "# Stage SW-8 Targeted Fine-tuning Report",
        "",
        "1. Executive summary",
        "",
        "- This is a subset diagnostic, not benchmark-grade validation.",
        "- This is lightweight fine-tuning, not full training reproduction.",
        "- No claim of beating the paper is made.",
        "- False-positive and pred/gt density tradeoffs are reported explicitly.",
        "",
        "2. Training entry audit",
        "",
        f"- Smoke checkpoint reload/eval ok: {training_entry_audit['checkpoint_reload_eval_ok']}",
        f"- Peak GPU memory MB: {training_entry_audit['peak_gpu_memory_mb']:.2f}" if training_entry_audit.get("peak_gpu_memory_mb") is not None else "- Peak GPU memory MB: unavailable",
        f"- Mean iter time sec: {training_entry_audit['mean_iter_time_sec']:.3f}" if training_entry_audit.get("mean_iter_time_sec") is not None else "- Mean iter time sec: unavailable",
        "",
        "3. Experiment summary",
        "",
    ]
    for s in experiment_summaries:
        lines += [
            f"- {s['experiment_id']}: clean occupied IoU delta={s.get('clean_occupied_iou_delta')}, clean semantic mIoU delta={s.get('clean_semantic_miou_delta')}, A10 small-object FF delta={s.get('a10_small_object_false_free_delta')}, C4 false-occupied delta={s.get('c4_false_occupied_delta')}",
        ]
    lines += [
        "",
        "4. SW-7 reliability retest",
        "",
        f"- Retested checkpoints: {', '.join(reliability_retest_summary.get('checkpoint_names', []))}",
        f"- Best candidate reliability headline: {reliability_retest_summary.get('headline', 'n/a')}",
        "",
        "5. Decision",
        "",
        f"- Decision: {decision['decision_type']}",
        f"- Safe claim: {decision['safe_claim']}",
        f"- Next action: {decision['next_action']}",
        "",
        "6. Limitations",
        "",
        "- Subset-only diagnostic; no full validation benchmark.",
        "- New-visible consistency used future-horizon reweight, not a stable explicit oracle mask.",
        "- Reliability retest is an internal diagnostic reliability indicator, not calibrated uncertainty.",
        "- If false-positive expansion appears, the candidate is marked unsafe rather than promoted.",
    ]
    payload = {
        "training_entry_audit": training_entry_audit,
        "experiment_summaries": experiment_summaries,
        "reliability_retest_summary": reliability_retest_summary,
        "decision": decision,
        "safe_claims": [
            "subset diagnostic, not benchmark-grade validation",
            "lightweight fine-tuning, not full training reproduction",
            "not deployment-ready",
            "no claim of beating paper",
            "false-positive tradeoff explicitly reported",
        ],
    }
    return "\n".join(lines) + "\n", payload


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    stage_logger = StageLogger()
    sample_indices = DEFAULT_SAMPLE_INDICES if not args.eval_sample_indices else [int(x) for x in args.eval_sample_indices.split(",") if x.strip()]
    if args.num_eval_samples > 0:
        sample_indices = sample_indices[: args.num_eval_samples]
    smoke_iters = [int(x.strip()) for x in args.smoke_iters.split(",") if x.strip()]
    if args.fast:
        smoke_iters = [min(smoke_iters[0], 20)]
    experiment_defs = build_experiment_defs(args.base_train_iters if not args.fast else min(args.base_train_iters, 30), args.combined_train_iters if not args.fast else min(args.combined_train_iters, 40))

    run_manifest = {
        "status": "running",
        "effective_samples": sample_indices,
        "effective_perturbations": CORE_PERTURBATIONS,
        "effective_horizons": DEFAULT_HORIZONS,
        "repo_root": str(REPO_ROOT),
        "config": str(CONFIG_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "effective_samples_per_gpu": args.samples_per_gpu,
        "effective_workers_per_gpu": args.workers_per_gpu,
    }
    write_json(REPORTS_DIR / "sw8_run_manifest.json", run_manifest)
    stage_logger.start("reference_eval", checkpoint=str(CHECKPOINT_PATH), sample_count=len(sample_indices), horizon_count=len(DEFAULT_HORIZONS))
    base_eval_rows, _, _ = evaluate_checkpoint(CHECKPOINT_PATH, "REF_epoch_56", sample_indices, DEFAULT_HORIZONS, capture_reliability=False)
    stage_logger.done("reference_eval", row_count=len(base_eval_rows))
    write_csv(REPORTS_DIR / "reference_epoch56_eval.csv", base_eval_rows)

    smoke_ckpt = None
    smoke_all_rows: list[dict[str, Any]] = []
    training_entry_audit: dict[str, Any] = {}
    for smoke_iter in smoke_iters:
        ckpt_path, iter_rows, summary = smoke_and_train(
            {"id": f"smoke_resume_{smoke_iter}iter", "train_iters": smoke_iter, "small_object_factor": 1.0, "future_scales": {}, "train_aug": []},
            samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=args.workers_per_gpu,
            seed=args.seed,
            stage_logger=stage_logger,
        )
        for row in iter_rows:
            row["smoke_iter_budget"] = smoke_iter
        smoke_all_rows.extend(iter_rows)
        if smoke_iter == max(smoke_iters):
            smoke_ckpt = ckpt_path
            stage_logger.start("smoke_eval", checkpoint=str(ckpt_path), sample_count=len(sample_indices))
            smoke_eval_rows, _, _ = evaluate_checkpoint(ckpt_path, f"smoke_resume_{smoke_iter}iter", sample_indices, DEFAULT_HORIZONS, capture_reliability=False)
            stage_logger.done("smoke_eval", row_count=len(smoke_eval_rows))
            training_entry_audit = summarize_training_entry(ckpt_path, iter_rows, smoke_eval_rows)
    write_csv(REPORTS_DIR / "training_smoke_metrics.csv", smoke_all_rows)
    write_json(REPORTS_DIR / "training_entry_audit.json", training_entry_audit)
    write_md(
        REPORTS_DIR / "training_entry_audit.md",
        "\n".join(
            [
                "# Training Entry Audit",
                "",
                f"- Effective Python/CUDA runtime: WSL + sparseworld_cu128",
                f"- Config: {to_windows_path(CONFIG_PATH)}",
                f"- Checkpoint: {to_windows_path(CHECKPOINT_PATH)}",
                f"- Effective batch size per GPU: {args.samples_per_gpu}",
                f"- Effective workers per GPU: {args.workers_per_gpu}",
                f"- Smoke checkpoint reload/eval ok: {training_entry_audit.get('checkpoint_reload_eval_ok')}",
                f"- Loss finite: {training_entry_audit.get('loss_finite')}",
                f"- Peak GPU memory MB: {training_entry_audit.get('peak_gpu_memory_mb')}",
                f"- Mean iter time sec: {training_entry_audit.get('mean_iter_time_sec')}",
            ]
        ) + "\n",
    )
    if smoke_all_rows:
        plot_training_curve(smoke_all_rows[-max(smoke_iters):], FIGURES_DIR / "training_smoke_loss_curve.png", "SW-8 smoke training loss")

    experiment_eval_rows: dict[str, list[dict[str, Any]]] = {}
    experiment_compare: dict[str, dict[str, Any]] = {}
    experiment_ckpts: dict[str, Path] = {}
    train_metric_rows_all: list[dict[str, Any]] = []
    experiment_summary_rows: list[dict[str, Any]] = []

    for exp_cfg in experiment_defs:
        if args.fast and exp_cfg["id"] not in {"E0_baseline_resume_control", "E1_small_object_reweight_light", "E4_future_horizon_reweight"}:
            continue
        ckpt_path, iter_rows, train_summary = smoke_and_train(
            exp_cfg,
            samples_per_gpu=args.samples_per_gpu,
            workers_per_gpu=args.workers_per_gpu,
            seed=args.seed,
            stage_logger=stage_logger,
            checkpoint_override=None,
        )
        experiment_ckpts[exp_cfg["id"]] = ckpt_path
        train_metric_rows_all.extend(iter_rows)
        eval_stage = f"eval_{exp_cfg['id']}"
        stage_logger.start(eval_stage, checkpoint=str(ckpt_path), sample_count=len(sample_indices))
        eval_rows, _, _ = evaluate_checkpoint(ckpt_path, exp_cfg["id"], sample_indices, DEFAULT_HORIZONS, capture_reliability=False)
        stage_logger.done(eval_stage, row_count=len(eval_rows))
        experiment_eval_rows[exp_cfg["id"]] = eval_rows
        cmp = compare_against_reference(eval_rows, base_eval_rows, exp_cfg["id"])
        experiment_compare[exp_cfg["id"]] = cmp
        agg_map = {r["perturbation_id"]: r for r in cmp["aggregate"]}
        summary_row = {
            "experiment_id": exp_cfg["id"],
            "checkpoint_path": str(ckpt_path),
            "peak_gpu_memory_mb": train_summary["peak_gpu_memory_mb"],
            "mean_iter_time_sec": train_summary["mean_iter_time_sec"],
            "clean_occupied_iou_delta": agg_map.get("A0_clean", {}).get("mean_occupied_iou_delta"),
            "clean_semantic_miou_delta": agg_map.get("A0_clean", {}).get("mean_semantic_miou_delta"),
            "a10_small_object_false_free_delta": agg_map.get("A10_drop_front_triplet", {}).get("mean_small_object_false_free_delta"),
            "a10_new_visible_recall_delta": agg_map.get("A10_drop_front_triplet", {}).get("mean_new_visible_recall_delta"),
            "a10_front_sector_false_free_delta": agg_map.get("A10_drop_front_triplet", {}).get("mean_front_sector_false_free_delta"),
            "c4_false_occupied_delta": agg_map.get("C4_motion_blur_9", {}).get("mean_false_occupied_rate_delta"),
            "c4_pred_gt_ratio_delta": agg_map.get("C4_motion_blur_9", {}).get("mean_pred_gt_occupied_ratio_delta"),
        }
        experiment_summary_rows.append(summary_row)

    write_csv(REPORTS_DIR / "all_train_metrics.csv", train_metric_rows_all)
    for exp_id, rows in experiment_eval_rows.items():
        write_csv(REPORTS_DIR / f"{exp_id}_eval.csv", rows)
        write_csv(REPORTS_DIR / f"{exp_id}_delta_vs_reference.csv", experiment_compare[exp_id]["deltas"])
    write_csv(REPORTS_DIR / "experiment_summary_metrics.csv", experiment_summary_rows)

    # Named outputs required by task
    if "E0_baseline_resume_control" in experiment_eval_rows:
        write_csv(REPORTS_DIR / "e0_baseline_resume_control_eval.csv", experiment_eval_rows["E0_baseline_resume_control"])
        write_md(REPORTS_DIR / "e0_baseline_resume_control_summary.md", json.dumps(normalize_export([r for r in experiment_summary_rows if r["experiment_id"] == "E0_baseline_resume_control"][0]), indent=2, ensure_ascii=False))
    small_rows = [r for r in experiment_summary_rows if "small_object" in r["experiment_id"]]
    if small_rows:
        write_csv(REPORTS_DIR / "small_object_reweight_train_metrics.csv", [r for r in train_metric_rows_all if "small_object" in r["experiment_id"]])
        write_csv(REPORTS_DIR / "small_object_reweight_eval_metrics.csv", [r for r in sum((experiment_eval_rows[k] for k in experiment_eval_rows if "small_object" in k), [])])
        write_md(REPORTS_DIR / "small_object_reweight_summary.md", json.dumps(normalize_export(small_rows), indent=2, ensure_ascii=False))
    new_rows = [r for r in experiment_summary_rows if "future_horizon" in r["experiment_id"]]
    if new_rows:
        write_csv(REPORTS_DIR / "new_visible_future_train_metrics.csv", [r for r in train_metric_rows_all if "future_horizon" in r["experiment_id"]])
        write_csv(REPORTS_DIR / "new_visible_future_eval_metrics.csv", [r for r in sum((experiment_eval_rows[k] for k in experiment_eval_rows if "future_horizon" in k), [])])
        write_md(REPORTS_DIR / "new_visible_future_summary.md", json.dumps(normalize_export(new_rows), indent=2, ensure_ascii=False))
    sensor_rows = [r for r in experiment_summary_rows if ("front_dropout" in r["experiment_id"] or "motion_blur" in r["experiment_id"])]
    if sensor_rows:
        write_csv(REPORTS_DIR / "sensor_aug_train_metrics.csv", [r for r in train_metric_rows_all if ("front_dropout" in r["experiment_id"] or "motion_blur" in r["experiment_id"])])
        write_csv(REPORTS_DIR / "sensor_aug_eval_metrics.csv", [r for r in sum((experiment_eval_rows[k] for k in experiment_eval_rows if ("front_dropout" in k or "motion_blur" in k)), [])])
        write_md(REPORTS_DIR / "sensor_aug_summary.md", json.dumps(normalize_export(sensor_rows), indent=2, ensure_ascii=False))
    combined_rows = [r for r in experiment_summary_rows if r["experiment_id"].startswith("E10_")]
    if combined_rows:
        write_csv(REPORTS_DIR / "combined_finetune_eval_metrics.csv", [r for r in sum((experiment_eval_rows[k] for k in experiment_eval_rows if k.startswith("E10_")), [])])
        write_md(REPORTS_DIR / "combined_finetune_summary.md", json.dumps(normalize_export(combined_rows), indent=2, ensure_ascii=False))

    best_small = choose_best_experiment([(k, experiment_eval_rows[k], experiment_compare[k]) for k in experiment_eval_rows if "small_object" in k], "small_object")
    best_new = choose_best_experiment([(k, experiment_eval_rows[k], experiment_compare[k]) for k in experiment_eval_rows if "future_horizon" in k], "new_visible")
    best_sensor = choose_best_experiment([(k, experiment_eval_rows[k], experiment_compare[k]) for k in experiment_eval_rows if ("front_dropout" in k or "motion_blur" in k)], "sensor_aug")
    best_combined = choose_best_experiment([(k, experiment_eval_rows[k], experiment_compare[k]) for k in experiment_eval_rows if k.startswith("E10_")], "small_object")

    reliability_candidates: list[tuple[str, Path]] = []
    for name in ["E0_baseline_resume_control"]:
        if name in experiment_ckpts:
            reliability_candidates.append((name, experiment_ckpts[name]))
    for picked in [best_small, best_new, best_sensor, best_combined]:
        if picked and picked["experiment_id"] in experiment_ckpts and (picked["experiment_id"], experiment_ckpts[picked["experiment_id"]]) not in reliability_candidates:
            reliability_candidates.append((picked["experiment_id"], experiment_ckpts[picked["experiment_id"]]))

    reliability_retest_rows_all: list[dict[str, Any]] = []
    reliability_error_rows_all: list[dict[str, Any]] = []
    reliability_case_maps: list[dict[str, Any]] = []
    for ckpt_name, ckpt_path in reliability_candidates:
        rel_stage = f"reliability_retest_{ckpt_name}"
        stage_logger.start(rel_stage, checkpoint=str(ckpt_path), sample_count=len(sample_indices))
        eval_rows, rel_rows, debug_examples = evaluate_checkpoint(ckpt_path, ckpt_name, sample_indices, DEFAULT_HORIZONS, capture_reliability=True)
        stage_logger.done(rel_stage, eval_rows=len(eval_rows), reliability_rows=len(rel_rows))
        reliability_retest_rows_all.extend(rel_rows)
        if "best_case" in debug_examples:
            reliability_case_maps.append({
                "checkpoint_name": ckpt_name,
                "perturbation_id": "A10_drop_front_triplet",
                "sample_index": sample_indices[0],
                "horizon_s": 6,
                "pred": debug_examples["best_case"]["pred"],
                "maps": debug_examples["best_case"]["maps"],
            })
        for row in rel_rows:
            reliability_error_rows_all.append(row)
    write_csv(REPORTS_DIR / "sw7_reliability_retest_after_finetune.csv", reliability_retest_rows_all)
    topk_rows = topk_metrics_from_reliability_rows(reliability_case_maps)
    write_csv(REPORTS_DIR / "sw7_reliability_topk_after_finetune.csv", topk_rows)

    reliability_summary = {
        "checkpoint_names": [name for name, _ in reliability_candidates],
        "headline": "selected fine-tuned checkpoints were re-evaluated with the same internal diagnostic reliability indicator used in SW-7",
    }
    write_md(REPORTS_DIR / "sw7_reliability_retest_summary.md", json.dumps(normalize_export(reliability_summary), indent=2, ensure_ascii=False))

    if len(reliability_case_maps) >= 2:
        baseline_case = next((x for x in reliability_case_maps if x["checkpoint_name"] == "E0_baseline_resume_control"), reliability_case_maps[0])
        candidate_case = reliability_case_maps[-1]
        render_reliability_before_after("best_candidate", baseline_case, candidate_case, FIGURES_DIR / "reliability_before_after_best_candidate.png")
        render_reliability_before_after("best_candidate_score", baseline_case, candidate_case, FIGURES_DIR / "score_alpha_before_after_best_candidate.png")

    # decision
    decision_type = "T3_no_meaningful_improvement"
    safe_claim = "lightweight targeted fine-tuning produced only subset-level diagnostic evidence; no official benchmark claim is made."
    next_action = "if gains remain small after safe light-touch tuning, prioritize architecture-level sparse query / contributor modifications rather than longer blind fine-tuning."
    false_positive_headline = "false-positive and pred/gt density expansion remained the main safety gate"

    chosen = best_combined or best_small or best_new or best_sensor
    if chosen:
        agg_map = {r["perturbation_id"]: r for r in chosen["compare"]["aggregate"]}
        clean = agg_map.get("A0_clean", {})
        a10 = agg_map.get("A10_drop_front_triplet", {})
        a1 = agg_map.get("A1_drop_cam_front", {})
        c4 = agg_map.get("C4_motion_blur_9", {})
        safe_clean = float(clean.get("mean_occupied_iou_delta", 0.0) or 0.0) >= -0.005
        safe_fp = float(c4.get("mean_false_occupied_rate_delta", 0.0) or 0.0) <= 0.008
        better_front = float(a10.get("mean_front_sector_false_free_delta", 0.0) or 0.0) <= -0.03 or float(a1.get("mean_front_sector_false_free_delta", 0.0) or 0.0) <= -0.03
        better_small = float(a10.get("mean_small_object_false_free_delta", 0.0) or 0.0) <= -0.03
        better_new = float(a10.get("mean_new_visible_recall_delta", 0.0) or 0.0) >= 0.03
        if safe_clean and safe_fp and (better_front or better_small or better_new):
            decision_type = "T1_success_safe_improvement"
            safe_claim = "the best minimal candidate improved targeted subset diagnostics without a clear clean-set collapse, but this is still not an official benchmark result."
            next_action = "run a slightly longer targeted sweep around the winning recipe and then repeat the same SW-7 reliability retest before any full-val claim."
        elif (better_front or better_small or better_new) and (not safe_clean or not safe_fp):
            decision_type = "T2_partial_improvement_with_tradeoff"
            safe_claim = "some targeted gains appeared, but they came with non-trivial false-positive or clean-drift tradeoffs, so the candidate is not safe to promote."
            next_action = "tighten the loss/augmentation strength and add stronger density control before any larger fine-tune."
        elif not (better_front or better_small or better_new):
            decision_type = "T4_architecture_limited"
            safe_claim = "multiple lightweight routes failed to produce meaningful targeted gains, which points toward a sparse-query / contributor architecture bottleneck in this setup."
            next_action = "move to architecture-level proposals from SW-7 instead of extending the same fine-tune recipe."

    decision_payload = {
        "decision_type": decision_type,
        "best_checkpoint": chosen["experiment_id"] if chosen else None,
        "false_positive_headline": false_positive_headline,
        "safe_claim": safe_claim,
        "next_action": next_action,
    }
    write_json(REPORTS_DIR / "sw8_finetune_decision.json", decision_payload)
    write_md(REPORTS_DIR / "sw8_finetune_decision.md", json.dumps(normalize_export(decision_payload), indent=2, ensure_ascii=False))

    stage_logger.start("final_report_and_manifest")

    # figures
    if experiment_summary_rows:
        plot_delta_bars(experiment_summary_rows, "a10_small_object_false_free_delta", "A10 small-object false-free delta", "delta", FIGURES_DIR / "small_object_delta_bar.png")
        plot_delta_bars(experiment_summary_rows, "a10_new_visible_recall_delta", "A10 new-visible recall delta", "delta", FIGURES_DIR / "new_visible_delta_bar.png")
        plot_delta_bars(experiment_summary_rows, "a10_front_sector_false_free_delta", "A10 front-sector false-free delta", "delta", FIGURES_DIR / "front_sector_delta_bar.png")

    report_text, report_json = make_report(decision_payload, training_entry_audit, experiment_summary_rows, reliability_summary)
    write_md(REPORTS_DIR / "stage_sw8_targeted_finetune_report.md", report_text)
    write_json(REPORTS_DIR / "stage_sw8_targeted_finetune_report.json", report_json)

    # bridge outputs
    write_md(
        REPORTS_DIR / "sw8_recommended_training_plan_summary.md",
        "\n".join(
            [
                "# SW-8 recommended minimal directions",
                "",
                "- small-object activation reweighting remains the lowest-cost targeted intervention.",
                "- future-horizon reweight is the current fallback for new-visible consistency because a stable explicit new-visible mask was not promoted into training supervision here.",
                "- front-view sensor augmentation should stay light; otherwise clean drift and false-positive expansion become the main failure mode.",
            ]
        ) + "\n",
    )

    # tests
    run_manifest["status"] = "completed"
    write_json(REPORTS_DIR / "sw8_run_manifest.json", run_manifest)
    stage_logger.done("final_report_and_manifest", report_path=str(REPORTS_DIR / "stage_sw8_targeted_finetune_report.md"))


if __name__ == "__main__":
    main()
