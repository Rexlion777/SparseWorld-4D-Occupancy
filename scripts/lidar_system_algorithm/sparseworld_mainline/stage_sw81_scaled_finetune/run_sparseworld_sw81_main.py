from __future__ import annotations

import argparse
import csv
import gc
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


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPO_ROOT = PROJECT_ROOT / "external/SparseWorld"
CONFIG_PATH = REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"
CHECKPOINT_PATH = REPO_ROOT / "ckpts/epoch_56.pth"

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw81_scaled_finetune"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw81_scaled_finetune"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw81_scaled_finetune"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw81_scaled_finetune"

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
    p.add_argument("--samples-per-gpu", type=int, default=2)
    p.add_argument("--workers-per-gpu", type=int, default=6)
    p.add_argument("--quick-eval-count", type=int, default=10)
    p.add_argument("--core-eval-count", type=int, default=20)
    p.add_argument("--stress-eval-count", type=int, default=20)
    p.add_argument("--optional-eval-count", type=int, default=50)
    p.add_argument("--run-optional-eval50", action="store_true", default=False)
    p.add_argument("--e0-lr-scales", default="1.0,0.1,0.03")
    p.add_argument("--e0-iters", default="100,500,1000")
    p.add_argument("--targeted-iters", type=int, default=1000)
    p.add_argument("--combined-iters", type=int, default=3000)
    p.add_argument("--disable-combined", action="store_true", default=False)
    p.add_argument("--disable-e7", action="store_true", default=False)
    p.add_argument("--sample-offset-core", type=int, default=0)
    p.add_argument("--sample-offset-stress-tail", type=int, default=10)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--save-debug", action="store_true", default=True)
    p.add_argument("--save-figures", action="store_true", default=True)
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


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class StageLogger:
    def __init__(self) -> None:
        self.status_path = LOGS_DIR / "sw81_status.json"
        self.progress_path = LOGS_DIR / "sw81_progress.jsonl"
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
    print(f"[SW81] {msg}", flush=True)
    with (LOGS_DIR / "phase1_lr_stability_audit.log").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def safe_cuda_cleanup() -> None:
    try:
        gc.collect()
    except Exception:
        pass
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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


def scaled_lr_name(scale: float) -> str:
    if abs(scale - 1.0) < 1e-8:
        return "lr_original"
    if abs(scale - 0.1) < 1e-8:
        return "lr0.1x"
    if abs(scale - 0.03) < 1e-8:
        return "lr0.03x"
    return f"lrx{str(scale).replace('.', 'p')}"


def build_targeted_experiment_defs(targeted_iters: int, lr_scales: list[float]) -> list[dict[str, Any]]:
    exps: list[dict[str, Any]] = []
    for lr_scale in lr_scales:
        lr_tag = scaled_lr_name(lr_scale)
        exps.extend(
            [
                {
                    "id": f"E4_future_horizon_reweight_{lr_tag}",
                    "family": "E4_future_horizon_reweight",
                    "train_iters": targeted_iters,
                    "small_object_factor": 1.0,
                    "future_scales": {4: 1.15, 5: 1.30, 6: 1.45},
                    "train_aug": [],
                    "lr_scale": lr_scale,
                },
                {
                    "id": f"E8_motion_blur_aug_light_{lr_tag}",
                    "family": "E8_motion_blur_aug_light",
                    "train_iters": targeted_iters,
                    "small_object_factor": 1.0,
                    "future_scales": {},
                    "train_aug": [{"prob": 0.12, "spec_choices": ["C4_motion_blur_9"]}],
                    "lr_scale": lr_scale,
                },
            ]
        )
    exps.extend(
        [
            {
                "id": "E1_small_object_reweight_light_lr0.1x",
                "family": "E1_small_object_reweight_scaled",
                "train_iters": targeted_iters,
                "small_object_factor": 1.5,
                "future_scales": {},
                "train_aug": [],
                "lr_scale": 0.1,
            },
            {
                "id": "E1_small_object_reweight_light_lr0.03x",
                "family": "E1_small_object_reweight_scaled",
                "train_iters": targeted_iters,
                "small_object_factor": 1.5,
                "future_scales": {},
                "train_aug": [],
                "lr_scale": 0.03,
            },
            {
                "id": "E1_small_object_reweight_medium_lr0.03x",
                "family": "E1_small_object_reweight_scaled",
                "train_iters": targeted_iters,
                "small_object_factor": 2.0,
                "future_scales": {},
                "train_aug": [],
                "lr_scale": 0.03,
            },
        ]
    )
    return exps


def build_optional_e7_exp(targeted_iters: int, lr_scale: float) -> dict[str, Any]:
    return {
        "id": f"E7_front_dropout_aug_light_{scaled_lr_name(lr_scale)}",
        "family": "E7_front_dropout_aug_light",
        "train_iters": targeted_iters,
        "small_object_factor": 1.0,
        "future_scales": {},
        "train_aug": [{"prob": 0.08, "spec_choices": ["A1_drop_cam_front", "A10_drop_front_triplet"]}],
        "lr_scale": lr_scale,
    }


def build_combined_exp(targeted_iters: int, future_scale: float, small_scale: float | None, blur_scale: float | None) -> list[dict[str, Any]]:
    exps: list[dict[str, Any]] = []
    if blur_scale is not None:
        exps.append(
            {
                "id": "E12_best_future_plus_best_blur",
                "family": "combined_safe_candidates",
                "train_iters": targeted_iters,
                "small_object_factor": 1.0,
                "future_scales": {4: 1.15, 5: 1.30, 6: 1.45},
                "train_aug": [{"prob": 0.12, "spec_choices": ["C4_motion_blur_9"]}],
                "lr_scale": min(future_scale, blur_scale),
            }
        )
    if small_scale is not None:
        exps.append(
            {
                "id": "E13_best_future_plus_best_small",
                "family": "combined_safe_candidates",
                "train_iters": targeted_iters,
                "small_object_factor": 1.5,
                "future_scales": {4: 1.15, 5: 1.30, 6: 1.45},
                "train_aug": [],
                "lr_scale": min(future_scale, small_scale),
            }
        )
        if blur_scale is not None:
            exps.append(
                {
                    "id": "E14_best_small_plus_best_blur",
                    "family": "combined_safe_candidates",
                    "train_iters": targeted_iters,
                    "small_object_factor": 1.5,
                    "future_scales": {},
                    "train_aug": [{"prob": 0.12, "spec_choices": ["C4_motion_blur_9"]}],
                    "lr_scale": min(small_scale, blur_scale),
                }
            )
    return exps


def apply_experiment_config_overrides(exp_cfg: dict[str, Any], samples_per_gpu: int, workers_per_gpu: int) -> dict[str, Any]:
    cls_weights = current_base_cls_weights()
    factor = float(exp_cfg.get("small_object_factor", 1.0))
    if abs(factor - 1.0) > 1e-8:
        for cid in SMALL_OBJECT_CLASS_IDS:
            if 0 <= cid < len(cls_weights):
                cls_weights[cid] *= factor
    lr_scale = float(exp_cfg.get("lr_scale", 1.0))
    return {
        "data.samples_per_gpu": samples_per_gpu,
        "data.workers_per_gpu": workers_per_gpu,
        "model.train_cfg.pts.cls_weights": cls_weights,
        "optimizer.lr": float(2e-4 * lr_scale),
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


def build_eval_subsets(args: argparse.Namespace) -> tuple[dict[str, list[int]], list[int]]:
    quick = list(range(args.quick_eval_count))
    core = list(range(args.sample_offset_core, args.sample_offset_core + args.core_eval_count))
    stress_head = list(range(min(10, args.stress_eval_count // 2 or args.stress_eval_count)))
    stress_tail_count = max(0, args.stress_eval_count - len(stress_head))
    stress_tail = list(range(args.sample_offset_stress_tail, args.sample_offset_stress_tail + stress_tail_count))
    stress = (stress_head + stress_tail)[: args.stress_eval_count]
    optional50 = list(range(args.optional_eval_count))
    union = sorted(set(quick + core + stress + (optional50 if args.run_optional_eval50 else [])))
    return {
        "quick_eval_10": quick,
        "eval_core_20": core,
        "eval_stress_20": stress,
        "optional_eval_50": optional50,
    }, union


def sample_subset_memberships(sample_index: int, subsets: dict[str, list[int]]) -> list[str]:
    out: list[str] = []
    for subset_name, indices in subsets.items():
        if subset_name == "optional_eval_50":
            continue
        if sample_index in indices:
            out.append(subset_name)
    return out or ["union_eval"]


def filter_rows_by_samples(rows: list[dict[str, Any]], sample_indices: list[int]) -> list[dict[str, Any]]:
    allowed = set(sample_indices)
    return [r for r in rows if int(r["sample_index"]) in allowed]


def summarize_compare_by_subset(
    cmp_deltas: list[dict[str, Any]],
    subsets: dict[str, list[int]],
    checkpoint_name: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    metric_names = [
        "occupied_iou_delta",
        "semantic_miou_delta",
        "false_free_rate_delta",
        "false_occupied_rate_delta",
        "pred_gt_occupied_ratio_delta",
        "small_object_false_free_delta",
        "new_visible_recall_delta",
        "front_sector_false_free_delta",
    ]
    for subset_name, indices in subsets.items():
        if subset_name == "optional_eval_50":
            continue
        rows = [r for r in cmp_deltas if int(r["sample_index"]) in set(indices)]
        if not rows:
            continue
        out.extend(aggregate_eval(rows, ["checkpoint_name", "perturbation_id"], metric_names))
        for item in out[-len(CORE_PERTURBATIONS):]:
            item["subset_name"] = subset_name
            item["checkpoint_name"] = checkpoint_name
    return out


def pick_best_lr_from_quick_eval(stability_rows: list[dict[str, Any]]) -> float:
    grouped: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in stability_rows:
        if int(row.get("train_iters", 0)) != 100:
            continue
        grouped[float(row["lr_scale"])].append(row)
    scored: list[tuple[float, float]] = []
    for lr_scale, rows in grouped.items():
        clean = next((r for r in rows if r["perturbation_id"] == "A0_clean"), None)
        a10 = next((r for r in rows if r["perturbation_id"] == "A10_drop_front_triplet"), None)
        c4 = next((r for r in rows if r["perturbation_id"] == "C4_motion_blur_9"), None)
        score = 0.0
        if clean is not None:
            score += float(clean.get("mean_occupied_iou_delta", 0.0)) * 10.0
            score += float(clean.get("mean_semantic_miou_delta", 0.0)) * 8.0
            score -= max(0.0, -float(clean.get("mean_occupied_iou_delta", 0.0))) * 40.0
        if a10 is not None:
            score -= float(a10.get("mean_front_sector_false_free_delta", 0.0)) * 8.0
            score += float(a10.get("mean_new_visible_recall_delta", 0.0)) * 6.0
        if c4 is not None:
            score -= max(0.0, float(c4.get("mean_false_occupied_rate_delta", 0.0))) * 30.0
        scored.append((score, lr_scale))
    scored.sort(reverse=True)
    return scored[0][1] if scored else 0.03


def is_safe_candidate(summary_row: dict[str, Any]) -> bool:
    clean_iou = float(summary_row.get("clean_occupied_iou_delta") or 0.0)
    false_occ = float(summary_row.get("c4_false_occupied_delta") or 0.0)
    pred_gt = float(summary_row.get("c4_pred_gt_ratio_delta") or 0.0)
    return clean_iou >= -0.005 and false_occ <= 0.008 and pred_gt <= 0.10


def summarize_family_metric(summary_rows: list[dict[str, Any]], family_name: str) -> list[dict[str, Any]]:
    return [r for r in summary_rows if r.get("family") == family_name and not r.get("failed", False)]


def failure_row(exp_cfg: dict[str, Any], error: Exception) -> dict[str, Any]:
    return {
        "experiment_id": exp_cfg.get("id"),
        "family": exp_cfg.get("family"),
        "lr_scale": exp_cfg.get("lr_scale"),
        "train_iters": exp_cfg.get("train_iters"),
        "failed": True,
        "failed_reason": repr(error),
    }


def checkpoint_path_for_exp(exp_cfg: dict[str, Any]) -> Path:
    return ARTIFACTS_DIR / "checkpoints" / f"{exp_cfg['id']}_iter{int(exp_cfg['train_iters']):04d}.pth"


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
    debug_examples: dict[str, Any] = {"topk_rows": []}
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
                        if horizon_s == 6:
                            risk = maps["risk"].reshape(-1)
                            ff = maps["debug_masks"]["false_free"].reshape(-1)
                            fo = maps["debug_masks"]["false_positive"].reshape(-1)
                            total = int(risk.numel())
                            for frac in [0.01, 0.05, 0.10]:
                                k = max(1, int(total * frac))
                                idx = torch.topk(risk, k=k).indices
                                sel = torch.zeros_like(risk, dtype=torch.bool)
                                sel[idx] = True
                                debug_examples["topk_rows"].append(
                                    {
                                        "checkpoint_name": checkpoint_name,
                                        "perturbation_id": perturbation_id,
                                        "sample_index": sample_index,
                                        "horizon_s": horizon_s,
                                        "topk_frac": frac,
                                        "false_free_recall": safe_div((sel & ff).sum().item(), ff.sum().item()),
                                        "false_free_precision": safe_div((sel & ff).sum().item(), sel.sum().item()),
                                        "false_positive_recall": safe_div((sel & fo).sum().item(), fo.sum().item()),
                                        "false_positive_precision": safe_div((sel & fo).sum().item(), sel.sum().item()),
                                    }
                                )
                        if perturbation_id == "A10_drop_front_triplet" and sample_index == sample_indices[0] and horizon_s == 6:
                            debug_examples["representative_case"] = {
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
    def _eval_key(row: dict[str, Any]) -> tuple[str, int, int]:
        return (str(row["perturbation_id"]), int(row["sample_index"]), int(row["horizon_s"]))

    ref_map = {_eval_key(r): r for r in reference_rows}
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
        key = _eval_key(row)
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


def make_report(
    decision: dict[str, Any],
    lr_audit: dict[str, Any],
    eval_protocol: dict[str, Any],
    experiment_summaries: list[dict[str, Any]],
    reliability_retest_summary: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    lines = [
        "# Stage SW-8.1 Scaled Targeted Fine-tuning Report",
        "",
        "1. Executive summary",
        "",
        "- This is a subset diagnostic, not an official benchmark.",
        "- This is scaled targeted fine-tuning, not full official training reproduction.",
        "- No claim of beating the paper is made.",
        "- False-positive and pred/gt density tradeoffs are always reported.",
        "",
        "2. Training stability and LR audit",
        "",
        f"- Selected stable LR scale: {lr_audit.get('selected_lr_scale')}",
        f"- Longest E0 iter budget: {lr_audit.get('longest_e0_iter')}",
        f"- Throughput baseline used: samples_per_gpu={lr_audit.get('samples_per_gpu')}, workers_per_gpu={lr_audit.get('workers_per_gpu')}",
        "",
        "3. Fixed eval protocol",
        "",
        f"- quick_eval_10: {eval_protocol.get('quick_eval_10')}",
        f"- eval_core_20: {eval_protocol.get('eval_core_20')}",
        f"- eval_stress_20: {eval_protocol.get('eval_stress_20')}",
        "",
        "4. Experiment summary",
        "",
    ]
    for s in experiment_summaries:
        lines.append(
            f"- {s['experiment_id']}: clean occupied IoU delta={s.get('clean_occupied_iou_delta')}, clean semantic mIoU delta={s.get('clean_semantic_miou_delta')}, A10 small-object FF delta={s.get('a10_small_object_false_free_delta')}, A10 new-visible recall delta={s.get('a10_new_visible_recall_delta')}, A10 front-sector FF delta={s.get('a10_front_sector_false_free_delta')}, C4 false-occupied delta={s.get('c4_false_occupied_delta')}"
        )
    lines += [
        "",
        "5. SW-7 reliability retest",
        "",
        f"- Retested checkpoints: {', '.join(reliability_retest_summary.get('checkpoint_names', []))}",
        f"- Headline: {reliability_retest_summary.get('headline', 'n/a')}",
        "",
        "6. Decision",
        "",
        f"- Decision: {decision['decision_type']}",
        f"- Safe claim: {decision['safe_claim']}",
        f"- Next action: {decision['next_action']}",
        "",
        "7. Safe claims",
        "",
        "- subset diagnostic",
        "- scaled fine-tune but not full official benchmark",
        "- no claim of surpassing the original paper unless official full validation exists",
        "- false-positive / pred_gt_ratio tradeoff always reported",
        "",
        "8. Limitations",
        "",
        "- No official full-validation benchmark was run.",
        "- Reliability retest remains an internal diagnostic reliability indicator, not calibrated uncertainty.",
        "- Architecture-limited wording is reserved for scaled evidence only.",
    ]
    payload = {
        "lr_audit": lr_audit,
        "eval_protocol": eval_protocol,
        "experiment_summaries": experiment_summaries,
        "reliability_retest_summary": reliability_retest_summary,
        "decision": decision,
    }
    return "\n".join(lines) + "\n", payload


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    stage_logger = StageLogger()
    lr_scales = [float(x.strip()) for x in args.e0_lr_scales.split(",") if x.strip()]
    e0_iter_budgets = [int(x.strip()) for x in args.e0_iters.split(",") if x.strip()]
    if args.fast:
        e0_iter_budgets = [min(e0_iter_budgets[0], 10)]
    targeted_iters = args.targeted_iters if not args.fast else min(args.targeted_iters, 10)
    combined_iters = args.combined_iters if not args.fast else min(args.combined_iters, 15)

    eval_subsets, eval_union = build_eval_subsets(args)
    if args.fast:
        eval_subsets["quick_eval_10"] = eval_subsets["quick_eval_10"][:2]
        eval_subsets["eval_core_20"] = eval_subsets["eval_core_20"][:4]
        eval_subsets["eval_stress_20"] = eval_subsets["eval_stress_20"][:4]
        eval_union = sorted(set(eval_subsets["quick_eval_10"] + eval_subsets["eval_core_20"] + eval_subsets["eval_stress_20"]))

    run_manifest = {
        "status": "running",
        "effective_perturbations": CORE_PERTURBATIONS,
        "effective_horizons": DEFAULT_HORIZONS,
        "repo_root": str(REPO_ROOT),
        "config": str(CONFIG_PATH),
        "checkpoint": str(CHECKPOINT_PATH),
        "samples_per_gpu": args.samples_per_gpu,
        "workers_per_gpu": args.workers_per_gpu,
        "eval_subsets": eval_subsets,
        "eval_union": eval_union,
        "e0_lr_scales": lr_scales,
        "e0_iter_budgets": e0_iter_budgets,
        "targeted_iters": targeted_iters,
        "combined_iters": combined_iters,
    }
    write_json(REPORTS_DIR / "sw81_run_manifest.json", run_manifest)
    write_json(REPORTS_DIR / "sw81_eval_protocol.json", eval_subsets)
    manifest_rows = [{"subset_name": subset_name, "sample_index": idx} for subset_name, indices in eval_subsets.items() for idx in indices]
    write_csv(REPORTS_DIR / "eval_subset_manifest.csv", manifest_rows)

    ref_csv = REPORTS_DIR / "reference_epoch56_union_eval.csv"
    if ref_csv.exists():
        ref_eval_rows = read_csv_rows(ref_csv)
        if ref_eval_rows:
            stage_logger.state["stages"]["reference_eval_union"] = {
                "status": "done",
                "checkpoint": str(CHECKPOINT_PATH),
                "sample_count": len(eval_union),
                "horizon_count": len(DEFAULT_HORIZONS),
                "row_count": len(ref_eval_rows),
                "reused": True,
            }
            stage_logger.flush()
        else:
            stage_logger.start("reference_eval_union", checkpoint=str(CHECKPOINT_PATH), sample_count=len(eval_union), horizon_count=len(DEFAULT_HORIZONS))
            ref_eval_rows, _, _ = evaluate_checkpoint(CHECKPOINT_PATH, "REF_epoch_56", eval_union, DEFAULT_HORIZONS, capture_reliability=False)
            stage_logger.done("reference_eval_union", row_count=len(ref_eval_rows))
            write_csv(ref_csv, ref_eval_rows)
    else:
        stage_logger.start("reference_eval_union", checkpoint=str(CHECKPOINT_PATH), sample_count=len(eval_union), horizon_count=len(DEFAULT_HORIZONS))
        ref_eval_rows, _, _ = evaluate_checkpoint(CHECKPOINT_PATH, "REF_epoch_56", eval_union, DEFAULT_HORIZONS, capture_reliability=False)
        stage_logger.done("reference_eval_union", row_count=len(ref_eval_rows))
        write_csv(ref_csv, ref_eval_rows)

    e0_train_rows: list[dict[str, Any]] = []
    e0_summary_rows: list[dict[str, Any]] = []
    lr_quick_rows: list[dict[str, Any]] = []

    lr_audit_path = REPORTS_DIR / "lr_stability_audit.json"
    e0_eval_by_iter_path = REPORTS_DIR / "e0_control_eval_by_iter.csv"
    if lr_audit_path.exists() and e0_eval_by_iter_path.exists():
        lr_audit = json.loads(lr_audit_path.read_text(encoding="utf-8"))
        selected_lr_scale = float(lr_audit["selected_lr_scale"])
        selected_e0_ckpt = checkpoint_path_for_exp(
            {
                "id": f"E0_baseline_resume_control_{scaled_lr_name(selected_lr_scale)}_{int(lr_audit['longest_e0_iter'])}iter",
                "train_iters": int(lr_audit["longest_e0_iter"]),
            }
        )
        e0_summary_rows = read_csv_rows(e0_eval_by_iter_path)
        stage_logger.state["stages"]["phase1_lr_stability_audit"] = {
            "status": "done",
            "lr_scales": lr_scales,
            "iter_budgets": e0_iter_budgets,
            "selected_lr_scale": selected_lr_scale,
            "reused": True,
        }
        stage_logger.flush()
    else:
        stage_logger.start("phase1_lr_stability_audit", lr_scales=lr_scales, iter_budgets=e0_iter_budgets)
        quick_iters = min(100, e0_iter_budgets[0])
        for lr_scale in lr_scales:
            exp_cfg = {"id": f"E0_baseline_resume_control_{scaled_lr_name(lr_scale)}_{quick_iters}iter", "family": "E0_baseline_resume_control", "train_iters": quick_iters, "small_object_factor": 1.0, "future_scales": {}, "train_aug": [], "lr_scale": lr_scale}
            ckpt_path = checkpoint_path_for_exp(exp_cfg)
            if ckpt_path.exists():
                train_summary = {"peak_gpu_memory_mb": None, "mean_iter_time_sec": None}
            else:
                ckpt_path, train_rows, train_summary = smoke_and_train(exp_cfg, args.samples_per_gpu, args.workers_per_gpu, args.seed, stage_logger)
                e0_train_rows.extend(train_rows)
            quick_eval_rows, _, _ = evaluate_checkpoint(ckpt_path, exp_cfg["id"], eval_subsets["quick_eval_10"], DEFAULT_HORIZONS, capture_reliability=False)
            cmp = compare_against_reference(quick_eval_rows, ref_eval_rows, exp_cfg["id"])
            agg_rows = summarize_compare_by_subset(cmp["deltas"], {"quick_eval_10": eval_subsets["quick_eval_10"]}, exp_cfg["id"])
            for row in agg_rows:
                lr_quick_rows.append({**row, "lr_scale": lr_scale, "train_iters": quick_iters})
            e0_summary_rows.extend([{**row, "lr_scale": lr_scale, "train_iters": quick_iters, "checkpoint_path": str(ckpt_path), "peak_gpu_memory_mb": train_summary["peak_gpu_memory_mb"], "mean_iter_time_sec": train_summary["mean_iter_time_sec"]} for row in agg_rows])
            safe_cuda_cleanup()
        selected_lr_scale = pick_best_lr_from_quick_eval(lr_quick_rows)
        selected_e0_ckpt = None
        for train_iters in e0_iter_budgets:
            exp_cfg = {"id": f"E0_baseline_resume_control_{scaled_lr_name(selected_lr_scale)}_{train_iters}iter", "family": "E0_baseline_resume_control", "train_iters": train_iters, "small_object_factor": 1.0, "future_scales": {}, "train_aug": [], "lr_scale": selected_lr_scale}
            ckpt_path = checkpoint_path_for_exp(exp_cfg)
            if ckpt_path.exists():
                train_summary = {"peak_gpu_memory_mb": None, "mean_iter_time_sec": None}
            else:
                ckpt_path, train_rows, train_summary = smoke_and_train(exp_cfg, args.samples_per_gpu, args.workers_per_gpu, args.seed, stage_logger)
                e0_train_rows.extend(train_rows)
            selected_e0_ckpt = ckpt_path
            eval_rows, _, _ = evaluate_checkpoint(ckpt_path, exp_cfg["id"], eval_union, DEFAULT_HORIZONS, capture_reliability=False)
            cmp = compare_against_reference(eval_rows, ref_eval_rows, exp_cfg["id"])
            agg_rows = summarize_compare_by_subset(cmp["deltas"], eval_subsets, exp_cfg["id"])
            e0_summary_rows.extend([{**row, "lr_scale": selected_lr_scale, "train_iters": train_iters, "checkpoint_path": str(ckpt_path), "peak_gpu_memory_mb": train_summary["peak_gpu_memory_mb"], "mean_iter_time_sec": train_summary["mean_iter_time_sec"]} for row in agg_rows])
            safe_cuda_cleanup()
        stage_logger.done("phase1_lr_stability_audit", selected_lr_scale=selected_lr_scale)

        lr_audit = {
            "selected_lr_scale": selected_lr_scale,
            "longest_e0_iter": max(e0_iter_budgets) if e0_iter_budgets else None,
            "samples_per_gpu": args.samples_per_gpu,
            "workers_per_gpu": args.workers_per_gpu,
            "lr_variants_tested": lr_scales,
        }
        write_json(REPORTS_DIR / "lr_stability_audit.json", lr_audit)
        write_csv(REPORTS_DIR / "e0_control_train_curve.csv", e0_train_rows)
        write_csv(REPORTS_DIR / "e0_control_eval_by_iter.csv", e0_summary_rows)
        write_csv(
            REPORTS_DIR / "training_time_memory_profile.csv",
            [
                {
                    "experiment_id": r.get("experiment_id") or r.get("checkpoint_name") or "unknown",
                    "lr_scale": r.get("lr_scale"),
                    "train_iters": r.get("train_iters"),
                    "peak_gpu_memory_mb": r.get("peak_gpu_memory_mb"),
                    "mean_iter_time_sec": r.get("mean_iter_time_sec"),
                }
                for r in e0_summary_rows
                if "peak_gpu_memory_mb" in r
            ],
        )

    targeted_defs = build_targeted_experiment_defs(targeted_iters, [0.1, 0.03])
    targeted_train_rows: list[dict[str, Any]] = []
    targeted_summary_rows: list[dict[str, Any]] = []
    targeted_ckpts: dict[str, Path] = {}
    stage_logger.start("phase3_targeted_runs", experiment_count=len(targeted_defs))
    for idx, exp_cfg in enumerate(targeted_defs, start=1):
        try:
            ckpt_path = checkpoint_path_for_exp(exp_cfg)
            if ckpt_path.exists():
                train_rows = []
                train_summary = {"peak_gpu_memory_mb": None, "mean_iter_time_sec": None}
                targeted_ckpts[exp_cfg["id"]] = ckpt_path
            else:
                ckpt_path, train_rows, train_summary = smoke_and_train(exp_cfg, args.samples_per_gpu, args.workers_per_gpu, args.seed, stage_logger)
                targeted_train_rows.extend(train_rows)
                targeted_ckpts[exp_cfg["id"]] = ckpt_path
            eval_rows, _, _ = evaluate_checkpoint(ckpt_path, exp_cfg["id"], eval_union, DEFAULT_HORIZONS, capture_reliability=False)
            safe_cuda_cleanup()
            cmp = compare_against_reference(eval_rows, ref_eval_rows, exp_cfg["id"])
            agg_rows = summarize_compare_by_subset(cmp["deltas"], eval_subsets, exp_cfg["id"])
            agg_lookup = {(r["subset_name"], r["perturbation_id"]): r for r in agg_rows}
            targeted_summary_rows.append({
                "experiment_id": exp_cfg["id"],
                "family": exp_cfg["family"],
                "lr_scale": exp_cfg["lr_scale"],
                "train_iters": exp_cfg["train_iters"],
                "checkpoint_path": str(ckpt_path),
                "peak_gpu_memory_mb": train_summary["peak_gpu_memory_mb"],
                "mean_iter_time_sec": train_summary["mean_iter_time_sec"],
                "clean_occupied_iou_delta": agg_lookup.get(("eval_core_20", "A0_clean"), {}).get("mean_occupied_iou_delta"),
                "clean_semantic_miou_delta": agg_lookup.get(("eval_core_20", "A0_clean"), {}).get("mean_semantic_miou_delta"),
                "a10_small_object_false_free_delta": agg_lookup.get(("eval_stress_20", "A10_drop_front_triplet"), {}).get("mean_small_object_false_free_delta"),
                "a10_new_visible_recall_delta": agg_lookup.get(("eval_stress_20", "A10_drop_front_triplet"), {}).get("mean_new_visible_recall_delta"),
                "a10_front_sector_false_free_delta": agg_lookup.get(("eval_stress_20", "A10_drop_front_triplet"), {}).get("mean_front_sector_false_free_delta"),
                "a1_front_sector_false_free_delta": agg_lookup.get(("eval_stress_20", "A1_drop_cam_front"), {}).get("mean_front_sector_false_free_delta"),
                "c4_false_occupied_delta": agg_lookup.get(("eval_stress_20", "C4_motion_blur_9"), {}).get("mean_false_occupied_rate_delta"),
                "c4_pred_gt_ratio_delta": agg_lookup.get(("eval_stress_20", "C4_motion_blur_9"), {}).get("mean_pred_gt_occupied_ratio_delta"),
            })
        except Exception as exc:
            safe_cuda_cleanup()
            targeted_summary_rows.append(failure_row(exp_cfg, exc))
        stage_logger.progress("phase3_targeted_runs", idx, len(targeted_defs), experiment_id=exp_cfg["id"])
    stage_logger.done("phase3_targeted_runs", completed=len(targeted_defs))

    write_csv(REPORTS_DIR / "e4_future_horizon_reweight_train.csv", [r for r in targeted_train_rows if str(r["experiment_id"]).startswith("E4_")])
    write_csv(REPORTS_DIR / "e8_motion_blur_aug_train.csv", [r for r in targeted_train_rows if str(r["experiment_id"]).startswith("E8_")])
    write_csv(REPORTS_DIR / "e1_small_object_reweight_scaled_train.csv", [r for r in targeted_train_rows if str(r["experiment_id"]).startswith("E1_")])
    write_csv(REPORTS_DIR / "e4_future_horizon_reweight_eval.csv", summarize_family_metric(targeted_summary_rows, "E4_future_horizon_reweight"))
    write_csv(REPORTS_DIR / "e8_motion_blur_aug_eval.csv", summarize_family_metric(targeted_summary_rows, "E8_motion_blur_aug_light"))
    write_csv(REPORTS_DIR / "e1_small_object_reweight_scaled_eval.csv", summarize_family_metric(targeted_summary_rows, "E1_small_object_reweight_scaled"))
    write_md(REPORTS_DIR / "e4_future_horizon_reweight_summary.md", json.dumps(normalize_export(summarize_family_metric(targeted_summary_rows, "E4_future_horizon_reweight")), indent=2, ensure_ascii=False))
    write_md(REPORTS_DIR / "e8_motion_blur_aug_summary.md", json.dumps(normalize_export(summarize_family_metric(targeted_summary_rows, "E8_motion_blur_aug_light")), indent=2, ensure_ascii=False))
    write_md(REPORTS_DIR / "e1_small_object_reweight_scaled_summary.md", json.dumps(normalize_export(summarize_family_metric(targeted_summary_rows, "E1_small_object_reweight_scaled")), indent=2, ensure_ascii=False))

    best_future = max(summarize_family_metric(targeted_summary_rows, "E4_future_horizon_reweight"), key=lambda r: (float(r.get("a10_new_visible_recall_delta") or 0.0) - float(r.get("a10_front_sector_false_free_delta") or 0.0)), default=None)
    best_blur = max(summarize_family_metric(targeted_summary_rows, "E8_motion_blur_aug_light"), key=lambda r: max(0.0, -float(r.get("a10_front_sector_false_free_delta") or 0.0)), default=None)
    best_small = max(summarize_family_metric(targeted_summary_rows, "E1_small_object_reweight_scaled"), key=lambda r: max(0.0, -float(r.get("a10_small_object_false_free_delta") or 0.0)), default=None)

    combined_summary_rows: list[dict[str, Any]] = []
    combined_ckpts: dict[str, Path] = {}
    if not args.disable_combined and best_future and is_safe_candidate(best_future):
        combined_defs = build_combined_exp(combined_iters, float(best_future["lr_scale"]), float(best_small["lr_scale"]) if best_small and is_safe_candidate(best_small) else None, float(best_blur["lr_scale"]) if best_blur and is_safe_candidate(best_blur) else None)
        for exp_cfg in combined_defs:
            ckpt_path, train_rows, train_summary = smoke_and_train(exp_cfg, args.samples_per_gpu, args.workers_per_gpu, args.seed, stage_logger)
            eval_rows, _, _ = evaluate_checkpoint(ckpt_path, exp_cfg["id"], eval_union, DEFAULT_HORIZONS, capture_reliability=False)
            cmp = compare_against_reference(eval_rows, ref_eval_rows, exp_cfg["id"])
            agg_rows = summarize_compare_by_subset(cmp["deltas"], eval_subsets, exp_cfg["id"])
            agg_lookup = {(r["subset_name"], r["perturbation_id"]): r for r in agg_rows}
            combined_summary_rows.append({
                "experiment_id": exp_cfg["id"],
                "family": exp_cfg["family"],
                "lr_scale": exp_cfg["lr_scale"],
                "train_iters": exp_cfg["train_iters"],
                "checkpoint_path": str(ckpt_path),
                "peak_gpu_memory_mb": train_summary["peak_gpu_memory_mb"],
                "mean_iter_time_sec": train_summary["mean_iter_time_sec"],
                "clean_occupied_iou_delta": agg_lookup.get(("eval_core_20", "A0_clean"), {}).get("mean_occupied_iou_delta"),
                "clean_semantic_miou_delta": agg_lookup.get(("eval_core_20", "A0_clean"), {}).get("mean_semantic_miou_delta"),
                "a10_small_object_false_free_delta": agg_lookup.get(("eval_stress_20", "A10_drop_front_triplet"), {}).get("mean_small_object_false_free_delta"),
                "a10_new_visible_recall_delta": agg_lookup.get(("eval_stress_20", "A10_drop_front_triplet"), {}).get("mean_new_visible_recall_delta"),
                "a10_front_sector_false_free_delta": agg_lookup.get(("eval_stress_20", "A10_drop_front_triplet"), {}).get("mean_front_sector_false_free_delta"),
                "a1_front_sector_false_free_delta": agg_lookup.get(("eval_stress_20", "A1_drop_cam_front"), {}).get("mean_front_sector_false_free_delta"),
                "c4_false_occupied_delta": agg_lookup.get(("eval_stress_20", "C4_motion_blur_9"), {}).get("mean_false_occupied_rate_delta"),
                "c4_pred_gt_ratio_delta": agg_lookup.get(("eval_stress_20", "C4_motion_blur_9"), {}).get("mean_pred_gt_occupied_ratio_delta"),
            })
            combined_ckpts[exp_cfg["id"]] = ckpt_path
    write_csv(REPORTS_DIR / "combined_safe_candidates_eval.csv", combined_summary_rows)
    write_md(REPORTS_DIR / "combined_safe_candidates_summary.md", json.dumps(normalize_export(combined_summary_rows), indent=2, ensure_ascii=False))

    retest_targets: list[tuple[str, Path]] = []
    if selected_e0_ckpt is not None:
        retest_targets.append((f"E0_baseline_resume_control_{scaled_lr_name(selected_lr_scale)}_{max(e0_iter_budgets)}iter", selected_e0_ckpt))
    for candidate in [best_future, best_blur, best_small, max(combined_summary_rows, key=lambda r: max(0.0, -float(r.get("a10_front_sector_false_free_delta") or 0.0)), default=None)]:
        if candidate is None:
            continue
        exp_id = str(candidate["experiment_id"])
        ckpt = targeted_ckpts.get(exp_id) or combined_ckpts.get(exp_id)
        if ckpt is not None and (exp_id, ckpt) not in retest_targets:
            retest_targets.append((exp_id, ckpt))

    retest_rows_all: list[dict[str, Any]] = []
    retest_topk_all: list[dict[str, Any]] = []
    baseline_case = None
    candidate_case = None
    stage_logger.start("phase8_reliability_retest", checkpoint_count=len(retest_targets))
    for idx, (name, ckpt) in enumerate(retest_targets, start=1):
        _, rel_rows, dbg = evaluate_checkpoint(ckpt, name, eval_subsets["eval_stress_20"][: min(5, len(eval_subsets["eval_stress_20"]))], DEFAULT_HORIZONS, capture_reliability=True)
        retest_rows_all.extend(rel_rows)
        retest_topk_all.extend(dbg.get("topk_rows", []))
        rep = dbg.get("representative_case")
        if rep is not None and baseline_case is None:
            baseline_case = {"checkpoint_name": name, "perturbation_id": "A10_drop_front_triplet", "sample_index": eval_subsets["eval_stress_20"][0], "horizon_s": 6, **rep}
        if rep is not None and name != retest_targets[0][0]:
            candidate_case = {"checkpoint_name": name, "perturbation_id": "A10_drop_front_triplet", "sample_index": eval_subsets["eval_stress_20"][0], "horizon_s": 6, **rep}
        stage_logger.progress("phase8_reliability_retest", idx, len(retest_targets), checkpoint_name=name)
    stage_logger.done("phase8_reliability_retest", row_count=len(retest_rows_all))
    write_csv(REPORTS_DIR / "sw7_reliability_retest_scaled.csv", retest_rows_all)
    write_csv(REPORTS_DIR / "sw7_reliability_topk_scaled.csv", retest_topk_all)
    reliability_summary = {"checkpoint_names": [name for name, _ in retest_targets], "headline": "scaled candidates were re-evaluated with the same internal diagnostic reliability indicator used in SW-7"}
    write_md(REPORTS_DIR / "sw7_reliability_retest_scaled_summary.md", json.dumps(normalize_export(reliability_summary), indent=2, ensure_ascii=False))
    if baseline_case and candidate_case:
        render_reliability_before_after("sw81_best_candidate", baseline_case, candidate_case, FIGURES_DIR / "sw81_reliability_before_after_best.png")
        render_reliability_before_after("sw81_best_candidate_score", baseline_case, candidate_case, FIGURES_DIR / "sw81_score_alpha_before_after_best.png")

    all_summary_rows = targeted_summary_rows + combined_summary_rows
    decision_type = "S5_training_unstable"
    safe_claim = "the scaled training pipeline did not finish enough valid candidate runs to support a conclusion."
    next_action = "stabilize the scaled run matrix and rerun E4/E8 before making any targeted fine-tune claim."
    false_positive_headline = "no valid scaled candidate available"
    chosen = None
    candidates = [c for c in [best_future, best_blur, best_small, max(combined_summary_rows, key=lambda r: max(0.0, -float(r.get('a10_front_sector_false_free_delta') or 0.0)), default=None)] if c is not None]
    if candidates:
        chosen = max(candidates, key=lambda r: (float(r.get("a10_new_visible_recall_delta") or 0.0) * 4.0) + (max(0.0, -float(r.get("a10_front_sector_false_free_delta") or 0.0)) * 3.0) + (max(0.0, -float(r.get("a10_small_object_false_free_delta") or 0.0)) * 3.0) - (max(0.0, float(r.get("c4_false_occupied_delta") or 0.0)) * 6.0) - (max(0.0, -float(r.get("clean_occupied_iou_delta") or 0.0)) * 8.0))
    if chosen is not None:
        safe_clean = float(chosen.get("clean_occupied_iou_delta") or 0.0) >= -0.005
        safe_fp = float(chosen.get("c4_false_occupied_delta") or 0.0) <= 0.008 and float(chosen.get("c4_pred_gt_ratio_delta") or 0.0) <= 0.10
        better_front = min(float(chosen.get("a10_front_sector_false_free_delta") or 0.0), float(chosen.get("a1_front_sector_false_free_delta") or 0.0)) <= -0.03
        better_small = float(chosen.get("a10_small_object_false_free_delta") or 0.0) <= -0.03
        better_new = float(chosen.get("a10_new_visible_recall_delta") or 0.0) >= 0.03
        false_positive_headline = f"C4 false-occupied delta={float(chosen.get('c4_false_occupied_delta') or 0.0):+.4f}, pred_gt_ratio delta={float(chosen.get('c4_pred_gt_ratio_delta') or 0.0):+.4f}"
        if safe_clean and safe_fp and (better_front or better_small or better_new):
            decision_type = "S1_safe_targeted_improvement"
            safe_claim = "at least one scaled targeted candidate improved the target failure pattern without a clear clean collapse on the fixed subset protocol."
            next_action = "extend the winning recipe to 3000 iter and optional_eval_50 before any broader claim."
        elif (better_front or better_small or better_new) and (not safe_clean or not safe_fp):
            decision_type = "S2_partial_improvement_tradeoff"
            safe_claim = "scaled targeted gains appeared, but they came with false-positive expansion or clean drift, so the candidate is unsafe."
            next_action = "tighten the recipe and add density control before promoting any checkpoint."
        else:
            decision_type = "S3_no_clear_gain_after_scaled_training"
            safe_claim = "after scaled runs on fixed subsets, no candidate showed a clear targeted win that also stayed safe on clean and false-positive tradeoffs."
            next_action = "move to longer E4/E8 schedules or architecture-level changes instead of claiming improvement."
    if targeted_iters >= 3000 and all_summary_rows and all(not is_safe_candidate(r) for r in all_summary_rows):
        decision_type = "S4_architecture_limited_evidence_strengthened"
        safe_claim = "multiple scaled fine-tune routes still failed to yield a safe targeted gain, which strengthens the architecture-level bottleneck hypothesis in this setup."
        next_action = "shift effort to contributor/query architecture interventions guided by SW-7."
    decision_payload = {"decision_type": decision_type, "best_checkpoint": chosen["experiment_id"] if chosen else None, "false_positive_headline": false_positive_headline, "safe_claim": safe_claim, "next_action": next_action, "targeted_iters": targeted_iters, "samples_per_gpu": args.samples_per_gpu, "workers_per_gpu": args.workers_per_gpu}
    write_json(REPORTS_DIR / "sw81_scaled_finetune_decision.json", decision_payload)
    write_md(REPORTS_DIR / "sw81_scaled_finetune_decision.md", json.dumps(normalize_export(decision_payload), indent=2, ensure_ascii=False))

    if all_summary_rows:
        plot_delta_bars(all_summary_rows, "a10_small_object_false_free_delta", "A10 small-object false-free delta", "delta", FIGURES_DIR / "sw81_small_object_delta_bar.png")
        plot_delta_bars(all_summary_rows, "a10_new_visible_recall_delta", "A10 new-visible recall delta", "delta", FIGURES_DIR / "sw81_new_visible_delta_bar.png")
        plot_delta_bars(all_summary_rows, "a10_front_sector_false_free_delta", "A10 front-sector false-free delta", "delta", FIGURES_DIR / "sw81_front_sector_delta_bar.png")

    report_text, report_json = make_report(decision_payload, lr_audit, eval_subsets, all_summary_rows, reliability_summary)
    write_md(REPORTS_DIR / "stage_sw81_scaled_finetune_report.md", report_text)
    write_json(REPORTS_DIR / "stage_sw81_scaled_finetune_report.json", report_json)
    run_manifest["status"] = "completed"
    run_manifest["selected_lr_scale"] = selected_lr_scale
    run_manifest["retest_targets"] = [name for name, _ in retest_targets]
    write_json(REPORTS_DIR / "sw81_run_manifest.json", run_manifest)


if __name__ == "__main__":
    main()

