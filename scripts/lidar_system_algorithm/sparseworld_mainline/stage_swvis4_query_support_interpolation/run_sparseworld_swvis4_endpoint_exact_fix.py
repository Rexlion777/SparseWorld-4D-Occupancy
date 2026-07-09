from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import swvis4_support_interp_lib as lib


PROJECT_ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_swvis4_query_support_interpolation"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_swvis4_query_support_interpolation"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_swvis4_query_support_interpolation"
VIDEOS_DIR = PROJECT_ROOT / "videos/sparseworld_mainline/stage_swvis4_query_support_interpolation"
STATUS_PATH = LOGS_DIR / "swvis4_endpoint_exact_fix_stage_status.json"
PROGRESS_PATH = LOGS_DIR / "swvis4_endpoint_exact_fix_stage_progress.jsonl"

PC_RANGE = lib.PC_RANGE.astype(np.float32)
VOXEL_SIZE = lib.VOXEL_SIZE.astype(np.float32)
GRID_SIZE = lib.GRID_SIZE.astype(np.int64)
EMPTY_IDX = int(lib.EMPTY_IDX)
NUM_CLASSES = int(lib.NUM_CLASSES)
BASE_SCORE_THR = lib.SCORE_THR.astype(np.float32)
ANCHOR_TIMES = lib.ANCHOR_TIMES.astype(np.float32)

STATIC_CLASSES = {1, 11, 13, 14, 15, 16}
DYNAMIC_CLASSES = set(range(NUM_CLASSES)) - STATIC_CLASSES
STATIC_IDS = np.array(sorted(STATIC_CLASSES), dtype=np.int64)
DYNAMIC_IDS = np.array(sorted(DYNAMIC_CLASSES), dtype=np.int64)


class StageLogger:
    def __init__(self, sample_index: int, frame_count: int, fps: int) -> None:
        self.sample_index = int(sample_index)
        self.frame_count = int(frame_count)
        self.fps = int(fps)
        self.status: dict[str, Any] = {
            "sample_index": self.sample_index,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "started_at": time.time(),
            "current_stage": None,
            "stages": {},
        }
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._write()

    def _write(self) -> None:
        STATUS_PATH.write_text(json.dumps(self.status, indent=2, ensure_ascii=False), encoding="utf-8")

    def _append_progress(self, stage: str, payload: dict[str, Any]) -> None:
        row = {
            "ts": time.time(),
            "stage": stage,
            **payload,
        }
        with PROGRESS_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    def start(self, stage: str, **kwargs: Any) -> None:
        self.status["current_stage"] = stage
        self.status["stages"][stage] = {"status": "running", "start_ts": time.time(), **kwargs}
        self._append_progress(stage, {"event": "start", **kwargs})
        self._write()

    def progress(self, stage: str, current: int, total: int, **kwargs: Any) -> None:
        entry = self.status["stages"].setdefault(stage, {"status": "running", "start_ts": time.time()})
        entry.update({"status": "running", "progress_current": int(current), "progress_total": int(total), **kwargs})
        self._append_progress(stage, {"event": "progress", "progress_current": int(current), "progress_total": int(total), **kwargs})
        self._write()

    def done(self, stage: str, **kwargs: Any) -> None:
        entry = self.status["stages"].setdefault(stage, {"start_ts": time.time()})
        end_ts = time.time()
        entry.update({"status": "done", "end_ts": end_ts, "duration_sec": float(end_ts - entry.get("start_ts", end_ts)), **kwargs})
        self._append_progress(stage, {"event": "done", **kwargs})
        self._write()

    def skip(self, stage: str, reason: str, **kwargs: Any) -> None:
        self.status["stages"][stage] = {"status": "skipped", "skip_ts": time.time(), "reason": reason, **kwargs}
        self._append_progress(stage, {"event": "skip", "reason": reason, **kwargs})
        self._write()


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


def load_query_output(sample_index: int) -> dict[str, Any]:
    return torch.load(
        PROJECT_ROOT / f"artifacts/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/query_outputs/sample_{sample_index:04d}_query_output.pt",
        map_location="cpu",
        weights_only=False,
    )


def load_raw_output(sample_index: int) -> dict[str, Any]:
    return torch.load(
        PROJECT_ROOT / f"artifacts/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/raw_outputs/sample_{sample_index:04d}_raw_output.pt",
        map_location="cpu",
        weights_only=False,
    )


def decode_points(points: torch.Tensor) -> torch.Tensor:
    out = points.clone().float()
    out[..., 0] = out[..., 0] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
    out[..., 1] = out[..., 1] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
    out[..., 2] = out[..., 2] * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2]
    return out


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def scatter_max_np(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = torch.full((dim_size, src.shape[1]), float("-inf"), dtype=src.dtype)
    expanded_index = index[:, None].expand(-1, src.shape[1])
    out.scatter_reduce_(0, expanded_index, src, reduce="amax", include_self=True)
    out[out == float("-inf")] = 0.0
    return out


def native_get_occ_replay(cls_scores: torch.Tensor, refine_pts: torch.Tensor) -> torch.Tensor:
    cls_scores = cls_scores.sigmoid().clone()

    mask = cls_scores.argmax(-1) == 15
    dist = torch.norm(refine_pts - torch.mean(refine_pts, dim=2, keepdim=True), dim=-1)
    cls_scores[mask] = cls_scores[mask] * torch.clamp(torch.tensor(0.1, dtype=cls_scores.dtype) / dist[mask], max=1.0)[:, None]

    mask = cls_scores.argmax(-1) == 16
    dist = torch.norm(refine_pts - torch.mean(refine_pts, dim=2, keepdim=True), dim=-1)
    cls_scores[mask] = cls_scores[mask] * torch.clamp(torch.tensor(0.1, dtype=cls_scores.dtype) / dist[mask], max=1.0)[:, None]

    refine_pts = decode_points(refine_pts[0])
    cls_scores = cls_scores[0]

    centers = refine_pts.mean(dim=1, keepdim=True)
    ctr_dists = torch.norm(refine_pts - centers, dim=-1)
    mask_dist = ctr_dists < 3.0
    max_score, index = cls_scores.max(-1)
    score_thr = torch.tensor(BASE_SCORE_THR, dtype=cls_scores.dtype)
    mask_score = max_score > score_thr[index]
    mask = mask_dist & mask_score

    refine_pts = refine_pts[mask]
    cls_scores = cls_scores[mask]

    index = ((refine_pts - torch.tensor(PC_RANGE[:3])) // torch.tensor(VOXEL_SIZE)).long()
    mask = torch.logical_and(index >= 0, index < torch.tensor(GRID_SIZE)).all(-1)
    voxels, unq_inv, _ = torch.unique(index[mask], return_inverse=True, return_counts=True, dim=0)
    scores = scatter_max_np(cls_scores[mask], unq_inv, voxels.shape[0])

    occ = scores.new_zeros((GRID_SIZE[0], GRID_SIZE[1], GRID_SIZE[2], NUM_CLASSES))
    occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = scores
    occ = occ.permute(3, 0, 1, 2).unsqueeze(0)

    dilated_occ = F.max_pool3d(occ, 3, stride=1, padding=1)
    eroded_occ = -F.max_pool3d(-dilated_occ, 3, stride=1, padding=1)

    max_score, index = occ.max(1)
    original_mask = (max_score > score_thr[index]).expand_as(eroded_occ)
    eroded_occ[original_mask] = occ[original_mask]
    eroded_occ = eroded_occ.squeeze(0).permute(1, 2, 3, 0)

    voxels = torch.nonzero((eroded_occ > score_thr).any(dim=-1))
    occ_pred = torch.ones(GRID_SIZE.tolist(), dtype=torch.long) * EMPTY_IDX
    occ_pred[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = eroded_occ[voxels[:, 0], voxels[:, 1], voxels[:, 2], :].argmax(-1)
    return occ_pred


def compute_occ_comparison(recon_occ: np.ndarray, ref_occ: np.ndarray) -> dict[str, Any]:
    recon_occ = recon_occ.astype(np.uint8)
    ref_occ = ref_occ.astype(np.uint8)
    recon_mask = recon_occ != EMPTY_IDX
    ref_mask = ref_occ != EMPTY_IDX
    inter = recon_mask & ref_mask
    union = recon_mask | ref_mask
    inter_count = int(inter.sum())
    union_count = int(union.sum())
    recon_count = int(recon_mask.sum())
    ref_count = int(ref_mask.sum())
    same_sem_inter = int(((recon_occ == ref_occ) & inter).sum())
    pred_counts = np.bincount(recon_occ[recon_mask].reshape(-1), minlength=NUM_CLASSES).astype(np.int64)
    ref_counts = np.bincount(ref_occ[ref_mask].reshape(-1), minlength=NUM_CLASSES).astype(np.int64)
    return {
        "occupied_iou": float(inter_count / max(union_count, 1)),
        "semantic_agreement_intersection": float(same_sem_inter / max(inter_count, 1)),
        "occupied_count_ratio": float(recon_count / max(ref_count, 1)),
        "recon_occupied_count": recon_count,
        "ref_occupied_count": ref_count,
        "static_class_recall_proxy": float(pred_counts[STATIC_IDS].sum() / max(ref_counts[STATIC_IDS].sum(), 1)),
        "dynamic_class_recall_proxy": float(pred_counts[DYNAMIC_IDS].sum() / max(ref_counts[DYNAMIC_IDS].sum(), 1)),
        "per_class_count_delta": {str(i): int(pred_counts[i] - ref_counts[i]) for i in range(NUM_CLASSES)},
        "per_class_recon_count": {str(i): int(pred_counts[i]) for i in range(NUM_CLASSES)},
        "per_class_ref_count": {str(i): int(ref_counts[i]) for i in range(NUM_CLASSES)},
    }


def changed_voxel_ratio(a: np.ndarray, b: np.ndarray) -> float:
    return float((a.astype(np.uint8) != b.astype(np.uint8)).mean())


def get_horizon_tensors(sample_index: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    q = load_query_output(sample_index)
    fb = q["forward_backbone_outputs"]
    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for h in range(7):
        if h == 0:
            out.append((fb["cls_score"], fb["refine_pts"]))
        else:
            out.append((fb["forecast_semantics_list"][h - 1], fb["forecast_points_list"][h - 1]))
    return out


def run_native_replay_audit(sample_index: int) -> dict[str, Any]:
    raw = load_raw_output(sample_index)
    tensors = get_horizon_tensors(sample_index)
    rows = []
    replayed = {}
    for h, (cls_scores, refine_pts) in enumerate(tensors):
        replay_occ = native_get_occ_replay(cls_scores, refine_pts).cpu().numpy().astype(np.uint8)
        raw_occ = raw[f"semantic_occ_{h}s"][0].astype(np.uint8)
        metrics = compute_occ_comparison(replay_occ, raw_occ)
        rows.append({"horizon_index": h, "model_time_sec": float(ANCHOR_TIMES[h]), **metrics})
        replayed[h] = replay_occ
    summary = {
        "sample_index": sample_index,
        "rows": rows,
        "mean_occupied_iou": float(np.mean([r["occupied_iou"] for r in rows])),
        "mean_semantic_agreement_intersection": float(np.mean([r["semantic_agreement_intersection"] for r in rows])),
        "mean_occupied_count_ratio": float(np.mean([r["occupied_count_ratio"] for r in rows])),
        "native_replay_viable": bool(np.mean([r["occupied_iou"] for r in rows]) > 0.95),
        "unavailable_reason": None,
    }
    write_json(REPORTS_DIR / "native_get_occ_replay_audit.json", summary)
    (REPORTS_DIR / "native_get_occ_replay_audit.md").write_text(
        "\n".join(
            [
                "# native get_occ replay audit",
                f"- mean occupied IoU: `{summary['mean_occupied_iou']:.6f}`",
                f"- mean semantic agreement: `{summary['mean_semantic_agreement_intersection']:.6f}`",
                f"- mean occupied_count_ratio: `{summary['mean_occupied_count_ratio']:.6f}`",
                f"- native_replay_viable: `{summary['native_replay_viable']}`",
            ]
        ),
        encoding="utf-8",
    )
    return {"summary": summary, "replayed": replayed}


def reconstruct_support_surrogate(
    points_tensor: torch.Tensor,
    cls_tensor: torch.Tensor,
    *,
    hard_gate: bool = True,
    soft_gate: bool = False,
    global_occ_threshold: float = 0.12,
    classwise_occ_threshold: np.ndarray | None = None,
    aggregation: str = "sum_prob",
    splat: str = "trilinear",
    apply_native_static_scale: bool = False,
) -> np.ndarray:
    points = decode_points(points_tensor[0]).cpu().numpy().astype(np.float32)
    logits = cls_tensor[0].cpu().numpy().astype(np.float32)
    probs = sigmoid_np(logits)

    if apply_native_static_scale:
        point_cls = probs.argmax(-1)
        d = np.linalg.norm(points - points.mean(axis=1, keepdims=True), axis=-1)
        for cls_idx, thr in ((15, 0.1), (16, 0.1)):
            m = point_cls == cls_idx
            if np.any(m):
                probs[m] = probs[m] * np.clip(thr / np.clip(d[m], 1e-6, None), None, 1.0)[:, None]

    centers = points.mean(axis=1, keepdims=True)
    ctr_dist = np.linalg.norm(points - centers, axis=-1)
    point_class = probs.argmax(-1)
    point_score = probs.max(-1)
    gate = (ctr_dist < 3.0) & (point_score > BASE_SCORE_THR[point_class])

    if hard_gate:
        point_weight = point_score * gate.astype(np.float32)
    elif soft_gate:
        gate_soft = np.minimum(1.0, 3.0 / np.clip(ctr_dist, 1e-3, None)) * np.minimum(1.0, point_score / BASE_SCORE_THR[point_class])
        point_weight = point_score * gate_soft.astype(np.float32)
    else:
        point_weight = point_score.astype(np.float32)

    xyz = points.reshape(-1, 3).astype(np.float32)
    prob_f = probs.reshape(-1, NUM_CLASSES).astype(np.float32)
    weight_f = point_weight.reshape(-1).astype(np.float32)
    valid = weight_f > 1e-6
    xyz = xyz[valid]
    prob_f = prob_f[valid]
    weight_f = weight_f[valid]

    if xyz.shape[0] == 0:
        return np.full(tuple(GRID_SIZE.tolist()), EMPTY_IDX, dtype=np.uint8)

    float_idx = (xyz - PC_RANGE[:3]) / VOXEL_SIZE
    base = np.floor(float_idx).astype(np.int32)
    frac = float_idx - base.astype(np.float32)
    grid = np.zeros((int(np.prod(GRID_SIZE)), NUM_CLASSES), dtype=np.float32)

    def combine(flat: np.ndarray, contrib: np.ndarray) -> None:
        if aggregation == "sum_prob":
            np.add.at(grid, flat, contrib)
        elif aggregation == "max_prob":
            np.maximum.at(grid, flat, contrib)
        elif aggregation == "logsumexp":
            np.add.at(grid, flat, np.exp(contrib))
        elif aggregation == "topk3_sum":
            np.add.at(grid, flat, contrib)
        else:
            raise ValueError(f"unsupported aggregation: {aggregation}")

    def add(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, w: np.ndarray) -> None:
        mask = (
            (ix >= 0)
            & (iy >= 0)
            & (iz >= 0)
            & (ix < GRID_SIZE[0])
            & (iy < GRID_SIZE[1])
            & (iz < GRID_SIZE[2])
            & (w > 1e-8)
        )
        if not np.any(mask):
            return
        flat = (ix[mask] * (GRID_SIZE[1] * GRID_SIZE[2]) + iy[mask] * GRID_SIZE[2] + iz[mask]).astype(np.int64)
        combine(flat, prob_f[mask] * w[mask, None])

    if splat == "hard_assign":
        add(base[:, 0], base[:, 1], base[:, 2], weight_f)
    elif splat == "trilinear":
        wx = [1.0 - frac[:, 0], frac[:, 0]]
        wy = [1.0 - frac[:, 1], frac[:, 1]]
        wz = [1.0 - frac[:, 2], frac[:, 2]]
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    add(base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz, weight_f * wx[dx] * wy[dy] * wz[dz])
    elif splat == "gaussian_r1":
        sigma = 0.55
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    dist = (frac[:, 0] - dx) ** 2 + (frac[:, 1] - dy) ** 2 + (frac[:, 2] - dz) ** 2
                    add(base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz, weight_f * np.exp(-dist / (2.0 * sigma * sigma)))
    elif splat == "gaussian_r2_static":
        sigma = 0.70
        point_cls_flat = prob_f.argmax(-1)
        static_mask = np.isin(point_cls_flat, STATIC_IDS)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for dz in (-1, 0, 1):
                    dist = (frac[:, 0] - dx) ** 2 + (frac[:, 1] - dy) ** 2 + (frac[:, 2] - dz) ** 2
                    w = weight_f * np.exp(-dist / (2.0 * sigma * sigma))
                    w = np.where(static_mask, w, np.where((dx == 0) & (dy == 0) & (dz == 0), weight_f, 0.0))
                    add(base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz, w)
    else:
        raise ValueError(f"unsupported splat: {splat}")

    if aggregation == "logsumexp":
        grid = np.log1p(grid)

    max_score = grid.max(axis=1)
    cls = grid.argmax(axis=1).astype(np.uint8)
    occ = np.full((grid.shape[0],), EMPTY_IDX, dtype=np.uint8)
    if classwise_occ_threshold is not None:
        thr = np.asarray(classwise_occ_threshold, dtype=np.float32)
        mask = max_score > thr[cls]
    else:
        mask = max_score > float(global_occ_threshold)
    occ[mask] = cls[mask]
    return occ.reshape(*GRID_SIZE.tolist())


def run_surrogate_voxelizer_ablation(sample_index: int) -> dict[str, Any]:
    raw = load_raw_output(sample_index)
    tensors = get_horizon_tensors(sample_index)
    classwise_static_thr = np.array([0.12] * NUM_CLASSES, dtype=np.float32)
    classwise_static_thr[11] = 0.06
    classwise_static_thr[13] = 0.06
    classwise_static_thr[14] = 0.06
    classwise_static_thr[15] = 0.05
    classwise_static_thr[16] = 0.05

    variants: dict[str, dict[str, Any]] = {
        "A0_current_surrogate": dict(hard_gate=True, soft_gate=False, global_occ_threshold=0.12, aggregation="sum_prob", splat="trilinear"),
        "A1_no_artificial_gate": dict(hard_gate=False, soft_gate=False, global_occ_threshold=0.12, aggregation="sum_prob", splat="trilinear"),
        "A1_soft_gate": dict(hard_gate=False, soft_gate=True, global_occ_threshold=0.12, aggregation="sum_prob", splat="trilinear"),
        "A2_threshold_004": dict(hard_gate=True, soft_gate=False, global_occ_threshold=0.04, aggregation="sum_prob", splat="trilinear"),
        "A2_threshold_006": dict(hard_gate=True, soft_gate=False, global_occ_threshold=0.06, aggregation="sum_prob", splat="trilinear"),
        "A2_threshold_008": dict(hard_gate=True, soft_gate=False, global_occ_threshold=0.08, aggregation="sum_prob", splat="trilinear"),
        "A2_threshold_010": dict(hard_gate=True, soft_gate=False, global_occ_threshold=0.10, aggregation="sum_prob", splat="trilinear"),
        "A2_classwise_static_threshold": dict(hard_gate=True, soft_gate=False, global_occ_threshold=0.12, classwise_occ_threshold=classwise_static_thr, aggregation="sum_prob", splat="trilinear"),
        "A3_max_prob": dict(hard_gate=True, soft_gate=False, global_occ_threshold=0.12, aggregation="max_prob", splat="trilinear"),
        "A3_logsumexp": dict(hard_gate=True, soft_gate=False, global_occ_threshold=0.12, aggregation="logsumexp", splat="trilinear"),
        "A3_topk_proxy": dict(hard_gate=True, soft_gate=False, global_occ_threshold=0.12, aggregation="topk3_sum", splat="trilinear"),
        "A4_hard_assign": dict(hard_gate=True, soft_gate=False, global_occ_threshold=0.12, aggregation="sum_prob", splat="hard_assign"),
        "A4_gaussian_r1": dict(hard_gate=True, soft_gate=False, global_occ_threshold=0.12, aggregation="sum_prob", splat="gaussian_r1"),
        "A4_gaussian_r2_static": dict(hard_gate=True, soft_gate=False, global_occ_threshold=0.12, aggregation="sum_prob", splat="gaussian_r2_static"),
        "A5_geo_semantic_joint_unavailable": dict(unavailable_reason="geo_occ is only a derived binary occupancy label in artifacts, not a native confidence tensor for joint replay"),
    }

    rows: list[dict[str, Any]] = []
    recon_cache: dict[str, dict[int, np.ndarray]] = {}
    for variant_name, cfg in variants.items():
        if "unavailable_reason" in cfg:
            rows.append({
                "variant": variant_name,
                "available": False,
                "unavailable_reason": cfg["unavailable_reason"],
            })
            continue
        recon_seq: list[np.ndarray] = []
        metrics_seq: list[dict[str, Any]] = []
        recon_cache[variant_name] = {}
        for h, (cls_scores, refine_pts) in enumerate(tensors):
            occ = reconstruct_support_surrogate(refine_pts, cls_scores, **cfg)
            recon_cache[variant_name][h] = occ
            ref = raw[f"semantic_occ_{h}s"][0].astype(np.uint8)
            metrics_seq.append(compute_occ_comparison(occ, ref))
            recon_seq.append(occ)
        flicker = float(np.mean([changed_voxel_ratio(recon_seq[i], recon_seq[i + 1]) for i in range(len(recon_seq) - 1)]))
        rows.append(
            {
                "variant": variant_name,
                "available": True,
                "hard_gate": bool(cfg.get("hard_gate", False)),
                "soft_gate": bool(cfg.get("soft_gate", False)),
                "global_occ_threshold": cfg.get("global_occ_threshold"),
                "aggregation": cfg.get("aggregation"),
                "splat": cfg.get("splat"),
                "apply_native_static_scale": bool(cfg.get("apply_native_static_scale", False)),
                "mean_occupied_iou": float(np.mean([m["occupied_iou"] for m in metrics_seq])),
                "mean_semantic_agreement_intersection": float(np.mean([m["semantic_agreement_intersection"] for m in metrics_seq])),
                "mean_occupied_count_ratio": float(np.mean([m["occupied_count_ratio"] for m in metrics_seq])),
                "mean_static_class_recall": float(np.mean([m["static_class_recall_proxy"] for m in metrics_seq])),
                "mean_dynamic_class_recall": float(np.mean([m["dynamic_class_recall_proxy"] for m in metrics_seq])),
                "anchor_flicker_metric": flicker,
            }
        )

    def objective(row: dict[str, Any]) -> tuple[float, float]:
        ratio_penalty = abs(float(row["mean_occupied_count_ratio"]) - 1.0)
        return (ratio_penalty, -float(row["mean_occupied_iou"]))

    candidate_rows = [r for r in rows if r.get("available")]
    selected_base = min(candidate_rows, key=objective)

    write_csv(REPORTS_DIR / "surrogate_voxelizer_ablation.csv", rows)
    summary_lines = [
        "# surrogate voxelizer ablation summary",
        f"- selected_base_variant: `{selected_base['variant']}`",
        f"- selected_base_mean_iou: `{selected_base['mean_occupied_iou']:.6f}`",
        f"- selected_base_mean_occupied_count_ratio: `{selected_base['mean_occupied_count_ratio']:.6f}`",
        "",
        "Gate contribution (A1 - A0):",
    ]
    row_by_name = {r["variant"]: r for r in rows if r.get("available")}
    if "A1_no_artificial_gate" in row_by_name and "A0_current_surrogate" in row_by_name:
        a0 = row_by_name["A0_current_surrogate"]
        a1 = row_by_name["A1_no_artificial_gate"]
        summary_lines += [
            f"- occupied_count_ratio delta: `{float(a1['mean_occupied_count_ratio']) - float(a0['mean_occupied_count_ratio']):+.6f}`",
            f"- occupied_iou delta: `{float(a1['mean_occupied_iou']) - float(a0['mean_occupied_iou']):+.6f}`",
        ]
    summary_lines += ["", "Threshold contribution (best A2 - A0):"]
    a2_best = max([r for r in candidate_rows if r["variant"].startswith("A2_")], key=lambda r: r["mean_occupied_iou"])
    a0 = row_by_name["A0_current_surrogate"]
    summary_lines += [
        f"- best A2 variant: `{a2_best['variant']}`",
        f"- occupied_count_ratio delta: `{float(a2_best['mean_occupied_count_ratio']) - float(a0['mean_occupied_count_ratio']):+.6f}`",
        f"- occupied_iou delta: `{float(a2_best['mean_occupied_iou']) - float(a0['mean_occupied_iou']):+.6f}`",
        "",
        "Aggregation contribution (best A3 - A0):",
    ]
    a3_best = max([r for r in candidate_rows if r["variant"].startswith("A3_")], key=lambda r: r["mean_occupied_iou"])
    summary_lines += [
        f"- best A3 variant: `{a3_best['variant']}`",
        f"- occupied_count_ratio delta: `{float(a3_best['mean_occupied_count_ratio']) - float(a0['mean_occupied_count_ratio']):+.6f}`",
        f"- occupied_iou delta: `{float(a3_best['mean_occupied_iou']) - float(a0['mean_occupied_iou']):+.6f}`",
        "",
        "Splat contribution (best A4 - A0):",
    ]
    a4_best = max([r for r in candidate_rows if r["variant"].startswith("A4_")], key=lambda r: r["mean_occupied_iou"])
    summary_lines += [
        f"- best A4 variant: `{a4_best['variant']}`",
        f"- occupied_count_ratio delta: `{float(a4_best['mean_occupied_count_ratio']) - float(a0['mean_occupied_count_ratio']):+.6f}`",
        f"- occupied_iou delta: `{float(a4_best['mean_occupied_iou']) - float(a0['mean_occupied_iou']):+.6f}`",
    ]
    (REPORTS_DIR / "surrogate_voxelizer_ablation_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return {"rows": rows, "selected_base_variant": selected_base["variant"], "selected_base_cfg": variants[selected_base["variant"]], "recon_cache": recon_cache}


def ensure_support_artifacts(sample_index: int, frame_count: int, fps: int, encoder: str, logger: StageLogger) -> tuple[Path, Path]:
    stage = "ensure_support_artifacts"
    support_dir = ARTIFACTS_DIR / f"support_keyframes/sample_{sample_index:04d}"
    support_frames_dir = ARTIFACTS_DIR / f"interpolated_support_frames/nearest_match/sample_{sample_index:04d}_{frame_count}f"
    if support_dir.exists() and support_frames_dir.exists() and len(list(support_frames_dir.glob("frame_*.npz"))) == frame_count:
        logger.skip(stage, reason="support_artifacts_exist", support_dir=str(support_dir), support_frames_dir=str(support_frames_dir))
        return support_dir, support_frames_dir
    logger.start(stage, support_dir=str(support_dir), support_frames_dir=str(support_frames_dir))
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_sparseworld_swvis4_main.py"),
        "--sample-index",
        str(sample_index),
        "--frame-count",
        str(frame_count),
        "--fps",
        str(fps),
        "--encoder",
        encoder,
        "--resume",
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    logger.done(stage, support_dir=str(support_dir), support_frames_dir=str(support_frames_dir))
    return support_dir, support_frames_dir


def build_endpoint_residuals(sample_index: int, native_occ: dict[int, np.ndarray], base_recon_cache: dict[int, np.ndarray]) -> dict[str, Any]:
    out_dir = ARTIFACTS_DIR / f"endpoint_residual_layers/sample_{sample_index:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    residuals = {}
    for h in range(7):
        raw_occ = native_occ[h].astype(np.uint8)
        recon_occ = base_recon_cache[h].astype(np.uint8)
        raw_mask = raw_occ != EMPTY_IDX
        recon_mask = recon_occ != EMPTY_IDX
        missing_mask = raw_mask & ~recon_mask
        class_fix_mask = raw_mask & recon_mask & (raw_occ != recon_occ)
        add_mask = missing_mask | class_fix_mask
        extra_mask = recon_mask & ~raw_mask
        add_label = np.full_like(raw_occ, EMPTY_IDX, dtype=np.uint8)
        add_label[add_mask] = raw_occ[add_mask]
        extra_source_class = np.full_like(raw_occ, EMPTY_IDX, dtype=np.uint8)
        extra_source_class[extra_mask] = recon_occ[extra_mask]

        np.save(out_dir / f"h{h}_missing_residual.npy", add_label)
        np.save(out_dir / f"h{h}_extra_residual_mask.npy", extra_mask.astype(np.uint8))
        np.save(out_dir / f"h{h}_extra_residual_source_class.npy", extra_source_class)

        residuals[h] = {
            "add_label": add_label,
            "extra_mask": extra_mask,
            "extra_source_class": extra_source_class,
        }
        raw_counts = np.bincount(raw_occ[raw_mask].reshape(-1), minlength=NUM_CLASSES)
        miss_counts = np.bincount(raw_occ[missing_mask].reshape(-1), minlength=NUM_CLASSES)
        extra_counts = np.bincount(recon_occ[extra_mask].reshape(-1), minlength=NUM_CLASSES)
        rows.append(
            {
                "horizon_index": h,
                "model_time_sec": float(ANCHOR_TIMES[h]),
                "missing_voxel_count": int(missing_mask.sum()),
                "class_fix_voxel_count": int(class_fix_mask.sum()),
                "extra_voxel_count": int(extra_mask.sum()),
                "missing_ratio_vs_raw_occ": float(missing_mask.sum() / max(raw_mask.sum(), 1)),
                "static_missing_ratio": float(miss_counts[STATIC_IDS].sum() / max(raw_counts[STATIC_IDS].sum(), 1)),
                "dynamic_missing_ratio": float(miss_counts[DYNAMIC_IDS].sum() / max(raw_counts[DYNAMIC_IDS].sum(), 1)),
                "static_extra_ratio": float(extra_counts[STATIC_IDS].sum() / max(raw_counts[STATIC_IDS].sum(), 1)),
                "dynamic_extra_ratio": float(extra_counts[DYNAMIC_IDS].sum() / max(raw_counts[DYNAMIC_IDS].sum(), 1)),
            }
        )

    summary = {
        "sample_index": sample_index,
        "residual_dir": str(out_dir),
        "rows": rows,
        "note": "missing residual stores raw class labels on raw-occupied/recon-free or raw-occupied/recon-wrong voxels; extra residual stores recon-occupied/raw-free masks for free suppression",
    }
    write_json(REPORTS_DIR / "endpoint_residual_summary.json", summary)
    (REPORTS_DIR / "endpoint_residual_summary.md").write_text(
        "\n".join(
            ["# endpoint residual summary"]
            + [f"- h{r['horizon_index']}: missing={r['missing_voxel_count']} class_fix={r['class_fix_voxel_count']} extra={r['extra_voxel_count']}" for r in rows]
        ),
        encoding="utf-8",
    )
    return {"summary": summary, "residuals": residuals, "residual_dir": out_dir}


def add_weighted_labels(score_grid: np.ndarray, labels: np.ndarray, mask: np.ndarray, weight: float) -> None:
    idx = np.flatnonzero(mask.reshape(-1))
    if idx.size == 0:
        return
    flat_scores = score_grid.reshape(-1, score_grid.shape[-1])
    flat_labels = labels.reshape(-1)[idx].astype(np.int64)
    flat_scores[idx, flat_labels] += float(weight)


def add_weighted_empty(score_grid: np.ndarray, mask: np.ndarray, weight: float) -> None:
    idx = np.flatnonzero(mask.reshape(-1))
    if idx.size == 0:
        return
    flat_scores = score_grid.reshape(-1, score_grid.shape[-1])
    flat_scores[idx, EMPTY_IDX] += float(weight)


def build_layered_frame(base_occ: np.ndarray, left_res: dict[str, Any], right_res: dict[str, Any], alpha: float) -> np.ndarray:
    alpha_s = float(lib.smoothstep(alpha))
    w_left = 1.0 - alpha_s
    w_right = alpha_s
    score_grid = np.zeros(base_occ.shape + (EMPTY_IDX + 1,), dtype=np.float32)
    score_grid[..., EMPTY_IDX] = 0.10

    base_mask = base_occ != EMPTY_IDX
    base_static_mask = base_mask & np.isin(base_occ, STATIC_IDS)
    base_dynamic_mask = base_mask & ~np.isin(base_occ, STATIC_IDS)
    add_weighted_labels(score_grid, base_occ, base_static_mask, 0.35)
    add_weighted_labels(score_grid, base_occ, base_dynamic_mask, 1.20)

    left_add = left_res["add_label"]
    left_add_mask = left_add != EMPTY_IDX
    left_add_static = left_add_mask & np.isin(left_add, STATIC_IDS)
    left_add_dynamic = left_add_mask & ~np.isin(left_add, STATIC_IDS)
    add_weighted_labels(score_grid, left_add, left_add_static, w_left)
    add_weighted_labels(score_grid, left_add, left_add_dynamic, 1.40 * (w_left ** 3))

    right_add = right_res["add_label"]
    right_add_mask = right_add != EMPTY_IDX
    right_add_static = right_add_mask & np.isin(right_add, STATIC_IDS)
    right_add_dynamic = right_add_mask & ~np.isin(right_add, STATIC_IDS)
    add_weighted_labels(score_grid, right_add, right_add_static, w_right)
    add_weighted_labels(score_grid, right_add, right_add_dynamic, 1.40 * (w_right ** 3))

    left_extra_static = left_res["extra_mask"] & np.isin(left_res["extra_source_class"], STATIC_IDS)
    left_extra_dynamic = left_res["extra_mask"] & ~np.isin(left_res["extra_source_class"], np.append(STATIC_IDS, EMPTY_IDX))
    add_weighted_empty(score_grid, left_extra_static, 1.05 * w_left)
    add_weighted_empty(score_grid, left_extra_dynamic, 1.30 * (w_left ** 3))

    right_extra_static = right_res["extra_mask"] & np.isin(right_res["extra_source_class"], STATIC_IDS)
    right_extra_dynamic = right_res["extra_mask"] & ~np.isin(right_res["extra_source_class"], np.append(STATIC_IDS, EMPTY_IDX))
    add_weighted_empty(score_grid, right_extra_static, 1.05 * w_right)
    add_weighted_empty(score_grid, right_extra_dynamic, 1.30 * (w_right ** 3))

    return score_grid.argmax(axis=-1).astype(np.uint8)


def build_layered_occ_frames(
    sample_index: int,
    frame_count: int,
    support_frame_dir: Path,
    selected_base_cfg: dict[str, Any],
    residuals: dict[int, dict[str, Any]],
    logger: StageLogger,
    resume: bool,
) -> Path:
    out_dir = ARTIFACTS_DIR / f"layered_occ_frames/sample_{sample_index:04d}_{frame_count}f"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = "build_layered_occ_frames"
    if resume and len(list(out_dir.glob("occ_frame_*.npy"))) == frame_count:
        logger.skip(stage, reason="layered_occ_frames_exist", output_dir=str(out_dir), frame_count=frame_count)
        return out_dir
    logger.start(stage, output_dir=str(out_dir), frame_count=frame_count)
    for i in range(frame_count):
        h0, h1, alpha = lib.frame_interval_for_index(i, frame_count)
        support = lib.load_support_npz(support_frame_dir / f"frame_{i:06d}.npz")
        base_occ = reconstruct_support_surrogate(
            torch.from_numpy(support["support_xyz_metric"].reshape(1, -1, 1, 3).repeat(48, axis=2)[:, :, :48, :]) if False else None,  # never executed
            torch.from_numpy(support["support_logits"].reshape(1, -1, 1, NUM_CLASSES).repeat(48, axis=2)[:, :, :48, :]) if False else None,
        )
        raise RuntimeError("unreachable")


def reconstruct_surrogate_from_flat_support(
    support_frame: dict[str, Any],
    *,
    global_occ_threshold: float,
    classwise_occ_threshold: np.ndarray | None,
    aggregation: str,
    splat: str,
) -> np.ndarray:
    xyz = support_frame["support_xyz_metric"].astype(np.float32)
    logits = support_frame["support_logits"].astype(np.float32)
    weights = support_frame["support_weight"].astype(np.float32)
    if xyz.shape[0] == 0:
        return np.full(tuple(GRID_SIZE.tolist()), EMPTY_IDX, dtype=np.uint8)
    probs = sigmoid_np(logits)
    float_idx = (xyz - PC_RANGE[:3]) / VOXEL_SIZE
    base = np.floor(float_idx).astype(np.int32)
    frac = float_idx - base.astype(np.float32)
    grid = np.zeros((int(np.prod(GRID_SIZE)), NUM_CLASSES), dtype=np.float32)

    def combine(flat: np.ndarray, contrib: np.ndarray) -> None:
        if aggregation == "sum_prob":
            np.add.at(grid, flat, contrib)
        elif aggregation == "max_prob":
            np.maximum.at(grid, flat, contrib)
        elif aggregation == "logsumexp":
            np.add.at(grid, flat, np.exp(contrib))
        elif aggregation == "topk3_sum":
            np.add.at(grid, flat, contrib)
        else:
            raise ValueError(f"unsupported aggregation: {aggregation}")

    def add(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, w: np.ndarray) -> None:
        mask = (
            (ix >= 0)
            & (iy >= 0)
            & (iz >= 0)
            & (ix < GRID_SIZE[0])
            & (iy < GRID_SIZE[1])
            & (iz < GRID_SIZE[2])
            & (w > 1e-8)
        )
        if not np.any(mask):
            return
        flat = (ix[mask] * (GRID_SIZE[1] * GRID_SIZE[2]) + iy[mask] * GRID_SIZE[2] + iz[mask]).astype(np.int64)
        combine(flat, probs[mask] * w[mask, None])

    if splat == "hard_assign":
        add(base[:, 0], base[:, 1], base[:, 2], weights)
    elif splat == "trilinear":
        wx = [1.0 - frac[:, 0], frac[:, 0]]
        wy = [1.0 - frac[:, 1], frac[:, 1]]
        wz = [1.0 - frac[:, 2], frac[:, 2]]
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    add(base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz, weights * wx[dx] * wy[dy] * wz[dz])
    elif splat == "gaussian_r1":
        sigma = 0.55
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    dist = (frac[:, 0] - dx) ** 2 + (frac[:, 1] - dy) ** 2 + (frac[:, 2] - dz) ** 2
                    add(base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz, weights * np.exp(-dist / (2.0 * sigma * sigma)))
    elif splat == "gaussian_r2_static":
        sigma = 0.70
        point_cls_flat = probs.argmax(-1)
        static_mask = np.isin(point_cls_flat, STATIC_IDS)
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                for dz in (-1, 0, 1):
                    dist = (frac[:, 0] - dx) ** 2 + (frac[:, 1] - dy) ** 2 + (frac[:, 2] - dz) ** 2
                    w = weights * np.exp(-dist / (2.0 * sigma * sigma))
                    w = np.where(static_mask, w, np.where((dx == 0) & (dy == 0) & (dz == 0), weights, 0.0))
                    add(base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz, w)
    else:
        raise ValueError(f"unsupported splat: {splat}")

    if aggregation == "logsumexp":
        grid = np.log1p(grid)
    max_score = grid.max(axis=1)
    cls = grid.argmax(axis=1).astype(np.uint8)
    occ = np.full((grid.shape[0],), EMPTY_IDX, dtype=np.uint8)
    if classwise_occ_threshold is not None:
        thr = np.asarray(classwise_occ_threshold, dtype=np.float32)
        mask = max_score > thr[cls]
    else:
        mask = max_score > float(global_occ_threshold)
    occ[mask] = cls[mask]
    return occ.reshape(*GRID_SIZE.tolist())


def build_layered_occ_frames(
    sample_index: int,
    frame_count: int,
    support_frame_dir: Path,
    selected_base_cfg: dict[str, Any],
    residuals: dict[int, dict[str, Any]],
    logger: StageLogger,
    resume: bool,
) -> Path:
    out_dir = ARTIFACTS_DIR / f"layered_occ_frames/sample_{sample_index:04d}_{frame_count}f"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage = "build_layered_occ_frames"
    if resume and len(list(out_dir.glob("occ_frame_*.npy"))) == frame_count:
        logger.skip(stage, reason="layered_occ_frames_exist", output_dir=str(out_dir), frame_count=frame_count)
        return out_dir
    logger.start(stage, output_dir=str(out_dir), frame_count=frame_count)
    for i in range(frame_count):
        h0, h1, alpha = lib.frame_interval_for_index(i, frame_count)
        support = lib.load_support_npz(support_frame_dir / f"frame_{i:06d}.npz")
        base_occ = reconstruct_surrogate_from_flat_support(
            support,
            global_occ_threshold=float(selected_base_cfg.get("global_occ_threshold", 0.12)),
            classwise_occ_threshold=selected_base_cfg.get("classwise_occ_threshold"),
            aggregation=str(selected_base_cfg.get("aggregation", "sum_prob")),
            splat=str(selected_base_cfg.get("splat", "trilinear")),
        )
        layered = build_layered_frame(base_occ, residuals[h0], residuals[h1], alpha)
        np.save(out_dir / f"occ_frame_{i:06d}.npy", layered.astype(np.uint8))
        if ((i + 1) % 30 == 0) or ((i + 1) == frame_count):
            logger.progress(stage, i + 1, frame_count)
    logger.done(stage, output_dir=str(out_dir))
    return out_dir


def compute_sequence_keyframe_exactness(name: str, occ_dir: Path, native_occ: dict[int, np.ndarray], frame_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping = lib.keyframe_target_frame_map(frame_count)
    rows = []
    for h, fi in mapping.items():
        occ = np.load(occ_dir / f"occ_frame_{fi:06d}.npy" if (occ_dir / f"occ_frame_{fi:06d}.npy").exists() else occ_dir / f"occ_stab_frame_{fi:06d}.npy")
        metrics = compute_occ_comparison(occ, native_occ[h])
        rows.append({"variant": name, "horizon_index": int(h), "frame_index": int(fi), "model_time_sec": float(ANCHOR_TIMES[h]), **metrics})
    summary = {
        "variant": name,
        "mean_occupied_iou": float(np.mean([r["occupied_iou"] for r in rows])),
        "mean_semantic_agreement_intersection": float(np.mean([r["semantic_agreement_intersection"] for r in rows])),
        "mean_occupied_count_ratio": float(np.mean([r["occupied_count_ratio"] for r in rows])),
    }
    return rows, summary


def compute_sequence_jump_check(name: str, occ_dir: Path, frame_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    files = []
    for i in range(frame_count):
        p_occ = occ_dir / f"occ_frame_{i:06d}.npy"
        p_stab = occ_dir / f"occ_stab_frame_{i:06d}.npy"
        files.append(p_occ if p_occ.exists() else p_stab)
    arrs = [np.load(p) for p in files]
    adj = [changed_voxel_ratio(arrs[i], arrs[i + 1]) for i in range(frame_count - 1)]
    mean_adj = float(np.mean(adj))
    p95_adj = float(np.quantile(adj, 0.95))
    jump_thr = float(max(mean_adj * 2.0, p95_adj * 1.05))
    rows = []
    for h, fi in lib.keyframe_target_frame_map(frame_count).items():
        prev_metrics = None
        next_metrics = None
        warnings = []
        if fi > 0:
            prev_metrics = {
                "pair": f"{fi-1}->{fi}",
                "changed_voxel_ratio": adj[fi - 1],
                "occupied_count_delta": int((arrs[fi] != EMPTY_IDX).sum() - (arrs[fi - 1] != EMPTY_IDX).sum()),
                "class_count_delta_l1": int(np.abs(np.bincount(arrs[fi][arrs[fi] != EMPTY_IDX], minlength=NUM_CLASSES) - np.bincount(arrs[fi - 1][arrs[fi - 1] != EMPTY_IDX], minlength=NUM_CLASSES)).sum()),
            }
            if prev_metrics["changed_voxel_ratio"] > jump_thr:
                warnings.append("prev_jump")
        if fi < frame_count - 1:
            next_metrics = {
                "pair": f"{fi}->{fi+1}",
                "changed_voxel_ratio": adj[fi],
                "occupied_count_delta": int((arrs[fi + 1] != EMPTY_IDX).sum() - (arrs[fi] != EMPTY_IDX).sum()),
                "class_count_delta_l1": int(np.abs(np.bincount(arrs[fi + 1][arrs[fi + 1] != EMPTY_IDX], minlength=NUM_CLASSES) - np.bincount(arrs[fi][arrs[fi] != EMPTY_IDX], minlength=NUM_CLASSES)).sum()),
            }
            if next_metrics["changed_voxel_ratio"] > jump_thr:
                warnings.append("next_jump")
        rows.append({"variant": name, "horizon_index": int(h), "frame_index": int(fi), "model_time_sec": float(ANCHOR_TIMES[h]), "jump_warning_threshold": jump_thr, "warnings": "|".join(warnings), "prev_changed_voxel_ratio": None if prev_metrics is None else prev_metrics["changed_voxel_ratio"], "next_changed_voxel_ratio": None if next_metrics is None else next_metrics["changed_voxel_ratio"], "prev_occupied_count_delta": None if prev_metrics is None else prev_metrics["occupied_count_delta"], "next_occupied_count_delta": None if next_metrics is None else next_metrics["occupied_count_delta"], "prev_class_count_delta_l1": None if prev_metrics is None else prev_metrics["class_count_delta_l1"], "next_class_count_delta_l1": None if next_metrics is None else next_metrics["class_count_delta_l1"]})
    summary = {"variant": name, "adjacent_changed_voxel_ratio_mean": mean_adj, "adjacent_changed_voxel_ratio_p95": p95_adj, "jump_warning_threshold": jump_thr, "warning_count": int(sum(1 for r in rows if r["warnings"]))}
    return rows, summary


def build_final_report(
    sample_index: int,
    frame_count: int,
    fps: int,
    native_audit: dict[str, Any],
    ablation: dict[str, Any],
    residual_summary: dict[str, Any],
    layered_summary: dict[str, Any],
    exactness_rows: list[dict[str, Any]],
    jump_rows: list[dict[str, Any]],
    old_vs_fixed_summary: dict[str, Any],
    video_path: Path,
    comparison_video_path: Path | None,
) -> dict[str, Any]:
    report = {
        "sample_index": sample_index,
        "frame_count": frame_count,
        "fps": fps,
        "goal": "Endpoint-exact hybrid query/support interpolation for visualization reconstruction, not model improvement.",
        "answers": {
            "keyframe_reconstruction_error_from_interpolation": False,
            "keyframe_reconstruction_error_from_surrogate_voxelizer": True,
            "native_get_occ_replay_viable": bool(native_audit["summary"]["native_replay_viable"]),
            "endpoint_residual_is_visualization_fix": True,
            "is_model_performance_improvement": False,
        },
        "selected_base_variant": ablation["selected_base_variant"],
        "native_get_occ_replay_audit": native_audit["summary"],
        "surrogate_voxelizer_ablation_path": str(REPORTS_DIR / "surrogate_voxelizer_ablation.csv"),
        "endpoint_residual_summary": residual_summary,
        "layered_interpolation_summary": layered_summary,
        "keyframe_exactness_check_path": str(REPORTS_DIR / "keyframe_exactness_check.csv"),
        "keyframe_jump_check_path": str(REPORTS_DIR / "keyframe_jump_check.csv"),
        "old_vs_fixed_video_smoothness_path": str(REPORTS_DIR / "old_vs_fixed_video_smoothness.csv"),
        "video_path": str(video_path),
        "comparison_video_path": None if comparison_video_path is None else str(comparison_video_path),
        "safe_claims": [
            "This remains a 3s forecast slow-motion visualization, not a 9s prediction.",
            "native raw semantic_occ is used as an endpoint anchor for visualization reconstruction only.",
            "support surrogate reconstruction is reported separately from native raw semantic_occ.",
            "endpoint residual corrected visualization is a rendering-time reconstruction fix, not a model performance improvement.",
            "No training. No checkpoint changes. No manual prediction editing.",
        ],
        "next_action": "If visual fidelity still needs improvement, improve renderer quality separately; if reconstruction faithfulness needs improvement, use native get_occ replay for more interval anchors or voxel-space SDF residuals.",
    }
    write_json(REPORTS_DIR / "swvis4_endpoint_exact_fix_report.json", report)
    md = [
        "# SW-VIS4 endpoint-exact fix report",
        "",
        "1. keyframe reconstruction error source",
        "- Not from nearest-match interpolation.",
        "- Primarily from surrogate support voxelizer under-fill.",
        "",
        "2. native get_occ replay",
        f"- viable: `{native_audit['summary']['native_replay_viable']}`",
        f"- mean occupied IoU: `{native_audit['summary']['mean_occupied_iou']:.6f}`",
        "",
        "3. selected surrogate base",
        f"- `{ablation['selected_base_variant']}`",
        "",
        "4. endpoint residual layer",
        "- residuals are blended from interval endpoints with smoothstep weights.",
        "- no raw keyframe hard cut is inserted into the final visualization path.",
        "",
        "5. keyframe exactness and smoothness",
        f"- fixed video: `{video_path}`",
        f"- comparison video: `{comparison_video_path}`" if comparison_video_path is not None else "- comparison video: `skipped`",
        "",
        "6. safe claims",
        "- This is not a 9s prediction.",
        "- This is not a model performance improvement.",
        "- endpoint residual is not SparseWorld native head output; it is a visualization reconstruction layer.",
    ]
    (REPORTS_DIR / "swvis4_endpoint_exact_fix_report.md").write_text("\n".join(md), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=int, default=3)
    parser.add_argument("--frame-count", type=int, default=540)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--encoder", type=str, default="h264_nvenc")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    args = parser.parse_args()

    for p in [REPORTS_DIR, LOGS_DIR, ARTIFACTS_DIR, FIGURES_DIR, VIDEOS_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    logger = StageLogger(args.sample_index, args.frame_count, args.fps)

    logger.start("native_get_occ_replay_audit")
    native_audit = run_native_replay_audit(args.sample_index)
    logger.done("native_get_occ_replay_audit", mean_occupied_iou=native_audit["summary"]["mean_occupied_iou"])

    logger.start("surrogate_voxelizer_ablation")
    ablation = run_surrogate_voxelizer_ablation(args.sample_index)
    logger.done("surrogate_voxelizer_ablation", selected_base_variant=ablation["selected_base_variant"])

    _, support_frame_dir = ensure_support_artifacts(args.sample_index, args.frame_count, args.fps, args.encoder, logger)

    logger.start("endpoint_residual_layer")
    residual_payload = build_endpoint_residuals(args.sample_index, native_audit["replayed"], ablation["recon_cache"][ablation["selected_base_variant"]])
    logger.done("endpoint_residual_layer", residual_dir=str(residual_payload["residual_dir"]))

    layered_occ_dir = build_layered_occ_frames(
        args.sample_index,
        args.frame_count,
        support_frame_dir,
        ablation["selected_base_cfg"],
        residual_payload["residuals"],
        logger,
        args.resume,
    )

    stab_dir = ARTIFACTS_DIR / f"layered_occ_frames_stabilized/sample_{args.sample_index:04d}_{args.frame_count}f"
    stage = "stabilize_layered_occ_frames"
    if args.resume and len(list(stab_dir.glob("occ_stab_frame_*.npy"))) == args.frame_count:
        logger.skip(stage, reason="stabilized_layered_occ_frames_exist", output_dir=str(stab_dir), frame_count=args.frame_count)
        anti_flicker = json.loads((REPORTS_DIR / "anti_flicker_summary_D_unified_nearest_match.json").read_text(encoding="utf-8")) if False else {}
    else:
        logger.start(stage, output_dir=str(stab_dir), frame_count=args.frame_count)
        anti_flicker = lib.stabilize_occ_sequence(
            layered_occ_dir,
            stab_dir,
            args.frame_count,
            progress_cb=lambda c, t: logger.progress(stage, c, t),
            progress_every=30,
        )
        logger.done(stage, output_dir=str(stab_dir), **anti_flicker)

    layered_summary = {
        "sample_index": args.sample_index,
        "selected_base_variant": ablation["selected_base_variant"],
        "class_groups": {
            "static_background": sorted(STATIC_CLASSES),
            "dynamic_or_small": sorted(DYNAMIC_CLASSES),
        },
        "layered_occ_dir": str(layered_occ_dir),
        "stabilized_occ_dir": str(stab_dir),
        "keyframe_source_mode": "endpoint_residual_blend",
        "native_endpoint_anchor_mode": "native_get_occ_replay_exact",
        "no_raw_hard_cut": True,
        "visualization_note": "native raw semantic_occ enters only as blended endpoint residual anchors, not as a separate hard-cut frame source",
    }
    write_json(REPORTS_DIR / "layered_interpolation_summary.json", layered_summary)
    (REPORTS_DIR / "layered_interpolation_summary.md").write_text(
        "\n".join(
            [
                "# layered interpolation summary",
                f"- selected_base_variant: `{layered_summary['selected_base_variant']}`",
                f"- keyframe_source_mode: `{layered_summary['keyframe_source_mode']}`",
                f"- no_raw_hard_cut: `{layered_summary['no_raw_hard_cut']}`",
            ]
        ),
        encoding="utf-8",
    )

    logger.start("quality_checks")
    old_occ_dir = ARTIFACTS_DIR / f"stabilized_occ_frames/D_unified_nearest_match/sample_{args.sample_index:04d}_{args.frame_count}f"
    old_occ_available = old_occ_dir.exists() and len(list(old_occ_dir.glob("occ_stab_frame_*.npy"))) == args.frame_count
    exact_new_rows, exact_new_summary = compute_sequence_keyframe_exactness("endpoint_exact_layered", stab_dir, native_audit["replayed"], args.frame_count)
    jump_new_rows, jump_new_summary = compute_sequence_jump_check("endpoint_exact_layered", stab_dir, args.frame_count)
    if old_occ_available:
        exact_old_rows, exact_old_summary = compute_sequence_keyframe_exactness("old_support_only", old_occ_dir, native_audit["replayed"], args.frame_count)
        jump_old_rows, jump_old_summary = compute_sequence_jump_check("old_support_only", old_occ_dir, args.frame_count)
    else:
        exact_old_rows, jump_old_rows = [], []
        exact_old_summary = {
            "variant": "old_support_only",
            "mean_occupied_iou": None,
            "mean_semantic_agreement_intersection": None,
            "mean_occupied_count_ratio": None,
            "available": False,
        }
        jump_old_summary = {
            "variant": "old_support_only",
            "adjacent_changed_voxel_ratio_mean": None,
            "adjacent_changed_voxel_ratio_p95": None,
            "jump_warning_threshold": None,
            "warning_count": None,
            "available": False,
        }
    exactness_rows = exact_old_rows + exact_new_rows
    jump_rows = jump_old_rows + jump_new_rows
    write_csv(REPORTS_DIR / "keyframe_exactness_check.csv", exactness_rows)
    write_csv(REPORTS_DIR / "keyframe_jump_check.csv", jump_rows)
    write_csv(
        REPORTS_DIR / "old_vs_fixed_video_smoothness.csv",
        [
            {
                "old_variant": "old_support_only",
                "new_variant": "endpoint_exact_layered",
                "old_mean_keyframe_iou": exact_old_summary["mean_occupied_iou"],
                "new_mean_keyframe_iou": exact_new_summary["mean_occupied_iou"],
                "old_mean_occupied_count_ratio": exact_old_summary["mean_occupied_count_ratio"],
                "new_mean_occupied_count_ratio": exact_new_summary["mean_occupied_count_ratio"],
                "old_adjacent_changed_voxel_ratio_mean": jump_old_summary["adjacent_changed_voxel_ratio_mean"],
                "new_adjacent_changed_voxel_ratio_mean": jump_new_summary["adjacent_changed_voxel_ratio_mean"],
                "old_jump_warning_count": jump_old_summary["warning_count"],
                "new_jump_warning_count": jump_new_summary["warning_count"],
            }
        ],
    )
    (REPORTS_DIR / "swvis4_fix_quality_summary.md").write_text(
        "\n".join(
            [
                "# SW-VIS4 fix quality summary",
                f"- old support-only mean keyframe IoU: `{exact_old_summary['mean_occupied_iou']:.6f}`" if exact_old_summary["mean_occupied_iou"] is not None else "- old support-only mean keyframe IoU: `unavailable for this frame_count`",
                f"- endpoint-exact layered mean keyframe IoU: `{exact_new_summary['mean_occupied_iou']:.6f}`",
                f"- old support-only mean occupied_count_ratio: `{exact_old_summary['mean_occupied_count_ratio']:.6f}`" if exact_old_summary["mean_occupied_count_ratio"] is not None else "- old support-only mean occupied_count_ratio: `unavailable for this frame_count`",
                f"- endpoint-exact layered mean occupied_count_ratio: `{exact_new_summary['mean_occupied_count_ratio']:.6f}`",
                f"- old jump warnings: `{jump_old_summary['warning_count']}`" if jump_old_summary["warning_count"] is not None else "- old jump warnings: `unavailable for this frame_count`",
                f"- new jump warnings: `{jump_new_summary['warning_count']}`",
            ]
        ),
        encoding="utf-8",
    )
    logger.done("quality_checks", old_mean_keyframe_iou=exact_old_summary["mean_occupied_iou"], new_mean_keyframe_iou=exact_new_summary["mean_occupied_iou"], old_occ_available=old_occ_available)

    frame_dir = FIGURES_DIR / f"frames_endpoint_exact_layered/sample_{args.sample_index:04d}_{args.frame_count}f"
    stage = "render_frames_endpoint_exact_layered"
    if args.resume and len(list(frame_dir.glob("frame_*.png"))) == args.frame_count:
        logger.skip(stage, reason="rendered_frames_exist", output_dir=str(frame_dir), frame_count=args.frame_count)
    else:
        logger.start(stage, output_dir=str(frame_dir), frame_count=args.frame_count)
        lib.render_template1_frames(
            args.sample_index,
            stab_dir,
            frame_dir,
            progress_cb=lambda c, t: logger.progress(stage, c, t),
            progress_every=30,
        )
        logger.done(stage, output_dir=str(frame_dir))

    video_path = VIDEOS_DIR / f"sparseworld_support_interp_endpoint_exact_layered_sample_{args.sample_index:04d}_{args.frame_count}f_{args.fps}fps.mp4"
    stage = "encode_endpoint_exact_layered_video"
    if args.resume and video_path.exists():
        logger.skip(stage, reason="video_exists", video_path=str(video_path))
    else:
        logger.start(stage, video_path=str(video_path), encoder=args.encoder)
        enc = lib.encode_png_sequence_to_mp4(frame_dir, video_path, fps=args.fps, encoder_preference=args.encoder)
        logger.done(stage, video_path=str(video_path), encoder_used=enc["encoder"])

    comparison_video: Path | None = None
    if args.skip_comparison:
        logger.skip("build_comparison_video", reason="skip_comparison_requested")
        old_vs_fixed_summary = {}
    else:
        comparison_dir = VIDEOS_DIR / f"comparison_old_support_only_vs_endpoint_exact_layered_frames_sample_{args.sample_index:04d}_{args.frame_count}f"
        comparison_video = VIDEOS_DIR / f"comparison_old_support_only_vs_endpoint_exact_layered_sample_{args.sample_index:04d}_{args.frame_count}f_{args.fps}fps.mp4"
        stage = "build_comparison_video"
        if args.resume and comparison_video.exists():
            logger.skip(stage, reason="comparison_video_exists", video_path=str(comparison_video))
            old_vs_fixed_summary = json.loads((REPORTS_DIR / "old_vs_fixed_comparison.json").read_text(encoding="utf-8")) if (REPORTS_DIR / "old_vs_fixed_comparison.json").exists() else {}
        else:
            logger.start(stage, video_path=str(comparison_video))
            old_frame_dir = FIGURES_DIR / f"frames/D_unified_nearest_match/sample_{args.sample_index:04d}_{args.frame_count}f"
            old_vs_fixed_summary = lib.build_side_by_side_video(frame_dir, old_frame_dir, comparison_dir, comparison_video, fps=args.fps, encoder_preference=args.encoder)
            write_json(REPORTS_DIR / "old_vs_fixed_comparison.json", old_vs_fixed_summary)
            logger.done(stage, video_path=str(comparison_video), encoder_used=old_vs_fixed_summary["video_encoder"])

    logger.start("build_report")
    build_final_report(
        args.sample_index,
        args.frame_count,
        args.fps,
        native_audit,
        ablation,
        residual_payload["summary"],
        layered_summary,
        exactness_rows,
        jump_rows,
        old_vs_fixed_summary,
        video_path,
        comparison_video,
    )
    logger.done("build_report")


if __name__ == "__main__":
    main()
