from __future__ import annotations

import csv
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT_ROOT = Path(r"D:\ComputerVision\cv_lidar_transition")
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_swvis4_query_support_interpolation"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_swvis4_query_support_interpolation"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_swvis4_query_support_interpolation"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_swvis4_query_support_interpolation"
VIDEOS_DIR = PROJECT_ROOT / "videos/sparseworld_mainline/stage_swvis4_query_support_interpolation"

PC_RANGE = np.array([-40.0, -40.0, -1.0, 40.0, 40.0, 5.4], dtype=np.float32)
VOXEL_SIZE = np.array([0.4, 0.4, 0.4], dtype=np.float32)
GRID_SIZE = np.array([200, 200, 16], dtype=np.int64)
EMPTY_IDX = 17
NUM_CLASSES = 17
SCORE_THR = np.array([0.35] * 15 + [0.25, 0.30], dtype=np.float32)
ANCHOR_TIMES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)


def ensure_dirs() -> None:
    for p in [REPORTS_DIR, LOGS_DIR, SCRIPTS_DIR, FIGURES_DIR, ARTIFACTS_DIR, VIDEOS_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_renderer_module():
    return load_module(
        "swvis4_renderer",
        PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization/semantic_occupancy_renderer.py",
    )


def load_vis1_module():
    return load_module(
        "swvis4_vis1",
        PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_swvis1_paper_style_visualization/run_sparseworld_paper_style_visualization.py",
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for r in rows for k in r.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_query_output(sample_index: int) -> dict[str, Any]:
    p = PROJECT_ROOT / f"artifacts/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/query_outputs/sample_{sample_index:04d}_query_output.pt"
    return torch.load(p, map_location="cpu", weights_only=False)


def load_raw_output(sample_index: int) -> dict[str, Any]:
    p = PROJECT_ROOT / f"artifacts/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/raw_outputs/sample_{sample_index:04d}_raw_output.pt"
    return torch.load(p, map_location="cpu", weights_only=False)


def load_pred_occ(sample_index: int) -> torch.Tensor:
    p = PROJECT_ROOT / f"artifacts/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/standard_occ/sample_{sample_index:04d}_standard_pred_occ_temporal.pt"
    return torch.load(p, map_location="cpu", weights_only=False).long()


def load_gt_occ(sample_index: int) -> torch.Tensor:
    p = PROJECT_ROOT / f"artifacts/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/standard_occ/sample_{sample_index:04d}_standard_gt_occ_temporal.pt"
    return torch.load(p, map_location="cpu", weights_only=False).long()


def load_manifest_row(sample_index: int) -> dict[str, str]:
    rows = read_csv(PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/sw2_sample_manifest.csv")
    return next(r for r in rows if int(r["sample_index"]) == sample_index)


def decode_points(points: torch.Tensor) -> torch.Tensor:
    out = points.clone().float()
    out[..., 0] = out[..., 0] * (PC_RANGE[3] - PC_RANGE[0]) + PC_RANGE[0]
    out[..., 1] = out[..., 1] * (PC_RANGE[4] - PC_RANGE[1]) + PC_RANGE[1]
    out[..., 2] = out[..., 2] * (PC_RANGE[5] - PC_RANGE[2]) + PC_RANGE[2]
    return out


def softmax_np(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(x)
    return exp / np.clip(exp.sum(axis=-1, keepdims=True), 1e-12, None)


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))


def smoothstep(alpha: float | np.ndarray) -> float | np.ndarray:
    return alpha * alpha * (3.0 - 2.0 * alpha)


def build_support_keyframe_from_tensors(points_tensor: torch.Tensor, cls_tensor: torch.Tensor, horizon_index: int, model_time_sec: float) -> dict[str, Any]:
    points = decode_points(points_tensor[0]).cpu().numpy().astype(np.float32)  # [Q,48,3]
    logits = cls_tensor[0].cpu().numpy().astype(np.float32)  # [Q,48,C]
    probs_sig = sigmoid_np(logits)

    centers = points.mean(axis=1, keepdims=True)
    ctr_dist = np.linalg.norm(points - centers, axis=-1)
    point_class = probs_sig.argmax(axis=-1)
    point_score = probs_sig.max(axis=-1)
    thr = SCORE_THR[point_class]
    point_gate = (ctr_dist < 3.0) & (point_score > thr)
    point_weight = point_score * point_gate.astype(np.float32)

    query_logits = logits.mean(axis=1)
    query_probs = probs_sig.mean(axis=1)
    query_class = query_probs.argmax(axis=-1).astype(np.int16)
    query_weight = point_weight.mean(axis=1).astype(np.float32)
    query_center = np.where(
        point_gate[..., None],
        points,
        0.0,
    ).sum(axis=1)
    denom = np.clip(point_gate.sum(axis=1, keepdims=True).astype(np.float32), 1.0, None)
    query_center = (query_center / denom).astype(np.float32)

    return {
        "support_xyz_metric": points,
        "support_logits": logits,
        "support_prob": query_probs,
        "support_query_prob_48": probs_sig,
        "support_class": query_class,
        "support_weight": query_weight,
        "support_point_weight": point_weight.astype(np.float32),
        "support_query_center": query_center,
        "support_query_id": np.arange(points.shape[0], dtype=np.int32),
        "support_point_id": np.arange(points.shape[1], dtype=np.int16),
        "horizon_index": int(horizon_index),
        "model_time_sec": float(model_time_sec),
    }


def save_support_npz(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_support_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def query_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a:[Na,3], b:[Nb,3]
    aa = (a ** 2).sum(axis=1, keepdims=True)
    bb = (b ** 2).sum(axis=1, keepdims=True).T
    dist2 = np.clip(aa + bb - 2.0 * (a @ b.T), 0.0, None)
    return np.sqrt(dist2, dtype=np.float32)


def match_queries(a: dict[str, Any], b: dict[str, Any], distance_threshold: float = 6.0) -> dict[str, Any]:
    centers_a = a["support_query_center"].astype(np.float32)
    centers_b = b["support_query_center"].astype(np.float32)
    cls_a = a["support_class"].astype(np.int16)
    cls_b = b["support_class"].astype(np.int16)
    w_a = a["support_weight"].astype(np.float32)
    w_b = b["support_weight"].astype(np.float32)

    valid_a = w_a > 1e-5
    valid_b = w_b > 1e-5
    idx_a = np.where(valid_a)[0]
    idx_b = np.where(valid_b)[0]

    matches_a: list[int] = []
    matches_b: list[int] = []
    match_d: list[float] = []
    used_a: set[int] = set()
    used_b: set[int] = set()

    classes = sorted(set(cls_a[idx_a].tolist()) | set(cls_b[idx_b].tolist()))
    for cls in classes:
        ca = idx_a[cls_a[idx_a] == cls]
        cb = idx_b[cls_b[idx_b] == cls]
        if ca.size == 0 or cb.size == 0:
            continue
        dist = query_distance_matrix(centers_a[ca], centers_b[cb])
        nn_ab = dist.argmin(axis=1)
        nn_ba = dist.argmin(axis=0)
        for i_local, j_local in enumerate(nn_ab):
            if nn_ba[j_local] != i_local:
                continue
            d = float(dist[i_local, j_local])
            if d > distance_threshold:
                continue
            ai = int(ca[i_local])
            bj = int(cb[j_local])
            if ai in used_a or bj in used_b:
                continue
            used_a.add(ai)
            used_b.add(bj)
            matches_a.append(ai)
            matches_b.append(bj)
            match_d.append(d)

    unmatched_a = np.array([i for i in idx_a.tolist() if i not in used_a], dtype=np.int32)
    unmatched_b = np.array([i for i in idx_b.tolist() if i not in used_b], dtype=np.int32)
    out = {
        "matched_src_idx": np.array(matches_a, dtype=np.int32),
        "matched_dst_idx": np.array(matches_b, dtype=np.int32),
        "matched_distance_m": np.array(match_d, dtype=np.float32),
        "unmatched_src_idx": unmatched_a,
        "unmatched_dst_idx": unmatched_b,
        "distance_threshold_m": np.float32(distance_threshold),
        "matched_count": np.int32(len(matches_a)),
        "unmatched_src_count": np.int32(unmatched_a.size),
        "unmatched_dst_count": np.int32(unmatched_b.size),
    }
    return out


def save_match_npz(path: Path, match: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **match)


def load_match_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def flatten_support_keyframe(keyframe: dict[str, Any]) -> dict[str, Any]:
    return {
        "support_xyz_metric": keyframe["support_xyz_metric"].reshape(-1, 3).astype(np.float32),
        "support_logits": keyframe["support_logits"].reshape(-1, NUM_CLASSES).astype(np.float32),
        "support_weight": keyframe["support_point_weight"].reshape(-1).astype(np.float32),
        "support_class": np.repeat(keyframe["support_class"].astype(np.int16), keyframe["support_xyz_metric"].shape[1]),
    }


def compute_index_alignment_quality(a: dict[str, Any], b: dict[str, Any], distance_threshold: float = 6.0) -> dict[str, Any]:
    if a["support_query_center"].shape[0] != b["support_query_center"].shape[0]:
        return {
            "same_query_count": False,
            "index_usable": False,
            "mean_center_distance_m": None,
            "class_agreement_ratio": None,
        }
    dist = np.linalg.norm(a["support_query_center"] - b["support_query_center"], axis=-1)
    class_agree = (a["support_class"].astype(np.int16) == b["support_class"].astype(np.int16)).astype(np.float32)
    mean_dist = float(dist.mean()) if dist.size else 0.0
    class_ratio = float(class_agree.mean()) if class_agree.size else 1.0
    usable = mean_dist <= min(distance_threshold * 0.25, 1.5) and class_ratio >= 0.85
    return {
        "same_query_count": True,
        "index_usable": bool(usable),
        "mean_center_distance_m": mean_dist,
        "class_agreement_ratio": class_ratio,
    }


def build_index_aligned_match(a: dict[str, Any], b: dict[str, Any], distance_threshold: float = 6.0) -> dict[str, Any]:
    n = min(a["support_query_center"].shape[0], b["support_query_center"].shape[0])
    dist = np.linalg.norm(a["support_query_center"][:n] - b["support_query_center"][:n], axis=-1)
    valid = dist <= distance_threshold
    matched_src_idx = np.where(valid)[0].astype(np.int32)
    matched_dst_idx = np.where(valid)[0].astype(np.int32)
    unmatched_src_idx = np.where(~valid)[0].astype(np.int32)
    unmatched_dst_idx = np.where(~valid)[0].astype(np.int32)
    return {
        "matched_src_idx": matched_src_idx,
        "matched_dst_idx": matched_dst_idx,
        "matched_distance_m": dist[valid].astype(np.float32),
        "unmatched_src_idx": unmatched_src_idx,
        "unmatched_dst_idx": unmatched_dst_idx,
        "distance_threshold_m": np.float32(distance_threshold),
        "matched_count": np.int32(matched_src_idx.size),
        "unmatched_src_count": np.int32(unmatched_src_idx.size),
        "unmatched_dst_count": np.int32(unmatched_dst_idx.size),
        "alignment_mode": np.array("index_aligned", dtype="<U32"),
    }


def interpolate_interval_frame(a: dict[str, Any], b: dict[str, Any], match: dict[str, Any], alpha: float) -> dict[str, Any]:
    alpha_s = float(smoothstep(alpha))
    a_idx = match["matched_src_idx"]
    b_idx = match["matched_dst_idx"]

    xyz_list = []
    logits_list = []
    weight_list = []
    query_cls_list = []

    if a_idx.size > 0:
        xyz_m = (1.0 - alpha_s) * a["support_xyz_metric"][a_idx] + alpha_s * b["support_xyz_metric"][b_idx]
        logits_m = (1.0 - alpha_s) * a["support_logits"][a_idx] + alpha_s * b["support_logits"][b_idx]
        pw_m = (1.0 - alpha_s) * a["support_point_weight"][a_idx] + alpha_s * b["support_point_weight"][b_idx]
        xyz_list.append(xyz_m)
        logits_list.append(logits_m)
        weight_list.append(pw_m)
        query_cls_list.append(((1.0 - alpha_s) * a["support_prob"][a_idx] + alpha_s * b["support_prob"][b_idx]).argmax(axis=-1))

    ua = match["unmatched_src_idx"]
    if ua.size > 0:
        xyz_list.append(a["support_xyz_metric"][ua])
        logits_list.append(a["support_logits"][ua])
        weight_list.append(a["support_point_weight"][ua] * np.float32(1.0 - alpha_s))
        query_cls_list.append(a["support_class"][ua])

    ub = match["unmatched_dst_idx"]
    if ub.size > 0:
        xyz_list.append(b["support_xyz_metric"][ub])
        logits_list.append(b["support_logits"][ub])
        weight_list.append(b["support_point_weight"][ub] * np.float32(alpha_s))
        query_cls_list.append(b["support_class"][ub])

    if not xyz_list:
        return {
            "support_xyz_metric": np.zeros((0, 3), dtype=np.float32),
            "support_logits": np.zeros((0, NUM_CLASSES), dtype=np.float32),
            "support_weight": np.zeros((0,), dtype=np.float32),
            "support_class": np.zeros((0,), dtype=np.int16),
        }

    xyz = np.concatenate([x.reshape(-1, 3) for x in xyz_list], axis=0).astype(np.float32)
    logits = np.concatenate([x.reshape(-1, NUM_CLASSES) for x in logits_list], axis=0).astype(np.float32)
    weights = np.concatenate([x.reshape(-1) for x in weight_list], axis=0).astype(np.float32)
    query_classes = np.concatenate([np.repeat(c.astype(np.int16), 48) for c in query_cls_list], axis=0)
    valid = weights > 1e-6
    return {
        "support_xyz_metric": xyz[valid],
        "support_logits": logits[valid],
        "support_weight": weights[valid],
        "support_class": query_classes[valid],
    }


def interpolate_interval_frame_fade_only(a: dict[str, Any], b: dict[str, Any], alpha: float) -> dict[str, Any]:
    alpha_s = float(smoothstep(alpha))
    left = flatten_support_keyframe(a)
    right = flatten_support_keyframe(b)
    xyz = np.concatenate([left["support_xyz_metric"], right["support_xyz_metric"]], axis=0).astype(np.float32)
    logits = np.concatenate([left["support_logits"], right["support_logits"]], axis=0).astype(np.float32)
    weights = np.concatenate(
        [
            left["support_weight"] * np.float32(1.0 - alpha_s),
            right["support_weight"] * np.float32(alpha_s),
        ],
        axis=0,
    ).astype(np.float32)
    classes = np.concatenate([left["support_class"], right["support_class"]], axis=0).astype(np.int16)
    valid = weights > 1e-6
    return {
        "support_xyz_metric": xyz[valid],
        "support_logits": logits[valid],
        "support_weight": weights[valid],
        "support_class": classes[valid],
    }


def voxelize_support_frame(frame_support: dict[str, Any], mode: str = "trilinear_soft") -> np.ndarray:
    xyz = frame_support["support_xyz_metric"].astype(np.float32)
    logits = frame_support["support_logits"].astype(np.float32)
    weights = frame_support["support_weight"].astype(np.float32)
    if xyz.shape[0] == 0:
        return np.full(tuple(GRID_SIZE.tolist()), EMPTY_IDX, dtype=np.uint8)

    score = sigmoid_np(logits)
    float_idx = (xyz - PC_RANGE[:3]) / VOXEL_SIZE
    base = np.floor(float_idx).astype(np.int32)
    frac = float_idx - base.astype(np.float32)

    grid_scores = torch.zeros((int(np.prod(GRID_SIZE)), NUM_CLASSES), dtype=torch.float32)

    def add_contrib(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, w: np.ndarray) -> None:
        valid = (
            (ix >= 0) & (ix < GRID_SIZE[0]) &
            (iy >= 0) & (iy < GRID_SIZE[1]) &
            (iz >= 0) & (iz < GRID_SIZE[2]) &
            (w > 1e-8)
        )
        if not np.any(valid):
            return
        flat = (ix[valid] * (GRID_SIZE[1] * GRID_SIZE[2]) + iy[valid] * GRID_SIZE[2] + iz[valid]).astype(np.int64)
        contrib = torch.from_numpy((score[valid] * w[valid, None]).astype(np.float32))
        flat_t = torch.from_numpy(flat)
        grid_scores.index_add_(0, flat_t, contrib)

    if mode == "hard_nearest":
        ix, iy, iz = base[:, 0], base[:, 1], base[:, 2]
        add_contrib(ix, iy, iz, weights)
    else:
        wx = [1.0 - frac[:, 0], frac[:, 0]]
        wy = [1.0 - frac[:, 1], frac[:, 1]]
        wz = [1.0 - frac[:, 2], frac[:, 2]]
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = weights * wx[dx] * wy[dy] * wz[dz]
                    add_contrib(base[:, 0] + dx, base[:, 1] + dy, base[:, 2] + dz, w)

    max_score, cls = grid_scores.max(dim=-1)
    occ = torch.full((int(np.prod(GRID_SIZE)),), EMPTY_IDX, dtype=torch.uint8)
    occ[max_score > 0.12] = cls[max_score > 0.12].to(torch.uint8)
    return occ.view(*GRID_SIZE.tolist()).cpu().numpy()


def compute_anchor_frames(frame_count: int) -> np.ndarray:
    return np.rint(np.linspace(0, frame_count - 1, 7)).astype(np.int64)


def frame_interval_for_index(frame_idx: int, frame_count: int) -> tuple[int, int, float]:
    anchor_frames = compute_anchor_frames(frame_count)
    if frame_idx >= anchor_frames[-1]:
        return 5, 6, 1.0
    seg = int(np.searchsorted(anchor_frames, frame_idx, side="right") - 1)
    seg = max(0, min(seg, len(anchor_frames) - 2))
    f0, f1 = int(anchor_frames[seg]), int(anchor_frames[seg + 1])
    alpha = 0.0 if f1 <= f0 else (frame_idx - f0) / (f1 - f0)
    return seg, seg + 1, float(alpha)


def keyframe_target_frame_map(frame_count: int) -> dict[int, int]:
    anchor_frames = compute_anchor_frames(frame_count)
    return {int(h): int(anchor_frames[h]) for h in range(7)}


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
    same_sem_union = int(((recon_occ == ref_occ) & union & recon_mask & ref_mask).sum())
    pred_counts = np.bincount(recon_occ[recon_mask].reshape(-1), minlength=NUM_CLASSES).astype(np.int64)
    ref_counts = np.bincount(ref_occ[ref_mask].reshape(-1), minlength=NUM_CLASSES).astype(np.int64)
    return {
        "occupied_iou": float(inter_count / max(union_count, 1)),
        "semantic_agreement_intersection": float(same_sem_inter / max(inter_count, 1)),
        "semantic_agreement_union": float(same_sem_union / max(union_count, 1)),
        "occupied_count_ratio": float(recon_count / max(ref_count, 1)),
        "recon_occupied_count": recon_count,
        "ref_occupied_count": ref_count,
        "per_class_count_delta": {str(i): int(pred_counts[i] - ref_counts[i]) for i in range(NUM_CLASSES)},
        "per_class_recon_count": {str(i): int(pred_counts[i]) for i in range(NUM_CLASSES)},
        "per_class_ref_count": {str(i): int(ref_counts[i]) for i in range(NUM_CLASSES)},
    }


def changed_voxel_ratio(a: np.ndarray, b: np.ndarray) -> float:
    return float((a.astype(np.uint8) != b.astype(np.uint8)).mean())


def occupancy_count(a: np.ndarray) -> int:
    return int((a.astype(np.uint8) != EMPTY_IDX).sum())


def ffmpeg_has_encoder(encoder_name: str) -> bool:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=True)
    return encoder_name in result.stdout


def encode_png_sequence_to_mp4(frame_dir: Path, out_mp4: Path, fps: int, encoder_preference: str = "auto") -> dict[str, Any]:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    base_cmd = ["ffmpeg", "-y", "-framerate", str(fps), "-i", str(frame_dir / "frame_%06d.png")]
    attempts: list[dict[str, Any]] = []

    def run_cmd(cmd: list[str], encoder_name: str) -> dict[str, Any]:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            attempts.append({"encoder": encoder_name, "returncode": result.returncode, "stderr_tail": result.stderr[-1200:]})
            raise subprocess.CalledProcessError(result.returncode, cmd, output=result.stdout, stderr=result.stderr)
        return {"encoder": encoder_name, "returncode": 0}

    if encoder_preference in ("auto", "h264_nvenc") and ffmpeg_has_encoder("h264_nvenc"):
        nvenc_cmd = base_cmd + ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "23", "-pix_fmt", "yuv420p", str(out_mp4)]
        try:
            status = run_cmd(nvenc_cmd, "h264_nvenc")
            status["attempts"] = attempts
            return status
        except subprocess.CalledProcessError:
            pass

    x264_cmd = base_cmd + ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_mp4)]
    status = run_cmd(x264_cmd, "libx264")
    status["attempts"] = attempts
    return status


def stabilize_occ_sequence(frame_dir: Path, out_dir: Path, frame_count: int, progress_cb: Any | None = None, progress_every: int = 25) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(frame_dir.glob("occ_frame_*.npy"))
    key_locked = set(keyframe_target_frame_map(frame_count).values())
    metrics = []
    prev_prev = None
    prev = None
    for i, f in enumerate(files):
        curr = np.load(f)
        if i == 0 or i == len(files) - 1 or i in key_locked:
            out = curr
        else:
            nxt = np.load(files[i + 1])
            out = curr.copy()
            mask = (prev == nxt) & (prev != curr)
            out[mask] = prev[mask]
            occ = out != EMPTY_IDX
            occ_t = torch.from_numpy(occ.astype(np.float32))[None, None]
            neighbor_count = F.conv3d(occ_t, torch.ones((1, 1, 3, 3, 3)), padding=1)[0, 0].numpy()
            isolated = occ & (neighbor_count <= 1.0) & (prev == EMPTY_IDX) & (nxt == EMPTY_IDX)
            out[isolated] = EMPTY_IDX
        np.save(out_dir / f.name.replace("occ_frame_", "occ_stab_frame_"), out.astype(np.uint8))
        changed = float((out != curr).mean())
        metrics.append(
            {
                "frame_index": i,
                "occupied_count_before": int((curr != EMPTY_IDX).sum()),
                "occupied_count_after": int((out != EMPTY_IDX).sum()),
                "changed_voxel_ratio": changed,
                "small_component_proxy_after": int((((out != EMPTY_IDX).sum(axis=-1) > 0).sum())),
            }
        )
        if progress_cb is not None and ((i + 1) % max(1, progress_every) == 0 or (i + 1) == len(files)):
            progress_cb(i + 1, len(files))
        prev_prev = prev
        prev = out
    write_csv(REPORTS_DIR / "anti_flicker_metrics.csv", metrics)
    summary = {
        "frame_count": len(files),
        "mean_changed_voxel_ratio": float(np.mean([m["changed_voxel_ratio"] for m in metrics])) if metrics else 0.0,
        "max_changed_voxel_ratio": float(np.max([m["changed_voxel_ratio"] for m in metrics])) if metrics else 0.0,
        "keyframe_locked_frames": sorted(key_locked),
    }
    (REPORTS_DIR / "anti_flicker_summary.md").write_text(
        "\n".join(
            [
                "# anti-flicker summary",
                f"- frame_count: `{summary['frame_count']}`",
                f"- mean_changed_voxel_ratio: `{summary['mean_changed_voxel_ratio']:.6f}`",
                f"- max_changed_voxel_ratio: `{summary['max_changed_voxel_ratio']:.6f}`",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def render_template1_frames(sample_index: int, occ_dir: Path, out_dir: Path, progress_cb: Any | None = None, progress_every: int = 10) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    renderer = load_renderer_module()
    vis1 = load_vis1_module()
    style = renderer.RendererStyle(
        dpi=180,
        backend="open3d",
        mode="voxel_mesh",
        viewpoint="top",
        render_width=1400,
        render_height=900,
        open3d_point_size=6.8,
    )
    repo_root = PROJECT_ROOT / "external/SparseWorld"
    manifest_row = load_manifest_row(sample_index)
    mapped_info_index = int(manifest_row["mapped_info_index"])
    import pickle

    with (repo_root / "data/nuscenes/bevdetv2-nuscenes_infos_val.pkl").open("rb") as f:
        payload = pickle.load(f)
    infos = sorted(payload["infos"], key=lambda e: e["timestamp"])
    cam_paths = vis1.current_camera_paths_from_info(infos[mapped_info_index])
    cam_imgs = vis1.load_camera_strip(cam_paths, repo_root)
    cam_labels = [Path(p).parent.name for p in cam_paths]
    cam_panel = vis1.camera_grid(cam_imgs, cam_labels)
    pred_occ = load_pred_occ(sample_index)
    observation_panel = renderer.render_semantic_occ(pred_occ[0], "", style)
    base_future_panel = renderer.render_semantic_occ(pred_occ[0], "", style)
    legend_classes = ["car", "truck", "pedestrian", "driveable_surface", "manmade", "vegetation"]
    legend_strip = renderer.build_legend_strip(legend_classes, width=cam_panel.width, height=58)
    legend_strip = ImageOps.contain(legend_strip, (cam_panel.width, 52))
    margin = 22
    top_label_band = 52
    mid_label_band = 126
    bottom_margin = 28
    voxel_pad_top = 8
    panel_gap = 18
    target_bottom_width = cam_panel.width
    obs_w = int(target_bottom_width * 0.26)
    fut_w = target_bottom_width - panel_gap - obs_w
    base_aspect = base_future_panel.height / max(1, base_future_panel.width)
    bottom_h = max(300, int(fut_w * base_aspect))
    total_w = cam_panel.width + 2 * margin
    total_h = top_label_band + cam_panel.height + mid_label_band + voxel_pad_top + bottom_h + bottom_margin

    frame_files = sorted(occ_dir.glob("occ_stab_frame_*.npy"))
    for i, f in enumerate(frame_files):
        occ = torch.from_numpy(np.load(f).astype(np.int64))
        future_panel = renderer.render_semantic_occ(occ, "", style)
        obs_panel = renderer.fit_panel(observation_panel, (obs_w, bottom_h), inner_scale=0.995)
        fut_panel = renderer.fit_panel(future_panel, (fut_w, bottom_h), inner_scale=0.995)
        canvas = Image.new("RGB", (total_w, total_h), "white")
        draw = ImageDraw.Draw(canvas)
        try:
            font_t = ImageFont.truetype("times.ttf", 34)
            font_s = ImageFont.truetype("times.ttf", 30)
        except Exception:
            font_t = ImageFont.load_default()
            font_s = ImageFont.load_default()
        obs_bbox = draw.textbbox((0, 0), "Observations", font=font_s)
        obs_x = margin + max(0, obs_panel.width // 2 - (obs_bbox[2] - obs_bbox[0]) // 2)
        draw.text((obs_x, 10), "Present", fill="black", font=font_t)
        canvas.paste(cam_panel, (margin, top_label_band))
        obs_label_y = top_label_band + cam_panel.height + 46
        draw.text((obs_x, obs_label_y), "Observations", fill="black", font=font_s)
        pf_text = "Predicted Futures"
        pf_bbox = draw.textbbox((0, 0), pf_text, font=font_s)
        pf_x = margin + obs_w + panel_gap + max(0, fut_w // 2 - (pf_bbox[2] - pf_bbox[0]) // 2)
        draw.text((pf_x, obs_label_y), pf_text, fill="black", font=font_s)
        legend_y = obs_label_y + 38
        canvas.paste(legend_strip, (margin, legend_y))
        voxel_y = top_label_band + cam_panel.height + mid_label_band + voxel_pad_top
        canvas.paste(obs_panel, (margin, voxel_y))
        canvas.paste(fut_panel, (margin + obs_w + panel_gap, voxel_y))
        canvas.save(out_dir / f"frame_{i:06d}.png")
        if progress_cb is not None and ((i + 1) % max(1, progress_every) == 0 or (i + 1) == len(frame_files)):
            progress_cb(i + 1, len(frame_files))
    return {"frame_count": len(frame_files), "output_dir": str(out_dir)}


def build_side_by_side_video(ours_dir: Path, render_level_dir: Path, out_dir: Path, out_mp4: Path, fps: int = 60, encoder_preference: str = "auto") -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ours = sorted(ours_dir.glob("frame_*.png"))
    ref = sorted(render_level_dir.glob("frame_*.png"))
    n = min(len(ours), len(ref))
    if n == 0:
        raise RuntimeError("no frames available for side-by-side comparison")
    target_a = Image.open(ours[0]).convert("RGB").size
    target_b = Image.open(ref[0]).convert("RGB").size
    changed_ratios_ours = []
    changed_ratios_ref = []
    prev_ours = None
    prev_ref = None
    for i in range(n):
        img_a = Image.open(ours[i]).convert("RGB").resize(target_a, Image.Resampling.BILINEAR)
        img_b = Image.open(ref[i]).convert("RGB").resize(target_b, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (img_a.width + img_b.width, max(img_a.height, img_b.height)), "white")
        canvas.paste(img_b, (0, 0))
        canvas.paste(img_a, (img_b.width, 0))
        canvas.save(out_dir / f"frame_{i:06d}.png")
        arr_a = np.asarray(img_a)
        arr_b = np.asarray(img_b)
        if prev_ours is not None:
            changed_ratios_ours.append(float((np.abs(arr_a.astype(np.int16) - prev_ours.astype(np.int16)).max(axis=-1) > 8).mean()))
        if prev_ref is not None:
            changed_ratios_ref.append(float((np.abs(arr_b.astype(np.int16) - prev_ref.astype(np.int16)).max(axis=-1) > 8).mean()))
        prev_ours = arr_a
        prev_ref = arr_b
    encode_status = encode_png_sequence_to_mp4(out_dir, out_mp4, fps=fps, encoder_preference=encoder_preference)
    summary = {
        "frame_count": n,
        "mean_changed_pixel_ratio_support_level": float(np.mean(changed_ratios_ours)) if changed_ratios_ours else 0.0,
        "mean_changed_pixel_ratio_render_level": float(np.mean(changed_ratios_ref)) if changed_ratios_ref else 0.0,
        "comparison_video_path": str(out_mp4),
        "video_encoder": encode_status["encoder"],
        "video_encode_attempts": encode_status["attempts"],
    }
    return summary
