from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

if not hasattr(np, "Inf"):
    np.Inf = np.inf  # type: ignore[attr-defined]


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[4])))
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14d_candidate_reranker"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14d_candidate_reranker"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14d_candidate_reranker"
TEACHER_CACHE = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14c_residual_learning_audit/runtime_teacher_cache/audit_A10_drop_front_triplet"
SW14C_OEM_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14c_oracle_error_mask_residual"

EMPTY_IDX = 17
CORE_HORIZONS = [0, 2, 4, 6]
FEATURE_NAMES = [
    "confidence",
    "margin",
    "agreement",
    "raw_occ",
    "final_occ",
    "native_occ",
    "raw_final_addlike",
    "final_raw_suplike",
    "front",
    "future_h4h6",
    "x_norm",
    "abs_y_norm",
    "z_norm",
    "range_norm",
    "horizon_norm",
]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    lr: float = 3.0e-3
    batch_size: int = 65536
    max_pos_per_horizon: int = 2048
    max_neg_per_horizon: int = 2048
    add_precision_target: float = 0.55
    suppress_precision_target: float = 0.65
    max_add_ratio: float = 0.020
    max_suppress_ratio: float = 0.060
    add_suppress_balance: float = 0.50
    post_rerank_front_cap: bool = False
    front_cap_ratio: float = 1.30
    front_keep_conf: float = 0.78
    front_keep_agreement: float = 0.60
    front_keep_margin: float = 0.00
    front_add_keep_conf: float = 0.62
    front_add_keep_margin: float = 0.00
    protect_high_conf_adds: bool = True
    front_cap_protect_teacher_raw_final: bool = True


class CandidateScorer(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 48),
            nn.SiLU(),
            nn.Linear(48, 24),
            nn.SiLU(),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SW14D constrained candidate reranker")
    p.add_argument("--seed", type=int, default=23)
    p.add_argument("--train-start", type=int, default=0)
    p.add_argument("--train-end", type=int, default=99)
    p.add_argument("--val-start", type=int, default=100)
    p.add_argument("--val-end", type=int, default=149)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=65536)
    p.add_argument("--max-pos-per-horizon", type=int, default=2048)
    p.add_argument("--max-neg-per-horizon", type=int, default=2048)
    p.add_argument("--add-precision-target", type=float, default=0.55)
    p.add_argument("--suppress-precision-target", type=float, default=0.65)
    p.add_argument("--max-add-ratio", type=float, default=0.020)
    p.add_argument("--max-suppress-ratio", type=float, default=0.060)
    p.add_argument("--add-suppress-balance", type=float, default=0.50)
    p.add_argument("--post-rerank-front-cap", action="store_true")
    p.add_argument("--front-cap-ratio", type=float, default=1.30)
    p.add_argument("--front-keep-conf", type=float, default=0.78)
    p.add_argument("--front-keep-agreement", type=float, default=0.60)
    p.add_argument("--front-keep-margin", type=float, default=0.00)
    p.add_argument("--front-add-keep-conf", type=float, default=0.62)
    p.add_argument("--front-add-keep-margin", type=float, default=0.00)
    p.add_argument("--disable-protect-high-conf-adds", action="store_true")
    p.add_argument("--allow-front-cap-teacher-raw-final-prune", action="store_true")
    return p.parse_args()


def ensure_dirs() -> None:
    for path in [REPORTS_DIR, ARTIFACTS_DIR, CHECKPOINT_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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
                fields.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize(row))


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_div(a: float | int, b: float | int) -> float:
    return float(a) / float(b) if float(b) else 0.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def load_bundle(sample_index: int) -> dict[str, Any]:
    path = TEACHER_CACHE / f"A10_drop_front_triplet__sample{sample_index:03d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu", weights_only=False)
    except RuntimeError:
        return torch.load(path, map_location="cpu", weights_only=False)


def occ(x: torch.Tensor) -> torch.Tensor:
    return x.long() != EMPTY_IDX


def build_coord_features(shape: tuple[int, int, int]) -> dict[str, torch.Tensor]:
    x_n, y_n, z_n = shape
    x = torch.linspace(-50.0, 50.0, x_n)[:, None, None].expand(shape)
    y = torch.linspace(-50.0, 50.0, y_n)[None, :, None].expand(shape)
    z = torch.linspace(0.0, 1.0, z_n)[None, None, :].expand(shape)
    r = torch.sqrt(x.square() + y.square()).clamp(max=70.71)
    front = (x > 0.0) & (y.abs() <= 10.0)
    return {
        "x_norm": x / 50.0,
        "abs_y_norm": y.abs() / 50.0,
        "z_norm": z,
        "range_norm": r / 70.71,
        "front": front,
    }


def feature_tensor(teacher_h: dict[str, Any], horizon_s: int, idx: torch.Tensor, coords: dict[str, torch.Tensor]) -> torch.Tensor:
    raw = teacher_h["teacher_raw_semantic"].long()
    final = teacher_h["teacher_final_semantic"].long()
    native = teacher_h["native_semantic"].long()
    raw_occ = occ(raw)
    final_occ = occ(final)
    native_occ = occ(native)
    vals = [
        teacher_h["teacher_confidence"].float(),
        teacher_h["teacher_margin"].float(),
        teacher_h["agreement"].float(),
        raw_occ.float(),
        final_occ.float(),
        native_occ.float(),
        (raw_occ & ~final_occ).float(),
        (final_occ & ~raw_occ).float(),
        coords["front"].float(),
        torch.full_like(teacher_h["teacher_confidence"].float(), 1.0 if horizon_s in {4, 6} else 0.0),
        coords["x_norm"].float(),
        coords["abs_y_norm"].float(),
        coords["z_norm"].float(),
        coords["range_norm"].float(),
        torch.full_like(teacher_h["teacher_confidence"].float(), float(horizon_s) / 6.0),
    ]
    return torch.stack([v[idx[:, 0], idx[:, 1], idx[:, 2]] for v in vals], dim=1)


def candidate_masks(teacher_h: dict[str, Any], horizon_s: int, coords: dict[str, torch.Tensor], cfg: TrainConfig | None = None) -> dict[str, torch.Tensor]:
    raw = teacher_h["teacher_raw_semantic"].long()
    final = teacher_h["teacher_final_semantic"].long()
    raw_occ = occ(raw)
    final_occ = occ(final)
    conf = teacher_h["teacher_confidence"].float()
    margin = teacher_h["teacher_margin"].float()
    agreement = teacher_h["agreement"].float()
    front = coords["front"].bool()
    future = torch.ones_like(front) if horizon_s in {4, 6} else torch.zeros_like(front)
    add = (~final_occ) & raw_occ & (front | future | (conf > 0.25) | (margin > 0.02))
    suppress = final_occ & ((conf < 0.75) | (agreement < 0.75) | (~raw_occ) | front | future)
    if cfg is None:
        front_keep_conf = 0.78
        front_keep_agreement = 0.60
        front_keep_margin = 0.00
        front_add_keep_conf = 0.62
        front_add_keep_margin = 0.00
    else:
        front_keep_conf = cfg.front_keep_conf
        front_keep_agreement = cfg.front_keep_agreement
        front_keep_margin = cfg.front_keep_margin
        front_add_keep_conf = cfg.front_add_keep_conf
        front_add_keep_margin = cfg.front_add_keep_margin
    high_conf_keep = final_occ & raw_occ & (conf > 0.85) & (agreement > 0.80)
    front_keep = (
        final_occ
        & raw_occ
        & front
        & (conf >= front_keep_conf)
        & ((agreement >= front_keep_agreement) | (margin >= front_keep_margin))
    )
    high_conf_add_keep = (
        add
        & front
        & (conf >= front_add_keep_conf)
        & ((agreement >= front_keep_agreement) | (margin >= front_add_keep_margin))
    )
    return {
        "add": add,
        "suppress": suppress,
        "high_conf_keep": high_conf_keep | front_keep,
        "front_keep": front_keep,
        "high_conf_add_keep": high_conf_add_keep,
    }


def sample_indices(mask: torch.Tensor, label: torch.Tensor, max_pos: int, max_neg: int, seed: int) -> torch.Tensor:
    pos = torch.nonzero(mask & label, as_tuple=False)
    neg = torch.nonzero(mask & ~label, as_tuple=False)
    gen = torch.Generator().manual_seed(int(seed))
    if len(pos) > max_pos:
        pos = pos[torch.randperm(len(pos), generator=gen)[:max_pos]]
    if len(neg) > max_neg:
        neg = neg[torch.randperm(len(neg), generator=gen)[:max_neg]]
    if len(pos) == 0 and len(neg) == 0:
        return torch.zeros((0, 3), dtype=torch.long)
    return torch.cat([pos, neg], dim=0)


def build_training_matrix(indices: range, cfg: TrainConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    add_x, add_y, sup_x, sup_y = [], [], [], []
    rows: list[dict[str, Any]] = []
    coords = build_coord_features((200, 200, 16))
    for sample_index in indices:
        bundle = load_bundle(sample_index)
        for h in CORE_HORIZONS:
            th = bundle["by_horizon"][h]
            masks = candidate_masks(th, h, coords, cfg)
            gt_occ = occ(th["gt_h"].long())
            final_occ = occ(th["teacher_final_semantic"].long())
            add_label = gt_occ & ~final_occ
            sup_label = (~gt_occ) & final_occ
            add_idx = sample_indices(masks["add"], add_label, cfg.max_pos_per_horizon, cfg.max_neg_per_horizon, sample_index * 31 + h)
            sup_idx = sample_indices(masks["suppress"], sup_label, cfg.max_pos_per_horizon, cfg.max_neg_per_horizon, sample_index * 47 + h)
            if len(add_idx):
                add_x.append(feature_tensor(th, h, add_idx, coords))
                add_y.append(add_label[add_idx[:, 0], add_idx[:, 1], add_idx[:, 2]].float())
            if len(sup_idx):
                sup_x.append(feature_tensor(th, h, sup_idx, coords))
                sup_y.append(sup_label[sup_idx[:, 0], sup_idx[:, 1], sup_idx[:, 2]].float())
            rows.append(
                {
                    "split": "train",
                    "sample_index": sample_index,
                    "horizon_s": h,
                    "add_candidates": int(masks["add"].sum().item()),
                    "add_positive": int((masks["add"] & add_label).sum().item()),
                    "suppress_candidates": int(masks["suppress"].sum().item()),
                    "suppress_positive": int((masks["suppress"] & sup_label).sum().item()),
                    "front_keep_count": int(masks["front_keep"].sum().item()),
                    "high_conf_add_keep_count": int(masks["high_conf_add_keep"].sum().item()),
                }
            )
    return (
        torch.cat(add_x, dim=0),
        torch.cat(add_y, dim=0),
        torch.cat(sup_x, dim=0),
        torch.cat(sup_y, dim=0),
        rows,
    )


def train_model(name: str, x: torch.Tensor, y: torch.Tensor, cfg: TrainConfig, device: torch.device) -> tuple[CandidateScorer, list[dict[str, Any]]]:
    model = CandidateScorer(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1.0e-4)
    pos = float(y.sum().item())
    neg = float(len(y) - pos)
    pos_weight = torch.tensor([safe_div(neg, max(pos, 1.0))], device=device).clamp(1.0, 50.0)
    rows: list[dict[str, Any]] = []
    x_cpu = x.float().pin_memory() if torch.cuda.is_available() else x.float()
    y_cpu = y.float().pin_memory() if torch.cuda.is_available() else y.float()
    for epoch in range(cfg.epochs):
        gen = torch.Generator().manual_seed(1000 + epoch)
        order = torch.randperm(len(y_cpu), generator=gen)
        losses = []
        for start in range(0, len(order), cfg.batch_size):
            idx = order[start : start + cfg.batch_size]
            xb = x_cpu[idx].to(device, non_blocking=True)
            yb = y_cpu[idx].to(device, non_blocking=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        rows.append({"head": name, "epoch": epoch, "loss": float(np.mean(losses)), "positive_rate": safe_div(pos, len(y))})
    return model.eval(), rows


@torch.inference_mode()
def score_tensor(model: CandidateScorer, x: torch.Tensor, device: torch.device, batch_size: int = 262144) -> torch.Tensor:
    outs = []
    for start in range(0, len(x), batch_size):
        xb = x[start : start + batch_size].float().to(device, non_blocking=True)
        outs.append(torch.sigmoid(model(xb)).detach().cpu())
    return torch.cat(outs, dim=0) if outs else torch.empty((0,), dtype=torch.float32)


def choose_threshold(scores: torch.Tensor, labels: torch.Tensor, precision_target: float) -> dict[str, Any]:
    best = {"threshold": 0.99, "precision": 0.0, "recall": 0.0, "selected": 0}
    for th in torch.linspace(0.05, 0.99, 95):
        pred = scores >= th
        tp = int((pred & labels.bool()).sum().item())
        selected = int(pred.sum().item())
        pos = int(labels.sum().item())
        precision = safe_div(tp, selected)
        recall = safe_div(tp, pos)
        if precision >= precision_target and recall >= best["recall"]:
            best = {"threshold": float(th.item()), "precision": precision, "recall": recall, "selected": selected}
    return best


def evaluate_split(
    split: str,
    indices: range,
    add_model: CandidateScorer,
    sup_model: CandidateScorer,
    thresholds: dict[str, Any],
    cfg: TrainConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    coords = build_coord_features((200, 200, 16))
    for sample_index in indices:
        bundle = load_bundle(sample_index)
        for h in CORE_HORIZONS:
            th = bundle["by_horizon"][h]
            raw = th["teacher_raw_semantic"].long()
            final = th["teacher_final_semantic"].long()
            gt = th["gt_h"].long()
            final_occ = occ(final)
            gt_occ = occ(gt)
            masks = candidate_masks(th, h, coords, cfg)
            student = final.clone()
            added_mask = torch.zeros_like(final_occ, dtype=torch.bool)
            suppressed_mask = torch.zeros_like(final_occ, dtype=torch.bool)
            frontcap_suppressed_mask = torch.zeros_like(final_occ, dtype=torch.bool)
            sup_idx = torch.nonzero(masks["suppress"] & ~masks["high_conf_keep"], as_tuple=False)
            add_idx = torch.nonzero(masks["add"], as_tuple=False)
            selected_sup_count = 0
            selected_add_count = 0
            suppress_breaks_gt_occ_count = 0
            suppress_breaks_front_gt_occ_count = 0
            add_recovers_gt_occ_count = 0
            add_recovers_front_gt_occ_count = 0
            if len(sup_idx):
                sup_scores = score_tensor(sup_model, feature_tensor(th, h, sup_idx, coords), device)
                keep = sup_scores >= float(thresholds["suppress"]["threshold"])
                if bool(keep.any().item()):
                    candidates = sup_idx[keep]
                    scores = sup_scores[keep]
                    max_sup = min(len(candidates), int(max(1, final_occ.sum().item()) * cfg.max_suppress_ratio))
                    top = torch.topk(scores, k=max_sup).indices if len(scores) > max_sup else torch.arange(len(scores))
                    chosen = candidates[top]
                    student[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = EMPTY_IDX
                    suppressed_mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
                    selected_sup_count = int(len(chosen))
                    suppress_breaks_gt_occ_count = int(gt_occ[chosen[:, 0], chosen[:, 1], chosen[:, 2]].sum().item())
                    suppress_breaks_front_gt_occ_count = int((gt_occ & coords["front"].bool())[chosen[:, 0], chosen[:, 1], chosen[:, 2]].sum().item())
            if len(add_idx):
                add_scores = score_tensor(add_model, feature_tensor(th, h, add_idx, coords), device)
                keep = add_scores >= float(thresholds["add"]["threshold"])
                if bool(keep.any().item()):
                    candidates = add_idx[keep]
                    scores = add_scores[keep]
                    max_add_by_density = int(max(1, final_occ.sum().item()) * cfg.max_add_ratio)
                    max_add_by_sup = int(selected_sup_count * cfg.add_suppress_balance) + 32
                    max_add = min(len(candidates), max_add_by_density, max_add_by_sup)
                    if max_add > 0:
                        top = torch.topk(scores, k=max_add).indices if len(scores) > max_add else torch.arange(len(scores))
                        chosen = candidates[top]
                        student[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = raw[chosen[:, 0], chosen[:, 1], chosen[:, 2]]
                        added_mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
                        selected_add_count = int(len(chosen))
                        add_recovers_gt_occ_count = int(gt_occ[chosen[:, 0], chosen[:, 1], chosen[:, 2]].sum().item())
                        add_recovers_front_gt_occ_count = int((gt_occ & coords["front"].bool())[chosen[:, 0], chosen[:, 1], chosen[:, 2]].sum().item())
            front = coords["front"].bool()
            native_front = max(1, int((occ(th["native_semantic"].long()) & front).sum().item()))
            selected_frontcap_count = 0
            frontcap_breaks_gt_occ_count = 0
            frontcap_breaks_added_count = 0
            frontcap_breaks_added_gt_occ_count = 0
            if cfg.post_rerank_front_cap:
                target_front = int(math.floor(float(cfg.front_cap_ratio) * native_front))
                student_occ_now = occ(student)
                extra_front = int((student_occ_now & front).sum().item()) - target_front
                if extra_front > 0:
                    conf = th["teacher_confidence"].float()
                    agreement = th["agreement"].float()
                    raw_occ = occ(raw)
                    high_conf_keep = masks["high_conf_keep"]
                    if cfg.protect_high_conf_adds:
                        high_conf_keep = high_conf_keep | (added_mask & masks["high_conf_add_keep"])
                    if cfg.front_cap_protect_teacher_raw_final:
                        high_conf_keep = high_conf_keep | (final_occ & raw_occ & front)
                    cap_mask = student_occ_now & front & ~high_conf_keep
                    cap_idx = torch.nonzero(cap_mask, as_tuple=False)
                    if len(cap_idx):
                        low_value = (
                            (1.0 - conf[cap_idx[:, 0], cap_idx[:, 1], cap_idx[:, 2]])
                            + (1.0 - agreement[cap_idx[:, 0], cap_idx[:, 1], cap_idx[:, 2]])
                            + (~raw_occ[cap_idx[:, 0], cap_idx[:, 1], cap_idx[:, 2]]).float()
                        )
                        k = min(int(extra_front), len(cap_idx))
                        chosen = cap_idx[torch.topk(low_value, k=k).indices]
                        student[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = EMPTY_IDX
                        frontcap_suppressed_mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]] = True
                        selected_frontcap_count = int(len(chosen))
                        frontcap_breaks_gt_occ_count = int(gt_occ[chosen[:, 0], chosen[:, 1], chosen[:, 2]].sum().item())
                        frontcap_breaks_added_count = int(added_mask[chosen[:, 0], chosen[:, 1], chosen[:, 2]].sum().item())
                        frontcap_breaks_added_gt_occ_count = int((added_mask & gt_occ)[chosen[:, 0], chosen[:, 1], chosen[:, 2]].sum().item())
            teacher_occ = final_occ
            student_occ = occ(student)
            teacher_fn = gt_occ & ~teacher_occ
            teacher_fp = (~gt_occ) & teacher_occ
            student_fn = gt_occ & ~student_occ
            student_fp = (~gt_occ) & student_occ
            correct_occ = gt_occ & teacher_occ
            correct_free = (~gt_occ) & ~teacher_occ
            broken_occ = correct_occ & ~student_occ
            broken_free = correct_free & student_occ
            teacher_front_local_proxy = safe_div(int((teacher_occ & front).sum().item()), native_front)
            front_local_proxy = safe_div(int((student_occ & front).sum().item()), native_front)
            front_local_delta = front_local_proxy - teacher_front_local_proxy
            fn_reduction = safe_div(int(teacher_fn.sum().item()) - int(student_fn.sum().item()), int(teacher_fn.sum().item()))
            fp_reduction = safe_div(int(teacher_fp.sum().item()) - int(student_fp.sum().item()), int(teacher_fp.sum().item()))
            front_fn_reduction = safe_div(int((teacher_fn & front).sum().item()) - int((student_fn & front).sum().item()), int((teacher_fn & front).sum().item()))
            density_delta = safe_div(int(student_occ.sum().item()) - int(teacher_occ.sum().item()), teacher_occ.numel())
            fp_delta = safe_div(int(student_fp.sum().item()) - int(teacher_fp.sum().item()), int((~gt_occ).sum().item()))
            broken_rate = safe_div(int(broken_occ.sum().item()) + int(broken_free.sum().item()), teacher_occ.numel())
            front_local_safety = front_local_proxy <= 1.30 or front_local_delta <= 1.0e-9
            safety = density_delta <= 0.003 and fp_delta <= 0.001 and front_local_safety and broken_rate <= 0.002
            rows.append(
                {
                    "split": split,
                    "sample_index": sample_index,
                    "horizon_s": h,
                    "selected_add_count": selected_add_count,
                    "selected_suppress_count": selected_sup_count,
                    "selected_frontcap_suppress_count": selected_frontcap_count,
                    "selected_add_front_count": int((added_mask & front).sum().item()),
                    "selected_suppress_front_count": int((suppressed_mask & front).sum().item()),
                    "selected_frontcap_front_count": int((frontcap_suppressed_mask & front).sum().item()),
                    "add_recovers_gt_occ_count": add_recovers_gt_occ_count,
                    "add_recovers_front_gt_occ_count": add_recovers_front_gt_occ_count,
                    "suppress_breaks_gt_occ_count": suppress_breaks_gt_occ_count,
                    "suppress_breaks_front_gt_occ_count": suppress_breaks_front_gt_occ_count,
                    "frontcap_breaks_gt_occ_count": frontcap_breaks_gt_occ_count,
                    "frontcap_breaks_added_count": frontcap_breaks_added_count,
                    "frontcap_breaks_added_gt_occ_count": frontcap_breaks_added_gt_occ_count,
                    "front_keep_count": int(masks["front_keep"].sum().item()),
                    "high_conf_add_keep_count": int(masks["high_conf_add_keep"].sum().item()),
                    "teacher_FN_count": int(teacher_fn.sum().item()),
                    "teacher_FP_count": int(teacher_fp.sum().item()),
                    "student_FN_count": int(student_fn.sum().item()),
                    "student_FP_count": int(student_fp.sum().item()),
                    "fn_reduction_rate": fn_reduction,
                    "front_fn_reduction_rate": front_fn_reduction,
                    "fp_reduction_rate": fp_reduction,
                    "density_delta_over_teacher": density_delta,
                    "false_positive_delta_over_teacher": fp_delta,
                    "teacher_front_local_proxy": teacher_front_local_proxy,
                    "front_local_proxy": front_local_proxy,
                    "front_local_delta_over_teacher": front_local_delta,
                    "front_local_safety_pass": bool(front_local_safety),
                    "broken_correct_occ_count": int(broken_occ.sum().item()),
                    "broken_correct_free_count": int(broken_free.sum().item()),
                    "broken_correct_rate": broken_rate,
                    "safety_pass": bool(safety),
                    "net_score": fn_reduction + fp_reduction - 2.0 * broken_rate - max(0.0, density_delta) - max(0.0, fp_delta),
                }
            )
    summary = summarize_eval_rows(split, rows)
    return rows, summary


def summarize_eval_rows(split: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(key: str) -> float:
        return float(np.mean([float(r[key]) for r in rows])) if rows else 0.0

    def total(key: str) -> int:
        return int(sum(int(r.get(key, 0)) for r in rows))

    safety_all = bool(rows) and all(bool(r["safety_pass"]) for r in rows)
    net = mean("net_score")
    front_fn_non_degraded = mean("front_fn_reduction_rate") >= 0.0
    if safety_all and front_fn_non_degraded and net >= 0.01:
        decision = "SW14D_R1_SAFE_STRONG_GAIN_READY_DEBUG"
    elif safety_all and front_fn_non_degraded and net >= 0.002:
        decision = "SW14D_R2_SAFE_SMALL_GAIN_READY_DEBUG"
    elif safety_all and net >= 0.002:
        decision = "SW14D_R2A_SUPPRESS_SAFE_FRONT_FN_DEGRADED"
    elif net >= 0.002:
        decision = "SW14D_R3_SIGNAL_ONLY_UNSAFE"
    elif net > 0.0:
        decision = "SW14D_R4_WEAK_SIGNAL_NO_DEBUG"
    else:
        decision = "SW14D_R5_NO_SIGNAL_KEEP_SW13"
    return {
        "split": split,
        "row_count": len(rows),
        "sample_count": len({int(r["sample_index"]) for r in rows}),
        "mean_fn_reduction_rate": mean("fn_reduction_rate"),
        "mean_front_fn_reduction_rate": mean("front_fn_reduction_rate"),
        "mean_fp_reduction_rate": mean("fp_reduction_rate"),
        "mean_density_delta_over_teacher": mean("density_delta_over_teacher"),
        "mean_false_positive_delta_over_teacher": mean("false_positive_delta_over_teacher"),
        "mean_teacher_front_local_proxy": mean("teacher_front_local_proxy"),
        "mean_front_local_proxy": mean("front_local_proxy"),
        "mean_front_local_delta_over_teacher": mean("front_local_delta_over_teacher"),
        "front_local_safety_pass_rate": safe_div(sum(bool(r["front_local_safety_pass"]) for r in rows), len(rows)),
        "mean_broken_correct_rate": mean("broken_correct_rate"),
        "front_fn_non_degraded": front_fn_non_degraded,
        "total_selected_add_count": total("selected_add_count"),
        "total_selected_suppress_count": total("selected_suppress_count"),
        "total_selected_frontcap_suppress_count": total("selected_frontcap_suppress_count"),
        "total_add_recovers_front_gt_occ_count": total("add_recovers_front_gt_occ_count"),
        "total_suppress_breaks_front_gt_occ_count": total("suppress_breaks_front_gt_occ_count"),
        "total_frontcap_breaks_gt_occ_count": total("frontcap_breaks_gt_occ_count"),
        "total_frontcap_breaks_added_gt_occ_count": total("frontcap_breaks_added_gt_occ_count"),
        "safety_pass_rate": safe_div(sum(bool(r["safety_pass"]) for r in rows), len(rows)),
        "safety_pass_all": safety_all,
        "mean_net_score": net,
        "decision": decision,
    }


def final_decision(val_summary: dict[str, Any]) -> dict[str, Any]:
    if val_summary["decision"] in {"SW14D_R1_SAFE_STRONG_GAIN_READY_DEBUG", "SW14D_R2_SAFE_SMALL_GAIN_READY_DEBUG"}:
        decision = val_summary["decision"]
        eval_debug = True
        next_action = "Freeze candidate-reranker protocol and run eval_debug only as a diagnostic."
    elif val_summary["decision"] == "SW14D_R3_SIGNAL_ONLY_UNSAFE":
        decision = "SW14D_R3_SIGNAL_ONLY_UNSAFE"
        eval_debug = False
        next_action = "Tighten hard budgets / protected keep rules before any eval_debug."
    elif val_summary["decision"] == "SW14D_R2A_SUPPRESS_SAFE_FRONT_FN_DEGRADED":
        decision = "SW14D_R2A_SUPPRESS_SAFE_FRONT_FN_DEGRADED"
        eval_debug = False
        next_action = "Suppress branch is safe but front FN still regresses; keep iterating high-confidence add / front-preserve rules before eval_debug."
    elif val_summary["decision"] == "SW14D_R4_WEAK_SIGNAL_NO_DEBUG":
        decision = "SW14D_R4_WEAK_SIGNAL_NO_DEBUG"
        eval_debug = False
        next_action = "Improve proxy features and calibration; do not run eval_debug yet."
    else:
        decision = "SW14D_R5_NO_SIGNAL_KEEP_SW13"
        eval_debug = False
        next_action = "Keep SW13 main; candidate reranker needs richer features or a learned mask before rerun."
    return {
        "decision": decision,
        "val_summary": val_summary,
        "whether_eval_debug_allowed": eval_debug,
        "whether_use_in_resume": False,
        "sw13_remains_main_result": True,
        "recommended_next_action": next_action,
        "front_fn_non_degraded": bool(val_summary.get("front_fn_non_degraded", False)),
        "front_local_safety_rule": "absolute <= 1.30 when SW13 teacher is <= 1.30; otherwise no worsening over SW13 teacher",
        "front_local_safety_pass_rate": val_summary.get("front_local_safety_pass_rate"),
        "mean_front_fn_reduction_rate": val_summary.get("mean_front_fn_reduction_rate"),
        "mean_fp_reduction_rate": val_summary.get("mean_fp_reduction_rate"),
        "mean_density_delta_over_teacher": val_summary.get("mean_density_delta_over_teacher"),
        "mean_false_positive_delta_over_teacher": val_summary.get("mean_false_positive_delta_over_teacher"),
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gt_used_for_training_labels": True,
        "gt_used_for_val_selection": False,
        "deployable_claim": False,
    }


def plot_outputs(train_summary: dict[str, Any], val_summary: dict[str, Any]) -> None:
    plt.figure(figsize=(7, 4))
    labels = ["FN", "front FN", "FP", "broken"]
    vals = [
        val_summary["mean_fn_reduction_rate"],
        val_summary["mean_front_fn_reduction_rate"],
        val_summary["mean_fp_reduction_rate"],
        val_summary["mean_broken_correct_rate"],
    ]
    plt.bar(labels, vals)
    plt.title("SW14D candidate reranker val behavior")
    plt.subplots_adjust(bottom=0.18, left=0.12, right=0.96, top=0.88)
    plt.savefig(FIGURES_DIR / "sw14d_candidate_reranker_behavior.png", dpi=180)
    plt.close()
    plt.figure(figsize=(5, 4))
    plt.bar(["train", "val"], [train_summary["mean_net_score"], val_summary["mean_net_score"]])
    plt.title("SW14D train vs val net score")
    plt.subplots_adjust(bottom=0.15, left=0.14, right=0.96, top=0.88)
    plt.savefig(FIGURES_DIR / "sw14d_candidate_reranker_train_val.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    cfg = TrainConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        max_pos_per_horizon=int(args.max_pos_per_horizon),
        max_neg_per_horizon=int(args.max_neg_per_horizon),
        add_precision_target=float(args.add_precision_target),
        suppress_precision_target=float(args.suppress_precision_target),
        max_add_ratio=float(args.max_add_ratio),
        max_suppress_ratio=float(args.max_suppress_ratio),
        add_suppress_balance=float(args.add_suppress_balance),
        post_rerank_front_cap=bool(args.post_rerank_front_cap),
        front_cap_ratio=float(args.front_cap_ratio),
        front_keep_conf=float(args.front_keep_conf),
        front_keep_agreement=float(args.front_keep_agreement),
        front_keep_margin=float(args.front_keep_margin),
        front_add_keep_conf=float(args.front_add_keep_conf),
        front_add_keep_margin=float(args.front_add_keep_margin),
        protect_high_conf_adds=not bool(args.disable_protect_high_conf_adds),
        front_cap_protect_teacher_raw_final=not bool(args.allow_front_cap_teacher_raw_final_prune),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inherited = {
        "stage": "SW14D candidate-level constrained reranker",
        "source": "SW14C OracleErrorMaskResidual showed candidate-level oracle potential but feature residual failed",
        "sw14c_oem_decision": json.loads((SW14C_OEM_REPORTS / "sw14c_oracle_error_mask_residual_final_decision.json").read_text(encoding="utf-8")).get("decision")
        if (SW14C_OEM_REPORTS / "sw14c_oracle_error_mask_residual_final_decision.json").exists()
        else None,
        "teacher_cache": str(TEACHER_CACHE),
        "train_range": [args.train_start, args.train_end],
        "val_range": [args.val_start, args.val_end],
        "no_sparseworld_backbone_training": True,
        "no_occupancy_head_training": True,
        "no_get_occ_or_f3_frontcap_modification": True,
        "no_eval_debug_or_core": True,
        "gpu_policy": "cache-first data build, GPU minibatch MLP training/scoring; future version should batch candidate scoring and postprocess over larger chunks",
    }
    write_json(REPORTS_DIR / "sw14d_candidate_reranker_inherited_state.json", inherited)
    add_x, add_y, sup_x, sup_y, candidate_rows = build_training_matrix(range(args.train_start, args.train_end + 1), cfg)
    write_csv(REPORTS_DIR / "sw14d_candidate_reranker_candidate_stats_train.csv", candidate_rows)
    add_model, add_train_rows = train_model("add", add_x, add_y, cfg, device)
    sup_model, sup_train_rows = train_model("suppress", sup_x, sup_y, cfg, device)
    add_scores = score_tensor(add_model, add_x, device)
    sup_scores = score_tensor(sup_model, sup_x, device)
    thresholds = {
        "add": choose_threshold(add_scores, add_y.bool(), cfg.add_precision_target),
        "suppress": choose_threshold(sup_scores, sup_y.bool(), cfg.suppress_precision_target),
        "config": asdict(cfg),
        "gt_used_for_threshold_selection": "train_split_only",
        "val_gt_used_for_threshold_selection": False,
    }
    write_csv(REPORTS_DIR / "sw14d_candidate_reranker_training_log.csv", add_train_rows + sup_train_rows)
    write_json(REPORTS_DIR / "sw14d_candidate_reranker_threshold_selection.json", thresholds)
    torch.save(
        {
            "add_state_dict": add_model.state_dict(),
            "suppress_state_dict": sup_model.state_dict(),
            "feature_names": FEATURE_NAMES,
            "thresholds": thresholds,
            "config": asdict(cfg),
        },
        CHECKPOINT_DIR / "sw14d_candidate_reranker_best.pth",
    )
    train_rows, train_summary = evaluate_split("train", range(args.train_start, args.train_end + 1), add_model, sup_model, thresholds, cfg, device)
    val_rows, val_summary = evaluate_split("val", range(args.val_start, args.val_end + 1), add_model, sup_model, thresholds, cfg, device)
    write_csv(REPORTS_DIR / "sw14d_candidate_reranker_eval_train.csv", train_rows)
    write_csv(REPORTS_DIR / "sw14d_candidate_reranker_eval_val.csv", val_rows)
    write_json(REPORTS_DIR / "sw14d_candidate_reranker_eval_summary.json", {"train": train_summary, "val": val_summary})
    final = final_decision(val_summary)
    write_json(REPORTS_DIR / "sw14d_candidate_reranker_final_decision.json", final)
    plot_outputs(train_summary, val_summary)
    write_md(
        REPORTS_DIR / "stage_sw14d_candidate_reranker_report.md",
        "\n".join(
            [
                "# SW14D Candidate-Level Constrained Reranker",
                "",
                "This is a train-derived diagnostic stage. It does not modify SparseWorld, get_occ, F3, FrontCap, or SW13 main results.",
                "",
                f"- final decision: `{final['decision']}`",
                f"- val net score: `{val_summary['mean_net_score']}`",
                f"- val front FN reduction rate: `{val_summary['mean_front_fn_reduction_rate']}`",
                f"- val FP reduction rate: `{val_summary['mean_fp_reduction_rate']}`",
                f"- val density delta over teacher: `{val_summary['mean_density_delta_over_teacher']}`",
                f"- val false-positive delta over teacher: `{val_summary['mean_false_positive_delta_over_teacher']}`",
                f"- val front-local safety pass rate: `{val_summary['front_local_safety_pass_rate']}`",
                f"- front-local safety rule: absolute <= 1.30 when SW13 teacher is <= 1.30; otherwise no worsening over SW13 teacher",
                f"- action counters: add recovered front GT occ `{val_summary['total_add_recovers_front_gt_occ_count']}`, suppress broke front GT occ `{val_summary['total_suppress_breaks_front_gt_occ_count']}`, front cap broke added GT occ `{val_summary['total_frontcap_breaks_added_gt_occ_count']}`",
                f"- val safety pass all: `{val_summary['safety_pass_all']}`",
                f"- eval_debug allowed: `{final['whether_eval_debug_allowed']}`",
                "",
                final["recommended_next_action"],
            ]
        )
        + "\n",
    )
    print(f"[sw14d] final decision {final['decision']} val_net={val_summary['mean_net_score']:.6f}", flush=True)


if __name__ == "__main__":
    main()
