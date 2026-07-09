from __future__ import annotations

import csv
import importlib.util
import json
import os
import random
import sys
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


PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(Path(__file__).resolve().parents[5])))
SW14E_SCRIPT = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw14e_three_route_rescue/run_sw14e_three_route_rescue.py"
REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw14e_three_route_rescue"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw14e_three_route_rescue"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw14e_three_route_rescue"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"


def load_sw14e_module():
    spec = importlib.util.spec_from_file_location("sw14e_mod", SW14E_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SW14E_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sw14e_mod"] = module
    spec.loader.exec_module(module)
    return module


sw14e = load_sw14e_module()

B2_ADD_FEATURES = sw14e.NO_GT_FEATURES + [
    "strong_add_candidate",
    "medium_add_candidate",
    "front_confidence",
    "future_confidence",
    "front_margin",
    "future_margin",
    "neighbor_confidence",
    "high_conf_raw_add_rule",
    "route2_raw_to_final_survival_gap",
    "route2_frontcap_boundary_proxy",
    "route2_f3_boundary_proxy",
    "route3_raw_final_reliability",
    "route3_native_raw_reliability",
    "route3_score_router_add_prior",
    "route3_density_supported_add_prior",
]


class AddRepairScorer(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 192),
            nn.SiLU(),
            nn.Dropout(p=0.03),
            nn.Linear(192, 96),
            nn.SiLU(),
            nn.Dropout(p=0.02),
            nn.Linear(96, 48),
            nn.SiLU(),
            nn.Linear(48, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sw14e.normalize(payload), indent=2, ensure_ascii=False), encoding="utf-8")


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
        writer.writerows([sw14e.normalize(r) for r in rows])


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def add_rule_score(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    conf = tensors["raw_confidence"].float()
    margin = tensors["raw_margin"].float()
    agreement = tensors["camera_view_agreement"].float()
    neighbor = tensors["neighbor_occ_count_norm"].float()
    front = tensors["front_region"].float()
    future = tensors["future_h4h6"].float()
    strong = tensors["strong_add_candidate"].float()
    return conf + 0.45 * margin + 0.20 * agreement + 0.25 * neighbor + 0.15 * front + 0.18 * future + 0.12 * strong


def normalize_score(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(score.float(), -torch.inf)
    vals = score[mask].float()
    out[mask] = (vals - vals.min()) / (vals.max() - vals.min() + 1.0e-9)
    return out


def feature_matrix(tensors: dict[str, torch.Tensor], idx: torch.Tensor) -> torch.Tensor:
    base = [tensors[name][idx].float() for name in sw14e.NO_GT_FEATURES]
    front = tensors["front_region"][idx].float()
    future = tensors["future_h4h6"][idx].float()
    conf = tensors["raw_confidence"][idx].float()
    margin = tensors["raw_margin"][idx].float()
    neighbor = tensors["neighbor_occ_count_norm"][idx].float()
    strong = tensors["strong_add_candidate"][idx].float()
    medium = tensors["medium_add_candidate"][idx].float()
    rule = add_rule_score(tensors)[idx].float()
    raw_occ = tensors["raw_occ"][idx].float()
    final_occ = tensors["teacher_final_occ"][idx].float()
    native_occ = tensors["native_final_occ"][idx].float()
    raw_but_final_empty = tensors["raw_but_final_empty"][idx].float()
    was_pruned_f3 = tensors["was_pruned_by_F3"][idx].float()
    was_pruned_fc = tensors["was_pruned_by_FrontCap"][idx].float()
    density = tensors["local_density_proxy"][idx].float()
    agreement = tensors["camera_view_agreement"][idx].float()
    route2_survival_gap = raw_but_final_empty * (conf + margin) * 0.5
    route2_frontcap_boundary = was_pruned_fc * front * (0.5 + 0.5 * density)
    route2_f3_boundary = was_pruned_f3 * (0.5 + 0.5 * agreement)
    route3_raw_final_reliability = raw_occ * (1.0 - final_occ) * agreement * conf
    route3_native_raw_reliability = raw_occ * native_occ * (0.5 + 0.5 * agreement)
    route3_score_router_add_prior = strong * conf + medium * 0.5 * (conf + density)
    route3_density_supported_add_prior = raw_but_final_empty * density * (0.5 + 0.5 * agreement)
    extra = [
        strong,
        medium,
        front * conf,
        future * conf,
        front * margin,
        future * margin,
        neighbor * conf,
        rule,
        route2_survival_gap,
        route2_frontcap_boundary,
        route2_f3_boundary,
        route3_raw_final_reliability,
        route3_native_raw_reliability,
        route3_score_router_add_prior,
        route3_density_supported_add_prior,
    ]
    return torch.stack(base + extra, dim=1)


def balanced_add_indices(tensors: dict[str, torch.Tensor], seed: int, max_pos: int = 900_000, max_neg: int = 1_200_000) -> torch.Tensor:
    add_mask = tensors["strong_add_candidate"].bool() | tensors["medium_add_candidate"].bool()
    label = tensors["GT_occ"].bool()
    pos = torch.nonzero(add_mask & label, as_tuple=False).flatten()
    neg = torch.nonzero(add_mask & ~label, as_tuple=False).flatten()
    gen = torch.Generator().manual_seed(seed)
    if len(pos) > max_pos:
        pos = pos[torch.randperm(len(pos), generator=gen)[:max_pos]]
    if len(neg) > max_neg:
        neg = neg[torch.randperm(len(neg), generator=gen)[:max_neg]]
    return torch.cat([pos, neg], dim=0)


def train_add_b2(train_tensors: dict[str, torch.Tensor], device: torch.device, seed: int) -> tuple[AddRepairScorer, list[dict[str, Any]]]:
    idx = balanced_add_indices(train_tensors, seed=seed)
    x_cpu = feature_matrix(train_tensors, idx).float()
    y_cpu = train_tensors["GT_occ"][idx].float()
    front = train_tensors["front_region"][idx].float()
    future = train_tensors["future_h4h6"][idx].float()
    strong = train_tensors["strong_add_candidate"][idx].float()
    # Positives in front/future are the recall-repair target; negatives there stay weighted to preserve precision.
    sample_weight = 1.0 + y_cpu * (1.5 * front + 2.0 * future + 0.5 * strong) + (1.0 - y_cpu) * (0.4 * front + 0.5 * future)
    if device.type == "cuda":
        x_cpu = x_cpu.pin_memory()
        y_cpu = y_cpu.pin_memory()
        sample_weight = sample_weight.pin_memory()
    model = AddRepairScorer(x_cpu.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    pos = float(y_cpu.sum().item())
    neg = float(len(y_cpu) - pos)
    pos_weight = torch.tensor([sw14e.safe_div(neg, max(pos, 1.0))], device=device).clamp(1.0, 50.0)
    rows: list[dict[str, Any]] = []
    batch_size = 262_144
    for epoch in range(5):
        gen = torch.Generator().manual_seed(seed * 100 + epoch)
        order = torch.randperm(len(y_cpu), generator=gen)
        losses: list[float] = []
        rank_losses: list[float] = []
        for start in range(0, len(order), batch_size):
            bidx = order[start : start + batch_size]
            xb = x_cpu[bidx].to(device, non_blocking=True)
            yb = y_cpu[bidx].to(device, non_blocking=True)
            wb = sample_weight[bidx].to(device, non_blocking=True)
            logits = model(xb)
            bce = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight, reduction="none")
            bce = (bce * wb).sum() / wb.sum().clamp_min(1.0)
            pos_logits = logits[yb > 0.5]
            neg_logits = logits[yb <= 0.5]
            if len(pos_logits) and len(neg_logits):
                k = min(len(pos_logits), len(neg_logits), 8192)
                rank_loss = F.relu(0.30 - pos_logits[:k] + neg_logits[:k]).mean()
            else:
                rank_loss = logits.new_tensor(0.0)
            loss = bce + 1.25 * rank_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            rank_losses.append(float(rank_loss.detach().cpu().item()))
        rows.append(
            {
                "head": "add_b2_recall_repair",
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "rank_loss": float(np.mean(rank_losses)),
                "train_rows": int(len(y_cpu)),
                "positive_rate": sw14e.safe_div(pos, len(y_cpu)),
            }
        )
    return model.eval(), rows


@torch.inference_mode()
def score_add_b2(model: AddRepairScorer, tensors: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
    mask = tensors["strong_add_candidate"].bool() | tensors["medium_add_candidate"].bool()
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    scores = torch.full((len(mask),), -torch.inf, dtype=torch.float32)
    batch = 524_288
    for start in range(0, len(idx), batch):
        chunk = idx[start : start + batch]
        xb = feature_matrix(tensors, chunk).float()
        if device.type == "cuda":
            xb = xb.pin_memory()
        scores[chunk] = torch.sigmoid(model(xb.to(device, non_blocking=True))).detach().cpu()
    return scores


def topk_precision_rows(tensors: dict[str, torch.Tensor], scores: dict[str, torch.Tensor], split: str) -> list[dict[str, Any]]:
    add_mask = tensors["strong_add_candidate"].bool() | tensors["medium_add_candidate"].bool()
    label = tensors["GT_occ"].bool()
    front = tensors["front_region"].bool()
    future = tensors["future_h4h6"].bool()
    rows: list[dict[str, Any]] = []
    for score_name, score in scores.items():
        idx = torch.nonzero(add_mask, as_tuple=False).flatten()
        order = idx[torch.argsort(score[idx], descending=True)]
        for k in [1000, 5000, 10000, 25000, 50000, 100000, 250000]:
            if len(order) < k:
                continue
            top = order[:k]
            rows.append(
                {
                    "split": split,
                    "score_name": score_name,
                    "topk": k,
                    "precision": float(label[top].float().mean().item()),
                    "front_precision": float(label[top][front[top]].float().mean().item()) if bool(front[top].any().item()) else 0.0,
                    "future_precision": float(label[top][future[top]].float().mean().item()) if bool(future[top].any().item()) else 0.0,
                    "front_count": int(front[top].sum().item()),
                    "future_count": int(future[top].sum().item()),
                }
            )
    return rows


def run_selection_search(val_table: dict[str, Any], base_scores: dict[str, torch.Tensor], add_scores: dict[str, torch.Tensor]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for score_name, add_score in add_scores.items():
        for max_sup in [0.02, 0.03, 0.04]:
            for bal in [0.50, 0.75, 1.0, 1.25]:
                for floor in [32, 64, 128]:
                    for strength in ["strong", "strong_medium"]:
                        cfg = sw14e.Config(max_suppress_ratio=max_sup, add_suppress_balance=bal, add_fixed_budget=floor)
                        budget = {
                            "name": f"b2_{score_name}_sup{max_sup}_bal{bal}_floor{floor}_{strength}",
                            "suppress_topk_ratio": 1.0,
                            "add_topk_ratio": 1.0,
                            "add_strength": strength,
                        }
                        _, summary = sw14e.evaluate_selection("val", val_table, add_score, base_scores["suppress_scores"], budget, cfg)
                        recall_repair_score = (
                            summary["mean_front_fn_reduction_rate"]
                            + summary["mean_future_h4h6_fn_reduction_rate"]
                            + summary["mean_fn_reduction_rate"]
                            + 0.25 * summary["mean_fp_reduction_rate"]
                            - max(0.0, summary["mean_broken_correct_rate"] - 0.001) * 5.0
                        )
                        row = {
                            **summary,
                            "add_score_name": score_name,
                            "max_suppress_ratio": max_sup,
                            "add_suppress_balance": bal,
                            "add_fixed_budget": floor,
                            "add_strength": strength,
                            "recall_repair_score": recall_repair_score,
                        }
                        rows.append(row)
                        ok = summary["safety_pass_all"] and summary["recall_nonregression_pass"]
                        improves_recall = summary["mean_front_fn_reduction_rate"] >= 0.00879 and summary["mean_future_h4h6_fn_reduction_rate"] >= -0.001
                        if ok and improves_recall and (best is None or recall_repair_score > best["recall_repair_score"]):
                            best = row
    return rows, best


def plot_outputs(rows: list[dict[str, Any]]) -> None:
    safe = [r for r in rows if r["safety_pass_all"] and r["recall_nonregression_pass"]]
    safe = sorted(safe, key=lambda r: r["recall_repair_score"], reverse=True)[:8]
    if not safe:
        return
    labels = [r["add_score_name"][:10] + f"_{r['max_suppress_ratio']}" for r in safe]
    plt.figure(figsize=(8, 4))
    plt.bar(labels, [float(r["mean_front_fn_reduction_rate"]) for r in safe], label="front FN")
    plt.plot(labels, [float(r["mean_future_h4h6_fn_reduction_rate"]) for r in safe], marker="o", label="future h4/h6 FN")
    plt.xticks(rotation=25, ha="right")
    plt.title("SW14D-B2 add recall repair candidates")
    plt.legend()
    plt.subplots_adjust(bottom=0.30, left=0.12, right=0.96, top=0.88)
    plt.savefig(FIGURES_DIR / "sw14d_b2_add_recall_repair.png", dpi=180)
    plt.close()


def main() -> None:
    seed = 37
    seed_everything(seed)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_table = sw14e.load_tensor_table(sw14e.table_path("train"))
    val_table = sw14e.load_tensor_table(sw14e.table_path("val"))
    base_scores = torch.load(ARTIFACTS_DIR / "sw14d_b_batched_scores.pt", map_location="cpu", weights_only=False)

    model, train_rows = train_add_b2(train_table["tensors"], device, seed)
    write_csv(REPORTS_DIR / "sw14d_b2_add_training_log.csv", train_rows)
    torch.save({"state_dict": model.state_dict(), "features": B2_ADD_FEATURES, "seed": seed}, CHECKPOINT_DIR / "sw14d_b2_add_recall_repair.pth")
    train_b2 = score_add_b2(model, train_table["tensors"], device)
    val_b2 = score_add_b2(model, val_table["tensors"], device)
    torch.save({"train_add_b2_scores": train_b2, "val_add_b2_scores": val_b2}, ARTIFACTS_DIR / "sw14d_b2_add_recall_scores.pt")

    val_tensors = val_table["tensors"]
    add_mask = val_tensors["strong_add_candidate"].bool() | val_tensors["medium_add_candidate"].bool()
    rule = normalize_score(add_rule_score(val_tensors), add_mask)
    b1 = normalize_score(base_scores["val"]["add_scores"], add_mask)
    b2 = normalize_score(val_b2, add_mask)
    add_score_variants = {
        "b1_model": base_scores["val"]["add_scores"],
        "b2_recall": val_b2,
        "b2_blend_rule25": 0.75 * b2 + 0.25 * rule,
        "b2_blend_b1": 0.60 * b2 + 0.40 * b1,
        "b2_front_future_boost": b2 + 0.06 * val_tensors["front_region"].float() + 0.08 * val_tensors["future_h4h6"].float(),
        "rule_high_precision": rule,
    }
    precision_rows = topk_precision_rows(val_tensors, add_score_variants, "val")
    write_csv(REPORTS_DIR / "sw14d_b2_add_topk_precision.csv", precision_rows)

    search_rows, best = run_selection_search(val_table, base_scores["val"], add_score_variants)
    write_csv(REPORTS_DIR / "sw14d_b2_add_recall_repair_search.csv", search_rows)
    plot_outputs(search_rows)

    current = json.loads((REPORTS_DIR / "sw14d_b_final_decision.json").read_text(encoding="utf-8"))["selected_budget"]
    if best is None:
        decision = "SW14D_B2_ADD_0_NO_SAFE_RECALL_IMPROVEMENT"
        best_payload: dict[str, Any] = {}
    else:
        net_improved = best["mean_net_score"] > current["mean_net_score"]
        fn_improved = best["mean_fn_reduction_rate"] > current["mean_fn_reduction_rate"]
        future_improved = best["mean_future_h4h6_fn_reduction_rate"] > current["mean_future_h4h6_fn_reduction_rate"]
        if net_improved and fn_improved and future_improved:
            decision = "SW14D_B2_ADD_3_RECALL_AND_NET_IMPROVE"
        elif fn_improved or future_improved:
            decision = "SW14D_B2_ADD_2_RECALL_IMPROVES_NET_LOWER"
        elif best["mean_front_fn_reduction_rate"] > current["mean_front_fn_reduction_rate"]:
            decision = "SW14D_B2_ADD_1_FRONT_GAIN_ONLY"
        else:
            decision = "SW14D_B2_ADD_0_NO_SAFE_RECALL_IMPROVEMENT"
        best_payload = best
    final = {
        "decision": decision,
        "best": best_payload,
        "current_sw14d_b": current,
        "checkpoint": str(CHECKPOINT_DIR / "sw14d_b2_add_recall_repair.pth"),
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "whether_eval_debug_allowed": decision in {"SW14D_B2_ADD_3_RECALL_AND_NET_IMPROVE"},
        "whether_resume_claim_allowed": False,
        "sw13_remains_main_result": True,
        "focus": "add / recall repair only; suppress score is reused from SW14D-B.",
    }
    write_json(REPORTS_DIR / "sw14d_b2_add_recall_repair_final_decision.json", final)
    write_md(
        REPORTS_DIR / "stage_sw14d_b2_add_recall_repair_report.md",
        "\n".join(
            [
                "# SW14D-B2 Add Recall Repair",
                "",
                "This stage strengthens add/recall repair on the existing SW14E candidate table. No eval_debug/core was run.",
                f"- decision: `{decision}`",
                f"- best: `{best_payload}`",
                f"- current SW14D-B front FN: `{current['mean_front_fn_reduction_rate']}`",
                f"- current SW14D-B future h4/h6 FN: `{current['mean_future_h4h6_fn_reduction_rate']}`",
                f"- current SW14D-B net: `{current['mean_net_score']}`",
                "",
            ]
        ),
    )
    print(f"[sw14d-b2] decision {decision}", flush=True)


if __name__ == "__main__":
    main()
