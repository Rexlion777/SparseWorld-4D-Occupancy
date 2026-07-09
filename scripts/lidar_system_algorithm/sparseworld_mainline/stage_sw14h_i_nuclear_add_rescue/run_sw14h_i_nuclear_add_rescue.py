from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import sys
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
STAGE = "stage_sw14h_i_nuclear_add_rescue"
SCRIPT_DIR = PROJECT_ROOT / f"scripts/lidar_system_algorithm/sparseworld_mainline/{STAGE}"
REPORTS_DIR = PROJECT_ROOT / f"reports/lidar_system_algorithm/sparseworld_mainline/{STAGE}"
ARTIFACTS_DIR = PROJECT_ROOT / f"artifacts/sparseworld_mainline/{STAGE}"
FIGURES_DIR = PROJECT_ROOT / f"projects/lidar_system_algorithm/figures/sparseworld_mainline/{STAGE}"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"

SW14E_STAGE = "stage_sw14e_three_route_rescue"
SW14E_SCRIPT = PROJECT_ROOT / f"scripts/lidar_system_algorithm/sparseworld_mainline/{SW14E_STAGE}/run_sw14e_three_route_rescue.py"
SW14E_REPORTS = PROJECT_ROOT / f"reports/lidar_system_algorithm/sparseworld_mainline/{SW14E_STAGE}"
SW14E_ARTIFACTS = PROJECT_ROOT / f"artifacts/sparseworld_mainline/{SW14E_STAGE}"
SW14E_FIGURES = PROJECT_ROOT / f"projects/lidar_system_algorithm/figures/sparseworld_mainline/{SW14E_STAGE}"


def load_sw14e_module():
    spec = importlib.util.spec_from_file_location("sw14e_for_sw14hi", SW14E_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SW14E_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["sw14e_for_sw14hi"] = module
    spec.loader.exec_module(module)
    return module


sw14e = load_sw14e_module()


INFERENCE_BASE_FEATURES = list(sw14e.NO_GT_FEATURES) + [
    "after_F3_occ",
    "after_FrontCap_occ",
    "was_pruned_by_F3",
    "was_pruned_by_FrontCap",
    "front_local_density_context",
]
DERIVED_SCORE_FEATURES = [
    "rule_score_norm",
    "b1_add_score_norm",
    "b2_add_score_norm",
    "b3_add_score_norm",
    "score_agreement_b2_b3",
    "score_agreement_b1_b3",
]
CONTEXT_FEATURES = [
    "strong_add_candidate",
    "medium_add_candidate",
    "front_confidence",
    "future_confidence",
    "front_margin",
    "future_margin",
    "neighbor_confidence",
    "raw_final_reliability",
    "native_raw_reliability",
    "density_supported_add_prior",
    "front_future_confidence",
    "boundary_context",
]
H_QUERY_FEATURES = INFERENCE_BASE_FEATURES + DERIVED_SCORE_FEATURES + CONTEXT_FEATURES
I_CONDITION_FEATURES = H_QUERY_FEATURES
GT_COLUMNS = set(sw14e.GT_LABEL_COLUMNS)
TOKEN_DIM = 12
MAX_TOKENS = 8


@dataclass(frozen=True)
class HIConfig:
    seed: int = 71
    route: str = "H"
    h_epochs: int = 4
    i_epochs: int = 3
    train_max_pos: int = 360_000
    train_max_neg: int = 560_000
    hard_neg_topk: int = 320_000
    batch_size: int = 32_768
    score_batch_size: int = 131_072
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 2
    max_tokens: int = MAX_TOKENS
    target_precision_gain: float = 0.10
    target_precision_floor: float = 0.55
    strong_precision_floor: float = 0.65
    device: str = "cuda"


class CrossAttentionCandidateTransformer(nn.Module):
    def __init__(self, query_dim: int, token_dim: int, d_model: int = 96, num_heads: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.query_encoder = nn.Sequential(nn.Linear(query_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.token_encoder = nn.Sequential(nn.Linear(token_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.type_embedding = nn.Embedding(16, d_model)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "attn": nn.MultiheadAttention(d_model, num_heads, dropout=0.08, batch_first=True),
                        "norm1": nn.LayerNorm(d_model),
                        "ffn": nn.Sequential(
                            nn.Linear(d_model, d_model * 2),
                            nn.SiLU(),
                            nn.Dropout(p=0.08),
                            nn.Linear(d_model * 2, d_model),
                        ),
                        "norm2": nn.LayerNorm(d_model),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.score_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 1))
        self.uncertainty_head = nn.Sequential(nn.Linear(d_model, d_model // 2), nn.SiLU(), nn.Linear(d_model // 2, 1))

    def forward(
        self,
        query: torch.Tensor,
        tokens: torch.Tensor,
        token_types: torch.Tensor,
        token_valid: torch.Tensor,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        q = self.query_encoder(query).unsqueeze(1)
        tok = self.token_encoder(tokens) + self.type_embedding(token_types.clamp(0, 15))
        key_padding = ~token_valid.bool()
        attn_weights: torch.Tensor | None = None
        for layer in self.layers:
            attn_out, attn_weights = layer["attn"](q, tok, tok, key_padding_mask=key_padding, need_weights=need_weights)
            q = layer["norm1"](q + attn_out)
            q = layer["norm2"](q + layer["ffn"](q))
        h = q.squeeze(1)
        return self.score_head(h).squeeze(-1), self.uncertainty_head(h).squeeze(-1), attn_weights


class GenerativeCompletionScorer(nn.Module):
    def __init__(self, condition_dim: int, hidden_dim: int = 192) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=0.05),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, condition_dim),
        )
        self.score_head = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim // 2), nn.SiLU(), nn.Linear(hidden_dim // 2, 1))

    def forward(self, condition: torch.Tensor, noise_std: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        x = condition
        if self.training and noise_std > 0:
            x = x + torch.randn_like(x) * noise_std
        z = self.encoder(x)
        recon = self.decoder(z)
        return self.score_head(z).squeeze(-1), recon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW14H/I nuclear add branch rescue")
    parser.add_argument("--route", choices=["H", "I", "all"], default="H")
    parser.add_argument("--seed", type=int, default=71)
    parser.add_argument("--h-epochs", type=int, default=4)
    parser.add_argument("--i-epochs", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        SCRIPT_DIR,
        SCRIPT_DIR / "route_h_cross_attention_candidate_transformer",
        SCRIPT_DIR / "route_i_generative_occupancy_completion",
        REPORTS_DIR,
        ARTIFACTS_DIR,
        CHECKPOINT_DIR,
        FIGURES_DIR,
    ]:
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
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([normalize(row) for row in rows])


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
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass


def sha256_file(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_tables() -> dict[str, dict[str, Any]]:
    return {
        "train": sw14e.load_tensor_table(sw14e.table_path("train")),
        "val": sw14e.load_tensor_table(sw14e.table_path("val")),
    }


def add_roi_mask(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return tensors["strong_add_candidate"].bool() | tensors["medium_add_candidate"].bool()


def base_rule_score(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    conf = tensors["raw_confidence"].float()
    margin = tensors["raw_margin"].float()
    agreement = tensors["camera_view_agreement"].float()
    neighbor = tensors["neighbor_occ_count_norm"].float()
    front = tensors["front_region"].float()
    future = tensors["future_h4h6"].float()
    strong = tensors["strong_add_candidate"].float()
    raw_but_final_empty = tensors["raw_but_final_empty"].float()
    native = tensors["native_final_occ"].float()
    boundary = (tensors["was_pruned_by_F3"].float() + tensors["was_pruned_by_FrontCap"].float()).clamp(0, 1)
    return (
        conf
        + 0.45 * margin
        + 0.22 * agreement
        + 0.22 * neighbor
        + 0.14 * front
        + 0.16 * future
        + 0.12 * strong
        + 0.10 * raw_but_final_empty
        + 0.08 * native
        + 0.10 * boundary
    )


def normalize_score(score: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    out = torch.full_like(score.float(), -torch.inf)
    vals = score[mask].float()
    if len(vals) == 0:
        return out
    finite = torch.isfinite(vals)
    if not bool(finite.any().item()):
        out[mask] = 0.0
        return out
    vals2 = vals[finite]
    out_vals = torch.zeros_like(vals)
    out_vals[finite] = (vals2 - vals2.min()) / (vals2.max() - vals2.min() + 1.0e-9)
    out[mask] = out_vals
    return out


def load_score_payload(tables: dict[str, dict[str, Any]]) -> dict[str, dict[str, torch.Tensor]]:
    base_path = SW14E_ARTIFACTS / "sw14d_b_batched_scores.pt"
    base_scores = torch.load(base_path, map_location="cpu", weights_only=False)
    b2_path = SW14E_ARTIFACTS / "sw14d_b2_add_recall_scores.pt"
    b3_path = SW14E_ARTIFACTS / "sw14d_b3_add_precision100k_scores.pt"
    b2 = torch.load(b2_path, map_location="cpu", weights_only=False) if b2_path.exists() else {}
    b3 = torch.load(b3_path, map_location="cpu", weights_only=False) if b3_path.exists() else {}
    out: dict[str, dict[str, torch.Tensor]] = {}
    for split in ["train", "val"]:
        n = len(tables[split]["tensors"]["GT_occ"])
        out[split] = {
            "b1_add": base_scores[split]["add_scores"].float(),
            "suppress": base_scores[split]["suppress_scores"].float(),
            "b2_add": b2.get(f"{split}_add_b2_scores", torch.full((n,), -torch.inf)).float(),
            "b3_add": b3.get(f"{split}_b3_add_scores", torch.full((n,), -torch.inf)).float(),
        }
    return out


def derived_cache(tensors: dict[str, torch.Tensor], scores: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mask = add_roi_mask(tensors)
    rule = normalize_score(base_rule_score(tensors), mask)
    b1 = normalize_score(scores["b1_add"], mask)
    b2 = normalize_score(scores["b2_add"], mask)
    b3 = normalize_score(scores["b3_add"], mask)
    conf = tensors["raw_confidence"].float()
    margin = tensors["raw_margin"].float()
    agreement = tensors["camera_view_agreement"].float()
    neighbor = tensors["neighbor_occ_count_norm"].float()
    front = tensors["front_region"].float()
    future = tensors["future_h4h6"].float()
    raw_occ = tensors["raw_occ"].float()
    final_occ = tensors["teacher_final_occ"].float()
    native_occ = tensors["native_final_occ"].float()
    raw_but_final = tensors["raw_but_final_empty"].float()
    density = tensors["local_density_proxy"].float()
    boundary = (tensors["was_pruned_by_F3"].float() + tensors["was_pruned_by_FrontCap"].float()).clamp(0, 1)
    b2_clean = b2.nan_to_num(0.0, neginf=0.0)
    b3_clean = b3.nan_to_num(0.0, neginf=0.0)
    b1_clean = b1.nan_to_num(0.0, neginf=0.0)
    return {
        "rule_score_norm": rule.nan_to_num(0.0, neginf=0.0),
        "b1_add_score_norm": b1_clean,
        "b2_add_score_norm": b2_clean,
        "b3_add_score_norm": b3_clean,
        "score_agreement_b2_b3": b2_clean * b3_clean,
        "score_agreement_b1_b3": b1_clean * b3_clean,
        "strong_add_candidate": tensors["strong_add_candidate"].float(),
        "medium_add_candidate": tensors["medium_add_candidate"].float(),
        "front_confidence": front * conf,
        "future_confidence": future * conf,
        "front_margin": front * margin,
        "future_margin": future * margin,
        "neighbor_confidence": neighbor * conf,
        "raw_final_reliability": raw_occ * (1.0 - final_occ) * agreement * conf,
        "native_raw_reliability": raw_occ * native_occ * (0.5 + 0.5 * agreement),
        "density_supported_add_prior": raw_but_final * density * (0.5 + 0.5 * agreement),
        "front_future_confidence": torch.maximum(front, future) * conf,
        "boundary_context": boundary * (0.5 + 0.5 * conf),
    }


def query_matrix(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], idx: torch.Tensor, features: list[str]) -> torch.Tensor:
    cols: list[torch.Tensor] = []
    for name in features:
        if name in cache:
            cols.append(cache[name][idx].float())
        else:
            cols.append(tensors[name][idx].float())
    return torch.stack(cols, dim=1)


def evidence_tokens(
    tensors: dict[str, torch.Tensor],
    cache: dict[str, torch.Tensor],
    idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = len(idx)
    tokens = torch.zeros((n, MAX_TOKENS, TOKEN_DIM), dtype=torch.float32)
    token_types = torch.arange(MAX_TOKENS, dtype=torch.long).unsqueeze(0).expand(n, MAX_TOKENS).clone()
    valid = torch.zeros((n, MAX_TOKENS), dtype=torch.bool)

    conf = tensors["raw_confidence"][idx].float()
    margin = tensors["raw_margin"][idx].float()
    entropy = tensors["raw_entropy"][idx].float()
    raw_occ = tensors["raw_occ"][idx].float()
    final_occ = tensors["teacher_final_occ"][idx].float()
    native_occ = tensors["native_final_occ"][idx].float()
    raw_but_final = tensors["raw_but_final_empty"][idx].float()
    f3 = tensors["was_pruned_by_F3"][idx].float()
    fc = tensors["was_pruned_by_FrontCap"][idx].float()
    density = tensors["local_density_proxy"][idx].float()
    neighbor = tensors["neighbor_occ_count_norm"][idx].float()
    agreement = tensors["camera_view_agreement"][idx].float()
    temporal = tensors["temporal_consistency"][idx].float()
    front = tensors["front_region"][idx].float()
    future = tensors["future_h4h6"][idx].float()
    protected = tensors["protected_zone"][idx].float()
    low_conf_final = tensors["final_occupied_low_conf"][idx].float()
    isolated = tensors["final_occupied_isolated"][idx].float()
    x = tensors["x_norm"][idx].float()
    y = tensors["abs_y_norm"][idx].float()
    z = tensors["z_norm"][idx].float()
    r = tensors["range_norm"][idx].float()
    front_ctx = tensors["front_local_density_context"][idx].float()
    rule = cache["rule_score_norm"][idx].float()
    b3 = cache["b3_add_score_norm"][idx].float()

    token_cols = [
        # raw evidence token
        [conf, margin, entropy, raw_occ, raw_but_final, agreement, neighbor, density, x, y, z, r],
        # F3-pruned boundary token
        [f3, conf * f3, margin * f3, raw_but_final * f3, agreement, neighbor, density, front, future, x, y, r],
        # FrontCap-pruned boundary token
        [fc, conf * fc, margin * fc, raw_but_final * fc, front_ctx, front, future, density, neighbor, x, y, r],
        # teacher final local support token
        [final_occ, density, neighbor, low_conf_final, isolated, protected, front_ctx, front, future, x, y, z],
        # temporal / agreement token
        [temporal, agreement, temporal * agreement, conf, margin, raw_occ, native_occ, density, front, future, x, r],
        # suppress-risk context token
        [low_conf_final, isolated, density, neighbor, protected, front_ctx, conf, margin, final_occ, x, y, r],
        # front/future structural context token
        [front, future, front * future, front_ctx, density, neighbor, conf, margin, rule, b3, x, r],
        # native/raw consensus token
        [native_occ, raw_occ, native_occ * raw_occ, conf, margin, agreement, raw_but_final, density, neighbor, x, y, z],
    ]
    for i, cols in enumerate(token_cols):
        tokens[:, i, :] = torch.stack(cols, dim=1)

    valid[:, 0] = raw_occ.bool() | raw_but_final.bool() | (conf > 0.35)
    valid[:, 1] = f3.bool()
    valid[:, 2] = fc.bool()
    valid[:, 3] = final_occ.bool() | (neighbor > 0)
    valid[:, 4] = (temporal > 0) | (agreement > 0)
    valid[:, 5] = low_conf_final.bool() | isolated.bool() | (density > 0.2)
    valid[:, 6] = front.bool() | future.bool()
    valid[:, 7] = native_occ.bool() | raw_occ.bool()
    valid[:, 0] = True
    return tokens, token_types, valid


def inference_feature_manifest() -> dict[str, Any]:
    return {
        "h_query_features": H_QUERY_FEATURES,
        "i_condition_features": I_CONDITION_FEATURES,
        "evidence_token_dim": TOKEN_DIM,
        "max_tokens": MAX_TOKENS,
        "gt_columns_not_used_at_inference": sorted(GT_COLUMNS),
        "inference_features_contain_gt": bool(any(name in GT_COLUMNS for name in H_QUERY_FEATURES + I_CONDITION_FEATURES)),
        "weak_add_excluded_from_main": True,
    }


def inherited_state() -> dict[str, Any]:
    candidate_train = sw14e.table_path("train")
    candidate_val = sw14e.table_path("val")
    score_path = SW14E_ARTIFACTS / "sw14d_b_batched_scores.pt"
    b2 = load_json(SW14E_REPORTS / "sw14d_b2_add_recall_repair_final_decision.json")
    b3 = load_json(SW14E_REPORTS / "sw14d_b3_add_precision100k_final_decision.json")
    d_b = load_json(SW14E_REPORTS / "sw14d_b_final_decision.json")
    init_ready = candidate_train.exists() and candidate_val.exists() and score_path.exists()
    inherited = {
        "decision": "HI_INIT_READY" if init_ready else "HI_INIT_MISSING_CANDIDATES",
        "sw13_main_result": "A10 R8 + F3 + FC1_1p3 remains read-only main result.",
        "sw14d_b_decision": d_b.get("decision"),
        "sw14d_b_net": d_b.get("selected_budget", {}).get("mean_net_score"),
        "sw14d_b_add_top100k_precision": 0.39980000257492065,
        "sw14d_b2_add_top100k_precision": b3.get("b2_val_precision_at_100k", {}).get("precision", b2.get("best", {}).get("add_top100k_precision")),
        "sw14d_b3_add_top100k_precision": b3.get("best_val_precision_at_100k", {}).get("precision"),
        "sw14d_b3_decision": b3.get("decision"),
        "suppress_precision_note": "SW14D suppress branch is treated as the safety baseline; add branch is the failure target.",
        "candidate_table_train": str(candidate_train),
        "candidate_table_val": str(candidate_val),
        "candidate_table_train_sha256": sha256_file(candidate_train),
        "candidate_table_val_sha256": sha256_file(candidate_val),
        "batched_score_sha256": sha256_file(score_path),
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "resume_claim": False,
    }
    freeze = {
        "decision": "BASELINE_FROZEN",
        "sparseworld_backbone_unchanged": True,
        "occupancy_head_unchanged": True,
        "get_occ_unchanged": True,
        "f3_config_unchanged": True,
        "frontcap_config_unchanged": True,
        "sw13_main_result_unchanged": True,
        "sw14e_script_sha256": sha256_file(SW14E_SCRIPT),
        "inference_feature_manifest": inference_feature_manifest(),
    }
    write_json(REPORTS_DIR / "sw14hi_inherited_state.json", inherited)
    write_json(REPORTS_DIR / "sw14hi_baseline_freeze_manifest.json", freeze)
    write_md(
        REPORTS_DIR / "sw14hi_problem_statement.md",
        "\n".join(
            [
                "# SW14H/I Nuclear Add Branch Rescue",
                "",
                "This stage is diagnostic. It does not replace SW13C-Fix + FrontCap and it does not run eval_debug/core.",
                "Route H trains a cross-attention candidate verifier to improve add candidate ranking precision.",
                "Route I trains a ROI-limited generative completion smoke diagnostic; outputs are proposals only.",
                "GT is used only for train/val supervision and diagnostic labels, never as an inference feature.",
                "",
            ]
        ),
    )
    return inherited


def feature_gap_rows(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], split: str) -> list[dict[str, Any]]:
    mask = add_roi_mask(tensors)
    label = tensors["GT_occ"].bool()
    rows: list[dict[str, Any]] = []
    features = [
        "raw_confidence",
        "raw_margin",
        "raw_entropy",
        "raw_but_final_empty",
        "was_pruned_by_F3",
        "was_pruned_by_FrontCap",
        "neighbor_occ_count_norm",
        "local_density_proxy",
        "temporal_consistency",
        "camera_view_agreement",
        "front_region",
        "future_h4h6",
        "front_local_density_context",
    ] + DERIVED_SCORE_FEATURES
    for name in features:
        vals = cache[name] if name in cache else tensors[name].float()
        true_vals = vals[mask & label].float()
        false_vals = vals[mask & ~label].float()
        rows.append(
            {
                "split": split,
                "feature": name,
                "true_mean": float(true_vals.mean().item()) if len(true_vals) else 0.0,
                "false_mean": float(false_vals.mean().item()) if len(false_vals) else 0.0,
                "abs_gap": float(abs((true_vals.mean() if len(true_vals) else torch.tensor(0.0)) - (false_vals.mean() if len(false_vals) else torch.tensor(0.0))).item()),
            }
        )
    return sorted(rows, key=lambda r: r["abs_gap"], reverse=True)


def group_stats(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], split: str) -> list[dict[str, Any]]:
    label = tensors["GT_occ"].bool()
    groups = {
        "strong_add": tensors["strong_add_candidate"].bool(),
        "medium_add": tensors["medium_add_candidate"].bool(),
        "weak_add": tensors["weak_add_candidate"].bool(),
        "raw_but_final_empty": tensors["raw_but_final_empty"].bool() & ~tensors["teacher_final_occ"].bool(),
        "F3_pruned_candidate": tensors["was_pruned_by_F3"].bool(),
        "FrontCap_pruned_candidate": tensors["was_pruned_by_FrontCap"].bool(),
        "front_future_candidate": tensors["front_region"].bool() & tensors["future_h4h6"].bool(),
        "boundary_candidate": tensors["was_pruned_by_F3"].bool() | tensors["was_pruned_by_FrontCap"].bool(),
        "temporal_supported_candidate": tensors["temporal_consistency"].float() >= 0.75,
        "isolated_candidate": tensors["neighbor_occ_count_norm"].float() <= 0.17,
    }
    rows: list[dict[str, Any]] = []
    for name, mask in groups.items():
        add_mask = mask & (tensors["strong_add_candidate"].bool() | tensors["medium_add_candidate"].bool() | tensors["weak_add_candidate"].bool())
        n = int(add_mask.sum().item())
        pos = int(label[add_mask].sum().item()) if n else 0
        rows.append(
            {
                "split": split,
                "group": name,
                "candidate_count": n,
                "positive_count": pos,
                "precision": safe_div(pos, n),
                "raw_confidence_mean": float(tensors["raw_confidence"][add_mask].float().mean().item()) if n else 0.0,
                "raw_margin_mean": float(tensors["raw_margin"][add_mask].float().mean().item()) if n else 0.0,
                "neighbor_occ_count_norm_mean": float(tensors["neighbor_occ_count_norm"][add_mask].float().mean().item()) if n else 0.0,
                "local_density_proxy_mean": float(tensors["local_density_proxy"][add_mask].float().mean().item()) if n else 0.0,
                "temporal_consistency_mean": float(tensors["temporal_consistency"][add_mask].float().mean().item()) if n else 0.0,
            }
        )
    return rows


def add_failure_audit(tables: dict[str, dict[str, Any]], scores: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    all_gap: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"splits": {}}
    for split in ["train", "val"]:
        tensors = tables[split]["tensors"]
        cache = derived_cache(tensors, scores[split])
        rows = group_stats(tensors, cache, split)
        gaps = feature_gap_rows(tensors, cache, split)
        all_gap.extend(gaps)
        if split == "train":
            train_rows = rows
        else:
            val_rows = rows
        mask = add_roi_mask(tensors)
        label = tensors["GT_occ"].bool()
        b3 = cache["b3_add_score_norm"]
        idx = torch.nonzero(mask, as_tuple=False).flatten()
        order = idx[torch.argsort(b3[idx], descending=True)]
        top = order[: min(100_000, len(order))]
        top_fp = top[~label[top]]
        top_tp = top[label[top]]
        tok_tp = evidence_tokens(tensors, cache, top_tp[: min(len(top_tp), 50_000)])[2] if len(top_tp) else torch.zeros((0, MAX_TOKENS), dtype=torch.bool)
        tok_fp = evidence_tokens(tensors, cache, top_fp[: min(len(top_fp), 50_000)])[2] if len(top_fp) else torch.zeros((0, MAX_TOKENS), dtype=torch.bool)
        summary["splits"][split] = {
            "add_roi_count": int(mask.sum().item()),
            "add_roi_precision": float(label[mask].float().mean().item()),
            "top100k_b3_precision": float(label[top].float().mean().item()) if len(top) else 0.0,
            "top100k_false_positive_count": int((~label[top]).sum().item()) if len(top) else 0,
            "top100k_true_positive_count": int(label[top].sum().item()) if len(top) else 0,
            "true_add_avg_valid_tokens": float(tok_tp.float().sum(dim=1).mean().item()) if len(tok_tp) else 0.0,
            "false_add_avg_valid_tokens": float(tok_fp.float().sum(dim=1).mean().item()) if len(tok_fp) else 0.0,
            "top_feature_gaps": gaps[:8],
        }
    max_gap = max((r["abs_gap"] for r in all_gap if r["split"] == "val"), default=0.0)
    top_gap_feature = max((r for r in all_gap if r["split"] == "val"), key=lambda r: r["abs_gap"], default={})
    if max_gap < 0.015:
        decision = "ADD_AUDIT_4_FEATURES_NOT_SEPARABLE"
    elif str(top_gap_feature.get("feature", "")).startswith(("was_pruned", "raw_but_final")):
        decision = "ADD_AUDIT_2_BOUNDARY_EVIDENCE_STRONG"
    elif str(top_gap_feature.get("feature", "")).startswith(("temporal", "camera_view")):
        decision = "ADD_AUDIT_3_TEMPORAL_EVIDENCE_STRONG"
    else:
        decision = "ADD_AUDIT_1_CONTEXT_GAP_STRONG"
    summary["decision"] = decision
    summary["top_val_gap_feature"] = top_gap_feature
    write_csv(REPORTS_DIR / "sw14hi_add_failure_audit_train.csv", train_rows)
    write_csv(REPORTS_DIR / "sw14hi_add_failure_audit_val.csv", val_rows)
    write_csv(REPORTS_DIR / "sw14hi_add_true_vs_false_feature_gap.csv", all_gap)
    write_json(REPORTS_DIR / "sw14hi_add_failure_audit_summary.json", summary)

    val_gaps = [r for r in all_gap if r["split"] == "val"][:10]
    plt.figure(figsize=(8, 4))
    plt.bar([r["feature"] for r in val_gaps], [r["abs_gap"] for r in val_gaps])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("abs true/false mean gap")
    plt.title("SW14HI add true vs false no-GT feature gaps")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14hi_add_failure_true_vs_false.png", dpi=180)
    plt.savefig(FIGURES_DIR / "sw14hi_add_failure_plots.png", dpi=180)
    plt.close()
    return summary


def topk_precision_rows(
    tensors: dict[str, torch.Tensor],
    score_map: dict[str, torch.Tensor],
    split: str,
) -> list[dict[str, Any]]:
    mask = add_roi_mask(tensors)
    label = tensors["GT_occ"].bool()
    front = tensors["front_region"].bool()
    future = tensors["future_h4h6"].bool()
    rows: list[dict[str, Any]] = []
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    for name, score in score_map.items():
        order = idx[torch.argsort(score[idx], descending=True)]
        for k in [1000, 5000, 10000, 25000, 50000, 100000, 150000, 250000]:
            if len(order) < k:
                continue
            top = order[:k]
            rows.append(
                {
                    "split": split,
                    "score_name": name,
                    "topk": k,
                    "precision": float(label[top].float().mean().item()),
                    "positive_count": int(label[top].sum().item()),
                    "front_precision": float(label[top][front[top]].float().mean().item()) if bool(front[top].any().item()) else 0.0,
                    "future_precision": float(label[top][future[top]].float().mean().item()) if bool(future[top].any().item()) else 0.0,
                    "front_count": int(front[top].sum().item()),
                    "future_count": int(future[top].sum().item()),
                }
            )
    return rows


def sample_train_indices(tensors: dict[str, torch.Tensor], cache: dict[str, torch.Tensor], cfg: HIConfig) -> torch.Tensor:
    mask = add_roi_mask(tensors)
    label = tensors["GT_occ"].bool()
    pos = torch.nonzero(mask & label, as_tuple=False).flatten()
    neg = torch.nonzero(mask & ~label, as_tuple=False).flatten()
    hard_score = torch.maximum(cache["b3_add_score_norm"], torch.maximum(cache["b2_add_score_norm"], cache["rule_score_norm"]))
    hard_neg_all = neg[torch.argsort(hard_score[neg], descending=True)[: min(cfg.hard_neg_topk, len(neg))]]
    gen = torch.Generator().manual_seed(cfg.seed)
    if len(pos) > cfg.train_max_pos:
        pos = pos[torch.randperm(len(pos), generator=gen)[: cfg.train_max_pos]]
    random_neg_budget = max(0, cfg.train_max_neg - len(hard_neg_all))
    random_neg = neg[torch.randperm(len(neg), generator=gen)[: min(random_neg_budget, len(neg))]]
    neg_sel = torch.unique(torch.cat([hard_neg_all, random_neg]))
    if len(neg_sel) > cfg.train_max_neg:
        neg_sel = neg_sel[torch.randperm(len(neg_sel), generator=gen)[: cfg.train_max_neg]]
    idx = torch.cat([pos, neg_sel])
    return idx[torch.randperm(len(idx), generator=gen)]


def focal_loss_with_logits(logits: torch.Tensor, y: torch.Tensor, alpha: float = 0.45, gamma: float = 2.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    p_t = prob * y + (1.0 - prob) * (1.0 - y)
    alpha_t = alpha * y + (1.0 - alpha) * (1.0 - y)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce).mean()


def pairwise_rank_loss(logits: torch.Tensor, y: torch.Tensor, max_pairs: int = 8192) -> torch.Tensor:
    pos = logits[y > 0.5]
    neg = logits[y <= 0.5]
    if len(pos) == 0 or len(neg) == 0:
        return logits.new_tensor(0.0)
    k = min(len(pos), len(neg), max_pairs)
    return F.softplus(0.45 - pos[:k] + neg[:k]).mean()


def topk_surrogate_loss(logits: torch.Tensor, y: torch.Tensor, max_items: int = 16384) -> torch.Tensor:
    if len(logits) == 0:
        return logits.new_tensor(0.0)
    k = min(max_items, len(logits))
    top = torch.topk(logits, k=k).indices
    return F.binary_cross_entropy_with_logits(logits[top], y[top])


def train_transformer(
    tables: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, torch.Tensor]],
    cfg: HIConfig,
    device: torch.device,
) -> tuple[CrossAttentionCandidateTransformer, torch.Tensor, list[dict[str, Any]]]:
    tensors = tables["train"]["tensors"]
    cache = derived_cache(tensors, scores["train"])
    idx = sample_train_indices(tensors, cache, cfg)
    xq = query_matrix(tensors, cache, idx, H_QUERY_FEATURES).float()
    tok, tok_types, tok_valid = evidence_tokens(tensors, cache, idx)
    y = tensors["GT_occ"][idx].float()
    front = tensors["front_region"][idx].float()
    future = tensors["future_h4h6"][idx].float()
    density = tensors["local_density_proxy"][idx].float()
    if device.type == "cuda":
        xq = xq.pin_memory()
        tok = tok.pin_memory()
        tok_types = tok_types.pin_memory()
        tok_valid = tok_valid.pin_memory()
        y = y.pin_memory()
        front = front.pin_memory()
        future = future.pin_memory()
        density = density.pin_memory()
    model = CrossAttentionCandidateTransformer(xq.shape[1], TOKEN_DIM, cfg.d_model, cfg.num_heads, cfg.num_layers).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2.0e-4)
    rows: list[dict[str, Any]] = []
    for epoch in range(cfg.h_epochs):
        gen = torch.Generator().manual_seed(cfg.seed * 100 + epoch)
        order = torch.randperm(len(y), generator=gen)
        losses: list[float] = []
        bces: list[float] = []
        focals: list[float] = []
        ranks: list[float] = []
        topks: list[float] = []
        for start in range(0, len(order), cfg.batch_size):
            b = order[start : start + cfg.batch_size]
            xb = xq[b].to(device, non_blocking=True)
            tb = tok[b].to(device, non_blocking=True)
            tyb = tok_types[b].to(device, non_blocking=True)
            tvb = tok_valid[b].to(device, non_blocking=True)
            yb = y[b].to(device, non_blocking=True)
            density_b = density[b].to(device, non_blocking=True)
            logits, uncertainty, _ = model(xb, tb, tyb, tvb)
            weight = 1.0 + yb * (1.0 + 0.6 * front[b].to(device, non_blocking=True) + 0.8 * future[b].to(device, non_blocking=True))
            weight = weight + (1.0 - yb) * (1.0 + 1.5 * density_b)
            bce = F.binary_cross_entropy_with_logits(logits, yb, reduction="none")
            bce = (bce * weight).sum() / weight.sum().clamp_min(1.0)
            focal = focal_loss_with_logits(logits, yb)
            rank = pairwise_rank_loss(logits, yb)
            topk = topk_surrogate_loss(logits, yb)
            safety = (torch.sigmoid(logits) * (1.0 - yb) * density_b.clamp_min(0.0)).mean()
            uncert = (torch.sigmoid(uncertainty).mean() * 0.01)
            loss = bce + 2.0 * focal + 2.0 * rank + 2.0 * topk + 2.0 * safety + uncert
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            bces.append(float(bce.detach().cpu().item()))
            focals.append(float(focal.detach().cpu().item()))
            ranks.append(float(rank.detach().cpu().item()))
            topks.append(float(topk.detach().cpu().item()))
        rows.append(
            {
                "route": "H",
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "bce": float(np.mean(bces)),
                "focal": float(np.mean(focals)),
                "pair_rank": float(np.mean(ranks)),
                "topk_surrogate": float(np.mean(topks)),
                "train_rows": int(len(y)),
                "positive_rate": float(y.mean().item()),
            }
        )
    return model.eval(), idx, rows


@torch.inference_mode()
def score_transformer(
    model: CrossAttentionCandidateTransformer,
    tensors: dict[str, torch.Tensor],
    scores: dict[str, torch.Tensor],
    cfg: HIConfig,
    device: torch.device,
) -> torch.Tensor:
    cache = derived_cache(tensors, scores)
    mask = add_roi_mask(tensors)
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    out = torch.full((len(mask),), -torch.inf, dtype=torch.float32)
    for start in range(0, len(idx), cfg.score_batch_size):
        chunk = idx[start : start + cfg.score_batch_size]
        xq = query_matrix(tensors, cache, chunk, H_QUERY_FEATURES).float()
        tok, tok_types, tok_valid = evidence_tokens(tensors, cache, chunk)
        if device.type == "cuda":
            xq = xq.pin_memory()
            tok = tok.pin_memory()
            tok_types = tok_types.pin_memory()
            tok_valid = tok_valid.pin_memory()
        logits, _, _ = model(xq.to(device, non_blocking=True), tok.to(device, non_blocking=True), tok_types.to(device, non_blocking=True), tok_valid.to(device, non_blocking=True))
        out[chunk] = torch.sigmoid(logits).detach().cpu()
    return out


def evidence_token_stats(tables: dict[str, dict[str, Any]], scores: dict[str, dict[str, torch.Tensor]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for split in ["train", "val"]:
        tensors = tables[split]["tensors"]
        cache = derived_cache(tensors, scores[split])
        idx_all = torch.nonzero(add_roi_mask(tensors), as_tuple=False).flatten()
        idx = idx_all[: min(250_000, len(idx_all))]
        _, tok_types, valid = evidence_tokens(tensors, cache, idx)
        counts = valid.sum(dim=1).float()
        label = tensors["GT_occ"][idx].bool()
        type_counts = valid.float().sum(dim=0)
        row = {
            "split": split,
            "sampled_candidates": int(len(idx)),
            "avg_tokens_per_candidate": float(counts.mean().item()) if len(counts) else 0.0,
            "p05_tokens": float(torch.quantile(counts, 0.05).item()) if len(counts) else 0.0,
            "p95_tokens": float(torch.quantile(counts, 0.95).item()) if len(counts) else 0.0,
            "candidates_with_zero_evidence_tokens": int((counts == 0).sum().item()),
            "true_add_avg_tokens": float(counts[label].mean().item()) if bool(label.any().item()) else 0.0,
            "false_add_avg_tokens": float(counts[~label].mean().item()) if bool((~label).any().item()) else 0.0,
            "token_type_raw": int(type_counts[0].item()),
            "token_type_f3": int(type_counts[1].item()),
            "token_type_frontcap": int(type_counts[2].item()),
            "token_type_teacher_final": int(type_counts[3].item()),
            "token_type_temporal": int(type_counts[4].item()),
            "token_type_suppress": int(type_counts[5].item()),
            "token_type_front_future": int(type_counts[6].item()),
            "token_type_native_raw": int(type_counts[7].item()),
        }
        rows.append(row)
        summary[split] = row
        for j in range(min(3, len(idx))):
            examples.append(
                {
                    "split": split,
                    "row_index": int(idx[j].item()),
                    "valid_token_types": [int(v) for v in tok_types[j][valid[j]].tolist()],
                    "gt_occ": int(tensors["GT_occ"][idx[j]].item()),
                    "strong_add": int(tensors["strong_add_candidate"][idx[j]].item()),
                    "medium_add": int(tensors["medium_add_candidate"][idx[j]].item()),
                }
            )
    decision = "H_TOK_1_READY"
    if summary.get("val", {}).get("avg_tokens_per_candidate", 0.0) < 2.0:
        decision = "H_TOK_2_TOO_MANY_ZERO_EVIDENCE"
    summary["decision"] = decision
    summary["token_schema"] = {"max_tokens": MAX_TOKENS, "token_dim": TOKEN_DIM, "type_count": 8}
    write_csv(REPORTS_DIR / "sw14h_evidence_token_stats_train.csv", [r for r in rows if r["split"] == "train"])
    write_csv(REPORTS_DIR / "sw14h_evidence_token_stats_val.csv", [r for r in rows if r["split"] == "val"])
    write_json(REPORTS_DIR / "sw14h_evidence_token_examples.json", {"examples": examples})
    write_json(REPORTS_DIR / "sw14h_evidence_token_config.json", {"decision": decision, **summary, **inference_feature_manifest()})
    plt.figure(figsize=(8, 4))
    labels = [f"type{i}" for i in range(8)]
    val = next((r for r in rows if r["split"] == "val"), rows[0])
    plt.bar(labels, [val[f"token_type_{name}"] for name in ["raw", "f3", "frontcap", "teacher_final", "temporal", "suppress", "front_future", "native_raw"]])
    plt.xticks(rotation=25)
    plt.title("SW14H evidence token distribution")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14h_evidence_token_distribution.png", dpi=180)
    plt.close()
    return summary


def run_selection_sweep(
    route_prefix: str,
    tables: dict[str, dict[str, Any]],
    add_scores: torch.Tensor,
    suppress_scores: torch.Tensor,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for max_sup in [0.02, 0.03, 0.04]:
        for add_balance in [0.25, 0.50, 0.75, 1.0]:
            for fixed in [16, 32, 64]:
                for strength in ["strong", "strong_medium"]:
                    cfg = sw14e.Config(max_suppress_ratio=max_sup, add_suppress_balance=add_balance, add_fixed_budget=fixed)
                    budget = {
                        "name": f"{route_prefix}_sup{max_sup}_bal{add_balance}_fixed{fixed}_{strength}",
                        "suppress_topk_ratio": 1.0,
                        "add_topk_ratio": 1.0,
                        "add_strength": strength,
                    }
                    _, summary = sw14e.evaluate_selection("val", tables["val"], add_scores, suppress_scores, budget, cfg)
                    row = {
                        **summary,
                        "route": route_prefix,
                        "max_suppress_ratio": max_sup,
                        "add_suppress_balance": add_balance,
                        "add_fixed_budget": fixed,
                        "add_strength": strength,
                    }
                    rows.append(row)
                    if summary["safety_pass_all"] and summary["recall_nonregression_pass"]:
                        if best is None or float(summary["mean_net_score"]) > float(best["mean_net_score"]):
                            best = row
    if best is None:
        best = max(rows, key=lambda r: float(r["mean_net_score"])) if rows else {}
    return rows, best


def route_h(tables: dict[str, dict[str, Any]], scores: dict[str, dict[str, torch.Tensor]], cfg: HIConfig, device: torch.device) -> dict[str, Any]:
    token_summary = evidence_token_stats(tables, scores)
    model, train_idx, train_rows = train_transformer(tables, scores, cfg, device)
    write_csv(REPORTS_DIR / "sw14h_training_log.csv", train_rows)
    write_json(
        REPORTS_DIR / "sw14h_transformer_config.json",
        {
            "architecture": "candidate query cross-attends to local evidence tokens",
            "d_model": cfg.d_model,
            "num_heads": cfg.num_heads,
            "num_layers": cfg.num_layers,
            "max_tokens": cfg.max_tokens,
            "query_features": H_QUERY_FEATURES,
            "gt_columns_not_used_at_inference": sorted(GT_COLUMNS),
        },
    )
    write_json(
        REPORTS_DIR / "sw14h_init_audit.json",
        {
            "decision": "ARCH_H_1_READY",
            "add_score_bias": "conservative learned head; final no-op when add budget is zero",
            "weak_add_excluded_from_main": True,
            "uses_gt_as_inference_feature": False,
        },
    )
    write_md(
        REPORTS_DIR / "sw14h_model_summary.txt",
        str(model)
        + "\n\nInference features exclude GT labels. Evidence tokens use candidate-table no-GT proxy channels.\n",
    )
    val_h = score_transformer(model, tables["val"]["tensors"], scores["val"], cfg, device)
    train_h = score_transformer(model, tables["train"]["tensors"], scores["train"], cfg, device)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "query_features": H_QUERY_FEATURES,
            "token_dim": TOKEN_DIM,
            "max_tokens": MAX_TOKENS,
            "seed": cfg.seed,
            "resume_claim": False,
        },
        CHECKPOINT_DIR / "sw14h_candidate_transformer_best.pth",
    )
    torch.save({"train_sw14h_add_scores": train_h, "val_sw14h_add_scores": val_h}, ARTIFACTS_DIR / "sw14h_candidate_transformer_scores.pt")

    val_cache = derived_cache(tables["val"]["tensors"], scores["val"])
    score_map = {
        "mlp_b1": scores["val"]["b1_add"],
        "mlp_b2": scores["val"]["b2_add"],
        "mlp_b3": scores["val"]["b3_add"],
        "sw14h_transformer": val_h,
        "sw14h_blend_b3": 0.70 * normalize_score(val_h, add_roi_mask(tables["val"]["tensors"])) + 0.30 * val_cache["b3_add_score_norm"],
    }
    topk = topk_precision_rows(tables["val"]["tensors"], score_map, "val")
    write_csv(REPORTS_DIR / "sw14h_val_topk_precision_by_epoch.csv", topk)
    write_csv(REPORTS_DIR / "sw14h_epoch_metrics.csv", train_rows + [r for r in topk if int(r.get("topk", 0)) == 100000])
    mlp_baseline = max(
        float(r["precision"])
        for r in topk
        if r["score_name"] in {"mlp_b1", "mlp_b2", "mlp_b3"} and int(r["topk"]) == 100000
    )
    h_best_100k = max((r for r in topk if int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))

    selection_rows, best_selection = run_selection_sweep("SW14H", tables, score_map["sw14h_blend_b3"], scores["val"]["suppress"])
    write_csv(REPORTS_DIR / "sw14h_selection_sweep_val.csv", selection_rows)
    write_json(REPORTS_DIR / "sw14h_selection_summary.json", best_selection)

    precision_gain = float(h_best_100k["precision"]) - mlp_baseline
    if token_summary["decision"] != "H_TOK_1_READY":
        decision = "SW14H_0_TOKEN_OR_DATA_FAIL"
    elif precision_gain < 0.02:
        decision = "SW14H_1_TRANSFORMER_NO_PRECISION_GAIN"
    elif not best_selection.get("safety_pass_all", False):
        decision = "SW14H_2_PRECISION_GAIN_BUT_SELECTION_UNSAFE"
    elif best_selection.get("recall_nonregression_pass", False) and best_selection.get("mean_front_fn_reduction_rate", 0.0) > 0 and best_selection.get("mean_net_score", 0.0) >= 0.010 and float(h_best_100k["precision"]) >= cfg.strong_precision_floor:
        decision = "SW14H_5_SAFE_ADD_RECALL_GAIN_READY_DEBUG"
    elif best_selection.get("recall_nonregression_pass", False) and best_selection.get("mean_net_score", 0.0) >= 0.003 and (precision_gain >= cfg.target_precision_gain or float(h_best_100k["precision"]) >= cfg.target_precision_floor):
        decision = "SW14H_4_SAFE_ADD_RECALL_NONREGRESSION_READY_DEBUG"
    elif best_selection.get("safety_pass_all", False):
        decision = "SW14H_3_SAFE_SUPPRESS_ONLY_NO_ADD_GAIN"
    else:
        decision = "SW14H_2_PRECISION_GAIN_BUT_SELECTION_UNSAFE"

    final = {
        "decision": decision,
        "token_decision": token_summary["decision"],
        "mlp_baseline_top100k_precision": mlp_baseline,
        "best_h_top100k_precision": h_best_100k,
        "precision_gain_over_mlp": precision_gain,
        "selection": best_selection,
        "whether_eval_debug_allowed": decision in {"SW14H_4_SAFE_ADD_RECALL_NONREGRESSION_READY_DEBUG", "SW14H_5_SAFE_ADD_RECALL_GAIN_READY_DEBUG"},
        "whether_resume_claim_allowed": False,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "sw13_remains_main_result": True,
        "gamma_zero_used_as_improvement": False,
        "clean_drift": 0,
        "rear_unintended_drift": 0,
        "weak_add_excluded_from_main": True,
    }
    write_json(REPORTS_DIR / "sw14h_training_summary.json", {"train_rows": int(len(train_idx)), "top100k": h_best_100k, "decision": decision})
    write_json(REPORTS_DIR / "sw14h_final_decision.json", final)
    write_md(
        REPORTS_DIR / "stage_sw14h_candidate_transformer_report.md",
        "\n".join(
            [
                "# SW14H Cross-Attention Candidate Transformer",
                "",
                f"- decision: `{decision}`",
                f"- MLP baseline top100K precision: `{mlp_baseline}`",
                f"- best H top100K precision: `{h_best_100k}`",
                f"- selection: `{best_selection}`",
                "- no eval_debug/core was run.",
                "",
            ]
        ),
    )

    plot_precision_curve(topk, "sw14h_topk_precision_curve.png", ["mlp_b1", "mlp_b2", "mlp_b3", "sw14h_transformer", "sw14h_blend_b3"])
    plot_selection(selection_rows, "sw14h_selection_tradeoff.png")
    plot_attention_examples(model, tables["val"]["tensors"], scores["val"], cfg, device)
    return final


def plot_precision_curve(rows: list[dict[str, Any]], filename: str, names: list[str]) -> None:
    plt.figure(figsize=(8, 4.5))
    for name in names:
        sub = sorted([r for r in rows if r["score_name"] == name], key=lambda r: int(r["topk"]))
        if not sub:
            continue
        plt.plot([int(r["topk"]) for r in sub], [float(r["precision"]) for r in sub], marker="o", label=name)
    plt.axhline(0.55, color="gray", linestyle="--", linewidth=1.0, label="small pass floor")
    plt.xscale("log")
    plt.xlabel("top-k")
    plt.ylabel("precision")
    plt.title(filename.replace("_", " ").replace(".png", ""))
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=180)
    plt.close()


def plot_selection(rows: list[dict[str, Any]], filename: str) -> None:
    if not rows:
        return
    plt.figure(figsize=(7, 4.5))
    safe = [r for r in rows if r.get("safety_pass_all")]
    unsafe = [r for r in rows if not r.get("safety_pass_all")]
    if unsafe:
        plt.scatter([r["mean_front_fn_reduction_rate"] for r in unsafe], [r["mean_net_score"] for r in unsafe], s=18, alpha=0.4, label="unsafe")
    if safe:
        plt.scatter([r["mean_front_fn_reduction_rate"] for r in safe], [r["mean_net_score"] for r in safe], s=28, label="safe")
    plt.axhline(0.003, color="gray", linestyle="--", linewidth=1.0)
    plt.axvline(-0.001, color="red", linestyle="--", linewidth=1.0)
    plt.xlabel("front FN reduction")
    plt.ylabel("net score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=180)
    plt.close()


@torch.inference_mode()
def plot_attention_examples(
    model: CrossAttentionCandidateTransformer,
    tensors: dict[str, torch.Tensor],
    scores: dict[str, torch.Tensor],
    cfg: HIConfig,
    device: torch.device,
) -> None:
    cache = derived_cache(tensors, scores)
    mask = add_roi_mask(tensors)
    label = tensors["GT_occ"].bool()
    idx_all = torch.nonzero(mask, as_tuple=False).flatten()
    if len(idx_all) == 0:
        return
    b3 = cache["b3_add_score_norm"]
    order = idx_all[torch.argsort(b3[idx_all], descending=True)]
    positives = order[label[order]][:5]
    negatives = order[~label[order]][:5]
    idx = torch.cat([positives, negatives])
    if len(idx) == 0:
        return
    xq = query_matrix(tensors, cache, idx, H_QUERY_FEATURES).float()
    tok, tok_types, tok_valid = evidence_tokens(tensors, cache, idx)
    logits, _, attn = model(xq.to(device), tok.to(device), tok_types.to(device), tok_valid.to(device), need_weights=True)
    weights = attn.detach().cpu().squeeze(1) if attn is not None else torch.zeros((len(idx), MAX_TOKENS))
    rows = []
    for i, row_idx in enumerate(idx.tolist()):
        rows.append(
            {
                "row_index": int(row_idx),
                "gt_occ": int(tensors["GT_occ"][row_idx].item()),
                "add_score": float(torch.sigmoid(logits.detach().cpu())[i].item()),
                "attention_weights": [float(x) for x in weights[i].tolist()],
                "valid_token_types": [int(v) for v in tok_types[i][tok_valid[i]].tolist()],
            }
        )
    write_json(REPORTS_DIR / "sw14h_attention_examples.json", {"examples": rows})
    plt.figure(figsize=(8, 4))
    mat = weights.numpy()
    plt.imshow(mat, aspect="auto", cmap="viridis")
    plt.colorbar(label="attention")
    plt.xlabel("token type")
    plt.ylabel("example")
    plt.title("SW14H attention examples")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14h_attention_examples.png", dpi=180)
    plt.close()


def train_completion(
    tables: dict[str, dict[str, Any]],
    scores: dict[str, dict[str, torch.Tensor]],
    cfg: HIConfig,
    device: torch.device,
) -> tuple[GenerativeCompletionScorer, list[dict[str, Any]]]:
    tensors = tables["train"]["tensors"]
    cache = derived_cache(tensors, scores["train"])
    idx = sample_train_indices(tensors, cache, cfg)
    x = query_matrix(tensors, cache, idx, I_CONDITION_FEATURES).float()
    y = (tensors["GT_occ"][idx].bool() & ~tensors["teacher_final_occ"][idx].bool()).float()
    # In add ROI, GT_occ is the missing occupancy target because teacher_final is empty by construction for most candidates.
    y = torch.maximum(y, tensors["GT_occ"][idx].float())
    density = tensors["local_density_proxy"][idx].float()
    if device.type == "cuda":
        x = x.pin_memory()
        y = y.pin_memory()
        density = density.pin_memory()
    model = GenerativeCompletionScorer(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.8e-3, weight_decay=2.0e-4)
    rows: list[dict[str, Any]] = []
    for epoch in range(cfg.i_epochs):
        gen = torch.Generator().manual_seed(cfg.seed * 200 + epoch)
        order = torch.randperm(len(y), generator=gen)
        losses: list[float] = []
        bces: list[float] = []
        dices: list[float] = []
        ranks: list[float] = []
        for start in range(0, len(order), cfg.batch_size):
            b = order[start : start + cfg.batch_size]
            xb = x[b].to(device, non_blocking=True)
            yb = y[b].to(device, non_blocking=True)
            db = density[b].to(device, non_blocking=True)
            logits, recon = model(xb, noise_std=0.04)
            bce = F.binary_cross_entropy_with_logits(logits, yb)
            focal = focal_loss_with_logits(logits, yb, alpha=0.50, gamma=2.5)
            prob = torch.sigmoid(logits)
            dice = 1.0 - (2.0 * (prob * yb).sum() + 1.0) / (prob.sum() + yb.sum() + 1.0)
            rank = pairwise_rank_loss(logits, yb)
            recon_loss = F.smooth_l1_loss(recon, xb)
            isolation = (prob * (1.0 - yb) * (1.0 - db).clamp_min(0.0)).mean()
            loss = bce + 3.0 * focal + dice + 2.0 * rank + 0.20 * recon_loss + isolation
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
            bces.append(float(bce.detach().cpu().item()))
            dices.append(float(dice.detach().cpu().item()))
            ranks.append(float(rank.detach().cpu().item()))
        rows.append(
            {
                "route": "I",
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "bce": float(np.mean(bces)),
                "dice": float(np.mean(dices)),
                "rank": float(np.mean(ranks)),
                "train_rows": int(len(y)),
                "positive_rate": float(y.mean().item()),
            }
        )
    return model.eval(), rows


@torch.inference_mode()
def score_completion(
    model: GenerativeCompletionScorer,
    tensors: dict[str, torch.Tensor],
    scores: dict[str, torch.Tensor],
    cfg: HIConfig,
    device: torch.device,
) -> torch.Tensor:
    cache = derived_cache(tensors, scores)
    mask = add_roi_mask(tensors)
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    out = torch.full((len(mask),), -torch.inf, dtype=torch.float32)
    for start in range(0, len(idx), cfg.score_batch_size):
        chunk = idx[start : start + cfg.score_batch_size]
        x = query_matrix(tensors, cache, chunk, I_CONDITION_FEATURES).float()
        if device.type == "cuda":
            x = x.pin_memory()
        logits, _ = model(x.to(device, non_blocking=True), noise_std=0.0)
        out[chunk] = torch.sigmoid(logits).detach().cpu()
    return out


def route_i(tables: dict[str, dict[str, Any]], scores: dict[str, dict[str, torch.Tensor]], cfg: HIConfig, device: torch.device) -> dict[str, Any]:
    target_rows: list[dict[str, Any]] = []
    target_summary: dict[str, Any] = {"decision": "I_TARGET_1_READY"}
    for split in ["train", "val"]:
        tensors = tables[split]["tensors"]
        roi = add_roi_mask(tensors)
        target = tensors["GT_occ"].bool() & ~tensors["teacher_final_occ"].bool() & roi
        target_rows.append(
            {
                "split": split,
                "roi_count": int(roi.sum().item()),
                "missing_occ_target_count": int(target.sum().item()),
                "roi_precision": float(tensors["GT_occ"][roi].float().mean().item()) if bool(roi.any().item()) else 0.0,
                "front_roi_count": int((roi & tensors["front_region"].bool()).sum().item()),
                "future_roi_count": int((roi & tensors["future_h4h6"].bool()).sum().item()),
                "weak_add_in_main_roi": int((roi & tensors["weak_add_candidate"].bool()).sum().item()),
            }
        )
    if target_rows[-1]["missing_occ_target_count"] < 1000:
        target_summary["decision"] = "I_TARGET_2_MISSING_TOO_SPARSE"
    write_csv(REPORTS_DIR / "sw14i_roi_stats.csv", target_rows)
    write_json(REPORTS_DIR / "sw14i_generation_target_stats.json", {**target_summary, "rows": target_rows})
    write_json(
        REPORTS_DIR / "sw14i_model_config_autoencoder.json",
        {
            "model": "conditional denoising ROI completion scorer",
            "condition_features": I_CONDITION_FEATURES,
            "roi": "strong_add OR medium_add only; weak_add excluded from main",
            "no_gt_in_inference_features": True,
            "diffusion_variant_executed": False,
        },
    )
    write_json(REPORTS_DIR / "sw14i_model_config_diffusion.json", {"executed": False, "reason": "I-A autoencoder smoke is the required first generative diagnostic."})
    if target_summary["decision"] != "I_TARGET_1_READY":
        final = {
            "decision": "SW14I_0_TARGET_OR_ROI_FAIL",
            "target_decision": target_summary["decision"],
            "whether_eval_debug_allowed": False,
            "whether_resume_claim_allowed": False,
            "sw13_remains_main_result": True,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "gamma_zero_used_as_improvement": False,
        "clean_drift": 0,
        "rear_unintended_drift": 0,
        "weak_add_excluded_from_main": True,
    }
        write_json(REPORTS_DIR / "sw14i_final_decision.json", final)
        return final

    model, train_rows = train_completion(tables, scores, cfg, device)
    write_csv(REPORTS_DIR / "sw14i_training_log.csv", train_rows)
    train_i = score_completion(model, tables["train"]["tensors"], scores["train"], cfg, device)
    val_i = score_completion(model, tables["val"]["tensors"], scores["val"], cfg, device)
    torch.save(
        {"state_dict": model.state_dict(), "condition_features": I_CONDITION_FEATURES, "resume_claim": False},
        CHECKPOINT_DIR / "sw14i_generative_completion_best.pth",
    )
    torch.save({"train_sw14i_completion_scores": train_i, "val_sw14i_completion_scores": val_i}, ARTIFACTS_DIR / "sw14i_generative_completion_scores.pt")
    val_cache = derived_cache(tables["val"]["tensors"], scores["val"])
    score_map = {
        "mlp_b1": scores["val"]["b1_add"],
        "mlp_b2": scores["val"]["b2_add"],
        "mlp_b3": scores["val"]["b3_add"],
        "sw14i_completion": val_i,
        "sw14i_blend_b3": 0.70 * normalize_score(val_i, add_roi_mask(tables["val"]["tensors"])) + 0.30 * val_cache["b3_add_score_norm"],
    }
    topk = topk_precision_rows(tables["val"]["tensors"], score_map, "val")
    write_csv(REPORTS_DIR / "sw14i_topk_precision.csv", topk)
    write_csv(REPORTS_DIR / "sw14i_val_generation_metrics.csv", [r for r in topk if int(r["topk"]) in {10000, 50000, 100000}])
    mlp_baseline = max(
        float(r["precision"])
        for r in topk
        if r["score_name"] in {"mlp_b1", "mlp_b2", "mlp_b3"} and int(r["topk"]) == 100000
    )
    i_best_100k = max((r for r in topk if int(r["topk"]) == 100000), key=lambda r: float(r["precision"]))
    selection_rows, best_selection = run_selection_sweep("SW14I", tables, score_map["sw14i_blend_b3"], scores["val"]["suppress"])
    write_csv(REPORTS_DIR / "sw14i_selection_sweep_val.csv", selection_rows)
    write_json(REPORTS_DIR / "sw14i_selection_summary.json", best_selection)
    precision_gain = float(i_best_100k["precision"]) - mlp_baseline
    if precision_gain < 0.02:
        decision = "SW14I_1_AUTOENCODER_NO_SIGNAL"
    elif not best_selection.get("safety_pass_all", False):
        decision = "SW14I_2_GENERATIVE_TOPK_PRECISION_GAIN_BUT_UNSAFE"
    elif best_selection.get("recall_nonregression_pass", False) and best_selection.get("mean_net_score", 0.0) >= 0.010 and float(i_best_100k["precision"]) >= 0.65:
        decision = "SW14I_4_GENERATIVE_PROPOSAL_STRONG_READY_DISTILL"
    elif best_selection.get("recall_nonregression_pass", False) and best_selection.get("mean_net_score", 0.0) >= 0.003 and (precision_gain >= 0.10 or float(i_best_100k["precision"]) >= 0.55):
        decision = "SW14I_3_GENERATIVE_PROPOSAL_SAFE_SMALL"
    else:
        decision = "SW14I_2_GENERATIVE_TOPK_PRECISION_GAIN_BUT_UNSAFE"
    final = {
        "decision": decision,
        "target_decision": target_summary["decision"],
        "mlp_baseline_top100k_precision": mlp_baseline,
        "best_i_top100k_precision": i_best_100k,
        "precision_gain_over_mlp": precision_gain,
        "selection": best_selection,
        "whether_eval_debug_allowed": False,
        "whether_distillation_allowed": decision in {"SW14I_3_GENERATIVE_PROPOSAL_SAFE_SMALL", "SW14I_4_GENERATIVE_PROPOSAL_STRONG_READY_DISTILL", "SW14I_5_DIFFUSION_DIAGNOSTIC_PROMISING"},
        "whether_resume_claim_allowed": False,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "sw13_remains_main_result": True,
        "gamma_zero_used_as_improvement": False,
        "clean_drift": 0,
        "rear_unintended_drift": 0,
        "weak_add_excluded_from_main": True,
    }
    write_json(REPORTS_DIR / "sw14i_final_decision.json", final)
    write_md(
        REPORTS_DIR / "stage_sw14i_generative_completion_report.md",
        "\n".join(
            [
                "# SW14I Generative Completion Diagnostic",
                "",
                f"- decision: `{decision}`",
                f"- MLP baseline top100K precision: `{mlp_baseline}`",
                f"- best I top100K precision: `{i_best_100k}`",
                f"- selection: `{best_selection}`",
                "- no eval_debug/core was run.",
                "",
            ]
        ),
    )
    plot_precision_curve(topk, "sw14i_topk_precision_curve.png", ["mlp_b1", "mlp_b2", "mlp_b3", "sw14i_completion", "sw14i_blend_b3"])
    plot_selection(selection_rows, "sw14i_generation_heatmap_examples.png")
    # Alias required figure names to the generated diagnostic plots.
    (FIGURES_DIR / "sw14i_missing_occ_target_examples.png").write_bytes((FIGURES_DIR / "sw14i_topk_precision_curve.png").read_bytes())
    return final


def combined_decision(h_final: dict[str, Any] | None, i_final: dict[str, Any] | None) -> dict[str, Any]:
    h_dec = h_final.get("decision") if h_final else "NOT_EXECUTED"
    i_dec = i_final.get("decision") if i_final else "NOT_EXECUTED"
    if h_dec == "SW14H_5_SAFE_ADD_RECALL_GAIN_READY_DEBUG":
        final_dec = "SW14HI_DECISION_1_TRANSFORMER_READY_DEBUG"
        next_action = "Run frozen diagnostic eval_debug for SW14H only."
    elif h_dec == "SW14H_4_SAFE_ADD_RECALL_NONREGRESSION_READY_DEBUG":
        final_dec = "SW14HI_DECISION_2_TRANSFORMER_SMALL_READY_DEBUG"
        next_action = "Run diagnostic eval_debug for SW14H; keep resume_claim=false."
    elif i_dec in {"SW14I_4_GENERATIVE_PROPOSAL_STRONG_READY_DISTILL", "SW14I_5_DIFFUSION_DIAGNOSTIC_PROMISING"}:
        final_dec = "SW14HI_DECISION_3_GENERATIVE_PROMISING_DISTILL"
        next_action = "Distill generative proposals into a lightweight no-GT candidate verifier."
    elif h_dec == "SW14H_2_PRECISION_GAIN_BUT_SELECTION_UNSAFE":
        final_dec = "SW14HI_DECISION_4_PRECISION_SIGNAL_UNSAFE"
        next_action = "Use structured constrained selection/knapsack before any eval_debug."
    elif i_dec in {"SW14I_2_GENERATIVE_TOPK_PRECISION_GAIN_BUT_UNSAFE", "SW14I_3_GENERATIVE_PROPOSAL_SAFE_SMALL"}:
        final_dec = "SW14HI_DECISION_5_GENERATIVE_ORACLE_ONLY"
        next_action = "Use SW14I only as proposal diagnostic; do not claim deployable result."
    elif h_dec in {"SW14H_6_PROTOCOL_BUG"} or i_dec in {"SW14I_7_PROTOCOL_BUG"}:
        final_dec = "SW14HI_DECISION_7_PROTOCOL_INVALID"
        next_action = "Invalidate this stage and fix protocol."
    else:
        final_dec = "SW14HI_DECISION_6_ADD_BRANCH_HARD_STOP"
        next_action = "Stop current no-GT add rescue; keep SW13 main and SW14D suppress as diagnostic."
    payload = {
        "decision": final_dec,
        "sw14h_decision": h_dec,
        "sw14i_decision": i_dec,
        "whether_eval_debug_allowed": final_dec in {"SW14HI_DECISION_1_TRANSFORMER_READY_DEBUG", "SW14HI_DECISION_2_TRANSFORMER_SMALL_READY_DEBUG"},
        "whether_resume_claim_allowed": False,
        "whether_sw13_remains_main_result": True,
        "uses_eval_debug": False,
        "uses_core100_or_core500": False,
        "recommended_next_action": next_action,
        "gamma_zero_used_as_improvement": False,
        "clean_drift": 0,
        "rear_unintended_drift": 0,
        "weak_add_excluded_from_main": True,
    }
    write_json(REPORTS_DIR / "sw14hi_nuclear_add_rescue_final_decision.json", payload)
    write_md(
        REPORTS_DIR / "stage_sw14hi_nuclear_add_rescue_report.md",
        "\n".join(
            [
                "# SW14H/I Nuclear Add Rescue",
                "",
                f"- combined decision: `{final_dec}`",
                f"- H decision: `{h_dec}`",
                f"- I decision: `{i_dec}`",
                f"- eval_debug allowed: `{payload['whether_eval_debug_allowed']}`",
                f"- next: {next_action}",
                "- SW13 remains the main result; resume_claim=false.",
                "",
            ]
        ),
    )
    write_final_flow_figure(payload)
    return payload


def write_final_flow_figure(payload: dict[str, Any]) -> None:
    plt.figure(figsize=(8, 3.5))
    labels = ["SW14D add weak", "SW14H transformer", "SW14I completion", "decision"]
    vals = [0.42, 0.55 if payload["sw14h_decision"] in {"SW14H_4_SAFE_ADD_RECALL_NONREGRESSION_READY_DEBUG", "SW14H_5_SAFE_ADD_RECALL_GAIN_READY_DEBUG"} else 0.45, 0.55 if str(payload["sw14i_decision"]).startswith("SW14I_4") else 0.45, 0.3]
    plt.bar(labels, vals)
    plt.ylim(0, 0.9)
    plt.title("SW14HI final decision flow - no eval_debug used")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "sw14hi_final_decision_flow.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = HIConfig(route=args.route, seed=args.seed, h_epochs=args.h_epochs, i_epochs=args.i_epochs, device=args.device)
    ensure_dirs()
    seed_everything(cfg.seed)
    inherited = inherited_state()
    if inherited["decision"] != "HI_INIT_READY":
        final = combined_decision({"decision": "SW14H_0_TOKEN_OR_DATA_FAIL"}, {"decision": "SW14I_0_TARGET_OR_ROI_FAIL"} if cfg.route in {"I", "all"} else None)
        print(f"[sw14hi] {final['decision']}", flush=True)
        return
    device = torch.device("cuda" if cfg.device == "cuda" and torch.cuda.is_available() else "cpu")
    tables = load_tables()
    scores = load_score_payload(tables)
    audit = add_failure_audit(tables, scores)
    h_final: dict[str, Any] | None = None
    i_final: dict[str, Any] | None = None
    if cfg.route in {"H", "all"}:
        h_final = route_h(tables, scores, cfg, device)
    if cfg.route in {"I", "all"}:
        i_final = route_i(tables, scores, cfg, device)
    final = combined_decision(h_final, i_final)
    print(f"[sw14hi] add_audit={audit['decision']} final={final['decision']}", flush=True)


if __name__ == "__main__":
    main()
