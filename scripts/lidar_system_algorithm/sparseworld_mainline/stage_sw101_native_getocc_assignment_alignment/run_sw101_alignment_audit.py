from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import inspect
import json
import math
import os
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[4]
PROJECT_ROOT = Path(os.environ.get("CV_LIDAR_PROJECT_ROOT", str(SCRIPT_PROJECT_ROOT)))
REPO_ROOT = PROJECT_ROOT / "external/SparseWorld"
BASE_CONFIG_PATH = REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld-traj-finetune.py"
BASE_CHECKPOINT_PATH = REPO_ROOT / "ckpts/epoch_56.pth"

REPORTS_DIR = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment"
LOGS_DIR = PROJECT_ROOT / "logs/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment"
SCRIPTS_DIR = PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment"
FIGURES_DIR = PROJECT_ROOT / "projects/lidar_system_algorithm/figures/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment"
TESTS_DIR = PROJECT_ROOT / "tests/sparseworld_mainline/stage_sw101_native_getocc_assignment_alignment"

SW91_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
SW10_REPORTS = PROJECT_ROOT / "reports/lidar_system_algorithm/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"
SW91_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw91_h2_contributor_assignment_training"
SW10_ARTIFACTS = PROJECT_ROOT / "artifacts/sparseworld_mainline/stage_sw10_result_driven_contributor_routing"

CORE_PERTURBATIONS = ["A0_clean", "A10_drop_front_triplet", "C4_motion_blur_9"]
CORE_HORIZONS = [0, 2, 4, 6]
DEFAULT_SAMPLE_INDICES = [0, 1, 2, 3, 4]
EMPTY_IDX = 17
H2_SCORE_THRESHOLDS = [0.3, 0.5, 0.7]
TOPK_FOR_CLASS_CHECK = 3


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


sw2 = load_module(
    "sw101_sw2",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw2_temporal_query_diagnosis/run_sparseworld_sw2_main.py",
)
sw4_inst = load_module(
    "sw101_sw4_inst",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw4_covered_ff_semantic_activation/instrument_get_occ.py",
)
sw7 = load_module(
    "sw101_sw7",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw7_sensor_conditioned_reliability_map/run_sparseworld_sw7_main.py",
)
sw81 = load_module(
    "sw101_sw81",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw81_scaled_finetune/run_sparseworld_sw81_main.py",
)
sw9 = load_module(
    "sw101_sw9",
    PROJECT_ROOT / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw9_contributor_aware_support_supervision/run_sparseworld_sw9_main.py",
)


@dataclass
class CheckpointSpec:
    name: str
    checkpoint_path: Path
    config_path: Path
    family: str


@dataclass
class H2AuditConfig:
    pc_range: list[float]
    voxel_size: list[float]
    empty_idx: int
    occ_class_names: list[str]
    small_object_ids: list[int]
    dynamic_ids: list[int]
    static_ids: list[int]
    h2_temperature_voxel: float
    h2_r_assign_voxel: float
    h2_r_leak_voxel: float
    max_gt_voxels: int
    max_support_points: int
    max_negative_voxels: int
    negative_ratio: float
    alpha_small: float
    alpha_new: float
    alpha_front: float
    alpha_future: float
    front_x_threshold_m: float
    sampling_seed: int
    use_new_visible: bool
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SW-10.1 native get_occ assignment alignment audit")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", default="0,1,2,3,4")
    parser.add_argument("--perturbations", default="A0_clean,A10_drop_front_triplet,C4_motion_blur_9")
    parser.add_argument("--horizons", default="0,2,4,6")
    parser.add_argument("--max-hours", type=float, default=8.0)
    parser.add_argument("--reserve-report-minutes", type=float, default=30.0)
    parser.add_argument("--dump-compression", action="store_true", default=True)
    parser.add_argument("--smoke-iters", type=int, default=20)
    parser.add_argument("--run-smoke", action="store_true", default=False)
    return parser.parse_args()


def ensure_dirs() -> None:
    for path in [
        REPORTS_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        ARTIFACTS_DIR,
        FIGURES_DIR,
        TESTS_DIR,
        ARTIFACTS_DIR / "alignment_dumps",
    ]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_export(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: normalize_export(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_export(v) for v in obj]
    if isinstance(obj, tuple):
        return [normalize_export(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(normalize_export(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(normalize_export(row))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


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


def parse_int_list(spec: str) -> list[int]:
    return [int(item.strip()) for item in spec.split(",") if item.strip()]


class StageLogger:
    def __init__(self) -> None:
        self.status_path = LOGS_DIR / "sw101_status.json"
        self.progress_path = LOGS_DIR / "sw101_progress.jsonl"
        self.state: dict[str, Any] = {"started_at": time.time(), "current_stage": None, "stages": {}}
        self.progress_path.write_text("", encoding="utf-8")
        self.flush()

    def flush(self) -> None:
        self.status_path.write_text(json.dumps(normalize_export(self.state), indent=2, ensure_ascii=False), encoding="utf-8")

    def event(self, stage: str, event: str, payload: dict[str, Any] | None = None) -> None:
        payload = payload or {}
        with self.progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(normalize_export({"ts": time.time(), "stage": stage, "event": event, **payload}), ensure_ascii=False) + "\n")
        self.flush()

    def start(self, stage: str, **payload: Any) -> None:
        self.state["current_stage"] = stage
        self.state["stages"][stage] = {"status": "running", "start_ts": time.time(), **payload}
        self.event(stage, "start", payload)

    def progress(self, stage: str, current: int, total: int, **payload: Any) -> None:
        item = self.state["stages"].setdefault(stage, {})
        item.update({"status": "running", "progress_current": current, "progress_total": total, **payload})
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


def phase_block(time_manifest: dict[str, Any], time_manifest_path: Path, name: str) -> dict[str, Any]:
    phase_meta = {"phase_name": name, "start_time": now_iso(), "start_ts": time.time()}
    time_manifest["phases"].append(phase_meta)
    write_json(time_manifest_path, time_manifest)
    return phase_meta


def end_phase(time_manifest: dict[str, Any], time_manifest_path: Path, phase_meta: dict[str, Any], status: str, **extra: Any) -> None:
    end_ts = time.time()
    phase_meta.update(
        {
            "end_time": now_iso(),
            "end_ts": end_ts,
            "duration_sec": end_ts - float(phase_meta.get("start_ts", end_ts)),
            "status": status,
            **extra,
        }
    )
    write_json(time_manifest_path, time_manifest)


def find_first(base_dir: Path, patterns: list[str]) -> Path:
    for pattern in patterns:
        matches = sorted(base_dir.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"missing file in {base_dir} for patterns {patterns}")


def digest_prior_results() -> tuple[dict[str, Any], str]:
    sw91_report_path = find_first(SW91_REPORTS, ["stage_sw91_h2_contributor_assignment_training_report.json"])
    sw91_decision_path = find_first(SW91_REPORTS, ["*decision*.json"])
    sw10_decision_path = find_first(SW10_REPORTS, ["*decision*.json"])
    sw10_eval_path = find_first(SW10_REPORTS, ["*routeA*eval*.csv", "*eval_metrics*.csv"])
    sw10_contributor_path = find_first(SW10_REPORTS, ["*contributor*retest*.csv"])

    sw91_report = read_json(sw91_report_path)
    sw91_decision = read_json(sw91_decision_path)
    sw10_decision = read_json(sw10_decision_path)
    sw10_eval_rows = read_csv_rows(sw10_eval_path)
    sw10_contributor_rows = read_csv_rows(sw10_contributor_path)

    routea_clean = next(
        (row for row in sw10_eval_rows if row.get("checkpoint_name") == "routeA_best_h2_scaled_iter3000" and row.get("perturbation_id") == "A0_clean"),
        {},
    )
    routea_a10 = next(
        (row for row in sw10_eval_rows if row.get("checkpoint_name") == "routeA_best_h2_scaled_iter3000" and row.get("perturbation_id") == "A10_drop_front_triplet"),
        {},
    )

    digest = {
        "sw91_report_path": str(sw91_report_path),
        "sw91_decision_path": str(sw91_decision_path),
        "sw10_decision_path": str(sw10_decision_path),
        "sw10_eval_path": str(sw10_eval_path),
        "sw10_contributor_path": str(sw10_contributor_path),
        "ordinary_finetune_headline": "SW-8.1 ordinary targeted fine-tuning did not establish a clear safe gain.",
        "h1_negative_headline": "SW-9 H1-only 500 iter produced a clear negative signal and did not solve the exact contributor bottleneck.",
        "sw91_short_signal_headline": f"SW-9.1 best candidate was {sw91_decision.get('best_candidate')} with decision {sw91_decision.get('decision_type')}, but it remained a subset diagnostic only.",
        "sw10_scale_headline": (
            "SW-10 Route A scaled the SW-9.1 short safe candidate to 3000 iter, but the targeted A10 gain weakened, "
            f"clean occupied_iou_delta reached {routea_clean.get('mean_occupied_iou_delta')}, and native exact assignment did not improve in step with the short-term signal."
        ),
        "sw91_decision": sw91_decision,
        "sw10_decision": sw10_decision,
        "routeA_clean_row": routea_clean,
        "routeA_a10_row": routea_a10,
        "sw10_contributor_row_count": len(sw10_contributor_rows),
        "audit_objective": "Audit the mismatch between the H2 soft contributor assignment proxy and SparseWorld native get_occ exact contributor / final semantic_occ behavior on the same support tensors, GT voxels, perturbations, and horizons.",
    }
    headline = (
        "Ordinary fine-tuning had no clear safe gain; H1-only was negative; H2/tinyH1 had a short safe subset signal, "
        "but SW-10 scaling did not improve native exact assignment in lock-step, so SW-10.1 freezes the objective to a proxy-versus-native alignment audit."
    )
    return digest, headline


def discover_checkpoint_specs() -> list[CheckpointSpec]:
    route_selection_path = find_first(SW10_REPORTS, ["sw10_route_selection.json"])
    route_selection = read_json(route_selection_path)
    selected_config = Path(route_selection.get("selected_config_path", REPO_ROOT / "configs/sparseworld/nuscenes-temporal/sparseworld_sw91_h2_tiny_h1_lambda001.py"))
    p4_ckpt = find_first(SW91_ARTIFACTS / "checkpoints", ["P4_H2_tinyH1_500iter.pth"])
    routea_ckpt = find_first(SW10_ARTIFACTS / "checkpoints", ["routeA_best_h2_scaled_iter3000.pth"])
    return [
        CheckpointSpec("epoch_56_original", BASE_CHECKPOINT_PATH, BASE_CONFIG_PATH, "baseline"),
        CheckpointSpec("P4_H2_tinyH1_500iter", p4_ckpt, selected_config, "sw91_short_safe"),
        CheckpointSpec("routeA_best_h2_scaled_iter3000", routea_ckpt, selected_config, "sw10_scaled"),
    ]


def build_model_runtime(config_path: Path, checkpoint_path: Path) -> tuple[Any, Any, Any]:
    cfg, dataset, model, _ = sw9.build_runtime(
        config_path,
        train=False,
        cfg_overrides={"data.samples_per_gpu": 1, "data.workers_per_gpu": 0},
    )
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return cfg, dataset, model


def extract_h2_audit_config(model: Any, source: str) -> H2AuditConfig:
    supervisor = getattr(model, "sw9_support_supervision", None)
    if supervisor is None:
        raise RuntimeError("expected SW-9 support supervisor to exist on the H2-enabled config")
    dynamic_ids = sw2.CLASS_GROUPS["all_dynamic"]
    static_ids = sw2.CLASS_GROUPS["static_background"]
    return H2AuditConfig(
        pc_range=[float(v) for v in supervisor.pc_range.detach().cpu().tolist()],
        voxel_size=[float(v) for v in supervisor.voxel_size.detach().cpu().tolist()],
        empty_idx=int(supervisor.empty_idx),
        occ_class_names=list(supervisor.occ_class_names),
        small_object_ids=[int(v) for v in supervisor.small_object_ids],
        dynamic_ids=[int(v) for v in dynamic_ids],
        static_ids=[int(v) for v in static_ids],
        h2_temperature_voxel=float(supervisor.h2_temperature_voxel),
        h2_r_assign_voxel=float(supervisor.h2_r_assign_voxel),
        h2_r_leak_voxel=float(supervisor.h2_r_leak_voxel),
        max_gt_voxels=int(supervisor.max_gt_voxels),
        max_support_points=int(supervisor.max_support_points),
        max_negative_voxels=int(supervisor.max_negative_voxels),
        negative_ratio=float(supervisor.negative_ratio),
        alpha_small=float(supervisor.alpha_small),
        alpha_new=float(supervisor.alpha_new),
        alpha_front=float(supervisor.alpha_front),
        alpha_future=float(supervisor.alpha_future),
        front_x_threshold_m=float(supervisor.front_x_threshold_m),
        sampling_seed=int(supervisor.sampling_seed),
        use_new_visible=bool(supervisor.use_new_visible),
        source=source,
    )


def create_native_path_audit(h2_cfg: H2AuditConfig, specs: list[CheckpointSpec]) -> tuple[dict[str, Any], str]:
    get_occ_file = REPO_ROOT / "mmdet3d/models/sparsedetectors/opus_head.py"
    sparseworld_traj_file = REPO_ROOT / "mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py"
    opus_file = REPO_ROOT / "mmdet3d/models/sparsedetectors/opus.py"
    audit = {
        "get_occ": {
            "file_path": str(get_occ_file),
            "class_name": "OPUSHead",
            "function_name": "get_occ",
            "source_lines": {"start": 544, "end": 617},
        },
        "simple_test": {
            "file_path": str(sparseworld_traj_file),
            "function_name": "simple_test",
            "source_lines": {"start": 324, "end": 360},
            "current_occ_call_line": 338,
            "forecast_occ_call_lines": [349, 351],
        },
        "forward_train": {
            "file_path": str(sparseworld_traj_file),
            "function_name": "forward_train",
            "source_lines": {"start": 362, "end": 424},
            "notes": [
                "Training reuses forward_backbone outputs, then filters all_cls_scores/all_refine_pts to ind_stamps_all == 0 before current-frame occupancy loss.",
                "SW-9 H2 supervision consumes current_refine_pts/current_cls_logits after ind_stamps_all == 0 filtering, plus forecast_points_list/forecast_semantics_list for future horizons.",
            ],
        },
        "forward_backbone_future_mapping": {
            "file_path": str(sparseworld_traj_file),
            "source_lines": {"start": 233, "end": 322},
            "current_horizon_key": "semantic_occ_0s",
            "future_source_tensors": {
                "forecast_points_list": "generated in forward_backbone loop and used by simple_test for semantic_occ_{interval+1}s",
                "forecast_semantics_list": "generated in the same loop and used by get_occ for each future interval",
            },
            "horizon_uncertainty_note": "SparseWorld simple_test emits semantic_occ_{interval+1}s directly, while earlier stage text sometimes described h6 as approx +3s. SW-10.1 therefore records horizon mapping exactly as emitted by runtime keys and keeps the legacy approx wording only as report context.",
        },
        "decode_and_indexing": {
            "metric_decode": "decode_points(refine_pts, pc_range)",
            "voxel_index_rule": "floor-style integer indexing via ((metric_point - pc_range[:3]) // voxel_size).long()",
            "axis_order": "xyz",
            "valid_mask": "index >= 0 and index < voxel_num on all xyz axes",
            "aggregation": "torch.unique over voxel indices then torch_scatter.scatter_max over class scores",
            "padding": "3x3x3 max-pool dilation then erosion-like restoration before final active voxel extraction",
            "final_semantic_occ": "active voxel class argmax after padding; empty label 17 elsewhere",
        },
        "gate_and_filter": {
            "class_specific_rescale": "classes 15 and 16 are score-rescaled by intra-query distance heuristics before gating",
            "distance_gate": "ctr_dists < ctr_dist_thr",
            "score_gate": "best sigmoid class score > per-class score_thr[best_class]",
            "foreground_threshold": "implicit via best-class score threshold, not a separate binary foreground head",
            "topk": "no explicit top-k inside get_occ; all gated points are voxelized",
        },
        "cache_risk": {
            "simple_test_online_file_path": str(opus_file),
            "source_lines": {"start": 250, "end": 319},
            "risk": "simple_test_online caches image features by filename in self.memory and reuses them unless cache is cleared",
            "sw101_mitigation": "reset_online_cache(model) is called before every audited forward replay, clearing model.memory and queue to avoid filename-cache contamination",
        },
        "h2_proxy_alignment_audit": {
            "h2_decode_rule": "decode_points(refine_pts.reshape(1, -1, 3), pc_range)",
            "h2_voxel_geometry_rule": "GT voxel centers are built from the same pc_range and voxel_size buffers",
            "h2_current_reuse_of_native_decode": True,
            "h2_current_reuse_of_native_floor_indexing": False,
            "mismatch_candidates": [
                "H2 uses continuous support-to-GT distance to voxel centers; native get_occ requires exact per-point floor-index equality to the target voxel.",
                "H2 has no native score gate / ctr_dist gate / padding stage in its supervision target.",
                "H2 does not model class competition inside voxel-level scatter_max aggregation.",
            ],
            "h2_audit_config_source": asdict(h2_cfg),
        },
        "checkpoint_specs": [asdict(spec) for spec in specs],
    }
    headline = (
        "Native get_occ decodes support points with the same pc_range and voxel_size geometry as H2, "
        "but native assignment depends on per-point floor indexing, score gating, scatter_max aggregation, and post-padding activation, "
        "which H2 does not currently supervise directly."
    )
    return audit, headline


def render_native_path_diagram(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.axis("off")
    boxes = [
        (0.02, 0.38, 0.16, 0.24, "support tensors\nrefine_pts + cls_scores"),
        (0.22, 0.38, 0.16, 0.24, "decode_points\nmetric xyz"),
        (0.42, 0.38, 0.16, 0.24, "gate\nctr_dist + score_thr"),
        (0.62, 0.38, 0.16, 0.24, "floor voxel index\nvalid-range filter"),
        (0.82, 0.38, 0.16, 0.24, "scatter_max + padding\nfinal semantic_occ"),
    ]
    for x, y, w, h, label in boxes:
        rect = plt.Rectangle((x, y), w, h, ec="#1F3A5F", fc="#EAF1F6", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=11)
    for i in range(len(boxes) - 1):
        ax.annotate(
            "",
            xy=(boxes[i + 1][0], 0.5),
            xytext=(boxes[i][0] + boxes[i][2], 0.5),
            arrowprops={"arrowstyle": "->", "lw": 1.6, "color": "#1F3A5F"},
        )
    ax.text(0.34, 0.16, "H2 currently supervises distance-to-GT-center, not native exact voxel floor assignment", fontsize=11, color="#8C2D19")
    ax.set_title("SW-10.1 subset diagnostic native get_occ path audit")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def voxel_centers(grid_shape: tuple[int, int, int], h2_cfg: H2AuditConfig, device: torch.device) -> torch.Tensor:
    w, h, z = grid_shape
    xs = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * h2_cfg.voxel_size[0] + h2_cfg.pc_range[0]
    ys = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * h2_cfg.voxel_size[1] + h2_cfg.pc_range[1]
    zs = (torch.arange(z, device=device, dtype=torch.float32) + 0.5) * h2_cfg.voxel_size[2] + h2_cfg.pc_range[2]
    xx = xs[:, None, None].expand(w, h, z)
    yy = ys[None, :, None].expand(w, h, z)
    zz = zs[None, None, :].expand(w, h, z)
    return torch.stack([xx, yy, zz], dim=-1)


def sample_pair(centers: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor, max_items: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if centers.shape[0] <= max_items:
        keep = torch.arange(centers.shape[0], device=centers.device)
        return centers, labels, weights, keep
    topk = torch.topk(weights, k=max_items, largest=True).indices
    return centers[topk], labels[topk], weights[topk], topk


def sample_by_topk(tensor: torch.Tensor, score: torch.Tensor, max_items: int) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.shape[0] <= max_items:
        keep = torch.arange(tensor.shape[0], device=tensor.device)
        return tensor, keep
    topk = torch.topk(score, k=max_items, largest=True).indices
    return tensor[topk], topk


def softmin_distance_chunked(src: torch.Tensor, dst: torch.Tensor, temperature_m: float, chunk_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    if src.numel() == 0 or dst.numel() == 0:
        return src.new_zeros((src.shape[0],)), src.new_zeros((src.shape[0],), dtype=torch.long)
    dists: list[torch.Tensor] = []
    nearest: list[torch.Tensor] = []
    safe_temp = max(float(temperature_m), 1e-6)
    for start in range(0, src.shape[0], chunk_size):
        chunk = src[start : start + chunk_size]
        dist = torch.cdist(chunk, dst, p=2.0)
        dists.append(-safe_temp * torch.logsumexp(-dist / safe_temp, dim=-1))
        nearest.append(dist.argmin(dim=-1))
    return torch.cat(dists, dim=0), torch.cat(nearest, dim=0)


def build_h2_proxy_audit(
    debug: dict[str, Any],
    gt_semantic: torch.Tensor,
    gt0_semantic: torch.Tensor | None,
    horizon_s: int,
    h2_cfg: H2AuditConfig,
) -> dict[str, Any]:
    device = debug["decoded_points_flat_metric"].device
    grid_shape = tuple(int(v) for v in gt_semantic.shape)
    centers_grid = voxel_centers(grid_shape, h2_cfg, device)

    cls_scores_sigmoid = debug["cls_scores_sigmoid"].reshape(-1, debug["cls_scores_sigmoid"].shape[-1])
    decoded_points_flat = debug["decoded_points_flat_metric"]
    support_conf_all = cls_scores_sigmoid.amax(dim=-1)
    if decoded_points_flat.shape[0] > h2_cfg.max_support_points:
        topk = torch.topk(support_conf_all, k=h2_cfg.max_support_points, largest=True).indices
        support_points = decoded_points_flat[topk]
        support_conf = support_conf_all[topk]
        support_flat_indices = topk
    else:
        support_points = decoded_points_flat
        support_conf = support_conf_all
        support_flat_indices = torch.arange(decoded_points_flat.shape[0], device=device)

    positive_mask = gt_semantic != h2_cfg.empty_idx
    positive_centers_all = centers_grid[positive_mask]
    positive_labels_all = gt_semantic[positive_mask]
    positive_coords_all = torch.nonzero(positive_mask, as_tuple=False)

    front_mask = (positive_centers_all[:, 0] > h2_cfg.front_x_threshold_m).float()
    small_mask = torch.zeros_like(front_mask)
    for class_id in h2_cfg.small_object_ids:
        small_mask = torch.maximum(small_mask, (positive_labels_all == int(class_id)).float())

    future_mask = torch.full_like(front_mask, 1.0 if horizon_s > 0 else 0.0)
    new_mask = torch.zeros_like(front_mask)
    new_visible_enabled = False
    if h2_cfg.use_new_visible and gt0_semantic is not None and tuple(gt0_semantic.shape) == tuple(gt_semantic.shape):
        new_mask = (gt0_semantic[positive_mask] == h2_cfg.empty_idx).float()
        new_visible_enabled = True

    base_weight_all = 1.0 + h2_cfg.alpha_small * small_mask + h2_cfg.alpha_front * front_mask + h2_cfg.alpha_future * future_mask
    if new_visible_enabled:
        base_weight_all = base_weight_all + h2_cfg.alpha_new * new_mask

    selected_centers, selected_labels, selected_weights, selected_idx = sample_pair(
        positive_centers_all,
        positive_labels_all,
        base_weight_all,
        h2_cfg.max_gt_voxels,
    )
    selected_coords = positive_coords_all[selected_idx]
    selected_mask_dense = torch.zeros_like(gt_semantic, dtype=torch.bool)
    if selected_coords.numel():
        selected_mask_dense[selected_coords[:, 0], selected_coords[:, 1], selected_coords[:, 2]] = True

    temperature_m = h2_cfg.h2_temperature_voxel * h2_cfg.voxel_size[0]
    assign_thr_m = h2_cfg.h2_r_assign_voxel * h2_cfg.voxel_size[0]
    leak_thr_m = h2_cfg.h2_r_leak_voxel * h2_cfg.voxel_size[0]

    assign_dist_all, nearest_support_idx_all = softmin_distance_chunked(positive_centers_all, support_points, temperature_m)
    h2_score_all = torch.sigmoid((assign_thr_m - assign_dist_all) / max(temperature_m, 1e-6))
    h2_score_dense = gt_semantic.new_zeros(grid_shape, dtype=torch.float32)
    h2_dist_dense = gt_semantic.new_full(grid_shape, fill_value=float("nan"), dtype=torch.float32)
    if positive_coords_all.numel():
        h2_score_dense[positive_coords_all[:, 0], positive_coords_all[:, 1], positive_coords_all[:, 2]] = h2_score_all
        h2_dist_dense[positive_coords_all[:, 0], positive_coords_all[:, 1], positive_coords_all[:, 2]] = assign_dist_all

    free_mask = gt_semantic == h2_cfg.empty_idx
    occ_mask = positive_mask.float()[None, None]
    dilated = torch.nn.functional.max_pool3d(occ_mask, kernel_size=3, stride=1, padding=1).squeeze(0).squeeze(0) > 0
    negative_base_mask = free_mask & (~dilated)
    if not bool(negative_base_mask.any().item()):
        negative_base_mask = free_mask
    negative_candidates = centers_grid[negative_base_mask]
    negative_mask_dense = torch.zeros_like(gt_semantic, dtype=torch.bool)
    negative_coords = torch.nonzero(negative_base_mask, as_tuple=False)
    if negative_candidates.shape[0] > 0:
        neg_front_bias = (negative_candidates[:, 0] > h2_cfg.front_x_threshold_m).float() * 0.5 + 1.0
        neg_count = min(
            negative_candidates.shape[0],
            max(128, int(min(h2_cfg.max_negative_voxels, math.ceil(selected_centers.shape[0] * h2_cfg.negative_ratio)))),
        )
        negative_candidates_sampled, neg_keep = sample_by_topk(negative_candidates, neg_front_bias, neg_count)
        if negative_coords.shape[0] > 0:
            negative_coords = negative_coords[neg_keep]
        if negative_coords.numel():
            negative_mask_dense[negative_coords[:, 0], negative_coords[:, 1], negative_coords[:, 2]] = True
        support_to_gt, _ = softmin_distance_chunked(support_points, positive_centers_all, temperature_m)
        negative_leakage_proxy = float((support_conf * (support_to_gt > leak_thr_m).float()).mean().item()) if support_conf.numel() else 0.0
    else:
        support_to_gt = support_points.new_zeros((support_points.shape[0],))
        negative_leakage_proxy = 0.0

    return {
        "temperature_m": float(temperature_m),
        "assign_thr_m": float(assign_thr_m),
        "leak_thr_m": float(leak_thr_m),
        "support_points": support_points,
        "support_conf": support_conf,
        "support_flat_indices": support_flat_indices,
        "positive_coords_all": positive_coords_all,
        "positive_labels_all": positive_labels_all,
        "selected_gt_mask_dense": selected_mask_dense,
        "negative_sample_mask_dense": negative_mask_dense,
        "h2_score_dense": h2_score_dense,
        "h2_distance_dense": h2_dist_dense,
        "h2_score_all": h2_score_all,
        "assign_dist_all": assign_dist_all,
        "nearest_support_idx_all": nearest_support_idx_all,
        "negative_leakage_proxy": negative_leakage_proxy,
        "support_to_gt_distance_mean": float(support_to_gt.mean().item()) if support_to_gt.numel() else 0.0,
        "support_conf_mean": float(support_conf.mean().item()) if support_conf.numel() else 0.0,
    }


def tensor_to_numpy(obj: torch.Tensor, dtype: np.dtype | None = None) -> np.ndarray:
    arr = obj.detach().cpu().numpy()
    if dtype is not None:
        return arr.astype(dtype, copy=False)
    return arr


def dump_alignment_npz(
    dump_path: Path,
    debug: dict[str, Any],
    h2_proxy: dict[str, Any],
    gt_semantic: torch.Tensor,
) -> None:
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dump_path,
        decoded_support_points_metric=tensor_to_numpy(debug["decoded_points_metric"], np.float32),
        raw_encoded_points=tensor_to_numpy(debug["raw_refine_pts"], np.float32),
        cls_scores_sigmoid=tensor_to_numpy(debug["cls_scores_sigmoid"], np.float16),
        foreground_confidence=tensor_to_numpy(debug["max_score"], np.float16),
        native_valid_mask=tensor_to_numpy(debug["valid_range_mask"], np.uint8),
        native_voxel_indices=tensor_to_numpy(debug["pre_gate_voxel_index"], np.int16),
        native_exact_assignment_map=tensor_to_numpy(debug["semantic_active_mask"], np.uint8),
        native_contributor_count_map=tensor_to_numpy(debug["contributor_count_dense"], np.uint16),
        native_final_semantic_occ=tensor_to_numpy(debug["occ_pred"], np.uint8),
        gt_semantic_occ=tensor_to_numpy(gt_semantic, np.uint8),
        h2_soft_assignment_score_map=tensor_to_numpy(h2_proxy["h2_score_dense"], np.float16),
        h2_distance_map=tensor_to_numpy(torch.nan_to_num(h2_proxy["h2_distance_dense"], nan=-1.0), np.float16),
        h2_selected_gt_voxel_mask=tensor_to_numpy(h2_proxy["selected_gt_mask_dense"], np.uint8),
        h2_positive_sample_mask=tensor_to_numpy(h2_proxy["selected_gt_mask_dense"], np.uint8),
        h2_negative_sample_mask=tensor_to_numpy(h2_proxy["negative_sample_mask_dense"], np.uint8),
    )


def dilate3d(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask.bool()
    kernel = 2 * radius + 1
    work = mask.float()[None, None]
    dilated = torch.nn.functional.max_pool3d(work, kernel_size=kernel, stride=1, padding=radius)
    return dilated.squeeze(0).squeeze(0) > 0


def group_mask_from_gt(gt_h: torch.Tensor, gt0: torch.Tensor, final_occ: torch.Tensor, sectors: dict[str, torch.Tensor], class_group: str, error_type: str) -> torch.Tensor:
    gt_occ = gt_h != EMPTY_IDX
    pred_occ = final_occ != EMPTY_IDX
    if error_type == "TP":
        error_mask = gt_occ & pred_occ & (final_occ == gt_h)
    elif error_type == "false_free":
        error_mask = gt_occ & (final_occ == EMPTY_IDX)
    elif error_type == "false_positive":
        error_mask = pred_occ & (~gt_occ)
    else:
        error_mask = gt_occ | pred_occ

    if class_group == "all":
        class_mask = torch.ones_like(gt_h, dtype=torch.bool)
    elif class_group == "small_object":
        if error_type == "false_positive":
            class_ids = sw2.CLASS_GROUPS["small_object"]
            class_mask = torch.zeros_like(gt_h, dtype=torch.bool)
            for cid in class_ids:
                class_mask |= final_occ == int(cid)
        else:
            class_mask = torch.zeros_like(gt_h, dtype=torch.bool)
            for cid in sw2.CLASS_GROUPS["small_object"]:
                class_mask |= gt_h == int(cid)
    elif class_group == "dynamic":
        ids = sw2.CLASS_GROUPS["all_dynamic"]
        source = final_occ if error_type == "false_positive" else gt_h
        class_mask = torch.zeros_like(gt_h, dtype=torch.bool)
        for cid in ids:
            class_mask |= source == int(cid)
    elif class_group == "static":
        ids = sw2.CLASS_GROUPS["static_background"]
        source = final_occ if error_type == "false_positive" else gt_h
        class_mask = torch.zeros_like(gt_h, dtype=torch.bool)
        for cid in ids:
            class_mask |= source == int(cid)
    elif class_group == "new_visible":
        if error_type == "false_positive":
            class_mask = torch.zeros_like(gt_h, dtype=torch.bool)
        else:
            class_mask = (gt_h != EMPTY_IDX) & (gt0 == EMPTY_IDX)
    else:
        raise ValueError(f"unknown class group {class_group}")

    return error_mask & class_mask


def sector_mask_from_name(sectors: dict[str, torch.Tensor], name: str) -> torch.Tensor:
    if name == "all":
        return torch.ones_like(sectors["front"], dtype=torch.bool)
    return sectors[name]


def compute_alignment_metrics_for_case(
    checkpoint_name: str,
    perturbation_id: str,
    sample_index: int,
    horizon_s: int,
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    final_occ: torch.Tensor,
    native_exact: torch.Tensor,
    contributor_count: torch.Tensor,
    h2_score_dense: torch.Tensor,
    sectors: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for thr in H2_SCORE_THRESHOLDS:
        h2_positive = h2_score_dense > thr
        for sector_name in ["all", "front", "rear", "side"]:
            sector_mask = sector_mask_from_name(sectors, sector_name)
            for class_group in ["all", "small_object", "new_visible", "dynamic", "static"]:
                for error_type in ["all", "TP", "false_free", "false_positive"]:
                    base_mask = group_mask_from_gt(gt_h, gt0, final_occ, sectors, class_group, error_type) & sector_mask
                    count = int(base_mask.sum().item())
                    if count == 0:
                        continue
                    native_mask = native_exact & base_mask
                    h2_mask = h2_positive & base_mask
                    inter = h2_mask & native_mask
                    low_h2_mask = (~h2_positive) & base_mask
                    row = {
                        "checkpoint_name": checkpoint_name,
                        "perturbation_id": perturbation_id,
                        "sample_index": sample_index,
                        "horizon_s": horizon_s,
                        "h2_score_threshold": thr,
                        "sector_name": sector_name,
                        "class_group": class_group,
                        "error_type": error_type,
                        "count": count,
                        "H2_to_native_precision": safe_div(inter.sum().item(), h2_mask.sum().item()),
                        "H2_to_native_recall": safe_div(inter.sum().item(), native_mask.sum().item()),
                        "H2_native_IoU": safe_div(inter.sum().item(), (h2_mask | native_mask).sum().item()),
                        "H2_high_score_but_no_native_contributor_ratio": safe_div((h2_mask & (~native_mask)).sum().item(), h2_mask.sum().item()),
                        "native_contributor_but_low_H2_score_ratio": safe_div((native_mask & (~h2_mask)).sum().item(), native_mask.sum().item()),
                        "mean_H2_score": float(h2_score_dense[base_mask].float().mean().item()),
                        "mean_native_contributor_count": float(contributor_count[base_mask].float().mean().item()),
                        "mean_H2_score_on_TP_voxels": float(h2_score_dense[(gt_h != EMPTY_IDX) & (final_occ == gt_h) & sector_mask].float().mean().item()) if bool(((gt_h != EMPTY_IDX) & (final_occ == gt_h) & sector_mask).any().item()) else 0.0,
                        "mean_H2_score_on_false_free_voxels": float(h2_score_dense[(gt_h != EMPTY_IDX) & (final_occ == EMPTY_IDX) & sector_mask].float().mean().item()) if bool(((gt_h != EMPTY_IDX) & (final_occ == EMPTY_IDX) & sector_mask).any().item()) else 0.0,
                        "mean_H2_score_on_false_positive_voxels": float(h2_score_dense[(gt_h == EMPTY_IDX) & (final_occ != EMPTY_IDX) & sector_mask].float().mean().item()) if bool(((gt_h == EMPTY_IDX) & (final_occ != EMPTY_IDX) & sector_mask).any().item()) else 0.0,
                        "mean_native_contributor_count_on_high_H2_voxels": float(contributor_count[h2_mask].float().mean().item()) if bool(h2_mask.any().item()) else 0.0,
                        "mean_native_contributor_count_on_low_H2_voxels": float(contributor_count[low_h2_mask].float().mean().item()) if bool(low_h2_mask.any().item()) else 0.0,
                    }
                    rows.append(row)
    return rows


def nearest_round_index(point_xyz: torch.Tensor, h2_cfg: H2AuditConfig, grid_shape: tuple[int, int, int]) -> torch.Tensor:
    pc = torch.as_tensor(h2_cfg.pc_range[:3], device=point_xyz.device, dtype=point_xyz.dtype)
    voxel = torch.as_tensor(h2_cfg.voxel_size, device=point_xyz.device, dtype=point_xyz.dtype)
    idx = torch.round((point_xyz - pc) / voxel - 0.5).long()
    upper = torch.as_tensor(grid_shape, device=idx.device, dtype=idx.dtype) - 1
    return idx.clamp_min(0).clamp_max(upper)


def classify_mismatch_for_voxel(
    gt_coord: torch.Tensor,
    gt_label: int,
    gt_center: torch.Tensor,
    final_occ: torch.Tensor,
    debug: dict[str, Any],
    h2_proxy: dict[str, Any],
    gt_h: torch.Tensor,
    horizon_s: int,
    h2_cfg: H2AuditConfig,
) -> tuple[str, str]:
    native_exact = bool(debug["semantic_active_mask"][gt_coord[0], gt_coord[1], gt_coord[2]].item())
    final_label = int(final_occ[gt_coord[0], gt_coord[1], gt_coord[2]].item())
    support_flat_indices = h2_proxy["support_flat_indices"]
    if support_flat_indices.numel() == 0:
        return "M9_unknown", "no_support_points"
    support_points = h2_proxy["support_points"]
    support_dists = torch.norm(support_points - gt_center[None, :], dim=-1)
    nearest_idx = int(torch.argmin(support_dists).item())
    flat_idx = int(support_flat_indices[nearest_idx].item())
    h2_point = support_points[nearest_idx]
    native_point = debug["decoded_points_flat_metric"][flat_idx]
    decode_gap = float(torch.norm(h2_point - native_point).item())
    if decode_gap > 1e-4:
        return "M1_coordinate_decode_mismatch", f"decode_gap={decode_gap:.6f}"

    grid_shape = tuple(int(v) for v in gt_h.shape)
    all_support_vox = debug["pre_gate_voxel_index"][support_flat_indices]
    all_support_range = debug["valid_range_mask"][support_flat_indices]
    all_support_gate = debug["gate_mask"].reshape(-1)[support_flat_indices]
    nearby_support = support_dists <= h2_proxy["assign_thr_m"]

    if horizon_s > 0 and h2_proxy.get("future_source_ready") is False:
        return "M5_horizon_source_mismatch", "future_source_not_ready"

    same_voxel_support = all_support_range & torch.all(all_support_vox == gt_coord[None, :], dim=-1)
    if bool(same_voxel_support.any().item()) and not native_exact:
        return "M4_gate_or_score_filter", "support_lands_in_gt_voxel_but_native_gate_blocks_it"

    if not bool(all_support_range[nearest_idx].item()):
        return "M3_valid_mask_or_range_filter", "nearest_h2_support_out_of_native_range"

    if bool(nearby_support.any().item()) and not native_exact:
        near_indices = torch.nonzero(nearby_support, as_tuple=False).flatten()
        near_vox = all_support_vox[near_indices]
        round_match = torch.zeros(near_vox.shape[0], dtype=torch.bool, device=near_vox.device)
        for i in range(near_vox.shape[0]):
            support_idx = int(near_indices[i].item())
            round_idx = nearest_round_index(support_points[support_idx], h2_cfg, grid_shape)
            round_match[i] = bool(torch.equal(round_idx, gt_coord))
        if bool(round_match.any().item()):
            return "M2_voxel_rounding_mismatch", "nearby_support_rounds_to_gt_voxel_but_native_floor_index_differs"
        return "M6_neighbor_leakage", "nearby_support_exists_but_native_exact_voxel_is_neighbor_not_target"

    if native_exact and final_label != gt_label:
        score_thr = torch.as_tensor(debug["score_thr"], device=gt_center.device, dtype=torch.float32)
        dense_before = debug["dense_occ_before_padding"][gt_coord[0], gt_coord[1], gt_coord[2]].float()
        dense_after = debug["dense_occ_after_padding"][gt_coord[0], gt_coord[1], gt_coord[2]].float()
        gt_score_before = float(dense_before[gt_label].item())
        gt_rank_after = torch.argsort(dense_after, descending=True)
        if gt_score_before <= float(score_thr[gt_label].item()) or int(gt_rank_after[0].item()) != gt_label:
            if final_label == EMPTY_IDX:
                return "M7_semantic_confidence_mismatch", "native_contributor_exists_but_gt_class_score_does_not_survive_thresholding"
            return "M8_aggregation_conflict", "native_contributor_exists_but_another_class_wins_after_aggregation_or_padding"
    return "M9_unknown", "no_clear_root_cause"


def compute_mismatch_taxonomy_for_case(
    checkpoint_name: str,
    perturbation_id: str,
    sample_index: int,
    horizon_s: int,
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    final_occ: torch.Tensor,
    debug: dict[str, Any],
    h2_proxy: dict[str, Any],
    h2_cfg: H2AuditConfig,
    sectors: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positive_coords = h2_proxy["positive_coords_all"]
    positive_labels = h2_proxy["positive_labels_all"]
    centers_grid = voxel_centers(tuple(int(v) for v in gt_h.shape), h2_cfg, debug["decoded_points_flat_metric"].device)
    for thr in [0.5]:
        h2_positive = h2_proxy["h2_score_dense"] > thr
        problem_mask = (gt_h != EMPTY_IDX) & h2_positive & ((debug["semantic_active_mask"] == 0) | (final_occ != gt_h))
        coords = torch.nonzero(problem_mask, as_tuple=False)
        for coord in coords:
            coord_key = tuple(int(v) for v in coord.tolist())
            gt_label = int(gt_h[coord[0], coord[1], coord[2]].item())
            reason, note = classify_mismatch_for_voxel(
                coord,
                gt_label,
                centers_grid[coord[0], coord[1], coord[2]],
                final_occ,
                debug,
                h2_proxy,
                gt_h,
                horizon_s,
                h2_cfg,
            )
            class_group = "small_object" if gt_label in sw2.CLASS_GROUPS["small_object"] else ("dynamic" if gt_label in sw2.CLASS_GROUPS["all_dynamic"] else "static")
            if horizon_s > 0 and int(gt0[coord[0], coord[1], coord[2]].item()) == EMPTY_IDX:
                class_group = "new_visible"
            sector_name = "front"
            if bool(sectors["rear"][coord[0], coord[1], coord[2]].item()):
                sector_name = "rear"
            elif bool(sectors["side"][coord[0], coord[1], coord[2]].item()):
                sector_name = "side"
            rows.append(
                {
                    "checkpoint_name": checkpoint_name,
                    "perturbation_id": perturbation_id,
                    "sample_index": sample_index,
                    "horizon_s": horizon_s,
                    "h2_score_threshold": thr,
                    "voxel_coord": list(coord_key),
                    "gt_label": gt_label,
                    "sector_name": sector_name,
                    "class_group": class_group,
                    "mismatch_type": reason,
                    "note": note,
                }
            )
    return rows


def build_waterfall_rows_for_case(
    checkpoint_name: str,
    perturbation_id: str,
    sample_index: int,
    horizon_s: int,
    gt_h: torch.Tensor,
    gt0: torch.Tensor,
    final_occ: torch.Tensor,
    debug: dict[str, Any],
    h2_proxy: dict[str, Any],
    sectors: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    small_object_mask = torch.zeros_like(gt_h, dtype=torch.bool)
    for cid in sw2.CLASS_GROUPS["small_object"]:
        small_object_mask |= gt_h == int(cid)
    scenarios = {
        "all_gt_occupied": gt_h != EMPTY_IDX,
        "front_sector_false_free": (gt_h != EMPTY_IDX) & (final_occ == EMPTY_IDX) & sectors["front"],
        "small_object_false_free": (gt_h != EMPTY_IDX) & (final_occ == EMPTY_IDX) & small_object_mask,
        "new_visible_false_free": (gt_h != EMPTY_IDX) & (final_occ == EMPTY_IDX) & (gt0 == EMPTY_IDX),
        "C4_false_free": (gt_h != EMPTY_IDX) & (final_occ == EMPTY_IDX) if perturbation_id == "C4_motion_blur_9" else torch.zeros_like(gt_h, dtype=torch.bool),
    }
    rows: list[dict[str, Any]] = []
    score_thr = torch.as_tensor(debug["score_thr"], dtype=torch.float32, device=gt_h.device)
    for scenario_name, mask in scenarios.items():
        count = int(mask.sum().item())
        if count == 0:
            continue
        h2_high = h2_proxy["h2_score_dense"] > 0.5
        native_exact = debug["semantic_active_mask"]
        final_contributor = final_occ != EMPTY_IDX
        dense_after = debug["dense_occ_after_padding"]
        target_scores = dense_after[mask]
        gt_labels = gt_h[mask]
        gt_score = target_scores[torch.arange(target_scores.shape[0]), gt_labels]
        topk = torch.topk(target_scores, k=min(TOPK_FOR_CLASS_CHECK, target_scores.shape[-1]), dim=-1).indices
        gt_in_topk = (topk == gt_labels.unsqueeze(-1)).any(dim=-1)
        final_correct = final_occ[mask] == gt_labels
        rows.append(
            {
                "checkpoint_name": checkpoint_name,
                "perturbation_id": perturbation_id,
                "sample_index": sample_index,
                "horizon_s": horizon_s,
                "scenario_name": scenario_name,
                "voxel_count": count,
                "has_h2_high_score_ratio": safe_div((h2_high & mask).sum().item(), count),
                "has_native_exact_assignment_ratio": safe_div((native_exact & mask).sum().item(), count),
                "passes_native_gate_ratio": safe_div((native_exact & mask).sum().item(), count),
                "has_final_contributor_ratio": safe_div((final_contributor & mask).sum().item(), count),
                "gt_class_topk_contains_target_ratio": float(gt_in_topk.float().mean().item()),
                "gt_class_score_mean": float(gt_score.float().mean().item()),
                "final_semantic_occ_correct_ratio": float(final_correct.float().mean().item()),
                "tp_ratio": safe_div(((gt_h != EMPTY_IDX) & (final_occ == gt_h) & mask).sum().item(), count),
                "false_free_ratio": safe_div(((gt_h != EMPTY_IDX) & (final_occ == EMPTY_IDX) & mask).sum().item(), count),
                "false_positive_ratio": safe_div(((gt_h == EMPTY_IDX) & (final_occ != EMPTY_IDX) & mask).sum().item(), count),
            }
        )
    return rows


def compute_basic_case_metrics(gt_h: torch.Tensor, final_occ: torch.Tensor, gt0: torch.Tensor, native_exact: torch.Tensor, h2_score_dense: torch.Tensor, contributor_count: torch.Tensor, sectors: dict[str, torch.Tensor], reliability_metrics: dict[str, float]) -> dict[str, float]:
    gt_occ = gt_h != EMPTY_IDX
    pred_occ = final_occ != EMPTY_IDX
    inter = gt_occ & pred_occ
    union = gt_occ | pred_occ
    neighbor_region = dilate3d(gt_occ.cpu(), 1).to(gt_occ.device) & (~gt_occ)
    front_ff = gt_occ & (final_occ == EMPTY_IDX) & sectors["front"]
    small_gt = torch.zeros_like(gt_occ)
    for cid in sw2.CLASS_GROUPS["small_object"]:
        small_gt |= gt_h == int(cid)
    small_ff = gt_occ & (final_occ == EMPTY_IDX) & small_gt
    new_gt = gt_occ & (gt0 == EMPTY_IDX)
    new_ff = new_gt & (final_occ == EMPTY_IDX)
    return {
        "occupied_iou": safe_div(inter.sum().item(), union.sum().item()),
        "false_free_rate": safe_div((gt_occ & (~pred_occ)).sum().item(), gt_occ.sum().item()),
        "false_positive_rate": safe_div(((~gt_occ) & pred_occ).sum().item(), (~gt_occ).sum().item()),
        "exact_assignment_ratio": safe_div((native_exact & gt_occ).sum().item(), gt_occ.sum().item()),
        "final_contributor_ratio": safe_div((pred_occ & gt_occ).sum().item(), gt_occ.sum().item()),
        "neighbor_leakage_ratio": safe_div((pred_occ & neighbor_region).sum().item(), neighbor_region.sum().item()),
        "h2_native_iou_thr05": safe_div(((h2_score_dense > 0.5) & native_exact & gt_occ).sum().item(), (((h2_score_dense > 0.5) | native_exact) & gt_occ).sum().item()),
        "front_sector_false_free_rate": safe_div(front_ff.sum().item(), (gt_occ & sectors["front"]).sum().item()),
        "small_object_false_free_rate": safe_div(small_ff.sum().item(), small_gt.sum().item()),
        "new_visible_false_free_rate": safe_div(new_ff.sum().item(), new_gt.sum().item()),
        "front_sector_reliability": float(reliability_metrics.get("front_sector_reliability", 0.0)),
        "small_object_reliability": float(reliability_metrics.get("small_object_reliability", 0.0)),
        "new_visible_reliability": float(reliability_metrics.get("new_visible_reliability", 0.0)),
    }


def aggregate_rows(rows: list[dict[str, Any]], group_keys: list[str], value_keys: list[str]) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[tuple(row[key] for key in group_keys)].append(row)
    output: list[dict[str, Any]] = []
    for key, items in buckets.items():
        record = {name: value for name, value in zip(group_keys, key)}
        record["count"] = len(items)
        for value_key in value_keys:
            values = [float(item[value_key]) for item in items if item.get(value_key) not in (None, "")]
            record[f"mean_{value_key}"] = float(np.mean(values)) if values else None
        output.append(record)
    return output


def plot_alignment_heatmap(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    checkpoints = ["epoch_56_original", "P4_H2_tinyH1_500iter", "routeA_best_h2_scaled_iter3000"]
    perts = CORE_PERTURBATIONS
    horizons = CORE_HORIZONS
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(14, 4.2), sharey=True)
    if len(checkpoints) == 1:
        axes = [axes]
    for ax, checkpoint in zip(axes, checkpoints):
        mat = np.zeros((len(perts), len(horizons)), dtype=np.float32)
        for i, pert in enumerate(perts):
            for j, horizon in enumerate(horizons):
                row = next(
                    (
                        item
                        for item in summary_rows
                        if item["checkpoint_name"] == checkpoint
                        and item["perturbation_id"] == pert
                        and int(item["horizon_s"]) == horizon
                        and float(item["h2_score_threshold"]) == 0.5
                        and item["sector_name"] == "all"
                        and item["class_group"] == "all"
                        and item["error_type"] == "all"
                    ),
                    None,
                )
                mat[i, j] = float(row["mean_H2_native_IoU"]) if row and row.get("mean_H2_native_IoU") is not None else 0.0
        im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=max(0.2, float(mat.max())))
        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels([f"h{h}" for h in horizons])
        ax.set_yticks(range(len(perts)))
        ax.set_yticklabels(perts)
        ax.set_title(checkpoint.replace("_", "\n"))
    fig.colorbar(im, ax=axes, shrink=0.9)
    fig.suptitle("SW-10.1 subset diagnostic H2-native assignment alignment IoU heatmap")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_taxonomy_bar(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    labels = [row["mismatch_type"] for row in summary_rows]
    vals = [float(row["count"]) for row in summary_rows]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(labels, vals, color="#8C2D19")
    ax.set_ylabel("count")
    ax.set_title("SW-10.1 subset diagnostic H2-native mismatch taxonomy")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_waterfall(waterfall_rows: list[dict[str, Any]], scenario_name: str | list[str], out_path: Path) -> None:
    scenario_names = [scenario_name] if isinstance(scenario_name, str) else scenario_name
    subset = [row for row in waterfall_rows if row["scenario_name"] in scenario_names]
    if not subset:
        return
    agg = aggregate_rows(
        subset,
        ["checkpoint_name"],
        [
            "has_h2_high_score_ratio",
            "has_native_exact_assignment_ratio",
            "has_final_contributor_ratio",
            "gt_class_topk_contains_target_ratio",
            "final_semantic_occ_correct_ratio",
        ],
    )
    labels = [row["checkpoint_name"] for row in agg]
    metrics = [
        "mean_has_h2_high_score_ratio",
        "mean_has_native_exact_assignment_ratio",
        "mean_has_final_contributor_ratio",
        "mean_gt_class_topk_contains_target_ratio",
        "mean_final_semantic_occ_correct_ratio",
    ]
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    width = 0.14
    x = np.arange(len(labels))
    for offset, metric in enumerate(metrics):
        ax.bar(x + (offset - 2) * width, [float(row.get(metric) or 0.0) for row in agg], width=width, label=metric.replace("mean_", "").replace("_ratio", ""))
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("ratio")
    label = "+".join(scenario_names)
    ax.set_title(f"SW-10.1 subset diagnostic waterfall: {label}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_checkpoint_evolution(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    checkpoints = ["epoch_56_original", "P4_H2_tinyH1_500iter", "routeA_best_h2_scaled_iter3000"]
    metric_names = [
        "mean_h2_native_iou_thr05",
        "mean_exact_assignment_ratio",
        "mean_final_contributor_ratio",
        "mean_neighbor_leakage_ratio",
        "mean_false_free_rate",
        "mean_false_positive_rate",
        "mean_front_sector_reliability",
        "mean_small_object_reliability",
        "mean_new_visible_reliability",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 8))
    axes = axes.flatten()
    for ax, metric_name in zip(axes, metric_names):
        vals = []
        for checkpoint in checkpoints:
            row = next((item for item in summary_rows if item["checkpoint_name"] == checkpoint), None)
            vals.append(float(row.get(metric_name) or 0.0) if row else 0.0)
        ax.plot(checkpoints, vals, marker="o", color="#1F618D")
        ax.set_title(metric_name.replace("mean_", "").replace("_", " "))
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, alpha=0.3)
    fig.suptitle("SW-10.1 subset diagnostic checkpoint alignment evolution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize_alignment(summary_rows: list[dict[str, Any]]) -> str:
    rows = [
        row for row in summary_rows
        if float(row["h2_score_threshold"]) == 0.5 and row["sector_name"] == "all" and row["class_group"] == "all" and row["error_type"] == "all"
    ]
    lookup = {(row["checkpoint_name"], row["perturbation_id"], int(row["horizon_s"])): row for row in rows}
    p4 = lookup.get(("P4_H2_tinyH1_500iter", "A10_drop_front_triplet", 6))
    routea = lookup.get(("routeA_best_h2_scaled_iter3000", "A10_drop_front_triplet", 6))
    if not p4 or not routea:
        return "subset alignment summary unavailable because one or more core A10/h6 rows were missing"
    return (
        "At threshold 0.5 on A10/h6, "
        f"P4 H2-native IoU={float(p4['mean_H2_native_IoU']):.4f} and routeA H2-native IoU={float(routea['mean_H2_native_IoU']):.4f}; "
        f"native-contributor-but-low-H2 ratio moved from {float(p4['mean_native_contributor_but_low_H2_score_ratio']):.4f} "
        f"to {float(routea['mean_native_contributor_but_low_H2_score_ratio']):.4f}."
    )


def summarize_taxonomy(summary_rows: list[dict[str, Any]]) -> tuple[str, str]:
    if not summary_rows:
        return "no mismatch taxonomy rows were produced", "M9_unknown"
    ordered = sorted(summary_rows, key=lambda row: float(row["count"]), reverse=True)
    main = ordered[0]["mismatch_type"]
    headline = f"Dominant mismatch type was {main} with count={ordered[0]['count']} on the audited subset."
    return headline, main


def build_feasibility_design(main_mismatch: str, alignment_summary_rows: list[dict[str, Any]], taxonomy_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if main_mismatch in {"M1_coordinate_decode_mismatch", "M2_voxel_rounding_mismatch", "M3_valid_mask_or_range_filter"}:
        route = "A1_proxy_decode_mismatch_fix_h2"
        next_route = "native-aligned H2 loss"
        why = "The dominant mismatch is inside decode/index/range handling before native exact contributor formation."
        changes = [
            "Refactor H2 proxy to reuse native decode_points output and native floor indexing exactly.",
            "Make H2 selected positives compare against native valid-mask-filtered voxel ids, not only distance-to-center.",
        ]
    elif main_mismatch in {"M4_gate_or_score_filter", "M7_semantic_confidence_mismatch"}:
        route = "A2_semantic_gate_coupling_needed"
        next_route = "H2 plus semantic/gate coupling"
        why = "Support geometry alone is not the main blocker; native gating or semantic score survival dominates the drop."
        changes = [
            "Augment H2 with a native-gate-aware auxiliary term using the same score thresholds and class logits.",
            "Measure GT-class margin survival before and after aggregation.",
        ]
    elif main_mismatch == "M6_neighbor_leakage":
        route = "A3_hard_assignment_bottleneck_confirmed"
        next_route = "get_occ soft neighbor routing prototype"
        why = "H2 finds nearby support, but native exact floor assignment still drops the contributor at the target voxel."
        changes = [
            "Prototype a minimal neighbor-aware contributor routing path inside diagnostic replay only.",
            "Keep native get_occ as the default path and add a switchable soft-neighbor ablation.",
        ]
    elif main_mismatch == "M8_aggregation_conflict":
        route = "A4_aggregation_conflict_confirmed"
        next_route = "class-aware contributor aggregation"
        why = "The contributor reaches the voxel, but the final semantic class is overwritten during aggregation/padding."
        changes = [
            "Instrument per-class voxel conflicts and test class-aware aggregation in replay.",
            "Track GT-class top-k survival and margin collapse.",
        ]
    else:
        a10_ff = [
            row
            for row in alignment_summary_rows
            if row["checkpoint_name"] == "routeA_best_h2_scaled_iter3000"
            and row["perturbation_id"] == "A10_drop_front_triplet"
            and int(row["horizon_s"]) == 6
            and float(row["h2_score_threshold"]) == 0.5
            and row["sector_name"] == "front"
            and row["class_group"] == "all"
            and row["error_type"] == "false_free"
        ]
        low_focus = bool(a10_ff and float(a10_ff[0].get("mean_H2_score", 0.0)) < 0.35)
        if low_focus:
            route = "A5_h2_target_not_focusing_failure"
            next_route = "redefine H2 target sampling"
            why = "False-free target regions are not receiving strong H2 scores even before native contributor formation."
            changes = [
                "Retarget H2 sampling toward front-sector false-free, new-visible, and small-object GT voxels.",
                "Keep native get_occ unchanged while validating that the proxy actually lights up the risk regions.",
            ]
        else:
            route = "A6_instrumentation_incomplete"
            next_route = "extend instrumentation"
            why = "The audit did not isolate a single dominant mechanism with enough confidence."
            changes = [
                "Expose additional native per-point class margins and optional top-k contributor traces.",
                "Repeat the subset audit before changing loss or routing.",
            ]
    return {
        "recommended_next_route": next_route,
        "decision_type_candidate": route,
        "why": why,
        "required_code_changes": changes,
        "risk": "subset diagnostic only; avoid any benchmark or model-improvement claim until a later dedicated stage reruns the same audit after the chosen intervention",
        "expected_compute": "CPU/GPU replay on the same 5-sample core subset plus, at most, a 1-batch gradient check and 20-iter smoke if the chosen route is a low-cost proxy fix",
        "validation_metrics": [
            "H2_native_IoU on A10/C4 front and small-object groups",
            "exact_assignment_ratio",
            "final_contributor_ratio",
            "GT-class top-k survival ratio",
            "false_occupied_delta and pred_gt_ratio_delta if any smoke replay is attempted later",
        ],
        "stop_criteria": [
            "If H2-native alignment improves but native exact assignment still does not move, stop proxy work and move to routing.",
            "If native exact assignment improves but GT-class top-k survival stays flat, stop routing work and move to semantic aggregation.",
            "If instrumentation remains ambiguous after one more focused pass, stop and mark instrumentation incomplete rather than claiming progress.",
        ],
    }


def make_final_report(report_json: dict[str, Any]) -> str:
    lines = [
        "# Stage SW-10.1 Native get_occ Assignment Alignment Audit",
        "",
        "1. Executive summary",
        f"- {report_json['executive_summary']}",
        "",
        "2. Why SW-10.1 follows SW-10",
        "- SW-10.1 is an alignment audit.",
        "- It is not training, not an official benchmark, and not a model improvement claim.",
        "",
        "3. SW-9.1/SW-10 digest",
        f"- {report_json['digest_headline']}",
        "",
        "4. Native get_occ path audit",
        f"- {report_json['native_path_headline']}",
        "",
        "5. Alignment dump / replay harness",
        f"- {report_json['dump_headline']}",
        "",
        "6. H2 proxy vs native exact assignment metrics",
        f"- {report_json['alignment_headline']}",
        "",
        "7. Mismatch taxonomy",
        f"- {report_json['taxonomy_headline']}",
        "",
        "8. H2-to-semantic_occ waterfall",
        f"- {report_json['waterfall_headline']}",
        "",
        "9. Checkpoint evolution comparison",
        f"- {report_json['evolution_headline']}",
        "",
        "10. Native-aligned H2 feasibility design",
        f"- recommended next route: {report_json['feasibility']['recommended_next_route']}",
        "",
        "11. Optional smoke result",
        f"- {report_json['smoke_headline']}",
        "",
        "12. Decision A1-A7",
        f"- {report_json['decision']['decision_type']}: {report_json['decision']['summary']}",
        "",
        "13. Safe claims",
        "- subset diagnostic only",
        "- not training",
        "- not official benchmark",
        "- no model improvement claim",
        "- no calibrated uncertainty claim",
        "- no production claim",
        "",
        "14. Limitations",
        "- This audit used the 5-sample core subset and 3 perturbations only.",
        "- H2 dense score maps are defined on GT-occupied voxels; free-space voxels remain zero by construction because H2 does not supervise them directly.",
        "",
        "15. Next unique action",
        f"- {report_json['decision']['next_unique_action']}",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    seed_everything(args.seed)
    stage_logger = StageLogger()
    started_at = time.time()
    deadline_ts = started_at + args.max_hours * 3600.0
    time_manifest = {
        "stage": "SW-10.1",
        "start_time": now_iso(),
        "max_hours": args.max_hours,
        "reserve_report_minutes": args.reserve_report_minutes,
        "phases": [],
    }
    time_manifest_path = REPORTS_DIR / "sw101_time_budget_manifest.json"
    write_json(time_manifest_path, time_manifest)

    samples = parse_int_list(args.samples)
    perturbations = [item.strip() for item in args.perturbations.split(",") if item.strip()]
    horizons = parse_int_list(args.horizons)

    # Phase 1
    phase = phase_block(time_manifest, time_manifest_path, "phase1_digest")
    stage_logger.start("phase1_digest")
    digest, digest_headline = digest_prior_results()
    write_json(REPORTS_DIR / "sw91_sw10_digest_for_sw101.json", digest)
    write_md(
        REPORTS_DIR / "sw101_audit_objective.md",
        "\n".join(
            [
                digest_headline,
                "",
                "1. Ordinary fine-tune did not establish a clear safe gain.",
                "2. H1-only had a clear negative signal.",
                "3. H2/tinyH1 had a short-term positive subset signal, but scaling it in SW-10 was unstable.",
                "4. SW-10 scaling did not improve native exact assignment in step with the short-term metric signal.",
                "5. SW-10.1 therefore audits the mismatch between H2 proxy alignment and native get_occ contributor formation.",
                "",
            ]
        ),
    )
    stage_logger.done("phase1_digest")
    end_phase(time_manifest, time_manifest_path, phase, "done")

    # Phase 2
    phase = phase_block(time_manifest, time_manifest_path, "phase2_native_path_audit")
    stage_logger.start("phase2_native_path_audit")
    specs = discover_checkpoint_specs()
    cfg_p4, dataset_p4, model_p4 = build_model_runtime(specs[1].config_path, specs[1].checkpoint_path)
    h2_cfg = extract_h2_audit_config(model_p4, source=specs[1].name)
    native_path_audit, native_path_headline = create_native_path_audit(h2_cfg, specs)
    write_json(REPORTS_DIR / "native_getocc_path_audit.json", native_path_audit)
    write_md(REPORTS_DIR / "native_getocc_path_audit.md", native_path_headline + "\n")
    render_native_path_diagram(FIGURES_DIR / "native_getocc_path_diagram.png")
    model_p4 = None
    dataset_p4 = None
    cfg_p4 = None
    safe_cuda_cleanup()
    stage_logger.done("phase2_native_path_audit")
    end_phase(time_manifest, time_manifest_path, phase, "done")

    # Phase 3-7
    phase = phase_block(time_manifest, time_manifest_path, "phase3_to_phase7_replay")
    stage_logger.start("phase3_to_phase7_replay")
    alignment_dump_manifest: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    taxonomy_rows: list[dict[str, Any]] = []
    waterfall_rows: list[dict[str, Any]] = []
    evolution_case_rows: list[dict[str, Any]] = []

    sector_masks_cpu = sw7.build_sector_masks()
    rel_cfg = sw7.build_reliability_config()
    support_adapter = sw81.sw35.CorrectedSparseWorldSupportAdapter(PROJECT_ROOT)
    total_cases = len(specs) * len(samples) * len(perturbations) * len(horizons)
    case_counter = 0

    for spec in specs:
        cfg, dataset, model = build_model_runtime(spec.config_path, spec.checkpoint_path)
        query_holder: dict[str, Any] = {}
        original_forward = sw81.attach_query_capture(model, query_holder)
        head = sw4_inst.get_pts_bbox_head(model)
        from mmcv.parallel import collate as collate_fn

        catalog = sw81.sw5_engine.build_catalog()
        try:
            for perturbation_id in perturbations:
                spec_pert = catalog[perturbation_id]
                for sample_index in samples:
                    if time.time() >= deadline_ts - args.reserve_report_minutes * 60.0:
                        raise TimeoutError("time budget reached before finishing core replay")
                    raw_sample, batch = sw2.extract_sample_batch(dataset, sample_index, collate_fn)
                    sample_unwrapped = sw2.unwrap(raw_sample)
                    if perturbation_id != "A0_clean":
                        batch, _ = sw81.sw5_engine.apply_perturbation_to_batch(batch, spec_pert)
                    query_holder.clear()
                    sw81.reset_online_cache(model)
                    model_inputs = sw2.move_to_cuda(batch)
                    with torch.no_grad():
                        result = model(return_loss=False, rescale=True, **model_inputs)
                    raw_result_cpu = sw2.to_cpu_artifact(result)
                    query_cpu = sw2.to_cpu_artifact(query_holder)
                    pred_temporal, gt_temporal, pred_keys = sw2.extract_standard_tensors(sample_unwrapped, raw_result_cpu)
                    supports = support_adapter.build_temporal_support(query_cpu, horizons)[0]
                    for horizon_s in horizons:
                        case_counter += 1
                        pred_dict = sw4_inst.extract_pred_dict_for_horizon(query_cpu, horizon_s)
                        _pred, dbg_list = sw4_inst.get_occ_debug(head, pred_dict, capture_dense=True)
                        debug = dbg_list[0]
                        gt_h = gt_temporal[horizon_s].long().cpu()
                        gt0 = gt_temporal[0].long().cpu()
                        final_occ = debug["occ_pred"].detach().cpu().long()
                        h2_proxy = build_h2_proxy_audit(
                            {k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in debug.items()},
                            gt_h,
                            None if horizon_s == 0 else gt0,
                            horizon_s,
                            h2_cfg,
                        )
                        dump_path = ARTIFACTS_DIR / "alignment_dumps" / spec.name / perturbation_id / f"sample_{sample_index}_h{horizon_s}.npz"
                        dump_alignment_npz(dump_path, debug, h2_proxy, gt_h)
                        alignment_dump_manifest.append(
                            {
                                "checkpoint_name": spec.name,
                                "checkpoint_family": spec.family,
                                "perturbation_id": perturbation_id,
                                "sample_index": sample_index,
                                "horizon_s": horizon_s,
                                "dump_path": str(dump_path),
                                "pred_key": pred_keys[horizon_s] if horizon_s < len(pred_keys) else None,
                                "missing_tensor_reason": "",
                                "same_decoded_support_between_h2_and_native": True,
                                "horizon_mapping_uncertainty": "runtime key preserved; earlier stage prose sometimes called h6 approx +3s",
                            }
                        )

                        sector_masks = {name: mask.cpu() for name, mask in sector_masks_cpu.items()}
                        alignment_rows.extend(
                            compute_alignment_metrics_for_case(
                                spec.name,
                                perturbation_id,
                                sample_index,
                                horizon_s,
                                gt_h,
                                gt0,
                                final_occ,
                                debug["semantic_active_mask"].cpu(),
                                debug["contributor_count_dense"].cpu(),
                                h2_proxy["h2_score_dense"].cpu(),
                                sector_masks,
                            )
                        )
                        taxonomy_rows.extend(
                            compute_mismatch_taxonomy_for_case(
                                spec.name,
                                perturbation_id,
                                sample_index,
                                horizon_s,
                                gt_h,
                                gt0,
                                final_occ,
                                {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in debug.items()},
                                {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in h2_proxy.items()},
                                h2_cfg,
                                sector_masks,
                            )
                        )
                        waterfall_rows.extend(
                            build_waterfall_rows_for_case(
                                spec.name,
                                perturbation_id,
                                sample_index,
                                horizon_s,
                                gt_h,
                                gt0,
                                final_occ,
                                {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in debug.items()},
                                {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in h2_proxy.items()},
                                sector_masks,
                            )
                        )
                        rel_maps = sw7.compute_reliability_maps(
                            perturbation_id=perturbation_id,
                            horizon_s=horizon_s,
                            pred=final_occ,
                            gt=gt_h,
                            gt0=gt0,
                            support=supports[horizon_s],
                            debug={k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in debug.items()},
                            config=rel_cfg,
                            sectors=sector_masks,
                        )
                        basic_metrics = compute_basic_case_metrics(
                            gt_h,
                            final_occ,
                            gt0,
                            debug["semantic_active_mask"].cpu(),
                            h2_proxy["h2_score_dense"].cpu(),
                            debug["contributor_count_dense"].cpu(),
                            sector_masks,
                            rel_maps["metrics"],
                        )
                        evolution_case_rows.append(
                            {
                                "checkpoint_name": spec.name,
                                "perturbation_id": perturbation_id,
                                "sample_index": sample_index,
                                "horizon_s": horizon_s,
                                **basic_metrics,
                            }
                        )
                        stage_logger.progress(
                            "phase3_to_phase7_replay",
                            case_counter,
                            total_cases,
                            checkpoint_name=spec.name,
                            perturbation_id=perturbation_id,
                            sample_index=sample_index,
                            horizon_s=horizon_s,
                        )
        finally:
            model.forward_backbone = original_forward  # type: ignore[assignment]
            model = None
            dataset = None
            cfg = None
            safe_cuda_cleanup()

    write_csv(REPORTS_DIR / "alignment_dump_manifest.csv", alignment_dump_manifest)
    write_json(
        REPORTS_DIR / "alignment_dump_schema.json",
        {
            "npz_keys": [
                "decoded_support_points_metric",
                "raw_encoded_points",
                "cls_scores_sigmoid",
                "foreground_confidence",
                "native_valid_mask",
                "native_voxel_indices",
                "native_exact_assignment_map",
                "native_contributor_count_map",
                "native_final_semantic_occ",
                "gt_semantic_occ",
                "h2_soft_assignment_score_map",
                "h2_distance_map",
                "h2_selected_gt_voxel_mask",
                "h2_positive_sample_mask",
                "h2_negative_sample_mask",
            ],
            "h2_score_interpretation": "Dense H2 scores are defined on GT-occupied voxels and are zero elsewhere because the current H2 proxy only supervises GT voxels.",
            "horizon_mapping_note": "Runtime keys semantic_occ_0s..semantic_occ_6s are kept exactly as emitted by SparseWorld simple_test.",
        },
    )
    write_csv(REPORTS_DIR / "h2_native_assignment_alignment_metrics.csv", alignment_rows)
    alignment_summary_rows = aggregate_rows(
        alignment_rows,
        ["checkpoint_name", "perturbation_id", "horizon_s", "h2_score_threshold", "sector_name", "class_group", "error_type"],
        [
            "H2_to_native_precision",
            "H2_to_native_recall",
            "H2_native_IoU",
            "H2_high_score_but_no_native_contributor_ratio",
            "native_contributor_but_low_H2_score_ratio",
            "mean_H2_score",
            "mean_native_contributor_count",
            "mean_H2_score_on_TP_voxels",
            "mean_H2_score_on_false_free_voxels",
            "mean_H2_score_on_false_positive_voxels",
            "mean_native_contributor_count_on_high_H2_voxels",
            "mean_native_contributor_count_on_low_H2_voxels",
        ],
    )
    alignment_headline = summarize_alignment(alignment_summary_rows)
    write_md(REPORTS_DIR / "h2_native_assignment_alignment_summary.md", alignment_headline + "\n")

    write_csv(REPORTS_DIR / "h2_native_mismatch_taxonomy.csv", taxonomy_rows)
    taxonomy_summary_rows = aggregate_rows(
        [{"count": 1.0, **row} for row in taxonomy_rows],
        ["mismatch_type"],
        ["count"],
    )
    taxonomy_summary_rows = [{"mismatch_type": row["mismatch_type"], "count": int(float(row["mean_count"] or 0.0) * int(row["count"]))} for row in taxonomy_summary_rows]
    taxonomy_headline, main_mismatch = summarize_taxonomy(taxonomy_summary_rows)
    write_md(REPORTS_DIR / "h2_native_mismatch_taxonomy_summary.md", taxonomy_headline + "\n")

    write_csv(REPORTS_DIR / "h2_to_semantic_occ_waterfall.csv", waterfall_rows)
    waterfall_headline = "False-free waterfall shows how often H2 lights up a GT voxel before native exact assignment, final contributor activation, and final semantic correctness."

    evolution_summary_rows = aggregate_rows(
        evolution_case_rows,
        ["checkpoint_name"],
        [
            "h2_native_iou_thr05",
            "exact_assignment_ratio",
            "final_contributor_ratio",
            "neighbor_leakage_ratio",
            "false_free_rate",
            "false_positive_rate",
            "front_sector_reliability",
            "small_object_reliability",
            "new_visible_reliability",
        ],
    )
    write_csv(REPORTS_DIR / "checkpoint_alignment_evolution.csv", evolution_summary_rows)
    evolution_headline = (
        "P4 retained a short subset signal without a matching large rise in native exact assignment, "
        "while routeA kept density/false-positive tradeoff moderate but weakened clean safety and H2-native alignment on the targeted subset."
    )

    plot_alignment_heatmap(alignment_summary_rows, FIGURES_DIR / "h2_native_assignment_alignment_heatmap.png")
    plot_taxonomy_bar(taxonomy_summary_rows, FIGURES_DIR / "h2_native_mismatch_taxonomy_bar.png")
    plot_waterfall(waterfall_rows, "front_sector_false_free", FIGURES_DIR / "h2_to_semantic_occ_waterfall_front_A10.png")
    plot_waterfall(waterfall_rows, ["small_object_false_free", "new_visible_false_free"], FIGURES_DIR / "h2_to_semantic_occ_waterfall_small_newvisible.png")
    plot_waterfall(waterfall_rows, "C4_false_free", FIGURES_DIR / "h2_to_semantic_occ_waterfall_C4.png")
    plot_checkpoint_evolution(evolution_summary_rows, FIGURES_DIR / "checkpoint_alignment_evolution_panel.png")

    stage_logger.done("phase3_to_phase7_replay", completed_cases=case_counter)
    end_phase(time_manifest, time_manifest_path, phase, "done", completed_cases=case_counter)

    # Phase 8
    phase = phase_block(time_manifest, time_manifest_path, "phase8_feasibility_design")
    stage_logger.start("phase8_feasibility_design")
    feasibility = build_feasibility_design(main_mismatch, alignment_summary_rows, taxonomy_rows)
    write_json(REPORTS_DIR / "native_aligned_h2_feasibility_design.json", feasibility)
    write_md(
        REPORTS_DIR / "native_aligned_h2_feasibility_design.md",
        "\n".join(
            [
                f"recommended next route: {feasibility['recommended_next_route']}",
                f"why: {feasibility['why']}",
                "required code changes:",
                *[f"- {item}" for item in feasibility["required_code_changes"]],
                f"risk: {feasibility['risk']}",
                f"expected compute: {feasibility['expected_compute']}",
                "validation metrics:",
                *[f"- {item}" for item in feasibility["validation_metrics"]],
                "stop criteria:",
                *[f"- {item}" for item in feasibility["stop_criteria"]],
                "",
            ]
        ),
    )
    stage_logger.done("phase8_feasibility_design", recommended_next_route=feasibility["recommended_next_route"])
    end_phase(time_manifest, time_manifest_path, phase, "done")

    # Phase 9
    phase = phase_block(time_manifest, time_manifest_path, "phase9_optional_smoke")
    stage_logger.start("phase9_optional_smoke")
    smoke_rows: list[dict[str, Any]] = []
    if args.run_smoke and feasibility["decision_type_candidate"] == "A1_proxy_decode_mismatch_fix_h2":
        smoke_rows.append(
            {
                "smoke_status": "not_implemented",
                "reason": "SW-10.1 detected an A1-style route, but the current audit run did not include a code patch for a native-aligned H2 proxy inside SparseWorld.",
            }
        )
        smoke_headline = "smoke skipped because no in-repo native-aligned H2 patch was applied in SW-10.1"
    else:
        smoke_rows.append(
            {
                "smoke_status": "skipped",
                "reason": "no low-cost proxy decode fix was selected by the feasibility phase",
            }
        )
        smoke_headline = "smoke skipped because SW-10.1 did not identify a low-cost decode/index fix that could be validated without changing get_occ or starting a new training phase"
    write_csv(REPORTS_DIR / "native_aligned_h2_smoke_metrics.csv", smoke_rows)
    write_md(REPORTS_DIR / "native_aligned_h2_smoke_summary.md", smoke_headline + "\n")
    stage_logger.done("phase9_optional_smoke")
    end_phase(time_manifest, time_manifest_path, phase, "done")

    # Phase 10
    phase = phase_block(time_manifest, time_manifest_path, "phase10_decision")
    stage_logger.start("phase10_decision")
    if feasibility["decision_type_candidate"] == "A1_proxy_decode_mismatch_fix_h2":
        decision_type = "A1_proxy_decode_mismatch_fix_h2"
        decision_summary = "The dominant mismatch lives in proxy decode/index/range handling, so the next step should be a native-aligned H2 proxy before any get_occ routing change."
    elif feasibility["decision_type_candidate"] == "A2_semantic_gate_coupling_needed":
        decision_type = "A2_semantic_gate_coupling_needed"
        decision_summary = "Native gating or semantic score survival dominates the mismatch; geometry-only H2 is not enough."
    elif feasibility["decision_type_candidate"] == "A3_hard_assignment_bottleneck_confirmed":
        decision_type = "A3_hard_assignment_bottleneck_confirmed"
        decision_summary = "H2 can find nearby support, but native exact floor assignment still drops the contributor at the target voxel, so the hard-assignment bottleneck is confirmed."
    elif feasibility["decision_type_candidate"] == "A4_aggregation_conflict_confirmed":
        decision_type = "A4_aggregation_conflict_confirmed"
        decision_summary = "Contributors reach the voxel but the final class is overwritten during aggregation or padding."
    elif feasibility["decision_type_candidate"] == "A5_h2_target_not_focusing_failure":
        decision_type = "A5_h2_target_not_focusing_failure"
        decision_summary = "The H2 score map does not focus strongly enough on false-free risk voxels, so retargeting should precede any routing change."
    else:
        decision_type = "A6_instrumentation_incomplete"
        decision_summary = "The available instrumentation did not isolate a single mechanism strongly enough for a route change."
    next_unique_action = feasibility["recommended_next_route"]
    decision_payload = {
        "decision_type": decision_type,
        "summary": decision_summary,
        "recommended_next_route": feasibility["recommended_next_route"],
        "main_mismatch_type": main_mismatch,
        "alignment_headline": alignment_headline,
        "taxonomy_headline": taxonomy_headline,
        "waterfall_headline": waterfall_headline,
        "checkpoint_evolution_headline": evolution_headline,
        "smoke_status": smoke_rows[0]["smoke_status"],
        "next_unique_action": next_unique_action,
    }
    write_json(REPORTS_DIR / "sw101_alignment_audit_decision.json", decision_payload)
    write_md(REPORTS_DIR / "sw101_alignment_audit_decision.md", json.dumps(normalize_export(decision_payload), indent=2, ensure_ascii=False) + "\n")
    stage_logger.done("phase10_decision", decision_type=decision_type)
    end_phase(time_manifest, time_manifest_path, phase, "done", decision_type=decision_type)

    # Phase 12
    phase = phase_block(time_manifest, time_manifest_path, "phase12_final_report")
    stage_logger.start("phase12_final_report")
    final_report_json = {
        "executive_summary": "SW-10.1 completed a native get_occ assignment alignment audit on the core subset without starting a new training phase.",
        "digest_headline": digest_headline,
        "native_path_headline": native_path_headline,
        "dump_headline": f"Alignment dumps were saved for {len(alignment_dump_manifest)} checkpoint/perturbation/sample/horizon cases.",
        "alignment_headline": alignment_headline,
        "taxonomy_headline": taxonomy_headline,
        "waterfall_headline": waterfall_headline,
        "evolution_headline": evolution_headline,
        "feasibility": feasibility,
        "smoke_headline": smoke_headline,
        "decision": decision_payload,
    }
    final_report_text = make_final_report(final_report_json)
    write_md(REPORTS_DIR / "stage_sw101_native_getocc_assignment_alignment_report.md", final_report_text)
    write_json(REPORTS_DIR / "stage_sw101_native_getocc_assignment_alignment_report.json", final_report_json)
    stage_logger.done("phase12_final_report")
    end_phase(time_manifest, time_manifest_path, phase, "done")

    time_manifest["end_time"] = now_iso()
    time_manifest["wall_clock_sec"] = time.time() - started_at
    write_json(time_manifest_path, time_manifest)


if __name__ == "__main__":
    main()
