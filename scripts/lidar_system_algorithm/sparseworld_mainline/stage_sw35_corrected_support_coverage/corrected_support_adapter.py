"""Corrected SparseWorld support adapter for SW-3.5.

Main correction relative to SW-2:
- do not collapse 48 refine points into a mean-center proxy
- use decoded_metric + xyz + noflip + all48
- preserve current-frame and forecast support separately per horizon
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


DEFAULT_SCORE_THR = [0.35] * 15 + [0.25, 0.3]


def _load_sw3_module(project_root: Path):
    script_path = (
        project_root
        / "scripts/lidar_system_algorithm/sparseworld_mainline/stage_sw3_query_support_validation/run_sparseworld_sw3_main.py"
    )
    spec = importlib.util.spec_from_file_location("sparseworld_sw3_main", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load SW-3 module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class SupportDefinition:
    tensor_name: str = "refine_pts_current"
    coord_mode: str = "decoded_metric"
    order_name: str = "xyz"
    flip_variant: str = "noflip"
    point_mode: str = "all48"
    class_filter_name: str = "all"
    use_score_gate: bool = True
    source: str = "Stage SW-3 best mapping"


class CorrectedSparseWorldSupportAdapter:
    """Build corrected support tensors from SparseWorld query artifacts."""

    def __init__(
        self,
        project_root: str | Path,
        score_thr: list[float] | None = None,
        support_definition: SupportDefinition | None = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.sw3 = _load_sw3_module(self.project_root)
        self.score_thr = list(score_thr or DEFAULT_SCORE_THR)
        self.support_definition = support_definition or SupportDefinition()
        self.order_lookup = {
            "xyz": (0, 1, 2),
            "yxz": (1, 0, 2),
            "xzy": (0, 2, 1),
            "zxy": (2, 0, 1),
            "yzx": (1, 2, 0),
            "zyx": (2, 1, 0),
        }

    def _tensor_sources(self, query_artifact: dict[str, Any], horizon_s: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        fb = query_artifact["forward_backbone_outputs"]
        if horizon_s == 0:
            return fb["refine_pts"], fb["cls_score"], "refine_pts_current", "cls_score_current"
        forecast_points = fb.get("forecast_points_list", [])
        forecast_scores = fb.get("forecast_semantics_list", [])
        if horizon_s - 1 >= len(forecast_points) or horizon_s - 1 >= len(forecast_scores):
            raise IndexError(f"forecast support missing for horizon {horizon_s}")
        return (
            forecast_points[horizon_s - 1],
            forecast_scores[horizon_s - 1],
            f"forecast_points_h{horizon_s}",
            f"forecast_semantics_h{horizon_s}",
        )

    def build_support(
        self,
        query_artifact: dict[str, Any],
        horizon_s: int,
        class_filter_name: str | None = None,
        use_score_gate: bool | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        points_tensor, cls_tensor, point_source, score_source = self._tensor_sources(query_artifact, horizon_s)
        class_filter_name = class_filter_name or self.support_definition.class_filter_name
        use_score_gate = self.support_definition.use_score_gate if use_score_gate is None else use_score_gate
        support = self.sw3.build_support_from_tensor(
            points_tensor=points_tensor,
            cls_scores_tensor=cls_tensor,
            score_thr=self.score_thr,
            point_mode=self.support_definition.point_mode,
            coord_mode=self.support_definition.coord_mode,
            order=self.order_lookup[self.support_definition.order_name],
            flip_x=self.support_definition.flip_variant in {"flip_x", "flip_xy"},
            flip_y=self.support_definition.flip_variant in {"flip_y", "flip_xy"},
            class_filter_name=class_filter_name,
            use_score_gate=use_score_gate,
        )
        support_points_metric = support["points_metric"]
        support_points_grid = support["grid_indices"]
        support_density_bev = support["geometric_mask"].any(dim=-1).to(torch.uint8)
        semantic_density_bev = support["semantic_active_mask"].any(dim=-1).to(torch.uint8)
        support.update(
            {
                "support_points_metric": support_points_metric,
                "support_points_grid": support_points_grid,
                "support_density_bev": support_density_bev,
                "semantic_active_density_bev": semantic_density_bev,
                "valid_range_mask": support["geometric_mask"],
                "support_semantic_scores": support["semantic_active_logits_sparse"],
                "support_tensor_source": point_source,
                "class_score_source": score_source,
                "semantic_active_threshold_definition": "get_occ-compatible: decoded_metric all48 points, per-point max score > class score_thr and center-distance < 3.0",
            }
        )
        manifest = {
            "horizon_s": horizon_s,
            "support_tensor_source": point_source,
            "class_score_source": score_source,
            "support_definition": asdict(self.support_definition),
            "input_points_shape": list(points_tensor.shape),
            "input_cls_shape": list(cls_tensor.shape),
            "support_points_metric_shape": list(support_points_metric.shape),
            "support_points_grid_shape": list(support_points_grid.shape),
            "coordinate_min": support_points_metric.min(dim=0).values.tolist() if support_points_metric.numel() else [None, None, None],
            "coordinate_max": support_points_metric.max(dim=0).values.tolist() if support_points_metric.numel() else [None, None, None],
            "valid_range_ratio": float(support["valid_ratio"]),
            "grid_range_xyz": [[0, 0, 0], [self.sw3.GRID_SIZE[0] - 1, self.sw3.GRID_SIZE[1] - 1, self.sw3.GRID_SIZE[2] - 1]],
            "class_filter_name": class_filter_name,
            "use_score_gate": bool(use_score_gate),
            "semantic_active_threshold_definition": support["semantic_active_threshold_definition"],
            "geometric_support_voxel_count": int(support["geometric_mask"].sum().item()),
            "semantic_active_voxel_count": int(support["semantic_active_mask"].sum().item()),
            "query_density_entropy": float(support["query_density_entropy"]),
            "query_com_xy": list(support["query_com_xy"]),
            "diagonal_line_score": float(support["diagonal_line_score"]),
        }
        return support, manifest

    def build_temporal_support(self, query_artifact: dict[str, Any], horizons: list[int]) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
        supports: dict[int, dict[str, Any]] = {}
        manifests: dict[int, dict[str, Any]] = {}
        for horizon_s in horizons:
            supports[horizon_s], manifests[horizon_s] = self.build_support(query_artifact, horizon_s=horizon_s)
        return supports, manifests

    def save_sample0_artifact(
        self,
        query_artifact: dict[str, Any],
        horizons: list[int],
        artifact_path: str | Path,
        manifest_path: str | Path,
    ) -> None:
        supports, manifests = self.build_temporal_support(query_artifact, horizons)
        serializable = {
            f"h{h}": {
                "support_points_metric": supports[h]["support_points_metric"].cpu(),
                "support_points_grid": supports[h]["support_points_grid"].cpu(),
                "support_density_bev": supports[h]["support_density_bev"].cpu(),
                "semantic_active_density_bev": supports[h]["semantic_active_density_bev"].cpu(),
                "valid_range_mask": supports[h]["valid_range_mask"].cpu(),
                "semantic_active_mask": supports[h]["semantic_active_mask"].cpu(),
                "support_semantic_scores": supports[h]["support_semantic_scores"].cpu(),
                "semantic_active_conf": supports[h]["semantic_active_conf"].cpu(),
                "semantic_active_class": supports[h]["semantic_active_class"].cpu(),
                "support_coords": supports[h].get("support_coords"),
            }
            for h in horizons
        }
        artifact_path = Path(artifact_path)
        manifest_path = Path(manifest_path)
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(serializable, artifact_path)
        manifest_path.write_text(
            json.dumps(
                {
                    "support_definition": asdict(self.support_definition),
                    "score_thr": self.score_thr,
                    "horizon_manifests": manifests,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


def summarize_support_bev(support: dict[str, Any]) -> dict[str, float]:
    density = support["support_density_bev"].float().cpu().numpy()
    semantic_density = support["semantic_active_density_bev"].float().cpu().numpy()
    return {
        "support_bev_active_cells": float(density.sum()),
        "semantic_active_bev_active_cells": float(semantic_density.sum()),
        "valid_range_ratio": float(support["valid_ratio"]),
        "query_density_entropy": float(support["query_density_entropy"]),
        "diagonal_line_score": float(support["diagonal_line_score"]),
        "query_com_x": float(support["query_com_xy"][0]) if np.isfinite(support["query_com_xy"][0]) else float("nan"),
        "query_com_y": float(support["query_com_xy"][1]) if np.isfinite(support["query_com_xy"][1]) else float("nan"),
    }
