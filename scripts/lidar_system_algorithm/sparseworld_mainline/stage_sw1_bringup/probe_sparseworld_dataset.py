"""Build SparseWorld dataset and dump sample payload summaries.

This is a bring-up utility for Stage SW-1. It avoids training and focuses on:
1. config load
2. dataset build
3. sample fetch
4. payload summary JSON artifacts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SparseWorld dataset build probe")
    parser.add_argument("--repo-root", required=True, help="SparseWorld repo root")
    parser.add_argument("--config", required=True, help="Config path")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-build-json", required=True)
    parser.add_argument("--output-sample-json", required=True)
    parser.add_argument("--output-temporal-json", required=True)
    return parser.parse_args()


def unwrap(value: Any) -> Any:
    try:
        from mmcv.parallel import DataContainer
    except Exception:
        DataContainer = None
    if DataContainer is not None and isinstance(value, DataContainer):
        return unwrap(value.data)
    if isinstance(value, list):
        if len(value) == 1:
            return unwrap(value[0])
        return [unwrap(v) for v in value]
    if isinstance(value, tuple):
        return [unwrap(v) for v in value]
    if isinstance(value, dict):
        return {k: unwrap(v) for k, v in value.items()}
    return value


def summarize(obj: Any) -> Any:
    try:
        import numpy as np
        import torch
    except Exception:  # pragma: no cover
        np = None
        torch = None
    if isinstance(obj, dict):
        return {k: summarize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 12:
            return {"type": "list", "length": len(obj)}
        return [summarize(v) for v in obj]
    if torch is not None and isinstance(obj, torch.Tensor):
        return {
            "type": "torch.Tensor",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "device": str(obj.device),
        }
    if np is not None and isinstance(obj, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
        }
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(type(obj).__name__)


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    cfg = Config.fromfile(str(config_path))
    cfg.data.samples_per_gpu = 1
    cfg.data.workers_per_gpu = 0
    split_cfg = cfg.data[args.split]
    split_cfg.test_mode = args.split != "train"

    dataset = build_dataset(split_cfg)

    build_payload = {
        "repo_root": str(repo_root),
        "config": str(config_path),
        "split": args.split,
        "dataset_type": type(dataset).__name__,
        "dataset_length": len(dataset),
        "has_temp2nusc_map": hasattr(dataset, "temp2nusc_map"),
        "temp2nusc_map_length": len(dataset.temp2nusc_map) if hasattr(dataset, "temp2nusc_map") else None,
    }

    sample = dataset[args.sample_index]
    sample_unwrapped = unwrap(sample)
    sample_payload = {
        "sample_index": args.sample_index,
        "keys": sorted(sample.keys()),
        "summary": summarize(sample_unwrapped),
    }

    temporal_payload: dict[str, Any] = {"available": False}
    if isinstance(sample_unwrapped, dict):
        temporal_payload["available"] = "temporal_semantics" in sample_unwrapped or "temporal_img_inputs" in sample_unwrapped
        if "temporal_semantics" in sample_unwrapped:
            temporal_payload["temporal_semantics_summary"] = summarize(sample_unwrapped["temporal_semantics"])
        if "temporal_ego_states" in sample_unwrapped:
            temporal_payload["temporal_ego_states_summary"] = summarize(sample_unwrapped["temporal_ego_states"])
        if "temporal_trajs" in sample_unwrapped:
            temporal_payload["temporal_trajs_summary"] = summarize(sample_unwrapped["temporal_trajs"])
        if "img_metas" in sample_unwrapped:
            temporal_payload["img_metas_summary"] = summarize(sample_unwrapped["img_metas"])

    Path(args.output_build_json).write_text(json.dumps(build_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.output_sample_json).write_text(json.dumps(sample_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    Path(args.output_temporal_json).write_text(json.dumps(temporal_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(build_payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
