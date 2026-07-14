"""SparseWorld Stage SW-1 forward probe.

This script performs a bounded inference bring-up:
1. load config
2. build dataset / dataloader
3. build model
4. load checkpoint
5. run one real sample forward
6. save raw output / query tensor artifacts / JSON manifests
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SparseWorld single-sample forward probe")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--raw-output-pt", required=True)
    parser.add_argument("--query-output-pt", required=True)
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
    import numpy as np
    import torch

    if isinstance(obj, dict):
        return {k: summarize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > 16:
            return {"type": "list", "length": len(obj)}
        return [summarize(v) for v in obj]
    if isinstance(obj, tuple):
        return [summarize(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        detached = obj.detach()
        cpu = detached.float().cpu()
        finite = torch.isfinite(detached).float().mean().item() if detached.numel() > 0 else 1.0
        return {
            "type": "torch.Tensor",
            "shape": list(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "finite_ratio": float(finite),
            "mean": float(cpu.mean().item()) if cpu.numel() else 0.0,
            "std": float(cpu.std().item()) if cpu.numel() > 1 else 0.0,
            "min": float(cpu.min().item()) if cpu.numel() else 0.0,
            "max": float(cpu.max().item()) if cpu.numel() else 0.0,
        }
    if isinstance(obj, np.ndarray):
        arr = obj.astype("float32", copy=False) if obj.size else obj
        return {
            "type": "numpy.ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "mean": float(arr.mean()) if obj.size else 0.0,
            "std": float(arr.std()) if obj.size else 0.0,
            "min": float(arr.min()) if obj.size else 0.0,
            "max": float(arr.max()) if obj.size else 0.0,
        }
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(type(obj).__name__)


def to_cpu_artifact(obj: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(obj, dict):
        return {k: to_cpu_artifact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu_artifact(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_cpu_artifact(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, np.ndarray):
        return obj
    return obj


def move_to_cuda(obj: Any) -> Any:
    import torch
    try:
        from mmcv.parallel import DataContainer
    except Exception:
        DataContainer = None

    if (
        DataContainer is not None
        and isinstance(obj, list)
        and len(obj) == 1
        and isinstance(obj[0], DataContainer)
    ):
        return move_to_cuda(obj[0])
    if DataContainer is not None and isinstance(obj, DataContainer):
        return move_to_cuda(obj.data)
    if isinstance(obj, dict):
        return {k: move_to_cuda(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_cuda(v) for v in obj]
    if isinstance(obj, tuple):
        return [move_to_cuda(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.cuda(non_blocking=False)
    return obj


def ckpt_state_audit(model: Any, checkpoint_path: Path) -> dict[str, Any]:
    import torch

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model_state = model.state_dict()
    matched = []
    missing = []
    unexpected = []
    shape_mismatch = []

    for key, value in state_dict.items():
        if key not in model_state:
            unexpected.append(key)
            continue
        if tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatch.append(
                {
                    "key": key,
                    "ckpt_shape": list(value.shape),
                    "model_shape": list(model_state[key].shape),
                }
            )
            continue
        matched.append(key)

    for key in model_state.keys():
        if key not in state_dict:
            missing.append(key)

    return {
        "checkpoint_top_keys": sorted(list(ckpt.keys())) if isinstance(ckpt, dict) else None,
        "matched_key_count": len(matched),
        "missing_key_count": len(missing),
        "unexpected_key_count": len(unexpected),
        "shape_mismatch_count": len(shape_mismatch),
        "missing_keys_preview": missing[:20],
        "unexpected_keys_preview": unexpected[:20],
        "shape_mismatch_preview": shape_mismatch[:20],
        "checkpoint_meta_keys": sorted(list(ckpt.get("meta", {}).keys())) if isinstance(ckpt, dict) and isinstance(ckpt.get("meta", {}), dict) else [],
    }


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    output_json = Path(args.output_json).resolve()
    raw_output_pt = Path(args.raw_output_pt).resolve()
    query_output_pt = Path(args.query_output_pt).resolve()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    raw_output_pt.parent.mkdir(parents=True, exist_ok=True)
    query_output_pt.parent.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo_root))
    os.chdir(repo_root)

    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.utils import compat_cfg, setup_multi_processes
    from mmdet3d.datasets import build_dataloader, build_dataset
    from mmdet3d.models import build_model
    from mmdet3d.utils import patch_config

    cfg = Config.fromfile(str(config_path))
    cfg = compat_cfg(cfg)
    cfg = patch_config(cfg)
    setup_multi_processes(cfg)
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    cfg.gpu_ids = [0]
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    if "4D" in cfg.model.type:
        cfg.model.align_after_view_transfromation = True

    split_cfg = cfg.data[args.split]
    split_cfg.test_mode = args.split != "train"
    if cfg.data.get(f"{args.split}_dataloader", {}).get("samples_per_gpu", 1) > 1:
        split_cfg.pipeline = replace_ImageToTensor(split_cfg.pipeline)
    test_loader_cfg = {
        "samples_per_gpu": 1,
        "workers_per_gpu": 0,
        "dist": False,
        "shuffle": False,
    }
    dataset = build_dataset(split_cfg)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    audit = ckpt_state_audit(model, checkpoint_path)
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, str(checkpoint_path), map_location="cpu")
    model = model.cuda()
    model.eval()

    captured: dict[str, Any] = {}
    original_forward_backbone = model.forward_backbone

    def wrapped_forward_backbone(*fb_args: Any, **fb_kwargs: Any) -> Any:
        outputs = original_forward_backbone(*fb_args, **fb_kwargs)
        captured["forward_backbone_outputs"] = to_cpu_artifact(outputs)
        return outputs

    model.forward_backbone = wrapped_forward_backbone  # type: ignore[assignment]

    batch = None
    for idx, data in enumerate(data_loader):
        if idx == args.sample_index:
            batch = data
            break
    if batch is None:
        raise IndexError(f"sample_index {args.sample_index} out of range")

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model_inputs = move_to_cuda(batch)
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **model_inputs)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    raw_result_cpu = to_cpu_artifact(result)
    torch.save(raw_result_cpu, raw_output_pt)
    torch.save(captured, query_output_pt)

    batch_unwrapped = unwrap(batch)
    batch_summary = summarize(batch_unwrapped)
    result_summary = summarize(raw_result_cpu)
    query_summary = summarize(captured)

    pred_hist = {}
    pred_keys = [k for k in result.keys() if k.startswith("semantic_occ_")]
    for key in pred_keys:
        arr = result[key][0]
        uniq, counts = torch.unique(torch.as_tensor(arr), return_counts=True)
        pred_hist[key] = {
            "shape": list(arr.shape),
            "class_hist": {str(int(k.item())): int(v.item()) for k, v in zip(uniq, counts)},
            "non_empty_count": int((torch.as_tensor(arr) != 17).sum().item()),
        }

    payload = {
        "repo_root": str(repo_root),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "sample_index": args.sample_index,
        "dataset_length": len(dataset),
        "model_type": cfg.model.type,
        "checkpoint_meta_keys": sorted(list(checkpoint.get("meta", {}).keys())) if isinstance(checkpoint, dict) and isinstance(checkpoint.get("meta", {}), dict) else [],
        "state_dict_audit": audit,
        "latency_ms": elapsed_ms,
        "peak_memory_mb": peak_mem_mb,
        "batch_summary": batch_summary,
        "result_keys": sorted(list(result.keys())),
        "result_summary": result_summary,
        "query_summary": query_summary,
        "pred_histograms": pred_hist,
        "raw_output_pt": str(raw_output_pt),
        "query_output_pt": str(query_output_pt),
    }
    output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"latency_ms": elapsed_ms, "peak_memory_mb": peak_mem_mb, "result_keys": payload["result_keys"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
