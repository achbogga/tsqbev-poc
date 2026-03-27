"""TensorRT conversion and benchmark helpers.

References:
- NVIDIA TensorRT installation guide:
  https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-1020/pdf/TensorRT-Installation-Guide.pdf
- HotBEV deployment-minded latency discipline:
  https://proceedings.neurips.cc/paper_files/paper/2023/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tsqbev.config import ModelConfig
from tsqbev.export import (
    EXPORT_INPUT_NAMES,
    ExportableCore,
    build_export_input_dict,
    export_core_to_onnx,
)
from tsqbev.model import TSQBEVModel

Tensor = torch.Tensor

try:
    import tensorrt as trt
except ImportError:  # pragma: no cover - exercised only when TensorRT is absent.
    trt = None


TRT_OUTPUT_NAMES = ("object_logits", "object_boxes", "lane_logits", "lane_polylines")


def _require_tensorrt() -> Any:
    if trt is None:
        raise RuntimeError(
            "TensorRT Python bindings are not installed. "
            "Install tensorrt-cu12 to enable engine build and benchmarking."
        )
    return trt


def _torch_dtype_from_trt(trt_dtype: Any) -> torch.dtype:
    mapping = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.bool_): torch.bool,
    }
    np_dtype = np.dtype(_require_tensorrt().nptype(trt_dtype))
    if np_dtype not in mapping:
        raise TypeError(f"Unsupported TensorRT dtype: {np_dtype}")
    return mapping[np_dtype]


def benchmark_callable(
    fn: Any,
    *,
    warmup: int = 20,
    steps: int = 100,
    device: torch.device | str = "cuda",
) -> dict[str, float]:
    """Benchmark a callable with CUDA synchronization."""

    resolved_device = torch.device(device)
    for _ in range(warmup):
        fn()
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)

    samples_ms: list[float] = []
    for _ in range(steps):
        start = time.perf_counter()
        fn()
        if resolved_device.type == "cuda":
            torch.cuda.synchronize(resolved_device)
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    samples_ms.sort()
    return {
        "mean_ms": sum(samples_ms) / len(samples_ms),
        "p50_ms": samples_ms[len(samples_ms) // 2],
        "p95_ms": samples_ms[min(len(samples_ms) - 1, max(0, int(len(samples_ms) * 0.95) - 1))],
    }


def benchmark_exportable_core(
    config: ModelConfig,
    *,
    image_height: int = 256,
    image_width: int = 704,
    precision: str = "fp32",
    warmup: int = 20,
    steps: int = 100,
    device: str = "cuda",
) -> dict[str, float | str]:
    """Benchmark the exportable core path in eager PyTorch."""

    resolved_device = torch.device(device)
    model = TSQBEVModel(config).to(resolved_device).eval()
    wrapper = ExportableCore(model.core).to(resolved_device).eval()
    dtype = torch.float16 if precision == "fp16" else torch.float32
    if precision == "fp16":
        wrapper = wrapper.half()
    input_dict = build_export_input_dict(
        config,
        image_height=image_height,
        image_width=image_width,
        device=resolved_device,
        dtype=dtype,
    )
    inputs = tuple(input_dict[name] for name in EXPORT_INPUT_NAMES)

    with torch.inference_mode():
        metrics: dict[str, float | str] = dict(
            benchmark_callable(
                lambda: wrapper(*inputs),
                warmup=warmup,
                steps=steps,
                device=resolved_device,
            )
        )
    metrics["precision"] = precision
    metrics["backend"] = "pytorch-exportable-core"
    return metrics


def build_trt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    workspace_bytes: int = 1 << 30,
    enable_fp16: bool = True,
) -> dict[str, int | bool | str | float]:
    """Build a TensorRT engine from an ONNX file."""

    trt_mod = _require_tensorrt()
    logger = trt_mod.Logger(trt_mod.Logger.INFO)
    builder = trt_mod.Builder(logger)
    network = builder.create_network(1 << int(trt_mod.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt_mod.OnnxParser(network, logger)

    onnx_bytes = Path(onnx_path).read_bytes()
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parse failed:\n" + "\n".join(errors))

    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(trt_mod.MemoryPoolType.WORKSPACE, workspace_bytes)
    if enable_fp16 and builder.platform_has_fast_fp16:
        builder_config.set_flag(trt_mod.BuilderFlag.FP16)

    started = time.perf_counter()
    engine_memory = builder.build_serialized_network(network, builder_config)
    build_time_s = time.perf_counter() - started
    if engine_memory is None:
        raise RuntimeError("TensorRT engine build returned None")

    engine_path = Path(engine_path)
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_bytes = bytes(engine_memory)
    engine_path.write_bytes(engine_bytes)
    return {
        "engine_path": str(engine_path),
        "parse_ok": True,
        "fp16_enabled": bool(enable_fp16 and builder.platform_has_fast_fp16),
        "engine_bytes": len(engine_bytes),
        "build_time_s": build_time_s,
    }


def benchmark_trt_engine(
    engine_path: str | Path,
    input_dict: dict[str, Tensor],
    *,
    warmup: int = 20,
    steps: int = 100,
    device: str = "cuda",
) -> dict[str, float | str]:
    """Benchmark a serialized TensorRT engine using torch-owned CUDA buffers."""

    trt_mod = _require_tensorrt()
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("TensorRT benchmarking requires a CUDA device")

    logger = trt_mod.Logger(trt_mod.Logger.ERROR)
    runtime = trt_mod.Runtime(logger)
    engine_bytes = Path(engine_path).read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Failed to create TensorRT execution context")

    stream = torch.cuda.current_stream(resolved_device)
    io_tensors: dict[str, Tensor] = {}
    for name in EXPORT_INPUT_NAMES:
        tensor = input_dict[name].contiguous()
        context.set_input_shape(name, tuple(tensor.shape))
        io_tensors[name] = tensor
        context.set_tensor_address(name, int(tensor.data_ptr()))

    for name in TRT_OUTPUT_NAMES:
        shape = tuple(context.get_tensor_shape(name))
        dtype = _torch_dtype_from_trt(engine.get_tensor_dtype(name))
        output = torch.empty(shape, device=resolved_device, dtype=dtype)
        io_tensors[name] = output
        context.set_tensor_address(name, int(output.data_ptr()))

    def execute() -> None:
        ok = context.execute_async_v3(stream.cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

    metrics: dict[str, float | str] = dict(
        benchmark_callable(execute, warmup=warmup, steps=steps, device=resolved_device)
    )
    metrics["backend"] = "tensorrt"
    metrics["precision"] = "fp16-enabled-engine-fp32-io"
    return metrics


def run_trt_benchmark(
    config: ModelConfig,
    *,
    image_height: int = 256,
    image_width: int = 704,
    warmup: int = 20,
    steps: int = 100,
    output_dir: str | Path = "artifacts/trt",
) -> dict[str, Any]:
    """Export, build, and benchmark the default exportable deployment graph."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = TSQBEVModel(config).eval()
    onnx_path = export_core_to_onnx(
        model.core,
        config,
        output_dir / "core.onnx",
        image_height=image_height,
        image_width=image_width,
    )
    engine_metadata = build_trt_engine(onnx_path, output_dir / "core_fp16.plan", enable_fp16=True)
    trt_inputs = build_export_input_dict(
        config,
        image_height=image_height,
        image_width=image_width,
        device="cuda",
        dtype=torch.float32,
    )
    results = {
        "model": "tsqbev-exportable-core",
        "image_height": image_height,
        "image_width": image_width,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "engine": engine_metadata,
        "pytorch_fp32": benchmark_exportable_core(
            config,
            image_height=image_height,
            image_width=image_width,
            precision="fp32",
            warmup=warmup,
            steps=steps,
        ),
        "pytorch_fp16": benchmark_exportable_core(
            config,
            image_height=image_height,
            image_width=image_width,
            precision="fp16",
            warmup=warmup,
            steps=steps,
        ),
        "tensorrt_fp16": benchmark_trt_engine(
            str(engine_metadata["engine_path"]),
            trt_inputs,
            warmup=warmup,
            steps=steps,
        ),
    }
    benchmark_path = output_dir / "benchmark.json"
    benchmark_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    results["benchmark_path"] = str(benchmark_path)
    return results
