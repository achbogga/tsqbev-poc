"""Minimal CLI for smoke checks and intentionally disabled research entrypoints.

References:
- HotBEV deployment smoke-test discipline:
  https://proceedings.neurips.cc/paper_files/paper/2023/file/081b08068e4733ae3e7ad019fe8d172f-Paper-Conference.pdf
- Karpathy autoresearch workflow staging:
  https://github.com/karpathy/autoresearch
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich import print

from tsqbev.config import ModelConfig
from tsqbev.export import export_core_to_onnx
from tsqbev.latency import LatencyPredictor, features_from_config
from tsqbev.model import TSQBEVModel
from tsqbev.research_guard import ensure_research_loop_disabled
from tsqbev.runtime import benchmark_forward, run_eval_step, run_train_step
from tsqbev.synthetic import make_synthetic_batch
from tsqbev.trt import run_trt_benchmark


def smoke() -> None:
    """Run a compact synthetic forward pass."""

    config = ModelConfig.small()
    batch = make_synthetic_batch(config, batch_size=1)
    model = TSQBEVModel(config)
    outputs = model(batch)
    print(
        {
            "object_logits": tuple(outputs["object_logits"].shape),
            "object_boxes": tuple(outputs["object_boxes"].shape),
            "lane_logits": tuple(outputs["lane_logits"].shape),
            "lane_polylines": tuple(outputs["lane_polylines"].shape),
        }
    )


def export_smoke() -> None:
    """Export the core model with synthetic prepared inputs."""

    config = ModelConfig.small()
    model = TSQBEVModel(config)
    output = export_core_to_onnx(model.core, config, Path("artifacts/export/core.onnx"))
    print({"onnx_path": str(output)})


def latency_smoke() -> None:
    """Evaluate the simple latency predictor on the default config."""

    config = ModelConfig()
    predictor = LatencyPredictor()
    features = features_from_config(
        config, params_m=55.0, active_pillars=1200.0, activations_mb=220.0
    )
    prediction = predictor.predict_ms(features)
    print({"predicted_ms": prediction, "production_pass": predictor.production_pass(prediction)})


def train_step_smoke() -> None:
    """Run one manual synthetic training step."""

    print(run_train_step(ModelConfig.small(), batch_size=1))


def eval_smoke() -> None:
    """Run one manual synthetic evaluation step."""

    print(run_eval_step(ModelConfig.small(), batch_size=1))


def bench_smoke() -> None:
    """Run a short synthetic forward benchmark."""

    print(benchmark_forward(ModelConfig.small(), steps=3, warmup=1, batch_size=1))


def trt_bench_smoke() -> None:
    """Run ONNX export, TensorRT build, and GPU benchmark."""

    print(
        run_trt_benchmark(
            ModelConfig(),
            image_height=256,
            image_width=704,
            warmup=10,
            steps=50,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="tsqbev-poc utilities")
    parser.add_argument(
        "command",
        choices=(
            "smoke",
            "train-step",
            "eval",
            "bench",
            "trt-bench",
            "export-smoke",
            "latency",
            "research-loop",
        ),
    )
    args = parser.parse_args()

    if args.command == "smoke":
        smoke()
        return
    if args.command == "train-step":
        train_step_smoke()
        return
    if args.command == "eval":
        eval_smoke()
        return
    if args.command == "bench":
        bench_smoke()
        return
    if args.command == "trt-bench":
        trt_bench_smoke()
        return
    if args.command == "export-smoke":
        export_smoke()
        return
    if args.command == "latency":
        latency_smoke()
        return
    ensure_research_loop_disabled()
