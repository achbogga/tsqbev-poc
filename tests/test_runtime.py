from __future__ import annotations

from tsqbev.runtime import benchmark_forward, run_eval_step, run_train_step


def test_train_step_runs_on_cpu(small_config) -> None:
    metrics = run_train_step(small_config, batch_size=1, device="cpu")
    assert metrics["total"] >= 0.0
    assert metrics["kd_total"] >= 0.0


def test_eval_step_reports_shapes(small_config) -> None:
    metrics = run_eval_step(small_config, batch_size=1, device="cpu")
    assert metrics["total"] >= 0.0
    assert metrics["object_logits_shape"] == (
        1,
        small_config.max_object_queries,
        small_config.num_object_classes,
    )


def test_forward_benchmark_runs_on_cpu(small_config) -> None:
    metrics = benchmark_forward(small_config, steps=2, warmup=1, batch_size=1, device="cpu")
    assert metrics["device"] == "cpu"
    assert metrics["mean_ms"] > 0.0
