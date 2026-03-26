from __future__ import annotations

import importlib.util

import pytest

from tsqbev.export import ExportableCore, build_export_inputs, export_core_to_onnx
from tsqbev.model import TSQBEVModel


@pytest.mark.skipif(
    importlib.util.find_spec("onnx") is None or importlib.util.find_spec("onnxruntime") is None,
    reason="onnx dependencies are not installed",
)
def test_export_smoke(tmp_path, small_config) -> None:
    model = TSQBEVModel(small_config)
    output = export_core_to_onnx(model.core, small_config, tmp_path / "core.onnx")
    assert output.exists()


def test_exportable_core_runs(small_config) -> None:
    model = TSQBEVModel(small_config)
    wrapper = ExportableCore(model.core)
    outputs = wrapper(*build_export_inputs(small_config))
    assert len(outputs) == 4
