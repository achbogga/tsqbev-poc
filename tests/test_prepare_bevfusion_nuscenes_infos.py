from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_prepare_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "scripts"
        / "prepare_bevfusion_nuscenes_infos.py"
    )
    spec = importlib.util.spec_from_file_location("prepare_bevfusion_nuscenes_infos", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_inject_locations_adds_bevfusion_and_maptr_keys() -> None:
    module = _load_prepare_module()
    infos = [{"token": "sample-a"}, {"token": "sample-b", "location": "old"}]

    module._inject_locations(
        infos,
        {
            "sample-a": "boston-seaport",
            "sample-b": "singapore-onenorth",
        },
    )

    assert infos[0]["location"] == "boston-seaport"
    assert infos[0]["map_location"] == "boston-seaport"
    assert infos[1]["location"] == "singapore-onenorth"
    assert infos[1]["map_location"] == "singapore-onenorth"
