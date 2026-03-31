from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_parser_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "research"
        / "scripts"
        / "parse_bevfusion_eval_log.py"
    )
    spec = importlib.util.spec_from_file_location("parse_bevfusion_eval_log", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_detection_eval_log_extracts_headline_and_object_metrics() -> None:
    module = _load_parser_module()
    parsed = module.parse_log_text(
        "\n".join(
            [
                "mAP: 0.6730",
                "mATE: 0.2859",
                "NDS: 0.7072",
                "{'object/nds': 0.7072085062762249, 'object/map': 0.6729622385406919}",
            ]
        )
    )

    assert parsed["emitted_metrics"] is True
    assert parsed["eval_kind"] == "bbox"
    assert parsed["headline_metrics"]["mAP"] == 0.6730
    assert parsed["metrics"]["object/nds"] == 0.7072085062762249


def test_parse_map_eval_log_extracts_map_metrics_dict() -> None:
    module = _load_parser_module()
    parsed = module.parse_log_text(
        "{'map/drivable_area/iou@max': 0.72, 'map/mean/iou@max': 0.6295}"
    )

    assert parsed["emitted_metrics"] is True
    assert parsed["eval_kind"] == "map"
    assert parsed["metrics"]["map/mean/iou@max"] == 0.6295


def test_parse_log_text_reports_missing_metrics_for_noise() -> None:
    module = _load_parser_module()
    parsed = module.parse_log_text("training still running\nno metrics yet\n")

    assert parsed["emitted_metrics"] is False
    assert parsed["eval_kind"] is None
    assert parsed["metrics"] == {}
