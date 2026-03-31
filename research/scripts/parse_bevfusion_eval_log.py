#!/usr/bin/env python3
"""Parse BEVFusion eval logs into machine-readable summary artifacts."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

HEADLINE_PATTERN = re.compile(r"^(mAP|mATE|mASE|mAOE|mAVE|mAAE|NDS|Eval time):\s+([0-9.]+|nan)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to the raw BEVFusion eval log.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--require-kind",
        choices=["any", "bbox", "map"],
        default="any",
        help="Require metrics for a specific eval kind.",
    )
    parser.add_argument(
        "--config-rel",
        type=str,
        default=None,
        help="Config path used for the run.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint path used for the run.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Dataset root used for the run.",
    )
    parser.add_argument(
        "--upstream-repo-root",
        type=str,
        default=None,
        help="Local upstream repo root.",
    )
    parser.add_argument(
        "--upstream-head",
        type=str,
        default=None,
        help="Local upstream git HEAD.",
    )
    parser.add_argument(
        "--docker-exit-code",
        type=int,
        default=None,
        help="Raw docker command exit code.",
    )
    return parser.parse_args()


def _parse_float(raw: str) -> float:
    if raw == "nan":
        return float("nan")
    return float(raw)


def _safe_eval_metrics_dict(raw: str) -> dict[str, Any] | None:
    try:
        value = eval(raw, {"__builtins__": {}}, {"nan": float("nan")})  # noqa: S307
    except Exception:
        return None
    if not isinstance(value, dict):
        return None
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            return None
        if isinstance(item, int | float):
            normalized[key] = float(item)
        else:
            normalized[key] = item
    return normalized


def _extract_last_metrics_dict(lines: list[str]) -> dict[str, Any] | None:
    for line in reversed(lines):
        candidate = line.strip()
        if not candidate.startswith("{") or not candidate.endswith("}"):
            continue
        parsed = _safe_eval_metrics_dict(candidate)
        if parsed is not None:
            return parsed
    return None


def _extract_headline_metrics(lines: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in lines:
        match = HEADLINE_PATTERN.match(line.strip())
        if match is None:
            continue
        metrics[match.group(1)] = _parse_float(match.group(2))
    return metrics


def _infer_eval_kind(metrics_dict: dict[str, Any] | None) -> str | None:
    if metrics_dict is None:
        return None
    keys = metrics_dict.keys()
    if any(key.startswith("object/") for key in keys):
        return "bbox"
    if any(key.startswith("map/") for key in keys):
        return "map"
    return None


def _json_ready(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def parse_log_text(text: str) -> dict[str, Any]:
    lines = text.splitlines()
    metrics_dict = _extract_last_metrics_dict(lines)
    headline = _extract_headline_metrics(lines)
    eval_kind = _infer_eval_kind(metrics_dict)
    emitted = bool(metrics_dict) or bool(headline)
    return {
        "emitted_metrics": emitted,
        "eval_kind": eval_kind,
        "headline_metrics": headline,
        "metrics": metrics_dict or {},
    }


def build_summary(args: argparse.Namespace, parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "success" if parsed["emitted_metrics"] else "missing_metrics",
        "eval_kind": parsed["eval_kind"],
        "log_path": str(args.log),
        "config_rel": args.config_rel,
        "checkpoint_path": args.checkpoint_path,
        "dataset_root": args.dataset_root,
        "upstream_repo_root": args.upstream_repo_root,
        "upstream_head": args.upstream_head,
        "docker_exit_code": args.docker_exit_code,
        "headline_metrics": parsed["headline_metrics"],
        "metrics": parsed["metrics"],
    }


def main() -> None:
    args = parse_args()
    parsed = parse_log_text(args.log.read_text())
    if args.require_kind != "any" and parsed["eval_kind"] != args.require_kind:
        parsed["emitted_metrics"] = False
    summary = build_summary(args, parsed)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(_json_ready(summary), indent=2, sort_keys=True) + "\n")
    print(json.dumps(_json_ready(summary), indent=2, sort_keys=True))
    if not parsed["emitted_metrics"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
