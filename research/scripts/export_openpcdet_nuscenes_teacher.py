#!/usr/bin/env python3
"""Export nuScenes teacher predictions from OpenPCDet without running official evaluation.

This helper exists because the stock OpenPCDet `tools/test.py` path always evaluates
`v1.0-mini` against `mini_val`, while teacher-cache bootstrapping also needs plain export
coverage for `mini_train`.

Primary sources:
- OpenPCDet test entrypoint:
  https://github.com/open-mmlab/OpenPCDet/blob/master/tools/test.py
- OpenPCDet nuScenes dataset implementation:
  https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/datasets/nuscenes/nuscenes_dataset.py
- OpenPCDet nuScenes config:
  https://github.com/open-mmlab/OpenPCDet/blob/master/tools/cfgs/dataset_configs/nuscenes_dataset.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _info_pickle_name(version: str, split: str) -> str:
    if version == "v1.0-mini":
        if split == "mini_train":
            return "nuscenes_infos_10sweeps_train.pkl"
        if split == "mini_val":
            return "nuscenes_infos_10sweeps_val.pkl"
    if version == "v1.0-trainval":
        if split == "train":
            return "nuscenes_infos_10sweeps_train.pkl"
        if split == "val":
            return "nuscenes_infos_10sweeps_val.pkl"
    if version == "v1.0-test" and split == "test":
        return "nuscenes_infos_10sweeps_test.pkl"
    raise ValueError(f"unsupported version/split combination: version={version!r} split={split!r}")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openpcdet-repo-root", type=Path, required=True)
    parser.add_argument(
        "--cfg-file",
        type=Path,
        default=Path("tools/cfgs/nuscenes_models/cbgs_dyn_pp_centerpoint.yaml"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", choices=("v1.0-mini", "v1.0-trainval", "v1.0-test"), default="v1.0-mini")
    parser.add_argument("--split", required=True)
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main() -> None:
    args = _make_parser().parse_args()
    repo_root = args.openpcdet_repo_root.resolve()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from nuscenes.nuscenes import NuScenes
    from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
    from pcdet.datasets import build_dataloader
    from pcdet.datasets.nuscenes import nuscenes_utils
    from pcdet.models import build_network, load_data_to_gpu
    from pcdet.utils import common_utils
    import torch
    import tqdm

    cfg_file = args.cfg_file
    if not cfg_file.is_absolute():
        cfg_file = repo_root / cfg_file

    data_path = args.data_path.resolve() if args.data_path is not None else (repo_root / "data/nuscenes").resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg_from_yaml_file(str(cfg_file), cfg)
    cfg.TAG = Path(cfg_file).stem
    cfg.EXP_GROUP_PATH = "cfgs/nuscenes_models"
    cfg_from_list(
        [
            "DATA_CONFIG.VERSION",
            args.version,
            "DATA_CONFIG.DATA_PATH",
            str(data_path),
        ],
        cfg,
    )
    cfg.DATA_CONFIG.INFO_PATH["test"] = [_info_pickle_name(args.version, args.split)]
    cfg.DATA_CONFIG.DATA_SPLIT["test"] = args.split

    logger = common_utils.create_logger()
    dataset, dataloader, _ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=False,
        workers=args.workers,
        logger=logger,
        training=False,
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
    model.load_params_from_file(filename=str(checkpoint), logger=logger, to_cpu=args.device == "cpu")
    if args.device == "cuda":
        model.cuda()
    model.eval()

    det_annos: list[dict[str, object]] = []
    for batch_dict in tqdm.tqdm(
        dataloader,
        total=len(dataloader),
        desc=f"export_{args.version}_{args.split}",
        dynamic_ncols=True,
    ):
        if args.device == "cuda":
            load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, _ = model(batch_dict)
        det_annos.extend(
            dataset.generate_prediction_dicts(batch_dict, pred_dicts, cfg.CLASS_NAMES, output_path=output_dir)
        )

    dataroot = data_path / args.version
    nusc = NuScenes(version=args.version, dataroot=str(dataroot), verbose=False)
    nusc_annos = nuscenes_utils.transform_det_annos_to_nusc_annos(det_annos, nusc)
    nusc_annos["meta"] = {
        "use_camera": False,
        "use_lidar": True,
        "use_radar": False,
        "use_map": False,
        "use_external": False,
    }
    result_path = output_dir / "results_nusc.json"
    result_path.write_text(json.dumps(nusc_annos))

    summary = {
        "status": "completed",
        "version": args.version,
        "split": args.split,
        "num_samples": len(det_annos),
        "avg_pred_objects": float(sum(len(anno["name"]) for anno in det_annos) / max(1, len(det_annos))),
        "cfg_file": str(cfg_file),
        "checkpoint": str(checkpoint),
        "data_path": str(data_path),
        "result_path": str(result_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
