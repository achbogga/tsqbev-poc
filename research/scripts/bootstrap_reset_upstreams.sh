#!/usr/bin/env bash
set -euo pipefail

# Bootstrap the public upstream repos used by the dense-BEV reset stack.
#
# Primary sources:
# - BEVFusion: https://github.com/mit-han-lab/bevfusion
# - BEVDet: https://github.com/HuangJunJie2017/BEVDet
# - MapTR: https://github.com/hustvl/MapTR
# - PersFormer: https://github.com/OpenDriveLab/PersFormer_3DLane
# - DINOv2: https://github.com/facebookresearch/dinov2
# - DINOv3: https://github.com/facebookresearch/dinov3
# - EfficientViT: https://github.com/mit-han-lab/efficientvit

PROJECTS_ROOT="${PROJECTS_ROOT:-/home/achbogga/projects}"

clone_if_missing() {
  local url="$1"
  local dir="$2"
  if [[ -d "${PROJECTS_ROOT}/${dir}/.git" ]]; then
    echo "present ${dir}"
  else
    git clone --depth 1 "${url}" "${PROJECTS_ROOT}/${dir}"
  fi
}

clone_if_missing "https://github.com/mit-han-lab/bevfusion.git" "bevfusion"
clone_if_missing "https://github.com/HuangJunJie2017/BEVDet.git" "BEVDet"
clone_if_missing "https://github.com/hustvl/MapTR.git" "MapTR"
clone_if_missing "https://github.com/OpenDriveLab/PersFormer_3DLane.git" "PersFormer_3DLane"
clone_if_missing "https://github.com/facebookresearch/dinov2.git" "dinov2"
clone_if_missing "https://github.com/facebookresearch/dinov3.git" "dinov3"
clone_if_missing "https://github.com/mit-han-lab/efficientvit.git" "efficientvit"

# MapTRv2 is not the default branch in the official repo but is the branch we need.
git -C "${PROJECTS_ROOT}/MapTR" fetch origin maptrv2:maptrv2
git -C "${PROJECTS_ROOT}/MapTR" checkout maptrv2

for dir in bevfusion BEVDet MapTR PersFormer_3DLane dinov2 dinov3 efficientvit; do
  echo "== ${dir} =="
  git -C "${PROJECTS_ROOT}/${dir}" rev-parse --abbrev-ref HEAD
  git -C "${PROJECTS_ROOT}/${dir}" rev-parse HEAD
done
