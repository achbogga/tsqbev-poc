#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/achbogga/projects/tsqbev-poc"
DATASET_ROOT="/home/achbogga/projects/research/nuscenes"
ARTIFACT_ROOT="${1:-$REPO_ROOT/artifacts/foundation_v3_dinov3_teacher_vits16_36ep}"
FOUNDATION_WEIGHTS="${FOUNDATION_WEIGHTS:-/home/achbogga/projects/research/dinov3_weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth}"
TEACHER_CACHE_DIR="${TEACHER_CACHE_DIR:-$REPO_ROOT/artifacts/teacher_cache/centerpoint_pointpillar_mini}"
PRESET="rtx5000-nuscenes-dinov3-teacher"
CHECKPOINT_PATH="$ARTIFACT_ROOT/nuscenes/checkpoint_best.pt"
EVAL_ROOT="$ARTIFACT_ROOT/official_eval_s020_k40"
PREDICTION_PATH="$EVAL_ROOT/predictions.json"

mkdir -p "$ARTIFACT_ROOT" "$EVAL_ROOT"
cd "$REPO_ROOT"

uv run tsqbev train-nuscenes \
  --dataset-root "$DATASET_ROOT" \
  --artifact-dir "$ARTIFACT_ROOT" \
  --preset "$PRESET" \
  --foundation-weights "$FOUNDATION_WEIGHTS" \
  --teacher-cache-dir "$TEACHER_CACHE_DIR" \
  --version v1.0-mini \
  --train-split mini_train \
  --split mini_val \
  --device cuda \
  --epochs 36 \
  --lr 1e-4 \
  --weight-decay 0.0 \
  --optimizer-schedule constant \
  --grad-clip-norm 5.0 \
  --keep-best-checkpoint \
  --early-stop-patience 8 \
  --early-stop-min-delta 0.02 \
  --early-stop-min-epochs 12 \
  --official-eval-every-epochs 5 \
  --official-eval-score-threshold 0.20 \
  --official-eval-top-k 40 \
  --loss-mode quality_focal \
  --teacher-anchor-quality-class-weight 0.45 \
  --teacher-region-objectness-weight 0.12 \
  --teacher-region-class-weight 0.12 \
  --teacher-region-radius-m 6.0 \
  --no-teacher-distillation \
  --batch-size 1 \
  --grad-accum-steps 2 \
  --num-workers 4 \
  --seed 1337

uv run tsqbev export-nuscenes \
  --dataset-root "$DATASET_ROOT" \
  --checkpoint "$CHECKPOINT_PATH" \
  --preset "$PRESET" \
  --foundation-weights "$FOUNDATION_WEIGHTS" \
  --teacher-cache-dir "$TEACHER_CACHE_DIR" \
  --version v1.0-mini \
  --split mini_val \
  --device cuda \
  --score-threshold 0.20 \
  --top-k 40 \
  --output-path "$PREDICTION_PATH"

uv run tsqbev eval-nuscenes \
  --dataset-root "$DATASET_ROOT" \
  --version v1.0-mini \
  --split mini_val \
  --result-json "$PREDICTION_PATH" \
  --output-dir "$EVAL_ROOT"
