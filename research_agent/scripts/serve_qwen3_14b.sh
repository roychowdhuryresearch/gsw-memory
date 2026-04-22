#!/usr/bin/env bash
# Qwen3-14B dense reasoning model (public HF, Qwen3 base series).
# HF ID: Qwen/Qwen3-14B. Auto-pulls (~28 GB).
# Port: 8002 (reuses the former Qwen3.5-9B slot).
MODEL_ID="Qwen/Qwen3-14B"
PORT=${PORT:-8002}
TP=1
GPUS="${GPUS:-0}"
EXTRA_ARGS=(
  "--max-model-len" "40960"
  "--gpu-memory-utilization" "0.9"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "qwen3_coder"
  "--reasoning-parser" "qwen3"
)
source "$(dirname "$0")/serve_vllm_common.sh"
