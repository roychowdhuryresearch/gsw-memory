#!/usr/bin/env bash
# Qwen2.5-7B-Instruct — secondary small-model swap across the substitution grid.
# Weights: /mnt/SSD3/yigit/models/Qwen2.5-7B-Instruct.
# VRAM: ~15GB. Fits on any single A6000.
# Port: 8002.

MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
PORT=8002
TP=1
GPUS="${GPUS:-3}"
EXTRA_ARGS=(
  "--max-model-len" "32768"
  "--gpu-memory-utilization" "0.9"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "hermes"
)

source "$(dirname "$0")/serve_vllm_common.sh"
