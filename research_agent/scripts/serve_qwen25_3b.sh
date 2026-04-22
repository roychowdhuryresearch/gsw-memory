#!/usr/bin/env bash
# Qwen2.5-3B-Instruct — tiny sanity-check swap, mainly for Search-R1 (they ship a 3B ckpt).
# Weights: /mnt/SSD3/yigit/models/Qwen2.5-3B-Instruct.
# VRAM: ~6GB. Fits on any A6000 even with competing workloads.
# Port: 8003.

MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
PORT=8003
TP=1
GPUS="${GPUS:-3}"
EXTRA_ARGS=(
  "--max-model-len" "32768"
  "--gpu-memory-utilization" "0.85"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "hermes"
)

source "$(dirname "$0")/serve_vllm_common.sh"
