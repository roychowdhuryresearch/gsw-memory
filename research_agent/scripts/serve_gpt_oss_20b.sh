#!/usr/bin/env bash
# gpt-oss-20B MoE — our Tier 3 base + substitution target.
# Weights: /mnt/SSD3/yigit/models/gpt-oss-20b (already downloaded).
# VRAM: ~40GB full precision. Fits on a single A6000 (48GB) with TP=1.
# Port: 8001.

MODEL_ID="openai/gpt-oss-20b"
PORT=8001
TP=1
GPUS="${GPUS:-3}"  # default to GPU 3 (most free per nvidia-smi)
EXTRA_ARGS=(
  "--max-model-len" "32768"
  "--gpu-memory-utilization" "0.9"
  "--tool-call-parser" "openai"
  "--enable-auto-tool-choice"
)

source "$(dirname "$0")/serve_vllm_common.sh"
