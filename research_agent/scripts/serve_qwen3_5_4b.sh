#!/usr/bin/env bash
# Qwen3.5-4B — tiny *reasoning* swap (Downshift 3 / Search-R1 3B-class slot).
# Same reasoning-toggle story as the 9B variant.
# HF ID: Qwen/Qwen3.5-4B. Auto-pulls (~8GB).
# VRAM: ~9GB bf16. Single A6000.
# Port: 8003 (the former Qwen2.5-3B slot).

MODEL_ID="Qwen/Qwen3.5-4B"
PORT=8003
TP=1
GPUS="${GPUS:-3}"
# Same reasoning-parser story as the 9B script.
# Same recipe as the 9B script — see vLLM's Qwen3.5 guide.
EXTRA_ARGS=(
  "--max-model-len" "65536"
  "--gpu-memory-utilization" "0.85"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "qwen3_coder"
  "--reasoning-parser" "qwen3"
)

source "$(dirname "$0")/serve_vllm_common.sh"
