#!/usr/bin/env bash
# QwQ-32B — the "original scale" column for Search-o1 and ASearcher-prompt-mode swaps.
# HF ID: Qwen/QwQ-32B. Not yet downloaded — vLLM will auto-pull (~64GB) on first run.
# VRAM: ~64GB full precision. Requires TP=2 across two A6000s (or bf16 + quant).
# Port: 8004.

MODEL_ID="Qwen/QwQ-32B"
PORT=8004
TP=2
GPUS="${GPUS:-2,3}"   # pair the two with most free VRAM per nvidia-smi
EXTRA_ARGS=(
  "--max-model-len" "32768"
  "--gpu-memory-utilization" "0.9"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "hermes"
)

source "$(dirname "$0")/serve_vllm_common.sh"
