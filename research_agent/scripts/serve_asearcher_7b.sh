#!/usr/bin/env bash
# ASearcher-Web-7B — their trained small checkpoint, used as-shipped (no swap).
# HF ID: inclusionAI/ASearcher-Web-7B (v1; no -v2 variant exists on HF —
# only ASearcher-Web-QwQ-V2 got a v2 release). Auto-pulls (~14GB).
# VRAM: ~15GB. Single A6000.
# Port: 8005.

MODEL_ID="inclusionAI/ASearcher-Web-7B"
PORT=8005
TP=1
GPUS="${GPUS:-3}"
EXTRA_ARGS=(
  "--max-model-len" "32768"
  "--gpu-memory-utilization" "0.9"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "hermes"
)

source "$(dirname "$0")/serve_vllm_common.sh"
