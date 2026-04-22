#!/usr/bin/env bash
# ASearcher-Web-14B — their trained 14B checkpoint, as-shipped.
# HF ID: inclusionAI/ASearcher-Web-14B (v1; no -v2 variant exists on HF —
# only ASearcher-Web-QwQ-V2 got a v2 release). Auto-pulls (~28GB).
# VRAM: ~28GB bf16. Single A6000 OK, TP=2 preferred for headroom.
# Port: 8006.

MODEL_ID="inclusionAI/ASearcher-Web-14B"
PORT=8006
TP=1
GPUS="${GPUS:-3}"
EXTRA_ARGS=(
  "--max-model-len" "32768"
  "--gpu-memory-utilization" "0.9"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "hermes"
)

source "$(dirname "$0")/serve_vllm_common.sh"
