#!/usr/bin/env bash
# Tongyi DeepResearch 30B-A3B (MoE, 3.3B active) — run as-shipped.
# HF ID: Alibaba-NLP/Tongyi-DeepResearch-30B-A3B. Auto-pulls (~60GB).
# VRAM: ~60GB bf16. TP=2 on two A6000s.
# Port: 8008.

MODEL_ID="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"
PORT=8008
TP=2
GPUS="${GPUS:-2,3}"
EXTRA_ARGS=(
  "--max-model-len" "65536"
  "--gpu-memory-utilization" "0.9"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "hermes"
)

source "$(dirname "$0")/serve_vllm_common.sh"
