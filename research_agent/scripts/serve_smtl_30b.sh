#!/usr/bin/env bash
# SMTL-30B / AFM-MHQA — OPPO PersonalAI's trained Chain-of-Agents checkpoint.
# HF ID: PersonalAILab/SMTL-30B. Auto-pulls (~60GB) on first run.
# VRAM: ~60GB bf16. Requires TP=2 across two A6000s.
# Port: 8009.

MODEL_ID="PersonalAILab/SMTL-30B"
PORT=8009
TP=2
GPUS="${GPUS:-2,3}"
EXTRA_ARGS=(
  "--max-model-len" "32768"
  "--gpu-memory-utilization" "0.9"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "hermes"
)

source "$(dirname "$0")/serve_vllm_common.sh"
