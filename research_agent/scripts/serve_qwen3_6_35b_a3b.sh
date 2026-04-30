#!/usr/bin/env bash
# Qwen3.6-35B-A3B (MoE, 35B total / 3B active per token).
# HF ID: Qwen/Qwen3.6-35B-A3B. Auto-pulls (~70 GB BF16).
# Hardware: TP=4 over all 4× RTX A6000 (49 GB each). Weights split
# 17.5 GB/GPU, leaving ~31 GB/GPU for KV cache + activations.
# MTP (Multi-Token Prediction) speculative decoding enabled — Qwen3.6
# was specifically built with MTP heads → ~2× decode speed at no
# quality cost.
# Port: 8003.
MODEL_ID="Qwen/Qwen3.6-35B-A3B"
PORT=${PORT:-8003}
TP=2
GPUS="${GPUS:-0,1}"   # leaves GPUs 2+3 for embedding model + other uses
EXTRA_ARGS=(
  "--max-model-len" "32768"
  "--gpu-memory-utilization" "0.80"
  "--max-num-seqs" "32"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "qwen3_coder"
  "--reasoning-parser" "qwen3"
  "--language-model-only"
  "--seed" "0"
  # NOTE: MTP speculative decoding (`qwen3_next_mtp`) was tried and
  # broke tool-call emission — finish=stop with no tools captured even
  # at tool_choice=required. Disabled. If revisiting, try
  # `num_speculative_tokens=1` first.
)
source "$(dirname "$0")/serve_vllm_common.sh"
