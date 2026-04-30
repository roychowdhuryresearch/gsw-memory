#!/usr/bin/env bash
# Qwen3.6-27B dense reasoning model — vLLM Recipes config (text-only +
# tool calling + prefix caching + MTP-1 speculative decoding).
#
# Recipe source: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
#
# Hardware: TP=2 over GPUs 0+1 (RTX A6000, 49 GB each). Weights split
# ~27 GB/GPU, leaving ~12 GB/GPU at gpu-memory-utilization=0.80 for KV
# cache + activations. Leaves GPUs 2+3 free for the dense embedding
# model (~3 GB) + slack.
# Port: 8003.
MODEL_ID="Qwen/Qwen3.6-27B"
PORT=${PORT:-8003}
TP=2
GPUS="${GPUS:-0,1}"
EXTRA_ARGS=(
  "--max-model-len" "262144"
  "--gpu-memory-utilization" "0.80"
  "--max-num-seqs" "128"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "qwen3_coder"
  "--reasoning-parser" "qwen3"
  "--language-model-only"
  "--enable-prefix-caching"
  # NOTE: MTP speculative decoding (`mtp`) at both num_speculative_tokens=1
  # and =2 broke tool-call emission for Qwen3.6 — most calls returned
  # finish=stop with 0 tool calls captured. Disabled.
  "--seed" "0"
)
source "$(dirname "$0")/serve_vllm_common.sh"
