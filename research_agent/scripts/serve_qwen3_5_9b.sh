#!/usr/bin/env bash
# Qwen3.5-9B — small *reasoning* swap for the grid (Downshift 2 slot).
# Qwen3.5 small series has thinking/reasoning gated by a chat-template flag;
# we enable it at serve time via --chat-template-kwargs.
# HF ID: Qwen/Qwen3.5-9B. Auto-pulls (~18GB).
# VRAM: ~19GB bf16. Single A6000.
# Port: 8002 (the former Qwen2.5-7B slot).

MODEL_ID="Qwen/Qwen3.5-9B"
PORT=${PORT:-8002}
TP=1
GPUS="${GPUS:-3}"
# Qwen3.5 has thinking mode ON by default in the chat template — no
# serve-time flag needed. The --reasoning-parser flag tells vLLM to
# extract <think>...</think> from the output into message.reasoning_content
# so LLMClient sees it as ``delta.reasoning_content`` (same code path as
# Bedrock gpt-oss).
# Flags match vLLM's official Qwen3.5 recipe:
# https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html
#   --tool-call-parser qwen3_coder   (not hermes — Qwen3 uses XML tool format)
#   --reasoning-parser  qwen3
#   --enable-auto-tool-choice
EXTRA_ARGS=(
  # Qwen3.5-9B natively supports 128k ctx; 64k is plenty and fits on
  # an A6000 with room to spare. 32k was too tight: a 30k output budget
  # + multi-turn prompt accumulation overflowed context on turn 2+.
  "--max-model-len" "65536"
  "--gpu-memory-utilization" "0.9"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "qwen3_coder"
  "--reasoning-parser" "qwen3"
)

source "$(dirname "$0")/serve_vllm_common.sh"
