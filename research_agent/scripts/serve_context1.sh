#!/usr/bin/env bash
# Context-1 — Chroma's 20B MoE retrieval subagent, paired with a reasoner.
# HF ID: chromadb/context-1. Auto-pulls (~40GB).
# VRAM: ~40GB bf16. Single A6000 works TP=1; TP=2 for headroom.
# Port: 8007.

MODEL_ID="chromadb/context-1"
PORT=8007
TP=1
GPUS="${GPUS:-3}"
EXTRA_ARGS=(
  "--max-model-len" "32768"
  "--gpu-memory-utilization" "0.9"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "openai"
  # Context-1's tokenizer_config.json references a custom class
  # ``TokenizersBackend`` (Chroma's wrapper). Standard transformers can't
  # import it, so fall back to the base gpt-oss-20b tokenizer — Context-1
  # is a fine-tune of gpt-oss-20b and shares its vocabulary/tokenizer.
  "--trust-remote-code"
  "--tokenizer" "openai/gpt-oss-20b"
)

source "$(dirname "$0")/serve_vllm_common.sh"
