#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# Optional:
#   export CUDA_VISIBLE_DEVICES=0
#   export VLLM_BIN=.venv/bin/vllm

exec "${VLLM_BIN:-vllm}" serve Qwen/Qwen3.5-35B-A3B \
  --port 6379 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  "$@"
