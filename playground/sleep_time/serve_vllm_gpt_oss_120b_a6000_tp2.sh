#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

# Optional:
#   export CUDA_VISIBLE_DEVICES=0,1
#   export VLLM_BIN=.venv/bin/vllm

exec "${VLLM_BIN:-vllm}" serve openai/gpt-oss-120b \
  --tensor-parallel-size 2 \
  --port 6379 \
  --tool-call-parser openai \
  --enable-auto-tool-choice \
  "$@"
