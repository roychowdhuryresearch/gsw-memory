#!/usr/bin/env bash
# Shared vLLM serve bootstrap. Sourced by the per-model serve_*.sh scripts.
#
# Usage (in a per-model wrapper):
#   MODEL_ID="Qwen/Qwen2.5-7B-Instruct"
#   PORT=8002
#   TP=1
#   GPUS="3"         # CUDA_VISIBLE_DEVICES string
#   EXTRA_ARGS=("--max-model-len" "32768")
#   source scripts/serve_vllm_common.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# All model weights + HF cache live on /mnt/SSD3/yigit — both because space
# on / and /home is tight and because that's the project convention.
export HF_HOME="${HF_HOME:-/mnt/SSD3/yigit/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"

# If the user has a local-download directory under /mnt/SSD3/yigit/models/<name>
# (from `hf download ... --local-dir ...`), prefer it over a cache fetch.
if [[ -d "${VLLM_MODEL_DIR:-/mnt/SSD3/yigit/models}/${MODEL_ID##*/}" ]]; then
  RESOLVED_MODEL="${VLLM_MODEL_DIR:-/mnt/SSD3/yigit/models}/${MODEL_ID##*/}"
else
  # vLLM will auto-download from HF on first access into $HF_HOME.
  RESOLVED_MODEL="${MODEL_ID}"
fi

if [[ -n "${GPUS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${GPUS}"
fi

VLLM_BIN="${VLLM_BIN:-vllm}"

echo "==============================================="
echo " vLLM serve"
echo "   model_id:        ${MODEL_ID}"
echo "   resolved_model:  ${RESOLVED_MODEL}"
echo "   port:            ${PORT:-8000}"
echo "   tp:              ${TP:-1}"
echo "   CUDA_VISIBLE:    ${CUDA_VISIBLE_DEVICES:-unset}"
echo "   HF_HOME:         ${HF_HOME}"
echo "==============================================="

cd "${REPO_ROOT}"

exec "${VLLM_BIN}" serve "${RESOLVED_MODEL}" \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "${TP:-1}" \
  --served-model-name "${MODEL_ID}" \
  "${EXTRA_ARGS[@]:-}" \
  "$@"
