#!/usr/bin/env bash
# Serve a locally-trained Graph-R1 checkpoint via vLLM.
#
# Upstream (LHRLAB/Graph-R1) ships training code but NO HuggingFace
# checkpoints — you have to produce one yourself. See scripts/setup_graph_r1.sh
# for the full setup + train flow. Once you have a checkpoint, drop it at
# /mnt/SSD3/yigit/models/Graph-R1-Qwen2.5-7B (or override via env vars below)
# and run this script.
#
# Env vars (override as needed):
#   GRAPH_R1_CKPT_DIR  — absolute path to the trained weights.
#                        Default: /mnt/SSD3/yigit/models/Graph-R1-Qwen2.5-7B
#   GRAPH_R1_SERVED_ID — served-model-name vLLM reports. Default:
#                        Graph-R1-Qwen2.5-7B (matches --model in the pilot run)
#   PORT               — default 8010 (out of the way of existing scripts).
#   TP                 — tensor parallel size. Default 1 (fits on one A6000).
#   GPUS               — CUDA_VISIBLE_DEVICES. Default "3".

GRAPH_R1_CKPT_DIR="${GRAPH_R1_CKPT_DIR:-/mnt/SSD3/yigit/models/Graph-R1-Qwen2.5-7B}"
GRAPH_R1_SERVED_ID="${GRAPH_R1_SERVED_ID:-Graph-R1-Qwen2.5-7B}"

if [[ ! -d "${GRAPH_R1_CKPT_DIR}" ]]; then
  cat <<EOF >&2
[serve_graph_r1_trained] ERROR: checkpoint dir not found.
  expected: ${GRAPH_R1_CKPT_DIR}

Run scripts/setup_graph_r1.sh first and follow the manual steps to
train a checkpoint, then copy it into the models dir. Override the
path via GRAPH_R1_CKPT_DIR if it lives elsewhere.
EOF
  exit 1
fi

MODEL_ID="${GRAPH_R1_SERVED_ID}"
PORT="${PORT:-8010}"
TP="${TP:-1}"
GPUS="${GPUS:-3}"

# The shared bootstrap looks up /mnt/SSD3/yigit/models/<MODEL_ID##*/> and uses
# it if present. We set MODEL_ID to the friendly name and let it resolve.
EXTRA_ARGS=(
  "--max-model-len" "32768"
  "--gpu-memory-utilization" "0.9"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "hermes"
)

export VLLM_MODEL_DIR="$(dirname "${GRAPH_R1_CKPT_DIR}")"

source "$(dirname "$0")/serve_vllm_common.sh"
