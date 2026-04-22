#!/usr/bin/env bash
# Set up the upstream Graph-R1 repo under research_agent/third_party/Graph-R1
# for running their training scripts and producing a checkpoint we can then
# serve via vLLM for the `graph_r1_trained` pilot cell (E11b).
#
# What this does:
#   1. Clones LHRLAB/Graph-R1 into third_party/Graph-R1/ (gitignored).
#   2. Creates a venv at third_party/Graph-R1/.venv with their pinned deps.
#   3. Prints the TeraBox URLs and the exact training command.
#
# What this does NOT do (by design — requires manual steps):
#   - Download training data or pre-built hypergraphs from TeraBox. TeraBox
#     needs a manual login; the README lists the URLs but no CLI tool.
#   - Run the training itself. Training is a 4×48GB-GPU job — the operator
#     needs to schedule GPU time, not fire-and-forget.
#
# Usage:
#   bash scripts/setup_graph_r1.sh
#
# Then follow the printed next-steps to download data, train, and serve.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY="${REPO_ROOT}/third_party"
GRAPH_R1_DIR="${THIRD_PARTY}/Graph-R1"

mkdir -p "${THIRD_PARTY}"

if [[ ! -d "${GRAPH_R1_DIR}/.git" ]]; then
  echo "[setup_graph_r1] cloning LHRLAB/Graph-R1 -> ${GRAPH_R1_DIR}"
  git clone https://github.com/LHRLAB/Graph-R1.git "${GRAPH_R1_DIR}"
else
  echo "[setup_graph_r1] repo already present at ${GRAPH_R1_DIR}; pulling latest"
  git -C "${GRAPH_R1_DIR}" pull --ff-only
fi

# Create a venv if missing. Upstream's requirements include flash-attn and
# torch 2.4.0 — we keep it isolated from the main research_agent venv.
if [[ ! -d "${GRAPH_R1_DIR}/.venv" ]]; then
  echo "[setup_graph_r1] creating venv at ${GRAPH_R1_DIR}/.venv"
  python -m venv "${GRAPH_R1_DIR}/.venv"
fi

cat <<'EOF'

==================================================================
 Graph-R1 setup complete.
 Remaining manual steps (see third_party/Graph-R1/README.md):

 1. Activate the venv:
      source third_party/Graph-R1/.venv/bin/activate
      cd third_party/Graph-R1
      pip install -r requirements.txt

 2. Download training data + hypergraphs from TeraBox (manual login).
    URLs in the upstream README. Drop extracts into:
      third_party/Graph-R1/datasets/<benchmark>/
      third_party/Graph-R1/hypergraphs/<benchmark>/

    Six datasets shipped: 2WikiMultiHopQA, HotpotQA, MuSiQue,
    NQ, PopQA, TriviaQA. For our pilot, 2WikiMultiHopQA is the
    closest FRAMES-like multi-hop target. (No FRAMES-trained
    checkpoint is possible — they don't supply a FRAMES hypergraph.)

 3. Start their retrieve API server (needed by the training loop):
      bash script_api.sh       # binds 127.0.0.1:8001
    NOTE: conflicts with our serve_gpt_oss_20b.sh which also uses
    port 8001. Stop the 20B serve, or edit script_api.sh to
    rebind (e.g. --port 18001).

 4. Train on 4 × 48 GB GPUs (4–12h wall-clock depending on dataset):
      bash run_grpo.sh \
        -p Qwen/Qwen2.5-7B-Instruct \
        -m Qwen2.5-7B-Instruct \
        -d 2WikiMultiHopQA

    Output: third_party/Graph-R1/outputs/<run_name>/ with HF-format
    safetensors under the `global_step_*` subdirs. Pick the final
    checkpoint.

 5. Copy the final checkpoint into the vLLM models dir so our serve
    script auto-resolves it:
      cp -r third_party/Graph-R1/outputs/<run>/global_step_<N>/ \
            /mnt/SSD3/yigit/models/Graph-R1-Qwen2.5-7B

 6. Serve via:
      bash scripts/serve_graph_r1_trained.sh

 7. Run the pilot cell:
      PYTHONPATH=src python playground/run_substitution.py \
        --system graph_r1_trained --model Graph-R1-Qwen2.5-7B \
        --base-url http://127.0.0.1:8010/v1 --api-key dummy \
        --subset configs/pilot_subset.json

==================================================================
EOF
