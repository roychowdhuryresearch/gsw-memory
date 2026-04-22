#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${REPO_ROOT}"

exec "${PYTHON:-python}" playground/sleep_time/run_curriculum_guidance_experiment.py \
  --manifest playground_data/experiments/2wiki_lookup_diverse_44q.json \
  --output_root logs/experiments/2wiki_lookup_diverse_44q_b6_balanced \
  --gsw_path /mnt/SSD1/shreyas/SM_GSW/2wiki/networks \
  --model bedrock/openai.gpt-oss-120b-1:0 \
  --root_model bedrock/openai.gpt-oss-120b-1:0 \
  --worker_model bedrock/openai.gpt-oss-120b-1:0 \
  --pipeline_mode hybrid \
  --hybrid_scope doc_edge \
  --edge_max_depth 3 \
  --edge_max_calls 5 \
  --edge_max_tokens 12000 \
  --max_tokens 5000000 \
  --edge_parallel_enabled \
  --edge_parallel_workers 4 \
  --bridge_query_top_k 5 \
  --bridge_prompt_exemplar_top_k 3 \
  --curriculum_batch_size 6 \
  --curriculum_seed_batch_size 6 \
  --curriculum_generation_parallel_enabled \
  --curriculum_generation_parallel_workers 4 \
  --show-thinking \
  "$@"
