# RLVR Pipeline: End-to-End Guide

Training a LoRA + GRPO agent on Qwen3-30B-A3B-Thinking to learn sleep-time
multi-hop bridge compilation over GSW structures.

---

## Overview

```
musique.json + GSWs
       │
       ▼
[1] generate_gsws_multihop.py   ← build GSWs from corpus documents
       │
       ▼
[2] copy_gsw_with_global_ids.py ← rename to global doc_i IDs
       │
       ▼
[3a] data_prep.py               ← pair questions with GSW dirs → index.json
       │
       ▼
[3b] make_parquet.py            ← convert index.json → train/val .parquet (veRL format)
       │
       ▼
[4] verl.trainer.main_ppo       ← LoRA + GRPO training via veRL launcher
       │  (uses GSWInteraction + grpo_lora_qwen3.yaml)
       ▼
[5] run_sleep_time.py           ← inference with trained LoRA adapter
```

---

## Prerequisites

### Hardware
- Recommended: 8× H100 80GB (or 8× A100 80GB)
- Minimum for QLoRA: 2× A100 40GB

### Install
```bash
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install -e ".[dev]"
pip install verl peft
```

### Environment
```bash
# .env
OPENAI_API_KEY=your_key_here
```

---

## Step 1 — Generate GSWs

> Skip if GSWs are already on disk (e.g. `/mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini/`).

```bash
python playground/generate_gsws_multihop.py \
    --model-name gpt-4.1-mini \
    --vllm-base-url None
```

**Output structure:**
```
logs/<run_id>/gsw_output/
├── batch_001/networks/doc_0/gsw_0_0.json
├── batch_001/networks/doc_1/gsw_1_0.json
├── batch_002/networks/doc_100/gsw_100_0.json
...
```

To use a local vLLM server instead of OpenAI:
```bash
python playground/generate_gsws_multihop.py \
    --model-name Qwen/Qwen3-30B-A3B-Thinking \
    --vllm-base-url http://127.0.0.1:6379/v1
```

To recover documents that failed in a previous run:
```bash
python playground/generate_gsws_multihop.py \
    --recover-missing logs/<run_id>/gsw_output/full_corpus_<timestamp>/ \
    --corpus-offset 0
```

---

## Step 2 — Assign Global Document IDs

Batch outputs use batch-relative indices (`doc_0` inside each batch).
This script renames them to global indices (`doc_0`, `doc_100`, `doc_200`, ...).

Edit the constants at the top of the script:
```python
# playground/copy_gsw_with_global_ids.py
ORIGINAL_LOGS_DIR = "/home/yigit/codebase/gsw-memory/logs/<your_run_id>"
BATCH_SIZE = 100  # must match what was used in step 1
```

Then run:
```bash
python playground/copy_gsw_with_global_ids.py
```

**Output:**
```
logs/<run_id>/gsw_output_global_ids/networks/
├── doc_0/gsw_0_0.json
├── doc_1/gsw_1_0.json
├── doc_100/gsw_100_0.json
...
```

> For MuSiQue the pre-generated path is:
> `/mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini/`

---

## Step 3 — Build Training Index

Pairs each MuSiQue question with the GSW directories for its supporting paragraphs.

**Mapping:** `paragraph["idx"]` → `doc_{idx}` → `networks_4_1_mini/doc_{idx}/`

```bash
python -m gsw_memory.sleep_time.data_prep \
    --musique  playground_data/musique.json \
    --gsw_path /mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini \
    --output   data/rl_training/index.json \
    --min_hops 2
```

| Argument | Default | Description |
|---|---|---|
| `--musique` | `playground_data/musique.json` | Questions file (JSON array or JSONL) |
| `--gsw_path` | *(required)* | Directory containing `doc_i/` subdirectories |
| `--output` | `data/rl_training/index.json` | Output index path |
| `--limit` | None (all) | Max examples to read |
| `--min_hops` | 2 | Skip single-hop examples |

**Output sample (`index.json`):**
```json
[
  {
    "id": "2hop__13548_13529",
    "gsw_dirs": [
      "/mnt/.../networks_4_1_mini/doc_1",
      "/mnt/.../networks_4_1_mini/doc_2"
    ],
    "support_doc_indices": [1, 2],
    "question": "When was the person who Messi's goals were compared to signed by Barcelona?",
    "answer": "June 1982",
    "answer_aliases": ["1982"],
    "decomposition": [...],
    "num_hops": 2
  },
  ...
]
```

For larger training runs use the full train JSONL:
```bash
python -m gsw_memory.sleep_time.data_prep \
    --musique  playground_data/musique_ans_v1.0_train.jsonl \
    --gsw_path /mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini \
    --output   data/rl_training/index.json \
    --limit    5000
```

---

## Step 3b — Convert to Parquet (veRL format)

veRL requires `.parquet` files. This converts `index.json` → `train.parquet` + `val.parquet`.

```bash
pip install pandas pyarrow   # if not already installed

python -m gsw_memory.sleep_time.make_parquet \
    --index     data/rl_training/index.json \
    --output    data/rl_training/ \
    --val_split 0.05
```

**Output columns:**

| Column | Content |
|---|---|
| `prompt` | JSON-serialized chat messages (system prompt + question) |
| `ground_truth` | Gold answer string |
| `data_source` | `"musique"` |
| `extra_info` | JSON string with `interaction_kwargs` (contains `name`, `ground_truth`, `gsw_dirs`, `answer_aliases`, `decomposition`) |

---

## Step 4 — RL Training (LoRA + GRPO via veRL)

### How it works

veRL's `main_ppo` launcher handles everything automatically:
- Spawns vLLM rollout workers (tensor parallel across GPUs)
- Runs FSDP2 training workers in parallel
- Syncs LoRA weights from trainer → vLLM after each update
- Executes real GSW tool calls via `GSWInteraction` during each rollout turn
- Scores completed episodes via `GSWInteraction.calculate_score()` (bridge F1)
- Logs to W&B

### Config

Key parameters in [`conf/rl/grpo_lora_qwen3.yaml`](../conf/rl/grpo_lora_qwen3.yaml):

| Key | Value | Notes |
|---|---|---|
| `actor_rollout_ref.model.path` | `Qwen/Qwen3-30B-A3B-Thinking` | Base model |
| `actor_rollout_ref.model.lora_rank` | 256 | High rank for tool-use |
| `actor_rollout_ref.model.lora_alpha` | 512 | `2 × rank` |
| `actor_rollout_ref.model.target_modules` | `all-linear` | Excludes MoE router |
| `actor_rollout_ref.model.fsdp_config` | ZeRO-2 | ZeRO-3 breaks LoRA on MoE |
| `actor_rollout_ref.rollout.n` | 8 | GRPO group size |
| `algorithm.kl_ctrl.kl_coef` | 0.001 | KL penalty vs base |
| `actor_rollout_ref.actor.optim.lr` | 1e-5 | Learning rate |
| `data.max_response_length` | 16384 | Covers 30-turn tool call trace |
| `actor_rollout_ref.rollout.multi_turn.enable` | `true` | Live tool execution per turn |
| `actor_rollout_ref.rollout.multi_turn.interaction_config_path` | `conf/rl/gsw_interaction.yaml` | Points to `GSWInteraction` |
| `actor_rollout_ref.rollout.multi_turn.max_tool_response_length` | 2048 | GSW results are verbose |

### Launch

```bash
# --config-dir ADDS to Hydra's search path (--config-path would replace it)
python3 -m verl.trainer.main_ppo \
    --config-dir /home/yigit/codebase/gsw-memory/conf/rl \
    --config-name grpo_lora_qwen3
```

No `torchrun` needed — veRL's launcher manages distributed setup internally.

### What gets trained

Each training episode:
1. Agent receives a multi-hop question and the associated GSW corpus
2. Agent explores GSW structures using the 16 `GSWTools` (reconcile, get context, create bridges, etc.)
3. Episode ends when agent calls `mark_entity_explored` or hits `max_turns=30`
4. Reward is computed over bridges created:

```
reward = 10 × best_F1(bridge answers, gold answer)   [0 if F1 < 0.5]
       + Σ bridges: depth_bonus + novelty_bonus
       + decomp_coverage_bonus
```

Tool response tokens are **masked from the loss** (Search-R1 style) — only the
agent's own reasoning and tool call tokens are trained on.

### Checkpoints

```
logs/rl_training/
├── checkpoint_step_200/   ← LoRA adapter (saved every 200 steps)
├── checkpoint_step_400/
├── ...
└── final/                 ← final LoRA adapter
```

---

## Step 5 — Sleep-Time Inference with Trained Model

Use the trained LoRA adapter with the existing sleep-time runner:

```bash
# First: serve the merged model via vLLM
python -m vllm.entrypoints.openai.api_server \
    --model logs/rl_training/final \
    --tensor-parallel-size 4 \
    --port 6379

# Then: run sleep-time exploration
python playground/run_sleep_time.py \
    --model logs/rl_training/final \
    --base_url http://127.0.0.1:6379/v1 \
    --gsw_path /mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini \
    --num_docs 100 \
    --num_entities 50 \
    --max_iterations 30 \
    --output_dir logs/sleep_time_rl \
    --verbose
```

| Argument | Description |
|---|---|
| `--model` | Path to trained LoRA adapter (or any model name) |
| `--base_url` | vLLM server URL |
| `--gsw_path` | GSW networks directory |
| `--num_docs` | How many GSW documents to load |
| `--num_entities` | How many entities to explore |
| `--max_iterations` | Max tool calls per entity (default: 30) |
| `--reasoning_effort` | `low` / `medium` / `high` (Together AI models) |
| `--resume_from` | Path to checkpoint to resume from |

**Output:**
```
logs/sleep_time_rl/<run_id>/
├── bridges.csv          ← all generated bridge QA pairs
├── results.json         ← full results
├── summary.txt          ← stats summary
└── logs/agent_trace.log ← full tool call trace
```

---

## Quick Reference

```bash
# 1. (Skip if GSWs exist) Generate GSWs
python playground/generate_gsws_multihop.py --model-name gpt-4.1-mini

# 2. (Skip if GSWs exist) Assign global IDs
#    → Edit ORIGINAL_LOGS_DIR in copy_gsw_with_global_ids.py first
python playground/copy_gsw_with_global_ids.py

# 3a. Build training index
python -m gsw_memory.sleep_time.data_prep \
    --musique playground_data/musique.json \
    --gsw_path /mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini \
    --output data/rl_training/index.json

# 3b. Convert to parquet (veRL format)
python -m gsw_memory.sleep_time.make_parquet \
    --index data/rl_training/index.json \
    --output data/rl_training/

# 4. Train with veRL (--config-dir adds to Hydra search path, not replaces it)
python3 -m verl.trainer.main_ppo \
    --config-dir /home/yigit/codebase/gsw-memory/conf/rl \
    --config-name grpo_lora_qwen3

# 5. Inference with trained adapter
python playground/run_sleep_time.py \
    --model logs/rl_training/final \
    --base_url http://127.0.0.1:6379/v1 \
    --gsw_path /mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini \
    --num_docs 100 --num_entities 50 \
    --output_dir logs/sleep_time_rl
```
