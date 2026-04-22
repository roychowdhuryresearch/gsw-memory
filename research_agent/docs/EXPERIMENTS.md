# Pilot Experiment Plan — Substitution Grid

> **Scope.** Detailed per-cell spec for the inference-only pilot that runs 13 competitor frameworks × 1–3 base-model scales over a 30-question stratified FRAMES subset, plus our own GSW-based system at small scale. The pilot is **inference-only, no training** — everything below runs on released checkpoints or prompt-only wrappers.

---

## Table of contents

1. [Thesis](#thesis)
2. [Workspace map](#workspace-map)
3. [Pilot subset](#pilot-subset)
4. [The substitution grid](#the-substitution-grid)
5. [Execution order & cost](#execution-order--cost)
6. [Per-experiment details](#per-experiment-details)
   - [E1. Vanilla RAG+ReAct](#e1-vanilla-ragreact)
   - [E2. Search-o1](#e2-search-o1)
   - [E3. Search-R1](#e3-search-r1)
   - [E4. ASearcher prompt-mode](#e4-asearcher-prompt-mode)
   - [E5. ASearcher trained checkpoints](#e5-asearcher-trained-checkpoints)
   - [E6. Context-1 + reasoner](#e6-context-1--reasoner)
   - [E7. Tongyi DeepResearch](#e7-tongyi-deepresearch)
   - [E8. SMTL / AFM-MHQA](#e8-smtl--afm-mhqa)
   - [E9. EigentSearch Q+](#e9-eigentsearch-q)
   - [E10. Ours — Tier 3 v1 (focused GSW)](#e10-ours--tier-3-v1-focused-gsw)
   - [E11. Graph-R1 (prompt-mode)](#e11-graph-r1-prompt-mode)
   - [E11b. Graph-R1 (trained checkpoint)](#e11b-graph-r1-trained-checkpoint)
   - [E12. Agentic Reasoning + Mind-Map](#e12-agentic-reasoning--mind-map)
   - [E13. GAM — General Agentic Memory](#e13-gam--general-agentic-memory)
7. [Scoring & evaluation](#scoring--evaluation)
8. [Trace format & audit](#trace-format--audit)
9. [Aggregation — building the grid table](#aggregation--building-the-grid-table)
10. [Go/no-go criteria for scaling up](#gono-go-criteria-for-scaling-up)

---

## Thesis

The pilot is a **scientific substitution ablation**, not a leaderboard chase. For each existing agentic-search framework (10 total), we take their published prompt / tool contract and run it on the same 30-Q FRAMES subset at two or three model scales:

- **Original scale** — the model the paper reports (e.g. QwQ-32B, their 7B trained ckpt, Context-1 20B).
- **Downshift 1** — `openai/gpt-oss-20b` (MoE, same base as Context-1).
- **Downshift 2** — `Qwen/Qwen3.5-9B` (9B reasoning via `enable_thinking=true` in the chat-template kwargs; replaces the non-reasoning Qwen2.5-7B-Instruct so every row in the grid is a like-for-like *reasoner* swap).
- **Tiny sanity** (Search-R1 only) — `Qwen/Qwen3.5-4B` (4B reasoner; replaces Qwen2.5-3B-Instruct).

The cells we run answer two different questions:

1. **Does the framework's contribution survive downshift?** If Search-o1's `<|begin_search_query|>` loop gets 63.6 Avg@4 with QwQ-32B but only 12% with Qwen3.5-9B, the tags were carrying less load than the 32B reasoning. Conversely, if a framework only loses 3 pp going 32B→9B, its contribution is real and portable.
2. **Where does our focused-GSW scratchpad land in the all-small column?** Same base (gpt-oss-20B), same corpus, same subset — any difference vs Vanilla RAG+ReAct is attributable to the scratchpad.

Failure-mode traces (hallucination / wrong-retrieval / incomplete-decomposition / early-stop / loop / tool-error / wrong-synthesis / budget-exceeded / unknown) are recorded per question per cell to surface *why* a downshift loses, not just how much.

---

## Workspace map

All code lives at `/home/yigit/codebase/gsw-memory/research_agent/` — a subdirectory of the parent `gsw-memory` git repo (not a separate repo).

```
research_agent/
├── src/research_agent/
│   ├── adapters/                         # one per framework — 10 total
│   │   ├── base.py                       # Adapter ABC + registry (@register_adapter)
│   │   ├── vanilla_rag_react.py          # E1
│   │   ├── search_o1.py                  # E2
│   │   ├── search_r1.py                  # E3
│   │   ├── asearcher.py                  # E4 + E5 (two registered ids)
│   │   ├── context1.py                   # E6  (retrieval subagent + reasoner)
│   │   ├── tongyi_deep_research.py       # E7
│   │   ├── smtl.py                       # E8
│   │   ├── eigent_search_q_plus.py       # E9
│   │   └── ours_gsw_v1.py                # E10 — decompose → focused GSW → aggregate
│   ├── eval/
│   │   ├── frames_dataset.py             # FRAMES dev/full loader (reuses gsw-memory cache)
│   │   ├── subset.py                     # stratified subset selector + loader
│   │   ├── scoring.py                    # exact_match + token_f1 + alias match
│   │   ├── failure_classifier.py         # 9-category post-parse classifier
│   │   └── harness.py                    # run_cell(adapter, questions) → CellResult
│   ├── retrieval/
│   │   ├── corpus.py                     # FRAMES article cache + paragraph chunker
│   │   └── bm25.py                       # self-contained Okapi-BM25 (no external dep)
│   └── models/
│       ├── trace.py                      # Trajectory + QuestionResult + CellResult
│       ├── failure_modes.py              # FailureMode enum
│       └── llm_client.py                 # thin OpenAI-compatible wrapper
├── playground/
│   ├── select_pilot_subset.py            # CLI → writes configs/pilot_subset.json
│   ├── run_substitution.py               # CLI runs one (system, model) cell
│   └── aggregate_grid.py                 # CLI compiles logs/ → markdown grid + CSV
├── scripts/
│   ├── serve_vllm_common.sh              # shared bootstrap (HF_HOME, CUDA_VISIBLE_DEVICES, local-model detection)
│   ├── serve_gpt_oss_20b.sh              # port 8001, TP=1, GPU 3
│   ├── serve_qwen3_5_9b.sh                # port 8002
│   ├── serve_qwen3_5_4b.sh                # port 8003
│   ├── serve_qwq_32b.sh                  # port 8004, TP=2
│   ├── serve_asearcher_7b.sh             # port 8005
│   ├── serve_asearcher_14b.sh            # port 8006
│   ├── serve_context1.sh                 # port 8007
│   ├── serve_tongyi_30b_a3b.sh           # port 8008, TP=2
│   └── serve_smtl_30b.sh                 # port 8009, TP=2
├── configs/
│   └── pilot_subset.json                 # 30-Q stratified FRAMES subset (generated)
├── tests/                                # 12 passing unit tests
├── third_party/                          # upstream competitor repos (gitignored)
│   ├── Search-R1/                        # PeterGriffinJin/Search-R1
│   ├── Search-o1/                        # RUC-NLPIR/Search-o1
│   ├── ASearcher/                        # inclusionAI/ASearcher
│   ├── context-1-data-gen/               # chroma-core/context-1-data-gen
│   ├── DeepResearch/                     # Alibaba-NLP/DeepResearch (Tongyi)
│   ├── Agent_Foundation_Models/          # OPPO-PersonalAI/Agent_Foundation_Models (SMTL/AFM)
│   └── eigent_search/                    # camel-ai/eigent_search (Q+)
├── logs/                                 # cell_result.json + traces/ per run (gitignored)
├── .env                                  # HF_HOME=/mnt/SSD3/yigit/hf_cache + OPENAI_API_KEY
├── .venv/                                # python 3.12 venv
├── pyproject.toml                        # pydantic + typer + openai + faiss + pandas
└── README.md
```

**Model weights**: `/mnt/SSD3/yigit/models/`
- Already on disk (64 GB total): `gpt-oss-20b`, `Qwen2.5-7B-Instruct`, `Qwen2.5-3B-Instruct`, `bge-large-en-v1.5` (from an earlier iteration — kept as non-reasoning ablation fodder; the current grid uses `Qwen/Qwen3.5-9B` and `Qwen/Qwen3.5-4B` as its reasoning swaps, auto-pulled on first serve).
- Auto-pulled on first vLLM serve: `Qwen/QwQ-32B`, `inclusionAI/ASearcher-Web-7B`, `inclusionAI/ASearcher-Web-14B`, `chromadb/context-1`, `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`, `PersonalAILab/SMTL-30B`.

**Planning doc** (append-only iteration log): `/home/yigit/.claude/plans/eager-weaving-canyon.md`, section `Run N+10`.

---

## Pilot subset

30 questions from the 100-question FRAMES dev split, stratified by hop count. Regenerate with:

```bash
cd /home/yigit/codebase/gsw-memory/research_agent && source .venv/bin/activate
PYTHONPATH=src python playground/select_pilot_subset.py \
    --split dev --subset-id frames_pilot_v1 --seed 0 \
    --out configs/pilot_subset.json
```

Current subset (`configs/pilot_subset.json`):

| Bucket | Count | What it tests |
|---|---|---|
| 2-hop | 10 | Simple multi-hop (find entity A → look up attribute B) |
| 3-hop | 10 | Standard FRAMES difficulty, most common in the benchmark |
| 4-5-hop | 8 | Long chains — stresses turn limits + state tracking |
| 6+-hop | 2 | Extreme long-horizon — pathological cases (6, 7, 9, 10, 11, 13, 17 hops in full dev) |

Reasoning-type coverage (non-mutually-exclusive tags):

| Type | Count |
|---|---|
| Multiple constraints | 20 |
| Temporal reasoning | 14 |
| Tabular reasoning | 8 |
| Numerical reasoning | 7 |
| Post-processing | 4 |

**Why 30, not 100 or 824?** Pilot needs to be fast enough to iterate on adapter bugs. 30 Qs × (16 turns × 2k tokens) ≈ 1M tokens per cell ≈ ~30 min at 1 k tok/s on vLLM for a 7B. 26 cells × 30 min ≈ 13 hours total if everything ran sequentially — manageable in 1–2 days. Scaling to 100 (= dev split) is a 3× multiplier if the pilot signal is strong.

**Variance caveat**: 30 Qs gives ~±10 pp 95% CI on accuracy. A single cell landing at 0.50 could genuinely be 0.40–0.60. Treat score differences < 10 pp as noise at this sample size. That's why we look at **failure-mode histograms + trajectory diffs** alongside the headline number — they're much lower-variance signals at this N.

---

## The substitution grid

13 adapters × 1–3 base-model scales = **31–33 cells** total.

| # | system_id | Original | Downshift 1 (gpt-oss-20B) | Downshift 2 (Qwen3.5-9B) | Downshift 3 (Qwen3.5-4B) |
|---|---|---|---|---|---|
| E1 | `vanilla_rag_react` | GPT-5 (OpenAI API) | ✓ | ✓ | — |
| E2 | `search_o1` | QwQ-32B | ✓ | ✓ | — |
| E3 | `search_r1` | Qwen2.5-7B (their trained baseline) | ✓ | (skipped, duplicate) | ✓ |
| E4 | `asearcher_prompt` | QwQ-32B (prompt-only) | ✓ | ✓ | — |
| E5 | `asearcher_trained` | ASearcher-Web-7B + Web-14B (2 rows) | — | — | — |
| E6 | `context1_plus_reasoner` | Context-1 + GPT-5 | Context-1 + gpt-oss-20B | Context-1 + Qwen3.5-9B | — |
| E7 | `tongyi_deep_research` | Tongyi-30B-A3B | — | — | — |
| E8 | `smtl` | SMTL-30B | ✓ | ✓ | — |
| E9 | `eigent_search_q_plus` | GPT-5 + Q+ | ✓ | ✓ | — |
| E10 | `ours_gsw_v1` | — | ✓ | (optional extra) | — |
| E11 | `graph_r1` (prompt-mode) | Qwen2.5-7B-Instruct (paper's base) | ✓ | ✓ | — |
| E11b | `graph_r1_trained` | Self-trained Graph-R1-Qwen2.5-7B on 2WikiMultiHopQA (upstream's recipe) | — | — | — |
| E12 | `agentic_reasoning_mindmap` | GPT-5 (paper uses DeepSeek-R1 → GPT-5 is closest frontier match) | ✓ | ✓ | — |
| E13 | `gam` | GPT-5 (upstream paper uses frontier) | ✓ | ✓ | — |

Counting: E1=3, E2=3, E3=3, E4=3, E5=2, E6=3, E7=1, E8=3, E9=3, E10=1–2, E11=3, E11b=1, E12=3, E13=3 → **35 cells max, 32 with `ours_gsw_v1` single-row**.

**E11b is conditional on training completing.** Graph-R1 upstream ships training code and TeraBox-hosted hypergraphs + training data but **no HF checkpoints** — see E11b below for the 4-step setup-train-serve flow. Task #30 tracks the prerequisite training run. `graph_r1_trained` alias class is registered; just needs weights and a running serve.

---

## Execution order & cost

**Phase order (recommended):**

1. **Sanity: Vanilla RAG+ReAct on GPT-5 (E1, frontier ceiling).** Validates the eval harness, scoring, trace-dumping pipeline end-to-end. One cell, ~15 min, ~$0.50 of OpenAI spend.
2. **GPT-5 cells across the grid (E1-GPT5, E9-GPT5, E6-GPT5, E12-GPT5, E13-GPT5).** All frontier rows. No vLLM serves needed. Total ~$10–20 (E12 and E13 add ~$3–5 each since GAM tends to browse-heavy and Mind-Map fires an extra tool per turn).
3. **gpt-oss-20B cells.** Two routes available — pick one:
   - **Bedrock route** (no GPU needed): use `--model bedrock/openai.gpt-oss-20b-1:0`. AWS creds chain-load from parent `gsw-memory/.env`; `LLMClient` auto-detects the `bedrock/` prefix and routes through litellm with `reasoning_effort="medium"`. Same recipe sleep-time uses.
   - **Local vLLM route**: `scripts/serve_gpt_oss_20b.sh` on GPU 3, then `--model openai/gpt-oss-20b --base-url http://127.0.0.1:8001/v1 --api-key dummy`.
   Run: E1, E2, E4, E6, E8, E9, E10, E11, E12, E13. **Ten** cells × ~15 min = ~2.5 h.
4. **Qwen3.5-9B cells.** Swap the serve to `scripts/serve_qwen3_5_9b.sh`. Run: E1, E2, E4, E6, E8, E9, E11, E12, E13. **Nine** cells × ~15 min = ~2.25 h.
5. **Qwen2.5-7B-Instruct cell (E11 only).** Graph-R1 uses Qwen2.5-7B as the paper's base. Run `scripts/serve_qwen25_7b.sh`. One cell, ~15 min.
6. **Qwen3.5-4B cell (E3 only).** Swap to `scripts/serve_qwen3_5_4b.sh`. One cell, ~10 min.
7. **Trained checkpoints — serve each, run the cell.** ASearcher-7B, ASearcher-14B, QwQ-32B, Context-1-20B, Tongyi-30B-A3B, SMTL-30B. One at a time (serve → run → teardown). Total wall time ~5 h including weight downloads (~200 GB across the six).
8. **Aggregate.** Run `playground/aggregate_grid.py` to compile `logs/grid_summary.md` + CSV.

**Cost (OpenAI GPT-5 rows only)**: 30 Qs × 3 cells × ~16 turns × ~2 k prompt tokens × GPT-5 input+output pricing. Rough estimate: **$10–25 for all GPT-5 cells on the 30-Q pilot.**

**Compute (local vLLM)**: gpt-oss-20B and Qwen models run at ~400–1000 tok/s on a single A6000 (GPU 3 has 48 GB free per the last check). The 32B+ models need TP=2 across GPUs 2,3. All fine on current hardware once other users' processes free up the VRAM on GPUs 0–2.

---

## Per-experiment details

Each experiment below documents exactly one adapter — its source, what it tests, config, exact run command, expected behavior / scoring, and failure modes we expect to see if it degrades under model downshift.

### E1. Vanilla RAG+ReAct

**Code**: `src/research_agent/adapters/vanilla_rag_react.py`
**System id**: `vanilla_rag_react`
**Source**: Ours. Minimal baseline.

**What it tests.** The absolute baseline — retriever + ReAct tool-call loop + answer. Every other adapter either elaborates on this (by adding structured tools / prompts / decomposition) or replaces part of it (Context-1's subagent). If another adapter doesn't beat Vanilla RAG+ReAct at the same base model, its contribution is questionable.

**Tool schema** (2 OpenAI-style tools):
- `search(query, top_k=5)` → list of {chunk_id, title, score, text_preview}.
- `read(chunk_id)` → {title, article_text (≤ 12 k chars)}.

**Retriever.** BM25 over 366 FRAMES-cached articles chunked into ~2 k-char windows (avg ~5–10 chunks per article, ~2 k chunks total).

**Run commands.**
```bash
cd /home/yigit/codebase/gsw-memory/research_agent && source .venv/bin/activate

# E1-GPT5 (frontier ceiling — needs OPENAI_API_KEY in .env):
PYTHONPATH=src python playground/run_substitution.py \
    --system vanilla_rag_react --model gpt-5 \
    --subset configs/pilot_subset.json

# E1-gpt-oss-20B — two options (pick one):
#
# Option A — Bedrock (recommended, no GPU needed; AWS creds already in
# parent gsw-memory/.env). LLMClient detects the bedrock/ prefix and
# routes through litellm with reasoning_effort="medium".
python playground/run_substitution.py \
    --system vanilla_rag_react --model bedrock/openai.gpt-oss-20b-1:0 \
    --subset configs/pilot_subset.json

# Option B — local vLLM (start serve_gpt_oss_20b.sh first, port 8001):
python playground/run_substitution.py \
    --system vanilla_rag_react --model openai/gpt-oss-20b \
    --base-url http://127.0.0.1:8001/v1 --api-key dummy \
    --subset configs/pilot_subset.json

# E1-Qwen3.5-9B:
PYTHONPATH=src python playground/run_substitution.py \
    --system vanilla_rag_react --model Qwen/Qwen3.5-4B \
    --base-url http://127.0.0.1:8002/v1 --api-key dummy \
    --subset configs/pilot_subset.json
```

**Expected behavior.**
- GPT-5: ~0.60–0.70 Avg@4 (near Gemini-Pro-1.5's 0.66 from the original FRAMES paper on BM25 retrieval with multi-step).
- gpt-oss-20B: ~0.40–0.55 (reasoning is the bottleneck, BM25 is the same).
- Qwen3.5-9B: ~0.30–0.45.

**Failure modes to watch.**
- `hallucination` — if the model answers before calling `search` at all (more common at 7B).
- `loop` — same `search` call with near-identical queries (bad at 3B, less so at 20B+).
- `early_stop` — giving up at turn 2 on 4+-hop questions.

**Why this is the control.** Same 2-tool ReAct used for the frontier and the two downshifts. Every other row of the grid changes one axis — prompt, decomposition, or scratchpad — while holding retriever + ReAct constant. Gives a clean delta attribution.

---

### E2. Search-o1

**Code**: `src/research_agent/adapters/search_o1.py`
**System id**: `search_o1`
**Source**: *Search-o1: Agentic Search-Enhanced Large Reasoning Models* (`arxiv 2501.05366`, EMNLP 2025). [`github.com/RUC-NLPIR/Search-o1`](https://github.com/RUC-NLPIR/Search-o1). Our adapter ports `prompts.py::get_multiqa_search_o1_instruction` verbatim.

**What it tests.** Two things at once:
1. A special-token prompt (`<|begin_search_query|>…<|end_search_query|>`) replacing tool-calls — does structured-prompt tagging help small models without function-calling support?
2. **Reason-in-Documents** — a second LLM call per retrieval that distills the raw docs into a reason-ready note before injection. Extra cost, potentially better noise tolerance.

**Tool schema.** Inline tags only — no OpenAI tool-calls. The LLM emits `<|begin_search_query|>query<|end_search_query|>`; the adapter executes retrieval, distills, and re-prompts with `<|begin_search_result|>note<|end_search_result|>` appended.

**Distinctive config**: 2× LLM calls per retrieval (primary + distillation). Means a 16-turn cell costs roughly 2× a Vanilla RAG+ReAct cell in tokens.

**Run commands.**
```bash
# E2-QwQ32B (needs serve_qwq_32b.sh up on 8004, TP=2):
PYTHONPATH=src python playground/run_substitution.py \
    --system search_o1 --model Qwen/QwQ-32B \
    --base-url http://127.0.0.1:8004/v1 --api-key dummy \
    --subset configs/pilot_subset.json

# E2-gpt-oss-20B, E2-Qwen3.5-9B: same pattern, different --model + --base-url.
```

**Expected behavior.**
- QwQ-32B: ~0.60–0.65 (paper reports 63.6 Avg@4 for Search-o1 + QwQ on full 824 Qs; pilot variance widens this).
- gpt-oss-20B: ~0.40–0.55.
- Qwen3.5-9B: ~0.25–0.40.

**Failure modes.**
- Small models often **fail to emit the special tokens correctly** — closed-model SFT hasn't seen `<|begin_search_query|>` as a literal string. Watch for `no_tag` stops.
- The Reason-in-Documents distillation amplifies model weakness: a bad distill → bad injection → wrong answer. Log `reason_in_docs_errors` in `traj.extra`.

**Attack point for ours.** Search-o1 needs ~2× the LLM compute per turn. If E10 (ours) matches its accuracy at 1× compute per turn, that's a publishable efficiency win.

---

### E3. Search-R1

**Code**: `src/research_agent/adapters/search_r1.py`
**System id**: `search_r1`
**Source**: *Search-R1* (`arxiv 2503.09516`, COLM 2025). [`github.com/PeterGriffinJin/Search-R1`](https://github.com/PeterGriffinJin/Search-R1). Built on veRL + RAGEN. Our adapter ports the `infer.py` prompt — `<think>/<search>/<information>/<answer>` tags.

**What it tests.** The simplest inline-tag ReAct possible. Search-R1's contribution is the **RL-training recipe** (PPO over rule-based outcome reward) that teaches a small model to interleave search and reasoning, not the prompt itself. By running the prompt on untrained base models we isolate the prompt's baseline usefulness vs the trained-ckpt win.

**Tool schema.** Inline tags, single `<search>` per turn, `<information>` injected, `<answer>` terminates.

**Run commands.**
```bash
# E3-Qwen3.5-9B (their reported baseline scale — prompt only, no RL):
PYTHONPATH=src python playground/run_substitution.py \
    --system search_r1 --model Qwen/Qwen3.5-9B \
    --base-url http://127.0.0.1:8002/v1 --api-key dummy \
    --subset configs/pilot_subset.json

# E3-gpt-oss-20B, E3-Qwen3.5-4B: same pattern.
```

**Expected behavior.**
- Qwen3.5-9B (their baseline scale, untrained): low — maybe 0.20–0.35. The PPO-trained version is what performs; the prompt alone is weak.
- gpt-oss-20B: ~0.35–0.50 (more capable base model amortizes for missing training).
- Qwen3.5-4B: ~0.10–0.25 (tiny model + untrained prompt = stress test).

**Failure modes.**
- `no_tag` — Search-R1 prompt requires specific tag format; small untrained models often skip.
- `early_stop` — answers after 1 search on multi-hop questions.
- `incomplete_decomposition` — no planning step in their prompt.

**Attack point for ours.** If trained Search-R1-7B on their website shows N% and our `ours_gsw_v1` at gpt-oss-20B (no training) matches or beats it, structured decomposition compensates for missing RL.

---

### E4. ASearcher prompt-mode

**Code**: `src/research_agent/adapters/asearcher.py` (class `ASearcherAdapter`)
**System id**: `asearcher_prompt`
**Source**: *Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL* (`arxiv 2508.07976`). [`github.com/inclusionAI/ASearcher`](https://github.com/inclusionAI/ASearcher). Our adapter ports `SEARCH_ONLY_PROMPT_TEMPLATE` from their `train/prompts.py`.

**What it tests.** ASearcher's prompt is richer than Search-R1 — explicit `<think>` blocks, support for multiple answers (comma-separated), explicit invalid-question handling. The question: **does their prompt alone (no RL) help, or is all the FRAMES-Avg@4=70.9 due to the GRPO+async RL?**

**Tool schema.** Inline `<search>/<information>/<answer>` tags, with `<think>` as explicit scratchpad. No `<access>` / URL fetch (we're offline).

**Run commands.**
```bash
# E4-QwQ-32B (prompt-only, no training):
PYTHONPATH=src python playground/run_substitution.py \
    --system asearcher_prompt --model Qwen/QwQ-32B \
    --base-url http://127.0.0.1:8004/v1 --api-key dummy \
    --subset configs/pilot_subset.json

# E4-gpt-oss-20B, E4-Qwen3.5-9B: same pattern.
```

**Expected behavior.**
- QwQ-32B (prompt-only): ~0.50–0.62 (paper says pre-RL QwQ-32B around 55 on FRAMES; +15 from RL → 70.9).
- gpt-oss-20B: ~0.40–0.55.
- Qwen3.5-9B: ~0.30–0.45.

**Failure modes.**
- `loop` — ASearcher's prompt is most loop-prone because of "you can search as many times as you want" phrasing; untrained small models take this literally.
- `hallucination` — multiple-answer support encourages over-generating at low confidence.

**Delta vs E3 (Search-R1 prompt).** Both use inline tags. ASearcher adds `<think>` + multi-answer format. If E4 beats E3 at matched base model, the richer prompt is contributing; if they're equivalent, the prompt doesn't matter at this scale.

---

### E5. ASearcher trained checkpoints

**Code**: `src/research_agent/adapters/asearcher.py` (class `ASearcherTrainedAdapter`, same prompt loop, distinct system_id)
**System id**: `asearcher_trained`
**Weights**: `inclusionAI/ASearcher-Web-7B` and `inclusionAI/ASearcher-Web-14B`.

**What it tests.** The actual trained checkpoints — what ASearcher's GRPO + async RL actually buys. Held as its own grid row so the report is apples-to-apples against prompt-only E4.

**Run commands.**
```bash
# Start the 7B serve:
GPUS=3 ./scripts/serve_asearcher_7b.sh   # port 8005
# Then:
PYTHONPATH=src python playground/run_substitution.py \
    --system asearcher_trained --model inclusionAI/ASearcher-Web-7B \
    --base-url http://127.0.0.1:8005/v1 --api-key dummy \
    --subset configs/pilot_subset.json

# Then swap to 14B:
GPUS=3 ./scripts/serve_asearcher_14b.sh   # port 8006
PYTHONPATH=src python playground/run_substitution.py \
    --system asearcher_trained --model inclusionAI/ASearcher-Web-14B \
    --base-url http://127.0.0.1:8006/v1 --api-key dummy \
    --subset configs/pilot_subset.json
```

**Expected behavior.**
- Web-7B: No explicit 7B paper number is public. Estimate 0.55–0.70 based on scaling from 32B + RL boost.
- Web-14B: ~0.60–0.72.
- **Note:** The paper's 70.9 Avg@4 is the 32B model. At 7B/14B the numbers will be lower. This is the whole point of checking.

**Failure modes.** Should be minimal — these are the trained checkpoints. If we see bad behavior here, the eval harness has a bug.

**Verification**: E5-7B and E5-14B numbers are the **harness validation** — they must match ASearcher's published numbers within ±2 pp. If not, our BM25 retriever is far enough from their web retriever that the comparison isn't valid, and we need to revisit retrieval before trusting other cells.

---

### E6. Context-1 + reasoner

**Code**: `src/research_agent/adapters/context1.py`
**System id**: `context1_plus_reasoner`
**Sources**:
- Context-1 tech report: [trychroma.com/research/context-1](https://www.trychroma.com/research/context-1).
- Weights: [`chromadb/context-1`](https://huggingface.co/chromadb/context-1) (20B MoE, gpt-oss-20B base).
- Data-gen repo: [`github.com/chroma-core/context-1-data-gen`](https://github.com/chroma-core/context-1-data-gen).

**What it tests.** Unlike the other rows, Context-1 is explicitly designed as a **retrieval subagent** paired with an external reasoner. Our adapter is two-stage:
1. Context-1 runs a tool loop with {`search`, `read`, `prune`, `done`} to curate chunks.
2. A separate reasoner (GPT-5 / gpt-oss-20B / Qwen3.5-9B) reads ONLY the kept chunks and answers.

The substitution axis is **which reasoner sits on top of the same Context-1**. Isolates retrieval quality (Context-1's contribution) from reasoning quality (reasoner choice).

**Tool schema.** 4 tools for the retrieval stage:
- `search(query, top_k)` → candidate chunks.
- `read(chunk_id)` → full article body.
- `prune(chunk_ids)` → drop chunks from the kept set.
- `done(kept_chunk_ids)` → terminate retrieval, hand off to reasoner.

**Run commands.**
```bash
# E6-GPT5 (Context-1 via vLLM as retrieval; GPT-5 API as reasoner).
GPUS=3 ./scripts/serve_context1.sh   # port 8007
# Then run:
PYTHONPATH=src python playground/run_substitution.py \
    --system context1_plus_reasoner \
    --model chromadb/context-1 \
    --base-url http://127.0.0.1:8007/v1 --api-key dummy \
    --subset configs/pilot_subset.json
# Note: the reasoner is set via --extra. Pass "reasoner" config as JSON:
#   --extra '{"reasoner":{"model_name":"gpt-5","base_url":"","api_key":"$OPENAI_API_KEY"}}'
# (CLI currently supports a simple default — extend playground/run_substitution.py
#  if per-cell reasoner overrides are needed.)
```

Current CLI doesn't yet pass `reasoner` via `--extra`; to run the 3 reasoner rows, either (a) extend the CLI with `--reasoner-model / --reasoner-base-url` flags, or (b) instantiate `Context1PlusReasonerAdapter` directly in a small driver script per row.

**Expected behavior.**
- Context-1 + GPT-5: ~0.70–0.85 (tech report claims 0.87 F1 on FRAMES at single-shot; our pilot with BM25 vs their production retriever may be lower).
- Context-1 + gpt-oss-20B: ~0.55–0.70 (Context-1 IS gpt-oss-20B; adding it as reasoner is degenerate but worth running).
- Context-1 + Qwen3.5-9B: ~0.45–0.60.

**Failure modes.**
- Retrieval subagent never calls `done()` → we fall back to all gathered chunks; the reasoner gets noisy context.
- Retrieval subagent prunes everything → reasoner has empty context and hallucinates.
- Pruning heuristics in Context-1 are tuned on their synthetic web/SEC/patent/email distributions; FRAMES (Wikipedia) is out-of-domain.

**Attack point for ours.** Context-1 prunes *chunks*; ours prunes by *entity subgraph*. If GSW's per-entity-subgraph scratchpad produces comparable retrieval quality with cleaner interpretability, that's a side-claim.

---

### E7. Tongyi DeepResearch

**Code**: `src/research_agent/adapters/tongyi_deep_research.py`
**System id**: `tongyi_deep_research`
**Weights**: [`Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B). 30.5 B total, 3.3 B active per token (MoE).
**Framework**: [`github.com/Alibaba-NLP/DeepResearch`](https://github.com/Alibaba-NLP/DeepResearch).

**What it tests.** The **shipped** Tongyi agent, as-is. Their tools are {Search, Visit, Python, Scholar, FileParser}; for our offline FRAMES setup we only enable Search + Visit.

**Tool schema** (2 enabled):
- `Search(query, top_k)` → chunks.
- `Visit(chunk_id)` → full article.

Python, Scholar, FileParser tools are disabled — calling them returns a structured error. The model is instructed of this up front.

**Run commands.**
```bash
GPUS=2,3 ./scripts/serve_tongyi_30b_a3b.sh   # port 8008, TP=2
PYTHONPATH=src python playground/run_substitution.py \
    --system tongyi_deep_research \
    --model Alibaba-NLP/Tongyi-DeepResearch-30B-A3B \
    --base-url http://127.0.0.1:8008/v1 --api-key dummy \
    --subset configs/pilot_subset.json
```

**Expected behavior.**
- ~0.55–0.70. Published FRAMES numbers from the Tongyi blog claim SOTA on multiple deep-research benchmarks, but the exact FRAMES score isn't public. Single row (no swaps) — their MoE is the story.

**Failure modes.**
- Model confused by disabled tools (Python etc.) — may "stub" its answer instead of using real evidence.
- MoE routing + BF16 vLLM overhead; first retrieve is slow (~30s cold start).

**Attack point.** If ours matches or beats Tongyi at 1/10 the active params (3.3 B active vs our 20 B gpt-oss-MoE = closer to parity, but our full-model memory footprint is smaller than 30 B).

---

### E8. SMTL / AFM-MHQA

**Code**: `src/research_agent/adapters/smtl.py`
**System id**: `smtl`
**Sources**:
- *Search More, Think Less* (`arxiv 2602.22675`).
- *Chain-of-Agents: End-to-End Agent Foundation Models* (`arxiv 2508.13167`).
- [`github.com/OPPO-PersonalAI/Agent_Foundation_Models`](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models) — `AFM/data/mhqa_agent/sys_prompts.py::MHQA_PROMPT`.
- Weights: [`PersonalAILab/SMTL-30B`](https://huggingface.co/PersonalAILab/SMTL-30B).

**What it tests.** Two claims:
1. The **6-function prompt** (`<think>`/`<plan>`/`<wiki_search>`/`<observation>`/`<reflection>`/`<answer>`) — does explicit structured thinking help, or does it just eat tokens?
2. The **Search More, Think Less thesis** — the paper argues lighter per-step thinking + more exploration beats heavy reasoning per step. We test it at inference by keeping max_turns generous + trusting the prompt's "keep each <think> concise" instruction.

**Tool schema.** Inline tags only. `<wiki_search>` calls retrieve; `<observation>` injects. PRM-style reflection is free-form text in `<reflection>` tags — the adapter doesn't score it structurally, just records it as reasoning.

**Run commands.**
```bash
# E8-SMTL-30B (as-shipped):
GPUS=2,3 ./scripts/serve_smtl_30b.sh   # port 8009, TP=2
PYTHONPATH=src python playground/run_substitution.py \
    --system smtl --model PersonalAILab/SMTL-30B \
    --base-url http://127.0.0.1:8009/v1 --api-key dummy \
    --subset configs/pilot_subset.json

# E8-gpt-oss-20B, E8-Qwen3.5-9B: same pattern, different endpoint.
```

**Expected behavior.**
- SMTL-30B: ~0.55–0.70 (paper reports BrowseComp 48.6 %; FRAMES likely higher because Wiki > open web). First-ever FRAMES number for this checkpoint (as far as we can tell).
- gpt-oss-20B: ~0.40–0.55. Tests whether the prompt's structure helps an untrained 20 B.
- Qwen3.5-9B: ~0.25–0.40. Small model struggles with complex tag discipline.

**Failure modes.**
- Small models often **mis-emit the PRM reflection format** — scoring "good/average/poor" on 3 criteria is a lot to ask without training. Watch `no_tag` + long text without any special tags.
- `<plan>` block ignored at small scale — model just emits `<think>+<wiki_search>` and skips planning. Quantify by checking whether any `<plan>` tag appears in traces.

**Attack point for ours.** Both ours and SMTL do explicit decomposition. SMTL is free-form `<plan>` blocks, ours is a structured `SubQuestion` JSON list. If ours produces better sub-answers, structured decomposition > prose.

---

### E9. EigentSearch Q+

**Code**: `src/research_agent/adapters/eigent_search_q_plus.py`
**System id**: `eigent_search_q_plus`
**Source**: *EigentSearch-Q+* (`arxiv 2604.07927`). [`github.com/camel-ai/eigent_search`](https://github.com/camel-ai/eigent_search) — `eigent_search/toolkit/query_toolkit.py::QueryProcessingToolkit`.

**What it tests.** The Q+ thesis: **structured query-processing tools** (not just retrieval tools) are the active ingredient. Q+ adds four tools that are orthogonal to retrieval itself:
- `plan_next_searches(search_queries)` — frontier management.
- `select_query_and_search(query, top_k)` — explored-set tracking.
- `extract_relevant_details(notes)` — running note accumulator.
- `analyze_search_progress(summary, gaps)` — reflection.

The paper reports +3.0 / +3.8 / +0.6 pp accuracy gains on 4 benchmarks (including FRAMES) when Q+ tools are layered on top of GPT-4.1 / GPT-5.1 / Minimax M2.5. We expect the gain to be bigger at small model scales (more help needed) or to vanish (small models can't use the tools effectively).

**Run commands.**
```bash
# E9-GPT5:
PYTHONPATH=src python playground/run_substitution.py \
    --system eigent_search_q_plus --model gpt-5 \
    --subset configs/pilot_subset.json

# E9-gpt-oss-20B, E9-Qwen3.5-9B: vLLM + base-url.
```

**Expected behavior.**
- GPT-5 + Q+: ~0.60–0.75 (Q+ paper shows modest gain on GPT-5.1).
- gpt-oss-20B + Q+: ~0.40–0.60 (could go either way — the 4-tool schema is complex).
- Qwen3.5-9B + Q+: ~0.25–0.45 (likely fails to use the tools well).

**Failure modes.**
- `hallucination` — model calls `extract_relevant_details` with made-up notes.
- `tool_error` — model calls `select_query_and_search` with a query it already explored (our adapter returns an error, which the model must recover from).
- `loop` — model loops `plan_next_searches` without ever calling `select_query_and_search`.
- `budget_exceeded` — all 4 tools are "safe" to call, so the model may over-tool and hit max_turns before answering.

**Attack point for ours.** Q+ is the closest existing prior art to our "structured scratchpad" thesis. If ours does better at same base model, structured **content** (entity-relation triples) beats structured **process** (query-frontier management). If Q+ does better, the gain is in process, not content — useful to know.

---

### E10. Ours — Tier 3 v1 (focused GSW)

**Code**: `src/research_agent/adapters/ours_gsw_v1.py`
**System id**: `ours_gsw_v1`
**Source**: Ours. First instantiation of the query-driven research agent thesis (Run N+9 in the plan file).

**What it tests.** Our headline claim: **structured intermediate state helps small models on multi-hop QA.** Pipeline per question:

1. **Problem composer** — LLM decomposes into ≤ 6 sub-questions with entity focus + hop type (`lookup`/`compare`/`aggregate`/`temporal`/`other`). Returns JSON (`response_format={"type":"json_object"}`).
2. **Per-sub-question retrieval** — BM25 top-k with the sub-question's entity focus prepended.
3. **Focused GSW extraction** — LLM extracts entities + binary (subject, verb, object) triples from the retrieved chunks. Each triple cites `evidence_chunk_ids`. Returns JSON. ≤ 12 triples per sub-Q.
4. **Sub-answer** — LLM answers the sub-question from the triples + chunks only.
5. **Aggregator** — LLM combines sub-answers into the final answer.

5 distinct LLM calls per question (1 decompose + N_sub × 3 for retrieve/extract/answer + 1 aggregate). With N_sub = 3, that's 11 calls vs Vanilla RAG+ReAct's ~4–8.

**Run commands.**
```bash
# E10 — our v1 on gpt-oss-20B:
GPUS=3 ./scripts/serve_gpt_oss_20b.sh   # port 8001
PYTHONPATH=src python playground/run_substitution.py \
    --system ours_gsw_v1 --model openai/gpt-oss-20b \
    --base-url http://127.0.0.1:8001/v1 --api-key dummy \
    --subset configs/pilot_subset.json

# Optional second row on Qwen3.5-9B to test smaller base:
PYTHONPATH=src python playground/run_substitution.py \
    --system ours_gsw_v1 --model Qwen/Qwen3.5-9B \
    --base-url http://127.0.0.1:8002/v1 --api-key dummy \
    --subset configs/pilot_subset.json
```

**Expected behavior.**
- gpt-oss-20B + ours: **the key cell**. Target: beat Vanilla RAG+ReAct at the same base (E1-gpt-oss-20B) by ≥ 5 pp. If it doesn't, the scratchpad isn't pulling weight.
- Qwen3.5-9B + ours: tests whether the structured pipeline survives at even smaller scale. Expect bigger delta vs E1-Qwen3.5-9B because small models benefit more from being told *what to think about*.

**Failure modes.**
- `incomplete_decomposition` — composer emits 1 sub-Q on a 4-hop question → early_stop in disguise. Watch for mismatch between `num_hops` and `n_sub_questions` in traces.
- `wrong_synthesis` — triples are correct but aggregator ignores them. Look at the `gsw_snapshot` in `traj.extra`.
- `hallucination` on extraction — model invents triples not supported by chunks. Check `evidence_chunk_ids` coverage.
- `budget_exceeded` — 5+ LLM calls per Q × 30 Q = 150+ calls per run; if any fail with an OpenAI rate-limit or a vLLM timeout, the cell reports partial results.

**Headline test.** E10-gpt-oss-20B vs E1-gpt-oss-20B at equal compute per question. If ours wins by ≥ 5 pp, the scratchpad is doing real work. If it's ±2 pp, we're just burning more tokens for no gain and need to rethink the pipeline (probably move from flat triples to verb-phrase-centric structure, matching the linker v3 output).

---

### E11. Graph-R1 (prompt-mode)

**Code**: `src/research_agent/adapters/graph_r1.py`
**System id**: `graph_r1` · reserved alias `graph_r1_trained`
**Source**: *Graph-R1: Towards Agentic GraphRAG Framework via End-to-End RL*, [`arxiv 2507.21892`](https://arxiv.org/abs/2507.21892), [LHRLAB/Graph-R1](https://github.com/LHRLAB/Graph-R1).

**What it tests.** Graph-R1 is a head-on thesis competitor: multi-turn agent + graph intermediate state + RL-trained on multi-hop QA. The paper's headline is Qwen2.5-7B @ 57.82 avg F1 across HotpotQA/2Wiki/MuSiQue/NQ/PopQA/TriviaQA, beating Search-R1 at the same scale. We want to see (a) whether the *loop design* (think→query→retrieve→rethink→answer with hyperedge accumulation) alone contributes meaningfully, and (b) how it stacks against our GSW scratchpad at the same small-model base.

**Constraint — why prompt-mode, not trained-checkpoint.** Upstream ships training code (GRPO / REINFORCE++ / PPO bash scripts) + training data and a pre-built hypergraph on **TeraBox** — **no HuggingFace checkpoints**. Their 1.5B/3B/7B numbers come from weights *you train yourself* on the supplied hypergraphs. Training a FRAMES-native Graph-R1 is paper-scope, not pilot-scope. For the pilot we run the paper's inference-time loop against a base instruction-tuned model — same substitution pattern as `search_o1` wrapping QwQ without a specific ckpt. When a trained checkpoint exists, run under `graph_r1_trained` (alias class already in place).

Two implementation deviations from the paper:
1. **Hypergraph built on-the-fly from retrieved chunks**, not pre-indexed across the corpus. A cheap second LLM pass per retrieve-turn extracts (subject, predicate, object, context) tuples. Accumulated across turns and shown back to the agent in the `<information>` block.
2. **BM25 retrieval** via our shared retriever, same as every other adapter in the grid.

**Run commands.**
```bash
# E11-Qwen2.5-7B-Instruct (closest to paper's base):
GPUS=3 ./scripts/serve_qwen25_7b.sh   # port 8004
PYTHONPATH=src python playground/run_substitution.py \
    --system graph_r1 --model Qwen/Qwen2.5-7B-Instruct \
    --base-url http://127.0.0.1:8004/v1 --api-key dummy \
    --subset configs/pilot_subset.json

# E11-gpt-oss-20B (downshift 1 — actually upshift in param count):
PYTHONPATH=src python playground/run_substitution.py \
    --system graph_r1 --model openai/gpt-oss-20b \
    --base-url http://127.0.0.1:8001/v1 --api-key dummy \
    --subset configs/pilot_subset.json

# E11-Qwen3.5-9B (downshift 2, reasoner):
PYTHONPATH=src python playground/run_substitution.py \
    --system graph_r1 --model Qwen/Qwen3.5-9B \
    --base-url http://127.0.0.1:8002/v1 --api-key dummy \
    --subset configs/pilot_subset.json
```

**Expected behavior.**
- E11-Qwen2.5-7B: the paper's regime. On their native benchmarks they beat Search-R1 by ~11 pp avg F1; since FRAMES isn't in their training mix, we expect Graph-R1 prompt-mode to land *near* `search_r1` prompt-mode at same base (±3 pp), not at the paper's reported 57.8 F1.
- E11-gpt-oss-20B vs E10-gpt-oss-20B (ours): **the direct head-to-head.** Same base model, same corpus. If ours wins by ≥ 3 pp, query-focused GSW beats on-the-fly hypergraph as intermediate representation. If Graph-R1 wins, we need to reconsider the QA-pair-vs-triple primitive choice.

**Failure modes.**
- `extract_empty` — hyperedge extractor returns zero tuples (common when chunks are short or dense). Check `traj.extra.hyperedge_count == 0` on wrong-answer Qs.
- `seen_queries` — agent re-queries the same phrase twice; our adapter flags this with a `(rephrase)` suffix. If > 30% of queries are rephrase, the agent isn't using its accumulated hyperedges for planning.
- `no_tag` — model emits neither `<query>` nor `<answer>`; treated as premature stop.

**Artifacts captured.** `traj.extra.hyperedges` contains the full tuple list at stop time; `traj.extra.hyperedge_count` is the total. Aggregator can correlate hyperedge count with judge-correct for a cheap ablation of "more graph = better answers."

---

### E11b. Graph-R1 (trained checkpoint)

**Code**: [`graph_r1.py`](../src/research_agent/adapters/graph_r1.py) — same file as E11, but run under the `graph_r1_trained` alias.
**System id**: `graph_r1_trained`
**Source**: Same paper as E11 ([`arxiv 2507.21892`](https://arxiv.org/abs/2507.21892) · [LHRLAB/Graph-R1](https://github.com/LHRLAB/Graph-R1)). Distinct cell because the *weights* are different — user-trained, not stock Qwen.

**⚠️ Prerequisite: you must train the checkpoint.** As of April 20 2026:
- LHRLAB's HuggingFace account publishes **0 models** (verified: `huggingface.co/LHRLAB` → "None public yet").
- The arXiv paper is only at v1 and links no weights.
- No third-party upload of a Graph-R1 checkpoint on HuggingFace or elsewhere.

Their training code is open but their data + pre-built hypergraph are on **TeraBox** (manual login). Task #30 tracks this prerequisite.

**What it tests.** Same thesis-comparison as E11 (multi-turn agent + graph state, head-to-head vs our E10) — but with the RL training actually done. The difference between E11 and E11b isolates the contribution of RL training on top of the inference-time loop at the same base model. This is the cleanest version of "does their GRPO training actually buy more than the prompt structure?"

**4-step setup → train → serve → run flow.**

```bash
# 1. Clone upstream + create their venv. Prints the remaining manual steps.
bash scripts/setup_graph_r1.sh
# → third_party/Graph-R1/ cloned, third_party/Graph-R1/.venv created.

# 2. Inside their venv, install their deps:
source third_party/Graph-R1/.venv/bin/activate
cd third_party/Graph-R1
pip install -r requirements.txt
# NOTE: flash-attn + torch 2.4.0. Upstream is pinned to CUDA 12.1.

# 3. Download their data + pre-built hypergraphs from TeraBox (manual login,
#    their README has the URL). Extract to:
#      third_party/Graph-R1/datasets/2WikiMultiHopQA/
#      third_party/Graph-R1/hypergraphs/2WikiMultiHopQA/
#    For FRAMES eval, 2WikiMultiHopQA is the closest in-domain training set.
#    (No FRAMES hypergraph upstream — we'd have to build one ourselves.)

# 4. Start their retrieve API server (needed during training):
bash script_api.sh
# ⚠️ Binds 127.0.0.1:8001 — CONFLICTS with our serve_gpt_oss_20b.sh. Stop
# the gpt-oss serve before starting this, or edit script_api.sh to bind a
# different port.

# 5. Train on 4 × 48GB GPUs. Wall-clock: 4–12h depending on dataset size.
bash run_grpo.sh \
  -p Qwen/Qwen2.5-7B-Instruct \
  -m Qwen2.5-7B-Instruct \
  -d 2WikiMultiHopQA

# Output: third_party/Graph-R1/outputs/<run>/global_step_<N>/  (HF-format
# safetensors). Pick the final global_step dir.

# 6. Copy the final checkpoint into the vLLM models dir so our serve script
#    auto-resolves it via serve_vllm_common.sh:
cp -r third_party/Graph-R1/outputs/<run>/global_step_<N>/ \
      /mnt/SSD3/yigit/models/Graph-R1-Qwen2.5-7B

# 7. Serve on port 8010:
bash scripts/serve_graph_r1_trained.sh
# (Override weights path via GRAPH_R1_CKPT_DIR=/some/other/path if needed.)

# 8. Run the pilot cell:
PYTHONPATH=src python playground/run_substitution.py \
    --system graph_r1_trained --model Graph-R1-Qwen2.5-7B \
    --base-url http://127.0.0.1:8010/v1 --api-key dummy \
    --subset configs/pilot_subset.json
```

**Grid rows**: one row — the trained checkpoint. No downshift; the training already fixed the base model size. If we want 3B and 1.5B rows later, retrain with those `-p` / `-m` args.

**Expected behavior.**
- E11b is the **"FRAMES-vs-their-training-set generalization"** cell. The trained checkpoint saw 2WikiMultiHopQA during RL; FRAMES is a harder, more diverse distribution. We expect either (a) a clear lift over E11-Qwen2.5-7B (prompt-mode, untrained) if the RL training generalizes, or (b) a regression if the RL overfit to 2Wiki's question style.
- Either result is informative. (a) strengthens the "structured-agent RL works on small models" case. (b) shows the training doesn't cross-transfer — motivating our own recipe to train on a more diverse mixture.

**Failure modes.**
- Training OOM — 4×48GB is tight; upstream's default batch size may need to drop.
- Retrieve-API port conflict (8001) — rebind before training.
- Checkpoint format mismatch — upstream uses verl-style output dirs; vLLM should auto-load safetensors, but verify `global_step_<N>/model.safetensors` exists.
- Same inference-time failures as E11 (`extract_empty`, `seen_queries`, `no_tag`).

**Artifacts**: identical to E11 (`traj.extra.hyperedges`, `hyperedge_count`). The distinct `system_id` is what lets the aggregator report E11 vs E11b as separate rows.

**Defer policy.** If the FRAMES pilot has to ship before the training completes, skip E11b for the pilot report and note it as "pending training." The prompt-mode E11 already covers the loop design; E11b is the training-contribution check, which is more relevant to the paper than to the pilot go/no-go.

---

### E12. Agentic Reasoning + Mind-Map

**Code**: `src/research_agent/adapters/agentic_reasoning_mindmap.py`
**System id**: `agentic_reasoning_mindmap`
**Source**: *Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools*, [`arxiv 2502.04644`](https://arxiv.org/abs/2502.04644) · ACL 2025 long.

**What it tests.** Mind-Map is the closest *prompt-only* cousin to our GSW thesis. The original paper runs on DeepSeek-R1 and exposes three peer tools (Web-Search, Coding, Mind-Map); we swap Web-Search for BM25 and drop Coding (FRAMES rarely needs arithmetic). What remains is: does a persistent per-question mind-map — an entity-relation KG the agent explicitly reads and writes — help at the same base as the ReAct baseline?

The test is especially clean against E1 (Vanilla RAG+ReAct) at the same model: both have `search`, but E12 also has `mind_map_update(edges)` and `mind_map_query(focus)`. Any E12 - E1 delta at the same base attributable to the mind-map tool.

**Tool surface.**
- `search(query, top_k)` — BM25 retrieval, same shape as E1.
- `mind_map_update(edges)` — append (subject, relation, object) triples to the persistent per-question mind-map.
- `mind_map_query(focus)` — return all triples (max 30) whose subject or object contains `focus` (case-insensitive substring match). Persistent scratchpad across turns.

The mind-map is **per-question, not corpus-wide** — it survives context compression because the agent queries it explicitly instead of re-reading raw chunks.

**Run commands.**
```bash
# E12-GPT5 (closest frontier match to DeepSeek-R1 base):
PYTHONPATH=src python playground/run_substitution.py \
    --system agentic_reasoning_mindmap --model gpt-5 \
    --subset configs/pilot_subset.json

# E12-gpt-oss-20B:
PYTHONPATH=src python playground/run_substitution.py \
    --system agentic_reasoning_mindmap --model openai/gpt-oss-20b \
    --base-url http://127.0.0.1:8001/v1 --api-key dummy \
    --subset configs/pilot_subset.json

# E12-Qwen3.5-9B:
PYTHONPATH=src python playground/run_substitution.py \
    --system agentic_reasoning_mindmap --model Qwen/Qwen3.5-9B \
    --base-url http://127.0.0.1:8002/v1 --api-key dummy \
    --subset configs/pilot_subset.json
```

**Expected behavior.**
- E12-GPT5: since GPT-5 already answers ~0.53 on Vanilla RAG+ReAct (E1-GPT5 per pilot N=1 logs), the mind-map's ceiling contribution is capped — expect ±3 pp. Small models should benefit more.
- E12-gpt-oss-20B vs E1-gpt-oss-20B: the interesting cell. If mind-map lifts by ≥ 5 pp, the structured scratchpad concept is reproduced at small scale. If it drops, small models aren't maintaining the map well.

**Failure modes.**
- `empty_mind_map` — agent never calls `mind_map_update`. Check `traj.extra.mind_map_size == 0` on wrong-answer Qs.
- `update_then_ignore` — agent updates the map but never queries it back. Correlate `tool_calls.count('mind_map_query')` with correctness.
- `noisy_triples` — agent adds low-quality generic triples that don't chain. Manual audit required.

**Artifacts captured.** `traj.extra.mind_map` = full `[subject, relation, object]` list at stop time; `traj.extra.mind_map_size` is the total.

---

### E13. GAM — General Agentic Memory

**Code**: `src/research_agent/adapters/gam.py`
**System id**: `gam`
**Source**: *General Agentic Memory Via Deep Research*, [`arxiv 2511.18423`](https://arxiv.org/abs/2511.18423) · [VectorSpaceLab/general-agentic-memory](https://github.com/VectorSpaceLab/general-agentic-memory).

**What it tests.** GAM is the paper that explicitly names the **"JIT compilation"** paradigm — the thing our pivot is about. Two-phase design: (1) **Memorizer** does an offline lossless-storage pass with only lightweight hints built on top (title + short digest per chunk); (2) **Researcher** is an online multi-turn agent that uses hints to target the raw page-store only when it needs to fetch. Storage stays verbatim; compression happens at query time, controlled by the agent.

The substitution we make: GAM's paper evaluates on memory-grounded tasks (long conversational memory), not multi-hop QA. E13 tests whether the same paradigm holds up on FRAMES — which itself is a JIT-compilation regime (2–15 Wikipedia articles per question, most of the corpus irrelevant).

**Tool surface.**
- `browse_hints(query, top_k=10)` — cheap: returns `[{chunk_id, title, digest}]` rows. No reranking, no chunk-body access. Hints were built once at adapter init (one per chunk, title + first 160 chars).
- `fetch_page(chunk_id)` — expensive: pointer deref into the raw page-store, returns the full article text (up to 12k chars) for a chunk's article.

Agent is instructed to browse freely (cheap) and fetch selectively (expensive). The `browses_count` and `fetched_chunk_ids` metrics in `traj.extra` capture the browse/fetch ratio — GAM's headline claim is that a well-behaved Researcher browses 2–5× more than it fetches.

**Run commands.**
```bash
# E13-GPT5 (paper uses frontier for the Researcher):
PYTHONPATH=src python playground/run_substitution.py \
    --system gam --model gpt-5 \
    --subset configs/pilot_subset.json

# E13-gpt-oss-20B:
PYTHONPATH=src python playground/run_substitution.py \
    --system gam --model openai/gpt-oss-20b \
    --base-url http://127.0.0.1:8001/v1 --api-key dummy \
    --subset configs/pilot_subset.json

# E13-Qwen3.5-9B:
PYTHONPATH=src python playground/run_substitution.py \
    --system gam --model Qwen/Qwen3.5-9B \
    --base-url http://127.0.0.1:8002/v1 --api-key dummy \
    --subset configs/pilot_subset.json
```

**Memorizer cost at init.** The adapter builds ~2k hints once per `AdapterContext` instantiation (no LLM calls — just chunk iteration + string truncation). Cheap. If the harness re-instantiates per question, pre-compute with `extra={"hints": prebuilt_dict}` once and reuse across cells.

**Expected behavior.**
- E13 vs E1 at same base: E13 has a title-digest index the agent can scan before retrieving. Not a huge leg-up since BM25 already considers title + full text — the interesting metric is *browse/fetch ratio* rather than accuracy. Expect similar accuracy to E1 (±2 pp) but with a visible hint-browsing pattern in tool_calls.
- E13-GPT5: if GPT-5 browses freely and fetches selectively (3:1+), the JIT paradigm reproduces. If it browses once and fetches everything, Model doesn't internalize the cost asymmetry.

**Failure modes.**
- `fetch_storm` — agent fetches every hint returned. browse/fetch ratio ≤ 1. Mis-internalized the cost asymmetry.
- `browse_only` — agent browses N times and never fetches, answers from digests alone. Check `traj.extra.fetched_chunk_ids == []` on wrong-answer Qs.
- `bad_chunk_id` — agent invents a chunk_id not in the store. Caught by the `error` field on `fetch_page` tool calls.

**Artifacts captured.** `traj.extra.browses` (count), `traj.extra.fetched_chunk_ids` (list), `traj.extra.hint_store_size` (one-time sanity — should be constant across Qs in a run).

---

## Scoring & evaluation

All cells use the same scoring contract from `src/research_agent/eval/scoring.py`:

1. **Canonicalization** before any match: lowercase + strip articles (a/an/the) + remove punctuation + collapse whitespace.
2. **Exact match** (`exact_match`): canonicalized pred == canonicalized gold.
3. **Token F1** (`token_f1`): word-level F1 on canonicalized tokens.
4. **(Optional, not enabled in pilot)** LLM-as-judge tiebreaker on partial credit — wire up later if needed.

**Headline metric for FRAMES pilot**: `accuracy` = exact_match rate. `mean_f1` is reported alongside for questions where EM = 0 but F1 > 0 (answer contained gold but also had extra words).

**Pass@4 / Avg@4 not run in pilot.** The published numbers for ASearcher use 4 independent rollouts per Q; our pilot runs a single rollout (N_runs = 1). If the initial grid is worth scaling, we'll revisit with `--n-rollouts 4` added to the runner.

**Alias handling.** The FRAMES dataset doesn't ship a canonical alias list per question. Our harness uses exact-match-on-canonicalized with token F1 as the partial-credit signal. This is slightly stricter than the original paper's eval (which uses LLM-as-judge + alias matching). Expect our numbers to be 2–5 pp lower than the papers' numbers even for reproductions — the eval is tighter.

---

## Trace format & audit

Every cell run produces a directory under `logs/`:

```
logs/vanilla_rag_react__gpt-5__20260417_152345/
├── cell_result.json                 # CellResult (headline metrics + per-Q summary)
└── traces/
    ├── q_793.json                   # Trajectory JSON per question
    ├── q_510.json
    └── ...
```

**`cell_result.json` shape** (from `src/research_agent/models/trace.py::CellResult`):
- `system_id`, `model_id`, `benchmark`, `subset_id`.
- `n_total`, `n_correct`, `accuracy`, `mean_f1`.
- `mean_turns`, `mean_tool_calls`, `mean_prompt_tokens`, `mean_completion_tokens`, `mean_wall_time_s`.
- `failure_histogram`: `{FailureMode.value: count}`.
- `questions`: list of `QuestionResult` — one per question, scored.
- `config`: snapshot of `AdapterContext.max_turns / max_*_tokens / extra`.

**Per-question `Trajectory` shape**:
- `final_answer`, `reasoning`.
- `turns`, `tool_calls` (list of `ToolCall`), `prompt_tokens`, `completion_tokens`, `wall_time_s`.
- `messages`: full OpenAI message list (optional; adapters can skip).
- `extra`: adapter-specific state — `gold_articles`, `stopped_reason`, adapter-specific dumps like `gsw_snapshot`, `reason_in_docs_distilled`, `kept_chunk_ids`, `final_state` (Q+ frontier), etc.

**Failure-mode tags** (from `src/research_agent/models/failure_modes.py`):

| Tag | Meaning | Heuristic |
|---|---|---|
| `correct` | exact_match == True | deterministic |
| `hallucination` | wrong + zero retrieval calls + confident answer | no retrieval tools used |
| `wrong_retrieval` | wrong + retrieved, but gold article never appeared | gold titles not in any tool result |
| `incomplete_decomposition` | fewer sub-Qs than hops | adapter-dumped `n_sub_questions` < `num_hops` |
| `early_stop` | turns < num_hops × MIN_EXPECTED_TURNS_PER_HOP | hop heuristic |
| `loop` | ≥ 3 identical (name, args) tool calls | deterministic |
| `tool_error` | tool returned error + no recovery + no answer | deterministic |
| `budget_exceeded` | stopped_reason in {max_turns, max_tokens, budget_exceeded} | deterministic |
| `wrong_synthesis` | none of the above but wrong | fallback bucket |
| `unknown` | empty trajectory / adapter crash | deterministic |

**Compare-traces CLI** (deferred enhancement): `playground/compare_traces.py` — stub for a side-by-side diff between two cells' traces on the same question. Useful when auditing "why does GPT-5 get this right but gpt-oss-20B get it wrong." Not yet written; add in Phase 2 if the grid itself isn't enough.

---

## Aggregation — building the grid table

After all (or any subset of) the 26 cells have run:

```bash
cd /home/yigit/codebase/gsw-memory/research_agent && source .venv/bin/activate
PYTHONPATH=src python playground/aggregate_grid.py \
    --logs-dir logs \
    --out logs/grid_summary.md \
    --csv logs/grid_summary.csv
```

Outputs:
- `logs/grid_summary.md` — two tables:
  1. Headline metrics (system × model → accuracy / F1 / turns / tokens / wall).
  2. Failure-mode histogram per cell.
  Plus a "quick observations" footer (best / worst cells + accuracy spread).
- `logs/grid_summary.csv` — tidy long-format for further slicing.
- `logs/grid_summary.index.json` — one entry per cell with its log path.

For Obsidian reading: `logs/grid_summary.md` uses standard GH-flavored markdown, no Mermaid or custom extensions.

---

## Go/no-go criteria for scaling up

After the pilot grid is complete, the question becomes: do we scale to the full FRAMES 824 + add training (Tier 3 v2 SFT, Tier 3 v3 GRPO)?

**Proceed to full-FRAMES evaluation** (Phase 2, ~2 days additional wall time) if **any one** of the following holds:

1. **Ours v1 beats E1-gpt-oss-20B by ≥ 5 pp** → the scratchpad is pulling weight even zero-shot.
2. **At least one trained competitor ckpt cleanly degrades under prompt-mode swap** (e.g. E5-ASearcher-7B trained scores > 50, but E4-ASearcher-prompt-on-gpt-oss-20B < 30) → the training is the load-bearing axis, confirming the substitution setup is sensitive.
3. **Q+ or AFM prompt adds ≥ 5 pp over E1 at matched base** → structured process helps at small scale, worth exploring more.

**Abort / rethink** if:
- Ours ties or loses to E1 at same base → the scratchpad isn't helping; re-design the pipeline (probably move from flat triples to verb-phrase-centric structure — mirroring the linker v3 output in the gsw-memory sibling repo).
- All cells at gpt-oss-20B cluster within 5 pp → small-model downshift is a floor and we learn nothing from the variance; need to pick a stronger base (Qwen2.5-14B or QwQ-32B) as the "small" anchor.

**Proceed to training (Tier 3 v2 SFT + Tier 3 v3 GRPO)** only if the zero-shot ours wins cleanly. Training is ~3 weeks of work; no point investing if v1 isn't already competitive.

**Phase B — MoNaCo.** Only after FRAMES is published-quality. MoNaCo corpus ingest + eval port + rerun takes an additional ~2 weeks. See `Run N+10` in the plan file for the specific order.
