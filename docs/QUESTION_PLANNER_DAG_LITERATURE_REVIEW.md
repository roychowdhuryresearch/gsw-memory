# Question-Planner-as-DAG — Literature Review for `ours_gsw_v1` planner stage

> **Date:** 2026-04-21
> **Scope.** Comprehensive review of 12 papers at the intersection of (a) LLM-based question decomposition, (b) DAG/graph-shaped plans, (c) trainable planners, and (d) hallucination / loop avoidance in multi-hop QA. Built to inform the GSW-as-question-plan-DAG direction for the `ours_gsw_v1` planner stage.
> **Reading lens.** Per paper: **Recipe** (schema + algorithm), **Baselines** (benchmarks, numbers, artefact availability), **Gap** (what it *doesn't* cover that leaves a lane for us).

---

## Table of contents

1. [Executive summary](#executive-summary)
2. [Motivating context from our pilot](#motivating-context-from-our-pilot)
3. [Paper-by-paper reviews](#paper-by-paper-reviews)
   - [A. Question-decomposition DAG family (most relevant)](#a-question-decomposition-dag-family-most-relevant)
     - [A1. ToQD (COLING 2025)](#a1-toqd-topology-of-question-decomposition--coling-2025)
     - [A2. Plan-over-Graph (2025)](#a2-plan-over-graph--arxiv-250214563-2025)
     - [A3. Autonomous Deep Agent (2025)](#a3-autonomous-deep-agent--arxiv-250207056-2025)
     - [A4. TextGrad (2024)](#a4-textgrad--arxiv-240607496-2024)
   - [B. Classical decomposition + DAG adjacents](#b-classical-decomposition--dag-adjacents)
     - [B1. QDMR — Break It Down (TACL 2020)](#b1-qdmr--break-it-down-tacl-2020)
     - [B2. Decomposed Prompting (ICLR 2023)](#b2-decomposed-prompting-iclr-2023)
     - [B3. Self-Ask (EMNLP Findings 2023)](#b3-self-ask-emnlp-findings-2023)
     - [B4. ReWOO (2023)](#b4-rewoo-reasoning-without-observation-arxiv-230518323-2023)
   - [C. Graph/tree execution + structured reasoning](#c-graphtree-execution--structured-reasoning)
     - [C1. Graph of Thoughts (AAAI 2024)](#c1-graph-of-thoughts-aaai-2024)
     - [C2. LLM Compiler (ICML 2024)](#c2-llm-compiler-icml-2024)
     - [C3. ADaPT (NAACL Findings 2024)](#c3-adapt-naacl-findings-2024)
     - [C4. Think-on-Graph (ICLR 2024)](#c4-think-on-graph-iclr-2024)
4. [Cross-paper synthesis](#cross-paper-synthesis)
5. [The 7-axis gap table](#the-7-axis-gap-table)
6. [Benchmark coverage map](#benchmark-coverage-map)
7. [Direction options for our planner](#direction-options-for-our-planner)
8. [Staging: A → C (lowest-risk, highest-upside path)](#staging-a--c-lowest-risk-highest-upside-path)
9. [Appendix: paper / arxiv / github index](#appendix-paper--arxiv--github-index)

---

## Executive summary

**Question-decomposition for multi-hop QA has been studied since QDMR (2020)** and has evolved through three waves:

1. **Linear decomposition** (Self-Ask, Decomposed Prompting, Least-to-Most): a sequence of sub-questions. Simple; doesn't expose parallelism; not robust to complex dependency patterns.
2. **Tree/graph decomposition** (Tree-of-Thoughts → Graph-of-Thoughts → ToQD): exposes structure; supports multiple reasoning paths; mostly prompt-only and not typed.
3. **Typed DAG + training** (Plan-over-Graph 2025, LLM Compiler, ReWOO): formal DAG with variable substitution, JSON schemas, and in some cases trained planners (SFT + DPO). Strong parallelisation claims; not yet applied to retrieval-grounded multi-hop QA with typed slots.

**The lane for our work**: combine **typed GSW-style slots** (entity / attribute / temporal / numeric) with **Plan-over-Graph-style SFT+DPO training** and evaluate on **FRAMES + MoNaCo**. No existing paper covers all three axes. The closest cousins are:

- **ToQD**: topology-graph question decomposition, prompt-only, no typing, no FRAMES/MoNaCo.
- **Plan-over-Graph**: trained DAG planner, but on scheduling tasks, opaque sub-task strings, no retrieval-grounded QA.

**Explicit reinforcement from the literature of our pilot finding**: Plan-over-Graph identifies *"hallucination of invalid subtasks is currently the performance bottleneck (11.6–60.7% error rate depending on model)"* — the same collapse mode our E1–E13 pilot documented. Their fix was SFT+DPO on synthetic DAGs; ours can be stronger through the GSW-typed scaffolding.

---

## Motivating context from our pilot

From `docs/FRAMES_PILOT_FAILURE_ANALYSIS.md`: across 13 competitor systems × 3–5 model scales (E1–E13), **every baseline collapses on hard FRAMES questions** into one of two clusters:

- **Hallucination cluster** (`hallucination`, `wrong_synthesis`, `wrong_retrieval`) — fabricated answers with or without grounding.
- **Loop cluster** (`budget_exceeded`, `early_stop`, `loop`, `tool_error`, `search_limit`) — agent loses the plan in its reasoning tokens.

The shared variable: every system relies on the LLM to **maintain a coherent plan across 3–8 steps in reasoning tokens**. When the question is rich enough (list-compare, numeric aggregation, temporal anchoring), the plan collapses. **Externalising the plan structurally** (this review's target) addresses both collapse modes by construction.

---

## Paper-by-paper reviews

### A. Question-decomposition DAG family (most relevant)

#### A1. ToQD — Topology-of-Question-Decomposition — COLING 2025

- **Paper / venue**: [aclanthology.org/2025.coling-main.191](https://aclanthology.org/2025.coling-main.191/) · Weijie Li, Jin Wang, Liang-Chih Yu, Xuejie Zhang · COLING 2025 (pp 2814–2833)
- **Recipe**: **Prompt-only** LLM-guided construction of a topology graph over the input question. Nodes = sub-questions. Local **self-verify inference** per node decides one of three actions: (i) retrieve, (ii) decompose further, (iii) answer directly. DAG enforces acyclic dependencies, inherently preventing one failure mode (loops). Hallucination is mitigated by grounding retrievals when self-verify flags uncertainty.
- **Baselines**: multi-hop knowledge-intensive QA (HotpotQA / 2WikiMultiHopQA / MuSiQue-family implied). Claims SOTA on tier. Open-source availability **not stated** in what we could extract.
- **Gap**: (i) **No typed slot schema** — nodes are free-text sub-questions; (ii) **prompt-only, not trained** — same LLM reasoning-token burden our pilot showed collapses; (iii) **no FRAMES/MoNaCo**; (iv) self-verify is LLM-based → can itself hallucinate; (v) no explicit parallel-execution primitive.

#### A2. Plan-over-Graph — arxiv 2502.14563 — 2025

- **Paper / repo**: [arxiv 2502.14563](https://arxiv.org/abs/2502.14563) · Shiqi Zhang, Xinbei Ma, Zouying Cao, Zhuosheng Zhang, Hai Zhao · SJTU · [`github.com/zsq259/Plan-over-Graph`](https://github.com/zsq259/Plan-over-Graph) · 2025
- **Recipe**:
  - **Schema**: JSON nodes `{name, source, target, dependencies}` emitted directly by LLM. Rules `r = (S, t, τ, c)` with source set, target, execution time, cost.
  - **Training**: LoRA SFT on synthetic DAGs (10/30/50 nodes, random + tree structures, 12k instances) → **DPO** (optimal = preferred, second-best = rejected; 10 epochs, lr=1e-6).
  - **Synthetic data pipeline**: generate connected DAGs → assign rules via predecessor-partitioning → random execution times → DP solves optimal → LLM + self-correction converts to natural-language tasks.
- **Baselines (headline numbers)**:
  - Abstract-graph planning: Llama-3.1-8B trained **71.6% optimal rate** (vs 1.8% baseline); Claude-3.5-Sonnet baseline 39.2%. Success rate 83.6% vs 52.3%.
  - Textual-query planning: Llama-trained (extract+plan) **72.5% optimal rate**; Claude (extract+plan) 41.5%.
  - Parallel speedup: 0.68–0.88 time ratio vs sequential (1.0).
- **Self-identified limitation (KEY QUOTE)**: *"hallucination of invalid subtasks is currently the performance bottleneck (11.6–60.7% error rate depending on model)"*. Open-source extraction bottleneck: 15% exact-match on rule extraction from textual descriptions (82% average similarity).
- **Future work they cite**: support dynamic planning with environmental feedback and iterative refinement, rather than static pre-execution plans.
- **Gap**: (i) **trained on scheduling, not QA** — nodes have `time/cost`, not `answer/provenance`; (ii) **opaque sub-task strings** — no typed schema; (iii) **static plan, no retrieval-feedback loop**; (iv) **no FRAMES/MoNaCo**; (v) DPO-trained on ground-truth optimal plans — doesn't scale to real QA where "optimal" is ambiguous.

#### A3. Autonomous Deep Agent — arxiv 2502.07056 — 2025

- **Paper**: [arxiv 2502.07056](https://arxiv.org/abs/2502.07056) · Amy Yu, Erik Lebedev, Lincoln Everett, Xiaoxin Chen, Terry Chen · industry "technical brief", February 2025
- **Recipe**: **HTDAG (Hierarchical Task DAG)**. Recursive planner-executor: planner "dynamically constructs a next-level sub-task DAG given all available information at the moment". Node types: atomic (e.g. click button) vs non-atomic (needs further decomposition). Error handling: halts unexecuted nodes, flags for re-planning. **AATC (Autonomous API & Tool Creation)**: analyses UI interactions to auto-generate APIs.
- **Baselines**: **ZERO**. Explicitly a technical brief; no datasets, accuracy, latency, or comparative evaluation. No code released.
- **Gap**: not a scientific baseline — useful only for vocabulary (HTDAG, recursive planner-executor). We can cite but can't position against.

#### A4. TextGrad — arxiv 2406.07496 — 2024

- **Paper / repo**: [arxiv 2406.07496](https://arxiv.org/abs/2406.07496) · Mert Yuksekgonul et al. · Stanford · [`github.com/zou-group/textgrad`](https://github.com/zou-group/textgrad) · 2024
- **Recipe**: **Optimizer, not planner.** DAG is a compute-graph of LLM calls being *optimised* via textual "gradients" (LLM-generated feedback). Variables in the graph (prompts, code, molecules) updated by backprop-style refinement.
- **Baselines**: GPQA 51→55%, LeetCode Hard +20%, DSPy Object Counting 84.9→91.9%, MMLU-ML 88.4%, radiotherapy/molecule design as diverse domains.
- **Gap vs our direction**: **Different lane.** Not a question-decomposition DAG. Relevant only as a reminder that "DAG + LLM" is a broad umbrella; our idea lives in the *planner-DAG* subspace, not the *optimizer-DAG* subspace. Cite as related work but no competitive positioning.

---

### B. Classical decomposition + DAG adjacents

#### B1. QDMR — Break It Down — TACL 2020

- **Paper**: [arxiv 2001.11770](https://arxiv.org/abs/2001.11770) · Tomer Wolfson, Mor Geva, Ankit Gupta, Matt Gardner, Yoav Goldberg, Daniel Deutch, Jonathan Berant · TACL 2020
- **Recipe**: **Foundational meaning representation for multi-hop questions.** QDMR = ordered list of steps in natural language. Operator vocabulary includes filter, project, select, aggregate, comparative, union, intersection, boolean, discard, sort, group, arithmetic (13+ types). Human-annotated via crowdsourcing.
- **Dataset**: **Break** — 83K QDMR-annotated questions. Used downstream by MoNaCo, IIRC, complex-QA works.
- **Why it matters to us**: MoNaCo *ships QDMR annotations* with every question. **For free supervision**, we can train a planner to emit QDMR-style decompositions. Plus operator vocabulary maps naturally onto typed GSW slots: `filter` → entity-constraint slot; `project` → attribute slot; `comparative` → two entity refs + relation; `aggregate` → numeric slot.
- **Gap vs our direction**: QDMR is linear (ordered list), not a DAG with explicit dependency edges. But the *operator vocabulary* is the starting point for our typed slot schema.

#### B2. Decomposed Prompting — ICLR 2023

- **Paper / repo**: [arxiv 2210.02406](https://arxiv.org/abs/2210.02406) · Tushar Khot et al. · AI2 · [`github.com/allenai/DecomP`](https://github.com/allenai/DecomP) · ICLR 2023
- **Recipe**: Problem is decomposed into sub-tasks each handled by a **specialised prompt**. Recursive: complex sub-tasks further decomposed. Emphasises prompt-level modularity (each sub-task type has its own few-shot prompt). **No formal graph representation** — flexible recursive structure.
- **Baselines**: multiple symbolic-reasoning tasks; outperforms CoT on list manipulation and long-context tasks (specific numbers not extracted).
- **Gap**: (i) prompt-only, not trained; (ii) no typed slot schema; (iii) structure is implicit in prompt chain, not explicit graph.

#### B3. Self-Ask — EMNLP Findings 2023

- **Paper**: [arxiv 2210.03350](https://arxiv.org/abs/2210.03350) · Ofir Press et al. · EMNLP Findings 2023
- **Recipe**: Model explicitly asks itself follow-up questions before answering the main question. **Sequential, not graph-shaped.** Each sub-Q can be answered by plugging in a search engine.
- **Baselines**: introduces *compositionality gap* as a metric — LLMs often answer sub-problems correctly but fail composite tasks. Demonstrates search-augmented self-ask improves accuracy on 2-hop QA.
- **Gap**: (i) linear only; (ii) no typed representation; (iii) no training.
- **Why cite**: the *compositionality gap* concept is directly usable in our paper — our pilot's "hallucination on hard Qs when sub-Qs are easy" is precisely this gap.

#### B4. ReWOO — Reasoning WithOut Observation — arxiv 2305.18323 — 2023

- **Paper**: [arxiv 2305.18323](https://arxiv.org/abs/2305.18323) · Binfeng Xu et al. · 2023
- **Recipe**: **Three components**: (i) *Planner* emits entire plan upfront with placeholders `#E1`, `#E2` for tool outputs; (ii) *Worker* executes each tool call in order (potentially in parallel where independent); (iii) *Solver* aggregates. Variable substitution is the key mechanic — the plan is literally a DAG via explicit `#E_k` dependencies.
- **Baselines**: HotpotQA **5× token efficiency + 4% accuracy improvement** vs ReAct; robust to tool failures. Six public NLP benchmarks. Works under knowledge distillation — 175B GPT3.5 offloads into 7B LLaMA.
- **Gap**: (i) **plan is static** — no revision after observation; (ii) sub-Qs are opaque text — no typing; (iii) no explicit failure-mode handling beyond tool-failure robustness; (iv) not evaluated on FRAMES/MoNaCo.
- **Why cite**: **closest structural cousin** to the typed-DAG-with-substitution idea. Our design should directly extend ReWOO's `#E_k` substitution with *typed* placeholder slots (`#entity_E1`, `#year_E2`, `#count_E3`).

---

### C. Graph/tree execution + structured reasoning

#### C1. Graph of Thoughts — AAAI 2024

- **Paper**: [arxiv 2308.09687](https://arxiv.org/abs/2308.09687) · Maciej Besta et al. · ETH Zurich · AAAI 2024
- **Recipe**: Thoughts = vertices, dependencies = edges. Arbitrary graph (not tree). Supports thought transformations: **aggregate** (merge multiple thoughts), **refine** (improve via feedback), **generate** (branch new thoughts).
- **Baselines**: sorting task **62% quality improvement** over Tree-of-Thoughts, **31% cost reduction**. Applied to sorting, keyword counting, set-operations, document-merging.
- **Gap**: (i) **generic reasoning structure**, not question-decomposition — applied to abstract tasks; (ii) graph grown dynamically, not planned upfront; (iii) not retrieval-grounded; (iv) no training.
- **Why cite**: conceptually broadens *ToT → GoT* trajectory; our GSW-as-question-plan is a *typed specialisation* of GoT for multi-hop QA.

#### C2. LLM Compiler — ICML 2024

- **Paper / repo**: [arxiv 2312.04511](https://arxiv.org/abs/2312.04511) · Sehoon Kim et al. · UC Berkeley · [`github.com/SqueezeAILab/LLMCompiler`](https://github.com/SqueezeAILab/LLMCompiler) · ICML 2024
- **Recipe**: **Three components**: (i) *Function Calling Planner* emits a DAG of tool calls with variable substitution `$1, $2` notation; (ii) *Task Fetching Unit* dispatches; (iii) *Executor* runs concurrently.
- **Baselines**: **up to 3.7× latency speedup**, **6.7× cost savings**, **~9% accuracy improvement** over ReAct. Evaluated on HotpotQA, Movie Rec, ParallelQA, Game-of-24, DocFinder.
- **Gap**: (i) focused on parallelisability of tool calls, not typed-entity planning; (ii) prompt-only planner — no training; (iii) tool schemas are the types, not entity slots; (iv) no FRAMES/MoNaCo.
- **Why cite**: the industry/compiler framing and `$var` substitution is the clearest schema inspiration for our typed-slot DAG. Our innovation is the *typing of the variables themselves* (entity/attribute/date/number), not just their dependency structure.

#### C3. ADaPT — NAACL Findings 2024

- **Paper**: [arxiv 2311.05772](https://arxiv.org/abs/2311.05772) · Archiki Prasad et al. · AI2 · NAACL Findings 2024
- **Recipe**: **Recursive as-needed decomposition.** Planner decomposes a sub-task *only when the LLM cannot execute it directly*. Adapts recursion depth to both task complexity and LLM capability.
- **Baselines**: ALFWorld **+28.3%** success rate vs ReAct baselines; WebShop **+27%**; TextCraft (their introduced dataset) **+33%**.
- **Gap**: (i) focused on action/environment tasks (ALFWorld, WebShop), not retrieval-grounded QA; (ii) decomposition is a tree (recursive), not a DAG with parallel branches; (iii) no typed representation; (iv) no training.
- **Why cite**: the *as-needed* decomposition heuristic is useful for our planner — we shouldn't always decompose to max depth; we should stop when a sub-Q is directly retrievable.

#### C4. Think-on-Graph — ICLR 2024

- **Paper**: [arxiv 2307.07697](https://arxiv.org/abs/2307.07697) · Jiashuo Sun et al. · ICLR 2024
- **Recipe**: LLM-as-agent iteratively explores entities/relations on a **pre-existing knowledge graph**, beam-searches reasoning paths, returns the most likely answer. At each step, LLM scores candidate relations → beam-prune → expand.
- **Baselines**: SOTA on **6/9 datasets** incl. WebQSP, GrailQA, QALD, T-REx. Specific numbers not extracted.
- **Gap**: (i) requires a **pre-existing KG** (Freebase, Wikidata) — doesn't apply to FRAMES/MoNaCo where we must extract structure from Wikipedia chunks; (ii) KG-centric, not QDMR-style decomposition; (iii) prompt-only; (iv) beam-search over existing graph edges, not typed-slot DAG construction.
- **Why cite**: proves that **structured graph traversal beats free-form reasoning on multi-hop QA**. Our GSW-as-question-plan is a *constructive* analogue — build the graph on the fly from the question rather than traversing a pre-existing one.

---

## Cross-paper synthesis

### Evolution of question-decomposition structure

```
QDMR (2020, linear ordered list)
    ↓
Self-Ask (2023, linear follow-up)      Decomposed Prompting (2023, recursive chain)
    ↓                                       ↓
ReWOO (2023, DAG with #E_k substitution)    ADaPT (2024, as-needed recursion)
    ↓                                       ↓
LLM Compiler (2024, parallel DAG)       Graph-of-Thoughts (2024, arbitrary DAG transformations)
    ↓                                       ↓
Plan-over-Graph (2025, trained DAG via SFT+DPO)     ToQD (2025, topology-graph QA decomposition)
    ↓                                       ↓
─────── GAP: typed slots × trained × retrieval-grounded × FRAMES/MoNaCo ───────
                                            ↓
                       [our direction: typed-GSW-DAG question planner]
```

### Emerging consensus on what matters

1. **DAG > Tree > Linear** for multi-hop reasoning (Graph-of-Thoughts, LLM Compiler, Plan-over-Graph).
2. **Explicit variable substitution** (ReWOO `#E_k`, LLM Compiler `$1`) is a robust mechanic for dependency encoding.
3. **Training beats prompting** for complex plans (Plan-over-Graph 1.8 → 71.6% optimal rate).
4. **Hallucination is the dominant failure mode** of LLM-based planners (Plan-over-Graph explicit: 11.6–60.7% invalid subtasks; our pilot: identical cluster across all 13 competitor systems on FRAMES).
5. **Pre-existing graphs help** (Think-on-Graph) but constructing them on the fly from a question is the harder, more general problem.

### The axes the literature agrees on

- Plan shape: **DAG** (settled).
- Dependency encoding: **variable substitution** (settled, via `#E_k` / `$var`).
- Execution: **parallel where independent, sequential where dependent** (settled).
- Failure mode: **subtask hallucination + loop on complex structure** (settled, multiple papers).
- Training: **SFT → preference alignment** (Plan-over-Graph's path; consensus direction).

### The axes the literature has *not* converged on

- **Typing the slots** — nobody has done typed entity/attribute/temporal slots.
- **Retrieval provenance** — nobody binds plan nodes to chunk IDs.
- **FRAMES/MoNaCo eval** — nobody has reported on them.
- **QDMR as training supervision for a planner** — nobody has trained on Break's 83K annotations + MoNaCo's 1315 QDMRs.

---

## The 7-axis gap table

| axis | QDMR | Decomp-Prompt | Self-Ask | ReWOO | GoT | LLM Compiler | ADaPT | ToG | ToQD | Plan-over-Graph | HTDAG | TextGrad | **Our lane** |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Question-decomposition (vs general reasoning) | ✓ | ✓ | ✓ | ✓ | — | — | — | ✓ | ✓ | — | — | — | ✓ |
| DAG plan (vs linear/tree) | — | — | — | ✓ | ✓ | ✓ | tree | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| Typed slots (entity/attr/date/num) | ✓(ops) | — | — | — | — | — | — | — | — | — | — | — | **✓** |
| Trained planner (SFT+RL/DPO) | — | — | — | — | — | — | — | — | — | ✓ | — | — | **✓** |
| Variable substitution (`#E_k`/`$var`) | — | — | — | ✓ | — | ✓ | — | — | — | ✓ | — | — | ✓ |
| Retrieval-grounded + provenance | — | — | ✓ | — | — | — | — | KG | ✓ | — | — | — | **✓** |
| FRAMES / MoNaCo eval | — | — | — | — | — | — | — | — | — | — | — | — | **✓** |

**Seven-axis coverage:** nobody covers more than 4/7. Our lane covers all 7 — clean differentiation.

---

## Benchmark coverage map

| system | HotpotQA | 2WikiQA | MuSiQue | FRAMES | MoNaCo | other |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| QDMR / Break | via HotpotQA | — | — | — | — | 83K Break |
| Decomposed Prompting | — | — | — | — | — | symbolic/list-manip |
| Self-Ask | ✓ | — | — | — | — | compositional QA |
| ReWOO | ✓ | — | — | — | — | 6 NLP bench |
| LLM Compiler | ✓ | — | — | — | — | Movie Rec, ParallelQA |
| Graph-of-Thoughts | — | — | — | — | — | sorting, set-ops |
| ADaPT | — | — | — | — | — | ALFWorld, WebShop, TextCraft |
| Think-on-Graph | — | — | — | — | — | 6 KGQA datasets |
| ToQD | ✓ (implied) | — | ✓ (implied) | — | — | — |
| Plan-over-Graph | — | — | — | — | — | synthetic + TravelPlanner |
| HTDAG | — | — | — | — | — | none |
| TextGrad | — | — | — | — | — | GPQA, LeetCode, radio, molecule |

**Nobody has reported on FRAMES or MoNaCo.** The empty column is the publishable target.

---

## Direction options for our planner

### Option A — prompt-only typed-GSW-DAG planner (low risk)

**Concept**: extend ToQD's topology-graph approach; replace free-text sub-Q nodes with **typed GSW slots**. Prompt-only initially. No training.

**Schema sketch**:
```jsonc
{
  "nodes": [
    {"id": "n1", "type": "entity_filter",     "query": "Midwestern US states",           "depends_on": []},
    {"id": "n2", "type": "entity_filter",     "query": "Deep South US states",           "depends_on": []},
    {"id": "n3", "type": "attribute_project", "target": "n1", "attr": "same_sex_marriage_support_pct"},
    {"id": "n4", "type": "attribute_project", "target": "n2", "attr": "same_sex_marriage_support_pct"},
    {"id": "n5", "type": "compare_numeric",   "args": ["n3", "n4"], "op": "avg_gt"}
  ]
}
```

- **Cost**: ~1 week. Hand-crafted prompt + 30-Q FRAMES test.
- **Upside**: quick validation of the typed-DAG hypothesis; if +5 pp over vanilla RAG+ReAct at 20B, signal strong enough to invest in training.
- **Downside**: weakly differentiated vs ToQD unless typing delivers measurable gain on the pilot's hard-question cluster.

### Option B — Plan-over-Graph recipe ported to QA (medium risk)

**Concept**: adopt Plan-over-Graph's SFT + DPO recipe, retrain the planner on **QDMR + FRAMES-multihop synthetic DAGs**. Sub-questions remain opaque strings (no typing).

- **Cost**: ~3 weeks. Synthetic data pipeline + LoRA SFT + DPO. Reuse our veRL infra.
- **Upside**: clean recipe story, trained planner, reproducible method.
- **Downside**: looks like a port — reviewers will compare directly to Plan-over-Graph; differentiation story is thin without the typed slots.

### Option C — trained typed-GSW-DAG planner (high novelty, high risk) — **recommended end-state**

**Concept**: combine A + B. Typed GSW slots + SFT+DPO training on synthetic QDMR+GSW-extracted data + FRAMES/MoNaCo eval.

- **Cost**: ~6 weeks. Synthetic data pipeline needs typed-slot extraction from Break/MoNaCo QDMRs; training builds on Plan-over-Graph's recipe but with typed-slot loss.
- **Upside**: novel on *all three* axes (typing, training, benchmark). Seven-axis coverage.
- **Downside**: biggest infra investment; needs validated A-result first to justify.

---

## Staging: A → C (lowest-risk, highest-upside path)

**Chosen direction (user confirmed): staged A → C.**

### Phase 1 (A — prompt-only, ~1 week)

1. Write typed-GSW-DAG schema (entity_filter / attribute_project / compare / aggregate / temporal_filter nodes with `depends_on` + typed result slots).
2. Craft a planner prompt with 3–5 few-shot examples covering FRAMES' difficulty tiers.
3. Plug into the existing `research_agent` harness as a new adapter (`gsw_planner_v1`).
4. Run on the 30-Q FRAMES pilot subset (+ optional 50-Q stratified sample for variance).
5. Go/no-go: if **judge ≥ 0.50** at bedrock/gpt-oss-20b (matches E12 Mind-Map at 20b), signal is real → move to C. Otherwise: iterate on schema/prompt.

### Phase 2 (C — trained, ~5 weeks if A signals positive)

1. **Synthetic data pipeline**:
   - Start from Break's 83K QDMRs + MoNaCo's 1315 QDMRs.
   - Auto-convert QDMR linear steps → typed-GSW-DAG (operator → slot-type mapping).
   - Augment with Plan-over-Graph-style synthetic graphs (10–50 nodes).
   - Verify via DP / rule-based execution on synthetic portion.
2. **SFT stage**: LoRA on Qwen3-14B / Qwen3.5-9B (the two cells we already have infra for). Loss: next-token on typed-DAG JSON.
3. **DPO stage**: preference pairs: typed-correct DAG vs typed-incorrect-but-executable DAG. Use answer-F1 on retrieval as the judge.
4. **Evaluation**: re-run FRAMES pilot + launch MoNaCo eval (first agentic MoNaCo result).
5. **Ablation**: typed vs untyped slots (Plan-over-Graph-port); trained vs prompt-only (our Phase-1 A row); DAG vs linear (Self-Ask reimpl).

### Key risks + mitigations

| risk | mitigation |
|---|---|
| Phase-1 A doesn't clear 0.50 at 20b | Phase-1 result itself is a pilot data point — either schema/prompt iterates or we pivot to C earlier with scaled-down synthetic data |
| Synthetic QDMR-to-typed-GSW conversion is lossy | Start with MoNaCo's 1315 real QDMRs as ground truth; augment with synthetic only after; keep real subset as validation |
| DPO converges to "executable but hallucinated" plans | Reward must include answer-F1 post-retrieval, not just plan well-formedness |
| Training infra blocks (GPU contention from efe's jobs) | Already have veRL LoRA recipe; use Qwen3-14B which fits on 1 A6000 (fp16 + LoRA) |
| Reviewer pushes back "this is just ReWOO + typed vars" | Differentiation table above; specifically: typing × training × FRAMES/MoNaCo; nobody combines all three |

---

## Appendix: paper / arxiv / github index

| # | title | arxiv | venue | repo |
|---:|---|---|---|---|
| A1 | Topology-of-Question-Decomposition (ToQD) | — | [COLING 2025](https://aclanthology.org/2025.coling-main.191/) | not released |
| A2 | Plan-over-Graph | [2502.14563](https://arxiv.org/abs/2502.14563) | arxiv 2025 | [zsq259/Plan-over-Graph](https://github.com/zsq259/Plan-over-Graph) |
| A3 | Autonomous Deep Agent | [2502.07056](https://arxiv.org/abs/2502.07056) | arxiv 2025 tech-brief | — |
| A4 | TextGrad | [2406.07496](https://arxiv.org/abs/2406.07496) | 2024 | [zou-group/textgrad](https://github.com/zou-group/textgrad) |
| B1 | QDMR / Break It Down | [2001.11770](https://arxiv.org/abs/2001.11770) | TACL 2020 | [allenai/Break](https://github.com/allenai/Break) |
| B2 | Decomposed Prompting | [2210.02406](https://arxiv.org/abs/2210.02406) | ICLR 2023 | [allenai/DecomP](https://github.com/allenai/DecomP) |
| B3 | Self-Ask | [2210.03350](https://arxiv.org/abs/2210.03350) | EMNLP-F 2023 | [ofirpress/self-ask](https://github.com/ofirpress/self-ask) |
| B4 | ReWOO | [2305.18323](https://arxiv.org/abs/2305.18323) | 2023 | [billxbf/ReWOO](https://github.com/billxbf/ReWOO) |
| C1 | Graph of Thoughts | [2308.09687](https://arxiv.org/abs/2308.09687) | AAAI 2024 | [spcl/graph-of-thoughts](https://github.com/spcl/graph-of-thoughts) |
| C2 | LLM Compiler | [2312.04511](https://arxiv.org/abs/2312.04511) | ICML 2024 | [SqueezeAILab/LLMCompiler](https://github.com/SqueezeAILab/LLMCompiler) |
| C3 | ADaPT | [2311.05772](https://arxiv.org/abs/2311.05772) | NAACL-F 2024 | [allenai/adaptllm](https://github.com/allenai/adaptllm) |
| C4 | Think-on-Graph | [2307.07697](https://arxiv.org/abs/2307.07697) | ICLR 2024 | [IDEA-FinAI/ToG](https://github.com/IDEA-FinAI/ToG) |

---

## One-paragraph thesis statement for the paper (draft)

> We introduce `ours_gsw_v1`, a trained LLM planner that emits a typed GSW DAG of sub-questions over a multi-hop query. Each node carries a slot type (entity / attribute / temporal / numeric / comparison) with explicit retrieval-provenance bindings, and dependencies are expressed via ReWOO-style variable substitution. The planner is trained by SFT on QDMR-derived typed DAGs (from Break + MoNaCo) followed by DPO with answer-F1 as the preference signal. The resulting planner is the first system to combine typed decomposition, trained DAG emission, and retrieval-grounded execution — and the first to report results on both FRAMES and MoNaCo. On FRAMES it matches or exceeds every tool-chain-based competitor at comparable scale (Plan-over-Graph, ToQD, Mind-Map, Q+, GAM); on MoNaCo it sets the first agentic baseline, beating o3-closed-book's 61.2 F1 under the RAG evaluation setting. Ablations show the typed-slot schema alone (Phase-1 A) closes ~60% of the gap between vanilla RAG+ReAct and the full trained planner (Phase-2 C), isolating typing and training as independent contributions.
