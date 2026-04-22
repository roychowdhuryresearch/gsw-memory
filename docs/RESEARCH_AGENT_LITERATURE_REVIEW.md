# Research-Agent Literature Review — FRAMES & MoNaCo (April 16, 2026)

> **Scope.** Recent (2025–2026) agentic / research-agent work evaluated against two multi-hop QA benchmarks under active consideration for the `gsw-memory` research-agent pivot: **FRAMES** (Google DeepMind) and **MoNaCo** (AI2 + UPenn + Anthropic). Emphasis is on what each system actually does at the code level, what it claims, and — crucially — **where each paper has a gap we can attack.**

---

## Executive summary

- **FRAMES is the crowded arena.** It is the de-facto 2025–2026 agentic-multi-hop-RAG leaderboard. ~10 published systems report FRAMES numbers; the open-source bar is **ASearcher-QwQ-32B at 70.9 Avg@4 / 84.0 Pass@4** (`2508.07976`, async GRPO, 128-turn rollouts, ~7.6k H800-hours of compute). Chroma's **Context-1** (20B MoE, gpt-oss-20B base, SFT + CISPO RL) posts **0.87 F1 single-run / 0.96 at 4× parallel**.
- **MoNaCo is untouched by agentic work.** All 15 baselines in the original paper are zero-shot closed-book frontier LLMs (top F1: o3 @ 61.2%, GPT-5 @ 60.1%). Zero downstream papers report MoNaCo scores as of April 2026. The authors explicitly invite agentic systems in §6.
- **No paper uses both.** Submitting numbers on both is itself a contribution.
- **No competitor uses a GSW-style structured intermediate representation.** ASearcher is pure text trajectories; Context-1 is text trajectories + prune tool; Search-R1 / Tongyi DeepResearch / WebSailor operate over tool-call traces. The closest structured-memory lineage (Mem0ᵍ, AriGraph, GraphRAG, Zep, MAGMA) uses entity-relation triples, not QA-pair networks. **GSW's thesis — pre-structured question-answer pairs with entity-event awareness — is a white-space axis.**

---

## Part 1 — The two benchmarks

### 1.1 FRAMES (Factuality, Retrieval, And Reasoning MEasurement Set)

**Paper** Krishna et al., *Fact, Fetch, and Reason: A Unified Evaluation of Retrieval-Augmented Generation* [`arxiv 2409.12941`](https://arxiv.org/abs/2409.12941). Last revised Jan 2025.
**Dataset** [huggingface.co/datasets/google/frames-benchmark](https://huggingface.co/datasets/google/frames-benchmark) — 824 multi-hop questions, 2–15 Wikipedia articles per question. Tests *factuality* + *retrieval* + *reasoning* jointly. Covers temporal disambiguation, numerical reasoning, post-processing.

**Original-paper baselines** (Gemini-Pro-1.5-0514):
| Setup | Accuracy |
|---|---|
| No retrieval (closed-book) | 40.8% |
| BM25 retrieval (4 docs) | 47.4% |
| Oracle retrieval (gold docs) | 72.9% |
| Proposed multi-step retrieval + reasoning | 66.0% |

**What FRAMES does well**
- Real Wikipedia articles (not synthetic questions).
- Deliberately multi-hop (median 4+ articles per Q).
- Mixed reasoning types: temporal, numerical, multi-constraint.
- Lightweight eval (824 Qs, evaluator is exact-match with aliases).

**Where FRAMES falls short**
- **Max ~15 articles per Q**, with most questions at 2–5 docs. On that horizon, strong retrieval + a 72B closed-book model gets you a long way — the benchmark doesn't fully stress long-horizon aggregation.
- **No decomposition supervision.** You get question + answer only. Any decomposition-style training (like ours) has to rely on distillation from a teacher or learned search trajectories.
- **Wikipedia-only.** Distributional mismatch with web / long-form real research.
- **Leaderboard saturation risk.** ASearcher-QwQ (70.9) + Context-1 (0.87 F1) are closing on the 72.9 oracle ceiling. Marginal improvements are harder to claim novel.

**Attack surfaces for our research agent**
1. **Show that a <20B model with a GSW scratchpad beats ASearcher-QwQ-32B on FRAMES** — that's the tractable scale win since ASearcher needs 7.6k H800-hours of async RL.
2. **Compute-matched comparison**: run at ASearcher's budget but with GSW-scoped sub-question generation, report FRAMES/$ or FRAMES/GPU-hr.
3. **Interpretability contribution**: with Context-1 the "why did it answer X" is a text trajectory; with GSW it's a visible entity-relation graph with source spans. That's a publishable side-benefit on FRAMES.

### 1.2 MoNaCo (More Natural and Complex)

**Paper** Wolfson, Trivedi, Geva, Goldberg, Roth, Khot, Sabharwal, Tsarfaty, *MoNaCo: More Natural and Complex Questions for Reasoning Across Dozens of Documents* [`arxiv 2508.11133`](https://arxiv.org/abs/2508.11133) — TACL 2025 (published 2026).
**Dataset** [huggingface.co/datasets/allenai/MoNaCo_Benchmark](https://huggingface.co/datasets/allenai/MoNaCo_Benchmark) — **1315 human-written questions, avg 43.3 docs/Q (median 12), 36K distinct Wikipedia pages.**
**GitHub** [github.com/tomerwolgithub/monaco](https://github.com/tomerwolgithub/monaco)
**Project page** [tomerwolgithub.github.io/monaco](https://tomerwolgithub.github.io/monaco/)

**Construction**
- 4 expert annotators decomposed each question using **QDMR** (Question Decomposition Meaning Representation).
- Each intermediate step is labeled single-answer vs list-answer.
- Follow-up Qs are derived by substituting prior answers into placeholder positions (compositional).
- Execution engine automatically converts filter/group/sort/select steps into Python operations.
- **49.8% of questions are time-independent; 34.1% change every few years; 16.1% change yearly.**

**Example QDMR chain**:
> Q: *"Do people in Midwestern US states support same-sex marriage more than in the Deep South?"*
> Step 1 — filter: "What are the Midwestern states?" → list
> Step 2 — filter: "What are the states in the Deep South?" → list
> Steps 3–6 — query/group/compare (support percentages)

**Four official evaluation settings**
1. Parametric-only — just prompt the LLM, no context or retrieval.
2. Long-context — all 43 docs stuffed in context.
3. End-to-end RAG — retrieve then read.
4. Multi-document retrieval — retrieval metric only.

**Baselines in the paper (all zero-shot closed-book)**
| Reasoning models | F1 | Non-reasoning | F1 |
|---|---|---|---|
| o3 (2025-05) | **61.18** | GPT-4o (2025-03) | 48.98 |
| GPT-5 (2025-08) | 60.11 | Deepseek-V3 | 49.47 |
| Gemini-2.5-Pro (2025-07) | 59.11 | Llama-3.1-405B | 47.67 |
| Claude-4-Opus (2025-07) | 55.03 | Llama-3-70B | 44.76 |
| o4-mini (2025-04) | 54.92 | Qwen-2.5-72B | 42.85 |
| Deepseek-R1 | 53.82 | Qwen-2-72B | 42.64 |
| Gemini-2.5-Flash (2025-07) | 52.01 | GPT-4-Turbo (2024-05) | 42.57 |
| o3-mini (2025-04) | 48.75 | | |

**Failure modes identified by the authors**
- **Low recall + hallucination** — the 61.2 ceiling is hit by *loss of information*, not reasoning failures.
- **Sharp performance collapse** as the number of intermediate answers + evidence docs grows.
- **List-recall collapse** — GPT-4o precision is 75–80% but recall tanks with list length.

**Verbatim from §6** (direct invitation for agentic systems):
> "We did not evaluate any LLM-powered 'Deep Research' systems nor did we evaluate multi-step RAG systems that iteratively retrieve, decompose and answer complex questions … it can serve as a useful testbed for evaluating the capabilities of such Deep Research systems."

**Where MoNaCo falls short**
- **No trajectory / tool-call annotations** — only the final QDMR chain is labeled, not the sequence of searches a human would make.
- **Wikipedia-only** again.
- **Expensive to evaluate** — 43 docs/Q × 1315 Qs × iterative retrieval = significant compute to run even once.
- **Gold-answer variance** for list questions is high; F1 is the official metric but several steps produce set-valued answers that penalize legitimate variation.

**Attack surfaces for our research agent**
1. **First agentic system on MoNaCo.** Any score > o3's 61.2 F1 is a publishable claim. Our baseline bar is already set.
2. **Use QDMR as SFT supervision.** Problem composer learns to emit QDMR-style decompositions; each decomposition step becomes one focused-GSW target. This is a stronger training signal than anything FRAMES offers.
3. **Target the list-recall collapse explicitly.** A structured GSW scratchpad is the right aggregation substrate for "list all X satisfying Y"-style MoNaCo queries.
4. **Time-varying sub-split.** The 16.1% yearly-changing split is a natural knowledge-freshness evaluation where agentic retrieval beats parametric memory by construction.

---

## Part 2 — Systems reporting FRAMES scores

### 2.1 ASearcher / *Beyond Ten Turns* — current open-source FRAMES SOTA

**Paper** *Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL*, [`arxiv 2508.07976`](https://arxiv.org/abs/2508.07976).
**GitHub** [github.com/inclusionAI/ASearcher](https://github.com/inclusionAI/ASearcher)
**Supporting RL framework** [github.com/inclusionAI/AReaL](https://github.com/inclusionAI/AReaL)

**Architecture**

The agent is a **pure text-trajectory ReAct loop** with two external tools + one internal compression routine. No planner, no graph, no structured intermediate memory — everything the agent "knows" at turn `t` is what it has written / retrieved into the rolling text buffer up through turn `t-1`.

- **Base models**
  - **Qwen2.5-7B / 14B** — trained from scratch with online RL (zero-shot-RL style, no SFT cold-start).
  - **QwQ-32B** — used in a **prompt-based agent mode**; the LRM already "knows how to reason," RL is not applied on top (this is why 32B gets the headline FRAMES number despite not being RL-trained by them — the 7B/14B variants test the training recipe).
- **Tool surface (exactly 3 primitives)**
  | Tool | Signature | Returns |
  |---|---|---|
  | `search` | `query: str` | top-k snippets + URLs |
  | `browse` | `url: str` | full page content |
  | `summarize` (internal) | triggered by length, not model-callable | condenses one tool result in-place when the rolling buffer nears the context cap |
- **State representation — "append-only trajectory with sliding window"**
  - Each turn appends `<think>…</think>` + one `<action>tool(args)</action>` + `<observation>result</observation>` to a flat text buffer.
  - When the buffer exceeds the model's context (25k chars for QwQ), older `<observation>` blocks are replaced by one-line summaries; `<think>` blocks are preserved verbatim (they carry the reasoning chain).
  - No structured slot for "facts discovered so far" — facts live inline inside `<think>` blocks and the model has to re-extract them mentally each turn.
- **Inference loop (one rollout)**
  ```
  question → think → (search | browse) → observation → think → … → <answer>
  ```
  No hard cap on turns at inference time (128 is the training-time rollout horizon); the agent decides when to emit `<answer>` and stop.

**Training recipe**
- **Synthetic data generation.** 14,107 seed questions expanded via two transformations:
  - **Injection** (avg 6.3× / seed) — enriches with grounded Wikipedia facts to *increase verifiability*.
  - **Fuzzing** (avg 3.2× / seed) — blurs specific details (dates → "early X", names → "the second-born daughter") to *increase search difficulty / uncertainty*.
  - Three quality filters cascaded: (i) LLM quality check → (ii) QwQ-32B closed-book difficulty filter (drop anything QwQ solves without search) → (iii) answer-uniqueness filter.
  - Final corpus: **25,624 training samples.**
- **RL algorithm:** GRPO (group-relative PPO) — `N=16` rollouts per question per update step.
- **Async rollouts** (the headline infra contribution, shipped via the **AReaL** framework):
  - A pool of **rollout workers** generates 128-turn trajectories; a separate **learner worker** consumes completed trajectories asynchronously.
  - Without async, a single 128-turn rollout would starve the learner for minutes per sample. Async decouples them so the learner sees a steady stream.
- **Reward**
  - Base LLMs (7B / 14B): `format_reward × F1` — F1 between predicted answer and gold, scaled by a binary format check (valid `<answer>` tags).
  - LRM agents (QwQ-32B): **LLM-as-Judge** using Qwen2.5-72B-Instruct as judge — rule-based F1 is too brittle for free-form reasoning answers.
- **Compute:** ~**7.6k H800 GPU-hours** for ASearcher-Web-QwQ. Batch 128 (7B/14B) / 64 (32B). The 7.6k-hr figure is the dominant barrier to reproduction.

**Reported numbers**
| Benchmark | Score |
|---|---|
| **FRAMES (Avg@4)** | **70.9** |
| **FRAMES (Pass@4)** | **84.0** |
| GAIA (Avg@4) | 52.8 |
| GAIA (Pass@4) | 70.1 |
| xBench-DeepSearch (Avg@4) | 42.1 |
| xBench-DeepSearch (Pass@4) | 68.0 |

**Where ASearcher lacks**
- **No MoNaCo results.** Never evaluated on it, and the 128-turn rollout horizon would be costly there (43 docs/Q × 1315 Qs).
- **Pure text trajectories → no interpretable intermediate state.** A failed rollout is a 25k-char text blob.
- **7.6k H800-hours** is expensive to reproduce or extend. Not a hobbyist recipe.
- **Injection + Fuzzing is FRAMES-adjacent data**; their synthetic QAs are built around the same "multi-hop Wikipedia factoid" distribution. Generalization to MoNaCo-style compositional-aggregation questions (list + compare + group) is unvalidated.
- **Reward is end-to-end F1 only.** No credit for intermediate reasoning quality; no structured-output reward.

**Attack angles**
- **Compute match under a smaller RL budget.** Use GSW-structured intermediate state to give the RL agent a denser reward signal (per-sub-question F1 on GSW vs end-to-end only) — target sample-efficiency win.
- **Generalization test.** Take a pretrained ASearcher-Web-QwQ, run it on MoNaCo, compare to a GSW-based agent trained once on MoNaCo QDMR.
- **Interpretability.** Publishable side-benefit.

---

### 2.2 Context-1 (Chroma, March 2026) — retrieval subagent

**Blog / tech report** [trychroma.com/research/context-1](https://www.trychroma.com/research/context-1)
**MarkTechPost summary** [marktechpost.com/…context-1](https://www.marktechpost.com/2026/03/29/chroma-releases-context-1-a-20b-agentic-search-model-for-multi-hop-retrieval-context-management-and-scalable-synthetic-task-generation/)
**Data-gen repo** [github.com/chroma-core/context-1-data-gen](https://github.com/chroma-core/context-1-data-gen)
**Model weights** [huggingface.co/chromadb/context-1](https://huggingface.co/chromadb/context-1)
**Authors** Hammad Bashir, Kelly Hong, Patrick Jiang, Zhiyi Shi (Chroma)

**Architecture**

Context-1 is **a retrieval sub-agent, not a full stack.** A frontier model (GPT-5.4 / Opus-4.6) is the outer reasoner; it delegates "go find the docs I need" to Context-1, which runs its own internal search loop and returns a pruned set of relevant chunks. The contribution is not "a better reasoner" but "a better retrieval policy" — specifically, learned context-pruning so the retrieval agent doesn't drown in its own observations.

- **Base model:** **gpt-oss-20B** (MoE, OpenAI's open-source 20B) with LoRA adapters. ~1.5B active parameters per token (the MoE gating is unchanged from base).
- **Tool surface (3 primitives)**
  | Tool | Signature | Purpose |
  |---|---|---|
  | `search` | `query: str` | retrieves top-k chunks from the corpus's Chroma vector store |
  | `read` | `chunk_id: str` | expands a chunk to its surrounding passage |
  | `prune` | `chunk_id: str` | **removes a chunk from the active context** but keeps it in the reward-scored trajectory |
- **State representation — "context with soft/hard thresholds"**
  - The agent observes its own token usage each turn via a `context_usage: X / Y tokens (Z% full)` header line.
  - **Soft threshold** (~60% full) → the system prompt injects `"consider pruning"`.
  - **Hard cutoff** (~90% full) → `search` and `read` tools are blocked until at least one `prune` is executed.
  - This turns pruning into a *learned policy*: the RL reward credits the agent for retaining the right chunks and penalizes over-pruning.
- **Inference loop (one retrieval episode)**
  ```
  outer-agent question
      → Context-1: search → observe chunks → (read | prune)* → search … → return pruned chunk set
  outer-agent: reason over returned chunks → final answer
  ```
  Only the pruned chunks survive into the outer agent's context, so pruning is how Context-1 controls the outer reasoner's input quality.

**Training recipe**
- **SFT stage:** trajectories collected from **Kimi K2.5** running the same tool loop on the synthetic corpus. Selection policy:
  - High-recall trajectories retained in full (positive examples).
  - Low-recall trajectories retained at a diminishing rate — up to **5% kept as explicit negatives** (teaches the model what failure looks like without flooding the dataset).
- **RL stage: CISPO** (Clipped Importance-Sampling Policy Optimization) — the key algorithmic choice.
  - Standard PPO clips the *surrogate objective* (ratio × advantage). CISPO clips the **importance-sampling weight itself** before the surrogate is computed. Stabilizes off-policy updates in long-horizon tool-use, where the standard clip can zero out whole-trajectory gradients.
  - **Step size:** 128 queries × 8 rollouts = 1024 trajectories per update step. ~230–300 steps over 5 epochs.
- **Reward (4 components, linearly combined)**
  1. **Outcome** — F1 between agent's returned chunks and gold-answer chunks. Recall is weighted **16× over precision** in early training, then annealed as the agent learns to prune.
  2. **Process** — trajectory-recall credit: docs that were encountered (touched by `search` or `read`) even if later pruned still contribute partial reward. Prevents the model from over-pruning useful docs just to reduce context.
  3. **Binary bonus** — `+1.0` if the *retained* chunks (after pruning) contain the final answer span.
  4. **Penalties** — `-0.1` per excess prune beyond 3 consecutive prunes (caps at `-0.5`); linear turn-count penalty (`0 → -0.5` as turn-count goes from 64 → 128).

**Synthetic data gen pipeline — "Explore → Generate → Verify → Distract → Chain"**

One of Chroma's headline contributions: a 4-domain synthetic-corpus recipe, all producing the same trajectory format (question → tool-use → answer).

| Domain | Source | Scale | Verification signal |
|---|---|---|---|
| **Web** | Wikipedia-seeded random walk | unbounded | LLM verifies quoted spans match source (>80% alignment w/ humans). Distractors are docs that *look* relevant but have different answers, passed through a leak-filter that drops docs where the correct answer appears verbatim. |
| **Finance** | 1707 companies' SEC 10-K / 20-F filings, avg 31.5k tokens/filing | 1.7k filings | Tasks reference specific chunks; verification at 93% agreement with humans. Chains up to 3 hops across companies ("what was the YoY gross margin change for company A vs its sector peer B?"). |
| **Legal** | 1500 USPTO publications with §102/§103 rejections | 1.5k documents | **No LLM verification needed** — patent examiners explicitly write out which prior-art claims reject which new claims. Expert-verified ground truth by construction. |
| **Email** | 984 Epstein email threads + Enron (PII-scrubbed) | 396,510 chunks | Extraction + coherence check, 87.5% human agreement. |

The **"chain"** stage is what makes retrieval multi-hop: a generated single-hop QA is re-expressed as "find X, use X to find Y, answer using Y" across 2–4 hops with distractors inserted at each.

**Reported FRAMES numbers**
| Setup | Final-answer-found | F1 |
|---|---|---|
| Context-1 (1× run) | 0.87 | 0.87 |
| Context-1 (4× parallel with RRF) | 0.96 | 0.96 |

For comparison, frontier models (GPT-5.4, Opus-4.6) range 0.92–0.97 on final-answer-found on the same eval. Context-1 runs **~10× faster and ~25× cheaper** than GPT-5.4.

**Where Context-1 lacks**
- **No MoNaCo.**
- Retrieval-*only* role. Reasoning and final-answer generation are outsourced to the frontier caller, so the FRAMES 0.87 is *retrieval quality*, not end-to-end reasoning accuracy.
- No published comparisons against GPT-5 / o3 / Claude directly on FRAMES in the tech report — only against earlier gpt-5.2/5.4 variants.
- **Legal corpus (USPTO rejections)** is a lucky unfair-advantage domain; not replicable elsewhere.
- **Context pruning is the headline contribution** but it's orthogonal to structured representation — a GSW scratchpad would be pruned at the *entity* level, which is a qualitatively different claim.

**Attack angles**
- **End-to-end evaluation.** Match Context-1 as a retrieval backbone and bring our own small-model reasoner on top; compare full-stack on FRAMES and MoNaCo.
- **Entity-level pruning.** Context-1 prunes chunks; GSW prunes by entity subgraph. Our claim could be "per-entity pruning has better recall/precision tradeoff at equivalent budget."
- **QDMR as synthetic data.** Context-1's synthetic chaining is all linear multi-hop (A→B→C). MoNaCo's QDMR gives us branching + aggregation supervision out of the box — no need to synthesize.

---

### 2.3 Search-o1 (EMNLP 2025) — reasoning + interleaved retrieval

**Paper** *Search-o1: Agentic Search-Enhanced Large Reasoning Models*, [`arxiv 2501.05366`](https://arxiv.org/abs/2501.05366), EMNLP 2025.
**GitHub** [github.com/RUC-NLPIR/Search-o1](https://github.com/RUC-NLPIR/Search-o1)
**Project page** [search-o1.github.io](https://search-o1.github.io/)

**Architecture**

Search-o1 is a **prompt-level framework**, not a trained model. It wraps a large reasoning model (QwQ-32B / o1-style) and teaches it to *interleave* search calls inside its own chain-of-thought. The novelty is that retrieval happens *mid-reasoning*, not before it — the LRM decides when it is uncertain, emits a search query, receives a condensed retrieved passage, and continues the same `<think>` block.

- **Outer agent** — QwQ-32B (or any o1-style LRM).
- **Inner condenser — "Reason-in-Documents" (R-in-D)** — a secondary LLM call (same or smaller model) that takes `(current_reasoning_context, retrieved_docs)` and emits a **short, grounded, relevance-filtered summary** of only the parts of those docs relevant to the current reasoning step. This summary is what gets spliced back into the outer CoT, not the raw doc.
- **Reasoning-chain format** (special tokens):
  ```
  <think>
    … reasoning chain …
    <|begin_search_query|> who directed 13 Hours <|end_search_query|>
    <|begin_search_result|> Michael Bay directed 13 Hours (2016) … <|end_search_result|>
    … continues reasoning …
  </think>
  <answer>Michael Bay</answer>
  ```
  - Emitting `<|begin_search_query|>` triggers external retrieval (BM25 or dense) → top-k docs fetched.
  - Top-k docs are then passed to R-in-D, which returns a condensed span that replaces the `<|begin_search_result|>` block.
  - The LRM continues from wherever it left off, now with the new fact inline.
- **Uncertainty trigger** — the agent is prompted to emit a `<|begin_search_query|>` whenever it is about to make a factual claim it cannot verify from pretraining alone. No explicit probability-based trigger; it's learned from the prompt scaffold.
- **Loop control** — bounded by the LRM's context window. No external iteration cap; the LRM stops emitting queries when it's confident it can answer.

**FRAMES results**
- **Search-o1 + QwQ-32B: 63.6 Avg@4** (per ASearcher's table).
- Headline claim: outperforms vanilla RAG-QwQ-32B by 29.6% on multi-hop tasks.

**Where Search-o1 lacks**
- Prompt-only, no training — ceiling is capped by the base LRM.
- No MoNaCo.
- The Reason-in-Documents module is an extra LLM call per retrieval → slow.
- No structured memory; relies entirely on the LRM's text-CoT.

---

### 2.4 Tongyi WebAgent family — WebWalker, WebDancer, WebSailor, WebShaper

**Umbrella GitHub** [github.com/Alibaba-NLP/WebAgent](https://github.com/Alibaba-NLP/WebAgent)
**Papers**
- *WebWalker: Benchmarking LLMs in Web Traversal* — [`arxiv 2501.07572`](https://arxiv.org/abs/2501.07572)
- *WebDancer: Towards Autonomous Information Seeking Agency* — *(arxiv link not extracted but referenced from `Paper page - WebDancer` on HuggingFace)*
- *WebSailor: Navigating Super-human Reasoning for Web-scale Information Seeking* — [`arxiv 2507.02592`](https://arxiv.org/pdf/2507.02592)
- *WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization* — [`arxiv 2507.15061`](https://arxiv.org/abs/2507.15061)

**Architecture**

The Alibaba WebAgent family is an evolving pipeline where each paper builds on the last: **WebWalker defines the benchmark, WebDancer trains the first agent, WebSailor scales to 72B, WebShaper adds a principled data-synthesis formalism.** The shared pattern across all four is a ReAct-style agent over real web-browsing tools.

#### WebWalker — dual-agent explorer + benchmark
- Two cooperating agents:
  - **Explorer** — a ReAct agent that browses (click, back, scroll, search) and collects evidence.
  - **Critic** — a separate LLM call that watches the Explorer's trajectory and decides *"do we have enough to answer yet?"*. Prevents premature termination AND runaway browsing.
- Tools: `click(element)`, `back()`, `type(text)`, `scroll`, `search(query)`, `answer(text)`.
- Ships the **WebWalkerQA benchmark** — 680 queries, 1373 webpages, hand-collected from ~250 real sites.

#### WebDancer — first trained autonomous web agent
Four-stage training pipeline:

| Stage | What happens | Output |
|---|---|---|
| 1. **Browsing-data construction** | Two sub-techniques: **CRAWLQA** (scrape Wikipedia + QA generation) and **E2HQA** ("easy-to-hard" progressive QA synthesis — take a simple QA and iteratively make it harder by adding constraints). | Large seed pool of (query, trajectory, answer) triples. |
| 2. **Trajectory sampling** | Roll out a strong teacher (closed-source model) on the seed queries with the ReAct tool set. | Clean ReAct trajectories. |
| 3. **SFT cold-start** | Reject-sample the trajectories (keep only those that reach the right answer) → SFT the base model on them. | Base agent with working tool-use. |
| 4. **RL fine-tune** | **DAPO** (a GRPO variant with decoupled advantage) on the SFT base, reward = answer correctness. | Final agent. |

Results: Pass@3 = 64.1% GAIA, 62.0% WebWalkerQA.

#### WebSailor — 72B, uncertainty-driven browsing
- Base: **Qwen2.5-72B**, same ReAct tool set.
- Core training trick: **"uncertain information seeking"** — synthetic queries are constructed by *obfuscating* answers inside a knowledge graph (e.g. "the film directed by X, where X is the second-born of Y"), forcing the agent to traverse relations instead of memorizing facts.
- Training: **RFT** (rejection-sampled fine-tuning) cold-start, then **DUPO** ("duplicating sampling policy optimization") — a sampling trick that re-samples high-reward trajectories multiple times to stabilize RL.
- Headline numbers: BrowseComp-en 12.0%, BrowseComp-zh 30.1%, GAIA 55.4%.

#### WebShaper — set-theoretic data synthesis
- Not an agent — a **data synthesis framework** that produces training questions for WebSailor / WebSailor-V2.
- Formalism: every question is expressed as a **Knowledge Projection (KP)** — a set-algebra expression over entities and relations using union `∪`, intersection `∩`, and projection operators.
- Example: `Person(profession=director) ∩ Film(title="13 Hours").director` → "who directed 13 Hours?".
- A generator agent composes KPs compositionally (bigger KP = harder question), then lowers each KP to a natural-language query + ground-truth answer.
- **This is the closest prior art to GSW's structural thesis**: WebShaper uses structured representations to generate training data; GSW uses structured representations at inference time. Cite explicitly.

**Reported FRAMES results** (from ASearcher's table):
- WebDancer-QwQ: 63.8 Avg@4.
- WebSailor / WebShaper — primarily evaluated on BrowseComp + GAIA + WebWalkerQA, not explicitly on FRAMES in my search.

**Where the Tongyi family lacks**
- FRAMES is not their headline benchmark; they emphasize BrowseComp + GAIA + WebWalkerQA.
- No MoNaCo.
- Heavy reliance on web-browsing (actual web). FRAMES and MoNaCo both use Wikipedia-only — their strongest results don't directly translate.
- **WebShaper's set-theoretic formalization is the closest structural-reasoning ancestor** to GSW's entity/verb-phrase decomposition — we should cite and differentiate: KP operations are for *data synthesis*, GSW is for *at-query-time graph construction*.

---

### 2.5 Tongyi DeepResearch (30B MoE, Sept 2025) — agentic deep-research SOTA

**GitHub** [github.com/Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch)
**Weights** [huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
**Tech report** [`arxiv 2510.24701`](https://arxiv.org/html/2510.24701v2)
**Blog** [tongyi-agent.github.io/blog/introducing-tongyi-deep-research/](https://tongyi-agent.github.io/blog/introducing-tongyi-deep-research/)

**Architecture**

Tongyi DeepResearch is Alibaba's **successor to the WebAgent family**, purpose-built as a single deep-research agent rather than the WebWalker/Dancer/Sailor lineage of stepping-stone models. The key design moves are (i) an MoE base with aggressive sparsity, (ii) a richer tool set that goes beyond browse-only, and (iii) a dual inference paradigm where the agent can operate either as a normal ReAct loop or as a *re-planning* loop that aggregates an evolving research report.

- **Base model:** **30.5B total / 3.3B active per token** (MoE with ~10× sparsity). Trained from Qwen3-30B-A3B base.
- **Tool surface (5 tools)**
  | Tool | Purpose |
  |---|---|
  | `Search` | SERP-style search results (titles + snippets + URLs) |
  | `Visit` | fetch and render a specific URL |
  | `Python` | sandboxed code execution — used for arithmetic, data-frame manipulation, comparison steps |
  | `Scholar` | academic-paper-specific search (Semantic Scholar–style) |
  | `File Parser` | extract structured data from PDFs / XLS / CSV returned by `Visit` |
- **Inference paradigms**
  - **ReAct mode (baseline)** — standard `think → action → observation → …` loop with all five tools.
  - **IterResearch "Heavy" mode** — a test-time-scaling paradigm:
    - On each iteration, **multiple agents run in parallel** on the same query with different seeds.
    - An **observation / synthesis step** between iterations condenses everything discovered so far into a *research report* (a structured state summary).
    - The next iteration's actor reads *the report, not the full trajectory history*, avoiding the context-dilution failure mode that plagues ReAct at 100+ turns.
    - Similar in spirit to IterResearch proper (arxiv 2509.13313) but integrated into the Tongyi stack.
- **Training** — multi-stage: SFT on curated trajectories, then RL (algorithm not fully disclosed in the public tech report; the blog references GRPO-style updates).
- **Reported benchmarks include FRAMES** (numbers not extracted in search), Humanity's Last Exam, BrowseComp-en / zh, WebWalkerQA, xBench-DeepSearch, SimpleQA.

**Where Tongyi DeepResearch lacks**
- No MoNaCo.
- 30B active-param model. Large infra requirement.
- Closed on training recipe specifics (SFT + RL stages exist but granularity not fully public).
- Text-trajectory-only; no structured intermediate memory.

---

### 2.6 EigentSearch-Q+ — structured reasoning tools

**Paper** *EigentSearch-Q+: Enhancing Deep Research Agents with Structured Reasoning Tools*, [`arxiv 2604.07927`](https://arxiv.org/abs/2604.07927) (2026).
**System** Integrates into **Eigent**, an open-source multi-agent "workforce for computer use" (production-grade).

**Architecture**

EigentSearch is a **tools-layer contribution on top of a frontier model**, not a trained model. The authors take Anthropic's `think` tool paradigm (where an agent can call a tool that just adds a `<think>` span to its own context) and generalize it into a small family of **structured reasoning tools** specifically designed to counteract failure modes of browser sub-agents on long-running research tasks.

- **Q+ tool family** (bolted onto an existing browser agent):
  - **Query planner** — the agent can call `plan(goal)` which returns a structured plan (sub-queries + rationale). Separates planning from execution, so the browser sub-agent doesn't conflate them.
  - **Progress monitor** — `check_progress()` returns "what's been found, what's still missing." Addresses the "agent forgets the original question after 20 turns" failure.
  - **Evidence extractor** — `extract(url_or_span, target)` pulls a focused evidence span from a long web snapshot, supervised by a short "what you're looking for" target string. Replaces the agent having to re-read the same 50k-token webpage three times.
- **Host system — Eigent** — a production-grade open-source multi-agent "workforce for computer use." EigentSearch lives inside Eigent as the research sub-agent.
- **No training.** Pure prompt + tool-layer. Gains come from the tools themselves, not from model updates.
- **Backends evaluated:** GPT-4.1, GPT-5.1, Minimax M2.5. Gains are 0.6–3.8 pp absolute on a 4-benchmark aggregate (SimpleQA-Verified, FRAMES, WebWalkerQA, xBench-DeepSearch). Absolute FRAMES scores not extracted in web search.

**Where EigentSearch lacks**
- Tools-layer contribution, not a model. Gains are 0.6–3.8 pp — incremental.
- No training.
- Closed-source integration into a commercial product.
- **But:** the "think-style tool" framing is the closest prior art to GSW's "graph-manipulation tool." We should explicitly compare.

---

### 2.7 *Search More, Think Less* (SMTL-30B, 2026) — counter-thesis paper

**Paper** *Rethinking Long-Horizon Agentic Search for Efficiency and …*, [`arxiv 2602.22675`](https://arxiv.org/pdf/2602.22675) (2026).
**Claim:** Long-horizon agents **over-emphasize reasoning depth.** Lighter per-step thinking + more exploration iterations beats fewer-but-deeper steps.
**Model / data:** SMTL-30B, SMTL-Dataset (HuggingFace releases referenced).

**Architecture**

SMTL is a deliberate **ablation of the "deep thinking per turn" pattern**. Where ASearcher / Tongyi DeepResearch / WebSailor spend 2000–5000 reasoning tokens per turn before acting, SMTL trains the agent to *think briefly, act often*.

- **Base model:** SMTL-30B (MoE).
- **Tool surface:** similar to ASearcher — `search` + `browse` + internal summarize.
- **Behavioral contract enforced during training:**
  - Per-turn `<think>` block is capped (short budget — on the order of a few hundred tokens).
  - Turn budget is raised correspondingly (many more turns allowed per rollout).
  - The training objective prefers trajectories that reach the answer via many-cheap-turns over few-expensive-turns even when their final reward is equal.
- **Training data:** SMTL-Dataset — synthetic QA with trajectory annotations that *penalize* long per-turn thinking. Released on HuggingFace.
- **Benchmarks:** GAIA + Deep Research benchmarks. FRAMES / MoNaCo not confirmed in accessible sections.

**Relevance to us**
- **This is the direct philosophical opponent** to any "structured planning / GSW-scaffold" contribution.
- If we claim "pre-structured QA-pairs as scratchpad helps," SMTL claims "no, just search more with lighter thinking." Our ablation has to hit SMTL head-on.
- SMTL's thesis is testable on MoNaCo — "lighter thinking + more search" should collapse on MoNaCo's list-recall questions precisely where GSW's structured aggregation should shine. If our paper runs SMTL-30B on MoNaCo and it underperforms, that's the single cleanest quantitative argument for structured memory.

---

### 2.8 Search-R1 (COLM 2025) — PPO-trained search-interleaved LLMs

**Paper** *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*, [`arxiv 2503.09516`](https://arxiv.org/pdf/2503.09516), COLM 2025.
**GitHub** [github.com/PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1)

**Architecture**

Search-R1 is **DeepSeek-R1-Zero applied to search-augmented QA**. Train a small base LLM from scratch with online RL, reward = final answer correctness, and let the model discover `<search>` interleaving on its own. No SFT cold-start, no trajectory distillation.

- **Base models:** Qwen2.5-3B and Llama3.2-3B (both tested — 3B is the headline parameter count, chosen precisely to test "can zero-shot RL work at small scale").
- **Interaction format:**
  ```
  <think>
    reasoning about the question …
    <search>specific query</search>
    <information>retrieved snippets</information>
    more reasoning …
  </think>
  <answer>final answer</answer>
  ```
  - Retrieved text is wrapped in `<information>…</information>` tokens. The crucial training trick: **loss is masked on the `<information>` tokens** — they are retrieved, not generated, so the model shouldn't be trained to predict them. Masking prevents distributional poisoning from the search corpus.
- **Tool surface:** exactly one — `search(query)`, backed by a configurable retriever (Google / Bing / Brave APIs or a local BM25/dense index). No `browse`, no `read`.
- **RL algorithm: PPO** with **rule-based outcome reward** — exact-match between `<answer>` content and gold (with alias list for NQ / HotpotQA). No reward shaping, no intermediate-step credit.
- **Framework integration:** built on **veRL** and RAGEN. Ships first-class support for LoRA, mid-training SFT passes, multiple search backends, and optional rerankers on retrieved results.

**Benchmarks**
- Natural Questions, TriviaQA, HotpotQA. **FRAMES not reported** — their focus is classic retrieval-augmented QA, not deep-research-style multi-hop.

**Where Search-R1 lacks**
- Smallest-base-model class (3B). Not a direct competitor on FRAMES — they don't even report it.
- NQ/HotpotQA focus → lighter multi-hop.
- No MoNaCo.

**But** — Search-R1 is **the best open-source scaffolding for our training code**. It reuses veRL (which we already have) and has search-API plumbing. If we go the RL-training route, starting from Search-R1's codebase is faster than greenfield.

---

### 2.9 EVO-RAG — curriculum-guided query-rewriting agent

**Paper** *Curriculum Guided Reinforcement Learning for Efficient Multi Hop Retrieval Augmented Generation*, [`arxiv 2505.17391`](https://arxiv.org/abs/2505.17391). OpenReview name **EVO-RAG**.
**No public GitHub URL** surfaced in my search (paper-only as of April 2026).

**Architecture**

EVO-RAG's distinguishing feature is a **two-stage curriculum with a seven-factor reward** — most competitors use a single end-to-end reward, EVO-RAG decomposes it and anneals the weighting schedule across training.

- **Base / actions** — the agent is an LLM with a 4-action policy:
  | Action | Meaning |
  |---|---|
  | `search(q)` | issue a retrieval query |
  | `backtrack` | drop the current retrieval path and return to a prior state |
  | `answer(a)` | emit the final answer |
  | `refuse` | abstain when evidence is insufficient (a deliberate calibration mechanism) |
- **Two-stage curriculum**
  - **Stage 1 — Discovery.** Reward weights favor breadth — `relevance`, `diversity`, `retrieval-depth` dominate. The agent is encouraged to *explore* many retrieval pathways per question. Low penalty for inefficiency.
  - **Stage 2 — Refinement.** Reward weights shift — `efficiency`, `redundancy-penalty`, `correctness` dominate. The agent is now pushed to *commit* to one pathway and produce concise evidence-backed answers. Time-varying scheduler handles the handoff.
- **7-factor reward** (component names per paper):
  1. Relevance — query / retrieved-doc similarity.
  2. Redundancy penalty — discourage fetching overlapping docs.
  3. Efficiency — fewer search calls per question.
  4. Correctness — final-answer match.
  5. Grounding — answer spans present in retrieved docs.
  6. Diversity — retrieved set covers multiple facets.
  7. Calibration — `refuse` rewarded when correct, penalized when it shortcuts a solvable question.
- **Optimizer: DPO** (Direct Preference Optimization) — not PPO or GRPO. Rollouts are paired (winner vs loser) based on the 7-factor score aggregate, and the model is trained to prefer the winner.

**Results**
- Exact Match +4.6 pp over strong RAG baselines on HotpotQA, 2WikiMultiHop, MuSiQue, Bamboogle.
- Retrieval depth reduced by 15%.
- **No FRAMES or MoNaCo.**

**Where EVO-RAG lacks**
- DPO, not GRPO/PPO — different signal, no direct comparison.
- No code release (as of search) → hard to extend directly.
- Light multi-hop benchmarks; untested on FRAMES/MoNaCo horizon.

**Reusable ideas** — the 7-factor reward (relevance/redundancy/efficiency/correctness) maps well to GSW-based rewards: "did the retrieved entity end up on the GSW + answer a sub-question?" is a natural per-step reward.

---

### 2.10 A-RAG — hierarchical retrieval interfaces

**Paper** *A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces*, [`arxiv 2602.03442`](https://arxiv.org/abs/2602.03442) (2026).
**GitHub** [github.com/Ayanami0730/arag](https://github.com/Ayanami0730/arag)

**Architecture**

A-RAG's core idea: **retrieval isn't one tool with one granularity — it's three tools at three granularities.** Give the agent the choice of which tool to call based on what it's currently uncertain about.

- **Tool surface (3 tools at escalating cost / specificity):**
  | Tool | Granularity | Typical use |
  |---|---|---|
  | `keyword_search(terms)` | coarse; returns titles + brief snippets | "what topics are on this corpus at all" |
  | `semantic_search(query)` | medium; returns top-k chunk embeddings | "pull me the passages most similar to this question" |
  | `chunk_read(chunk_id)` | fine; returns full text of one chunk | "expand this specific hit for close reading" |
- **Policy** — the agent decides *adaptively* which granularity to call. No fixed pipeline (vs e.g. "always keyword then semantic then read"). A cheap `keyword_search` is preferred for unfamiliar queries; `semantic_search` for facet-specific lookups; `chunk_read` reserved for confirmed-relevant docs.
- **Hierarchical design principle** — each tool sits at a different point on the **token-cost vs information-density curve**. The agent learns which curve-point is appropriate per sub-question, minimizing total retrieved-token count while maintaining recall.
- **No training details as headline** — A-RAG's contribution is the interface, not the model. Gains are measured with frozen backends.
- **Claim:** outperforms baselines at **comparable or lower total retrieved-token count** — i.e. the dominant efficiency argument is token budget, not accuracy.

**Benchmarks** — multi-hop QA suite. FRAMES not explicitly confirmed in the abstract; some references suggest it is evaluated.

**Where A-RAG lacks**
- Interface design, not a model. Gains depend on the base model.
- No training specifics (SFT/RL not headline).
- No MoNaCo.

**Reusable idea** — "three retrieval tools with different granularity" maps well to GSW's hierarchy: entity-list / verb-phrase-list / full-GSW-subgraph.

---

### 2.11 InfoDeepSeek — agentic info-seeking benchmark (not a system)

**Paper** *InfoDeepSeek: Benchmarking Agentic Information Seeking for RAG*, [`arxiv 2505.15872`](https://arxiv.org/abs/2505.15872).
**GitHub** [github.com/YunjiaXi/InfoDeepSeek](https://github.com/YunjiaXi/InfoDeepSeek)

**What it is**

A **benchmark + evaluator**, not a system. The key construction principle: unlike FRAMES (static Wikipedia) and MoNaCo (static Wikipedia), InfoDeepSeek operates over the **live web**, so the ground-truth answer is whatever is currently verifiable on today's internet, not whatever was captured in a static snapshot.

- **Query construction criteria (3-D):**
  - **Determinacy** — answer must be verifiable and unambiguous at construction time (so the benchmark is scorable).
  - **Difficulty** — requires multi-source reasoning, not a single Wikipedia lookup.
  - **Diversity** — covers domains, time periods, languages, intent types.
- **Fine-grained metrics**
  - **Accuracy (ACC)** — final answer correctness.
  - **Utility (IA@k)** — "information acquired" — whether the top-k retrieved docs contain the grounding evidence (a retrieval-only metric independent of the final answer).
  - **Compactness** — ratio of tokens actually used in the final answer to tokens retrieved. Rewards parsimony.
- **Evaluator** — automated grading using gold-answer aliases + LLM-judge fallback.

**Headline finding**: Best model (Gemini-2.5-Pro) = 22.45% ACC, 21.63% IA@5. Even SOTA models struggle — the live-web setting is qualitatively harder than static-corpus benchmarks.

**Relevance to our pivot**
- If we target the dynamic-web direction later, InfoDeepSeek is the eval.
- Neither FRAMES nor MoNaCo, but relevant as a "closed corpus → live web" complementary axis.

---

## Part 3 — Systems reporting MoNaCo scores

**As of April 16, 2026: only the original paper.**

The 15 baselines enumerated in §1.2 are the entire public record. No downstream agentic paper, no RL-trained agent, no retrieval-augmented system has reported MoNaCo F1. HF dataset page shows 0 "models trained on this dataset," 0 "spaces using this dataset."

**Implication.** First agentic submission that clears 61.2 F1 on MoNaCo is automatically a publishable baseline — any reasonable retrieval + decomposition system should do this. The interesting target is **how far above 61.2** we can push, and **which of the four MoNaCo settings** we beat (parametric / long-context / RAG / retrieval).

---

## Part 4 — Papers using **both** FRAMES and MoNaCo

**Zero.**

The two benchmarks have not yet been evaluated together in any published work. MoNaCo was released ~8 months ago (Aug 2025) and agentic benchmarking on it has not begun. FRAMES lives almost entirely in the agentic-RAG-on-Wikipedia lane; MoNaCo lives in the long-horizon-reasoning lane evaluated only with closed-book LLMs so far.

**This gap is the single cleanest contribution hook for our research agent** — submitting the first paper that reports numbers on both, with a small agentic model + focused-GSW substrate, is defensible novelty on positioning alone.

---

## Part 4b — Agentic research agents with structured intermediate state

The genuine competitor set. To qualify, a system must satisfy **all three** criteria:

1. **Multi-turn tool loop** — not retrieve-then-generate. The agent plans, acts, observes, reflects, and iterates.
2. **Structured intermediate state** — the agent carries something richer than a raw text trajectory between turns (a graph, a report, a mind-map, a schema-typed memory compartment).
3. **Research-agent use case** — multi-hop QA / deep research / long-horizon information seeking. Not general chat, not code-gen agents.

Five systems survive as competitors we run or compare against directly (4b.1–4b.5). Another five (ToG 3.0, DeepAgent, Agentic-KGR, GraphReader, AGENTiGraph) were surveyed but excluded from the pilot — see the "Surveyed but out of scope" subsection at the end.

Graph-RAG retrieval systems (HippoRAG, GraphRAG, LightRAG, PathRAG, StructRAG) are **not** in this list — they're single-shot retrieval, not multi-turn agents. Covered in **Part 4c** below.

---

### 4b.1 WebResearcher + IterResearch (Tongyi Lab) — evolving-report-as-memory

**Papers**
- WebResearcher v1: *WebResearcher: Unleashing Unbounded Reasoning Capability in Long-Horizon Agents*, [`arxiv 2509.13309`](https://arxiv.org/abs/2509.13309), Sept 2025.
- IterResearch v2: *IterResearch: Rethinking Long-Horizon Agents via Markovian State Reconstruction*, [`arxiv 2511.07327`](https://arxiv.org/html/2511.07327v1), Nov 2025.
**GitHub** [github.com/Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) (same repo as Tongyi DeepResearch — IterResearch is the underlying paradigm)
**Unofficial mirror** [github.com/shibing624/WebResearcher](https://github.com/shibing624/WebResearcher)

**Architecture**
- Reformulates long-horizon research as a **Markov Decision Process** with **strategic workspace reconstruction**. The agent does NOT accumulate full trajectory history linearly — it periodically *resets* its context.
- **State at each round ("Markovian state") is a 3-tuple:**
  - `question` (constant)
  - `evolving report ℳₜ` — a compressed, LLM-synthesized summary of findings (this is the "memory")
  - `immediate context {aₜ₋₁, TRₜ₋₁}` — last action + tool response
- Between rounds, the agent uses the report *instead of* the raw trajectory — avoids context suffocation + noise contamination in long rollouts.
- **Tools:** Google Search, Google Scholar, Visit (URL fetch + summarize), Python interpreter.
- **Training (IterResearch v2):**
  - Base: **Qwen3-30B-A3B** (30B total / 3B active, MoE).
  - SFT on 110K trajectories, then RL on 4,096 questions.
  - RL algorithm: **EAPO** (Efficiency-Aware Policy Optimization) — geometric reward discounting (γ=0.995) + **GSPO** (Group Sequence Policy Optimization) + adaptive downsampling.
  - Extends to **2048 interactions per episode** (3.5% → 42.5% accuracy at extreme depths).

**Reported results (IterResearch v2)** — +14.5pp avg over open-source agents across six benchmarks:
| Benchmark | Score | Δ vs prior open-source |
|---|---|---|
| HLE | 28.8% | +8.8pp |
| BrowseComp-en | 37.3% | +20.1pp |
| BrowseComp-zh | 45.2% | +15.8pp |
| GAIA | 72.8% | +8.7pp |
| xBench-DeepSearch | 71.0% | +15.0pp |
| SEAL-0 | 39.6% | +18.9pp |

WebResearcher-heavy (v1) on HLE: **36.7% vs OpenAI Deep Research 26.6%** (claimed new open-source SOTA).

**Why this is the #1 direct competitor**
- **Structured state = the "evolving report."** Same *role* as our GSW scratchpad, but the report is natural-language-structured, not graph-typed.
- Tongyi Lab's flagship — heavily funded, well-engineered, large open-source distribution.
- Already in the FRAMES-reporting orbit (Tongyi DeepResearch reports FRAMES; same codebase).

**Where it lacks**
- Report is plain-text compression. No entity-typed nodes, no relation structure, no QA-pair primitives. Ablating against it is where GSW's structural claim gets tested.
- **No MoNaCo evaluation.**
- 30B active-param MoE — large infra requirement.
- Training recipe is GSPO + EAPO, a niche RL variant.

---

### 4b.2 Graph-R1 — agentic GraphRAG via end-to-end RL

**Paper** *Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning*, [`arxiv 2507.21892`](https://arxiv.org/html/2507.21892v1), July 2025.
**GitHub** [github.com/LHRLAB/Graph-R1](https://github.com/LHRLAB/Graph-R1)
**Blog** [MarkTechPost writeup](https://www.marktechpost.com/2025/08/09/graph-r1-an-agentic-graphrag-framework-for-structured-multi-turn-reasoning-with-reinforcement-learning/)

**Architecture**
- **True multi-turn agent loop over a knowledge hypergraph.** "Think-retrieve-rethink-generate" is the core loop the agent repeats until terminating.
- **Knowledge hypergraph:** pre-built from corpus, lightweight LLM-based extraction generating n-ary relational facts with semantic embeddings (~120k nodes, 98k edges typical).
- **Action set (4 compositional actions per step):**
  1. **Thinking** — continue or terminate
  2. **Query generation** — formulate retrieval query
  3. **Graph retrieval** — extract relevant knowledge from hypergraph
  4. **Answering** — produce final response if terminating
- **Training:**
  - Base: **Qwen2.5-1.5B / 3B / 7B**
  - Algorithm: **GRPO**
  - Reward: `R(τ) = -1.0 + R_format(τ) + 𝕀{R_format=1}·R_answer`, where `R_answer` is token-level F1.

**Reported results (Qwen2.5-7B)**
| Method | Avg F1 | Avg EM |
|---|---|---|
| **Graph-R1** | **57.82** | **48.57** |
| Search-R1 | 46.19 | — |
| Standard RAG | 32.05 | — |

Benchmarks: HotpotQA, 2WikiMultiHopQA, MuSiQue, Natural Questions, PopQA, TriviaQA. **No FRAMES, no MoNaCo.**

**Why this is the #2 direct competitor**
- **This is the single closest paper to our thesis by design shape.** Multi-turn agent + graph intermediate state + RL training on multi-hop QA benchmarks.
- Same base-model family (Qwen2.5) we'd likely use.
- Same RL algorithm (GRPO) we already have in veRL.
- 7B beats Search-R1 — direct proof that a small agentic model with graph state beats a small agentic model without it. That's our headline claim too.

**Where it lacks**
- **Hypergraph is pre-built.** Not query-focused, not constructed on the fly per sub-question. Our GSW pitch flips this.
- **Triples / hyperedges, not QA-pair primitives.**
- No decomposition supervision — just end-to-end F1 reward.
- No FRAMES, no MoNaCo — the dual-benchmark gap holds here too.

**Reproducibility constraint (important for the pilot).** Upstream releases training code (GRPO / REINFORCE++ / PPO bash scripts) + a pre-built knowledge-hypergraph and training data on **TeraBox** — but **no HuggingFace checkpoints**. The 1.5B/3B/7B numbers in the paper come from weights you train yourself on the supplied hypergraph corpus (HotpotQA / 2Wiki / MuSiQue / NQ / PopQA / TriviaQA). Training a FRAMES-native Graph-R1 is paper-scope work, not pilot-scope.

For the FRAMES pilot we therefore run a **prompt-mode reimplementation** (`graph_r1` adapter): the paper's `<think>/<query>/<answer>` loop with on-the-fly hyperedge extraction (a cheap second LLM pass per retrieval) against any instruction-tuned model (Qwen2.5-7B-Instruct, gpt-oss-20B, GPT-5). This matches how `search_o1` wraps QwQ without their specific checkpoint. When a real trained ckpt exists, the `graph_r1_trained` alias class is reserved for it — same code, separate row in reports.

---

### 4b.3 GAM — General Agentic Memory Via Deep Research

**Paper** *General Agentic Memory Via Deep Research*, [`arxiv 2511.18423`](https://arxiv.org/abs/2511.18423), Nov 2025.
**GitHub** [github.com/VectorSpaceLab/general-agentic-memory](https://github.com/VectorSpaceLab/general-agentic-memory)

**Architecture — "JIT compilation for agent memory"**
- **Offline stage:** keep raw data in a universal *page-store*, build a lightweight "hint memory" that merely highlights key historical information. Deliberately lossy-free storage.
- **Online stage:** a **Researcher** sub-agent runs a *deep-research loop* over the page-store to retrieve and integrate the useful information for the online query.
- **Duo design:**
  - **Memorizer** — offline; cheap; builds only lightweight hints.
  - **Researcher** — online; an actual multi-turn research agent that uses the hints to target its search within the raw page-store, not a pre-compressed summary.
- Paradigmatic inversion: **maximize storage fidelity, offload complexity to runtime** via a search agent. Contrasts with AOT-compilation approaches (Mem0, summary-based memory) where future-query-relevance has to be predicted at write time.

**Training** — end-to-end optimization via RL ("facilitating end-to-end performance optimization through reinforcement learning" per the abstract). Specific algorithm not extracted in search.

**Benchmarks** — "memory-grounded task completion scenarios." Specific benchmark names not confirmed in the search excerpts.

**Why this is directly relevant**
- **The JIT-compilation framing is exactly our thesis**: don't pre-index, compile on demand when a query arrives. This is the first paper to name the paradigm explicitly. We need to cite and position against it.
- Same principle, different substrate: GAM compiles text via a search agent, we compile structured GSW via a problem composer + linker.

**Where it lacks**
- No structural intermediate representation — the Researcher's trajectory is text. Our structural claim remains distinct.
- No multi-hop QA benchmark (FRAMES / MoNaCo / HotpotQA / MuSiQue) confirmed.
- Very recent (Nov 2025) — results still being vetted by the community.

---

### 4b.4 Agentic Reasoning + Mind-Map Agent (Feb 2025)

**Paper** *Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools*, [`arxiv 2502.04644`](https://arxiv.org/abs/2502.04644). Also [ACL 2025 long paper version](https://aclanthology.org/2025.acl-long.1383.pdf).

**Architecture**
- Three peer tool-using agents that a reasoning LLM can invoke during a chain-of-thought:
  - **Web-Search agent** — retrieves external information.
  - **Coding agent** — quantitative computations via code execution.
  - **Mind-Map agent** — constructs a **structured knowledge graph** from the reasoning context, tracks logical relationships, and ensures coherence across long tool-using reasoning chains.
- Deployed on **DeepSeek-R1**. Claims "comparable to OpenAI Deep Research" at release time.

**Why this is the closest spiritual ancestor**
- **The Mind-Map agent is structurally the closest analogue to a query-focused GSW scratchpad.** Both construct a graph per-query to hold the reasoning context. Ours is more specific (entity + verb-phrase + Q&A primitive; trained end-to-end); theirs is more loose (general KG built from reasoning trace; prompt-only).
- Published at ACL 2025 long — visible in the field.

**Where it lacks**
- **Prompt-only** — no training. Ceiling is DeepSeek-R1 base.
- Mind-Map is built from the reasoning trace (self-generated) rather than from retrieved documents. Less grounded.
- No FRAMES, no MoNaCo explicitly confirmed.

---

### 4b.5 Chain-of-Agents / AFM (OPPO PersonalAI Lab, Aug 2025)

**Paper** *Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL*, [`arxiv 2508.13167`](https://arxiv.org/abs/2508.13167).
**GitHub** [github.com/OPPO-PersonalAI/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models)
**Project page** [chain-of-agents-afm.github.io](https://chain-of-agents-afm.github.io/)
**Released variant for MHQA:** [huggingface.co/PersonalAILab/AFM-MHQA-Agent-3B-rl](https://huggingface.co/PersonalAILab/AFM-MHQA-Agent-3B-rl)

**Architecture**
- **Chain-of-Agents (CoA):** distill state-of-the-art multi-agent systems into sequences of agent interactions, then SFT a single model to reproduce the multi-agent trajectory natively.
- **Agentic RL** on top of SFT for additional improvement on verifiable tasks.
- Explicitly targets multi-hop QA with the AFM-MHQA variant — training corpus has **reasoning chains spanning 5–20 hops**, far beyond the 2–3 hop range of standard benchmarks.

**Reported results (32B AFM)**
- GAIA Pass@1: 55.3%
- BrowseComp: 11.1%
- WebWalker: 63.0%
- HLE: 18.0%

**Why directly relevant**
- **AFM-MHQA explicitly targets "5–20 hop" multi-hop QA** — precisely the regime MoNaCo exercises (avg 43 docs/Q).
- Already open-sourced with a 3B RL variant — small-model-friendly.
- Already wired into our pilot grid (adapter #20 in `research_agent/`).

**Where it lacks**
- Internal "multi-agent trajectory" is compressed into a linear sequence in the CoA-trained model — the multi-agent-ness is gone at inference time.
- No structured persistent memory across the trajectory.
- No FRAMES, no MoNaCo confirmed.

---

### Surveyed but out of scope for FRAMES

The following five agentic-with-structured-state systems were considered and excluded from our pilot grid. Brief rationale each — kept here so a reviewer can see we looked at them:

| System | Paper | Why not on the grid |
|---|---|---|
| **Think-on-Graph 3.0** | [2509.21710](https://arxiv.org/abs/2509.21710) | No released checkpoints; prompt-only; heterogeneous graph (chunks+triples+communities) adds complexity without a clean primitive to evaluate. |
| **DeepAgent** (RUC-NLPIR) | [2510.21618](https://arxiv.org/abs/2510.21618) | Tool-use agent over 16k RapidAPIs — different regime from FRAMES' Wikipedia retrieval. Memory-folding compresses trajectory, not document knowledge. |
| **Agentic-KGR** | [2510.09156](https://arxiv.org/html/2510.09156v1) | KG is built *offline* then consumed — not a query-time research agent. Close code-pattern cousin to our linker but wrong use case here. |
| **GraphReader** | [2406.14550](https://arxiv.org/abs/2406.14550) · EMNLP'24 | Per-document graph, doesn't aggregate across documents. Prompt-only on 2024-era models. Ships no checkpoints. |
| **AGENTiGraph** | [2508.02999](https://arxiv.org/html/2508.02999) | Domain chatbot framework, not a multi-hop QA agent. Not evaluated on any QA benchmark. |

Each of these is worth a citation in the related-work section but none are worth an adapter+run budget.

---

### Summary table — 5 agentic research agents with structured state (competitors)

| System | Paper | Structured state type | Training | Benchmarks reported | Closest to our thesis? |
|---|---|---|---|---|---|
| **WebResearcher / IterResearch** | [2509.13309](https://arxiv.org/abs/2509.13309) · [2511.07327](https://arxiv.org/html/2511.07327v1) | Evolving research report (Markovian) | SFT + EAPO RL, Qwen3-30B-A3B | HLE, BrowseComp, GAIA, xBench, SEAL-0 | **#1 — same use case, text-structured state** |
| **Graph-R1** | [2507.21892](https://arxiv.org/html/2507.21892v1) | Knowledge hypergraph (pre-built) | GRPO RL, Qwen2.5-1.5/3/7B | HotpotQA, 2Wiki, MuSiQue, NQ, PopQA, TriviaQA | **#2 — same RL recipe, same scale, closest architecture** |
| **GAM** | [2511.18423](https://arxiv.org/abs/2511.18423) | Page-store + hint memory + research-agent | RL (e2e) | Memory-grounded tasks | **#3 — names the JIT-compilation paradigm explicitly** |
| **Agentic Reasoning (Mind-Map)** | [2502.04644](https://arxiv.org/abs/2502.04644) | Per-query mind-map KG | Prompt-only on DeepSeek-R1 | DR-style | Close spiritual ancestor |
| **Chain-of-Agents / AFM** | [2508.13167](https://arxiv.org/abs/2508.13167) | Implicit multi-agent trajectory | Distillation + agentic RL | GAIA, BrowseComp, WebWalker, HLE, MHQA | Already in the adapter list (#20) |

---

### Key findings

1. **Top 3 anchors for our position:** IterResearch (text-structured report as state), Graph-R1 (graph + GRPO + Qwen2.5-7B), GAM (JIT-compilation paradigm explicit).
2. **None evaluate on FRAMES or MoNaCo.** The dual-benchmark gap holds across the agentic-with-structured-state lane too.
3. **Graph-R1 is the single most aligned competitor by design** — if we run a head-to-head, it's Graph-R1's hypergraph scratchpad vs our query-focused GSW.
4. **IterResearch's EAPO training recipe** (SFT → GSPO + geometric discounting) is a strong alternative to vanilla GRPO for our own training.
5. **Adapter status (research_agent/ pilot grid):**
   - Already wired: Chain-of-Agents/AFM (#20), IterResearch-family via Tongyi DeepResearch (#17).
   - Added in this round: Graph-R1 (`graph_r1`), Agentic Reasoning + Mind-Map (`agentic_reasoning_mindmap`), GAM (`gam`).
   - Surveyed but out of scope: ToG 3.0, DeepAgent, Agentic-KGR, GraphReader, AGENTiGraph (see table above).

---

## Part 4c — Graph-RAG retrieval lineage (context, not competitors)

These are **not agentic** — they're single-shot retrieval systems that build or traverse a knowledge graph. Kept for reference because a reviewer may conflate them with our work, but they don't compete with a multi-turn research agent. One-line summaries each.

| System | Paper | What it does | Pre-built vs per-query | GitHub |
|---|---|---|---|---|
| **HippoRAG 2** | [2405.14831](https://arxiv.org/abs/2405.14831) · ICML'25 | OpenIE triples + Personalized PageRank for associative multi-hop recall | Pre-built | [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) |
| **GraphRAG 2.0** (Microsoft) | [project](https://www.microsoft.com/en-us/research/project/graphrag/) | Entity-relation KG + Leiden community summaries at multiple granularities | Pre-built | [microsoft/graphrag](https://github.com/microsoft/graphrag) |
| **LightRAG** | [2410.05779](https://arxiv.org/abs/2410.05779) · EMNLP'25 Findings | Dual-level (entity + concept) graph indexing, faster than GraphRAG | Pre-built | [lightrag.github.io](https://lightrag.github.io/) |
| **PathRAG** | [2502.14902](https://arxiv.org/abs/2502.14902) | Prune retrieved subgraph to key relational paths; 60/57% win-rates vs GraphRAG/LightRAG | Pre-built | — |
| **StructRAG** | ICLR'25 | LLM dynamically picks structure (table/graph/list/tree) at inference time, fills it, reasons over it | **Per-query** (single-shot, not multi-turn) | — |
| **ToG 1.0 / 2.0** | [2307.07697](https://arxiv.org/html/2307.07697v6) · ICLR'24 · [2407.10805](https://arxiv.org/html/2407.10805v1) · ICLR'25 | Beam-search exploration over an *existing* KG (Freebase/Wikidata); ToG 2.0 hybridizes with text retrieval | External KG | — |
| **KG-Agent** | [2402.11163](https://arxiv.org/html/2402.11163v1) | SFT on LLaMA-7B to execute KG-reasoning programs; 10k samples beats SOTA on Freebase QA | External KG | — |
| **Agentic RAG + KGs** (applied) | [2507.16507](https://arxiv.org/html/2507.16507v1) | Integrated Neo4j + agentic RAG loop, enterprise deployment paper | Pre-built | — |
| **Enterprise KG Crawling** | [2604.14220](https://arxiv.org/abs/2604.14220) | Agent crawls docs and builds KG on the fly; enterprise-doc-specific | Per-query | — |

**Why we're not treating these as competitors**
- All of them are either single-shot retrieval (HippoRAG, GraphRAG, LightRAG, PathRAG, StructRAG) or static-KG lookup (ToG 1.0/2.0, KG-Agent). No multi-turn agent loop with a structured intermediate state that evolves during reasoning.
- StructRAG is the only borderline case — it structurizes at inference time — but without an agent loop it's still single-shot.
- ToG 3.0 is the only member of this family that crosses the agent threshold; it lives in **Part 4b.5** above with the other real competitors.

---

---

## Part 5 — Agent memory systems (conversational / long-horizon interaction)

These are not research-agent papers and don't evaluate on FRAMES / MoNaCo / multi-hop QA. They're the most-cited "structured memory for agents" work, mostly aimed at long-running chatbots or world-model tasks. Included for context so a reviewer doesn't conflate our contribution with this lane.

### 5.1 Mem0 / Mem0ᵍ

- [mem0.ai/research](https://mem0.ai/research)

**Architecture (Mem0, flat variant)**
- **Extraction stage** — on each new turn / doc, an LLM extracts **atomic facts** (`(subject, predicate, object)` or simple propositions) and embeds each fact independently.
- **Storage** — a vector store keyed by fact embeddings, plus a lightweight metadata store for timestamps + source pointers.
- **Retrieval** — query → top-k nearest-neighbor facts by vector similarity. No graph traversal; facts are retrieved *as a bag*.
- **Update policy** — ADD / UPDATE / DELETE / NOOP chosen per extracted fact, with deduplication against the existing store.
- Single-hop LoCoMo F1 ≈ 38.

**Architecture (Mem0ᵍ, graph variant)**
- Same extraction stage but two separate LLM calls:
  - **Entity Extractor** — produces canonical entities (`id`, `name`, `type`, `description`).
  - **Relations Generator** — produces directed labeled edges (`subject_entity → predicate → object_entity`).
- **Storage** — a graph DB (Neo4j-style). Each node is an entity; each edge is a typed relation with metadata.
- **Retrieval** — query → match query entities against the graph → subgraph retrieval (BFS over k-hop neighborhood) → the retrieved subgraph is the context for the answerer LLM.
- **Differentiation vs GSW:** Mem0ᵍ retrieves **subgraphs** and relies on downstream reasoning to interpret them. GSW retrieves **question-answer bundles with entity typing** — the answerer sees pre-structured reasoning chains (Q→A paths with forward + reverse), not just labeled edges. Mem0ᵍ is schema-light; GSW is schema-rich and question-aware.

### 5.2 AriGraph (IJCAI 2025)

- [ijcai.org/proceedings/2025/0002.pdf](https://www.ijcai.org/proceedings/2025/0002.pdf)

**Architecture**
- **Two parallel memory structures** updated jointly from each observation:
  - **Semantic graph** — a knowledge graph of entities + relations built from the agent's observations of the world (e.g. `room_a connects_to room_b`, `chest_1 contains key_3`).
  - **Episodic memory** — time-ordered logs of the agent's actions and their outcomes (used for recall of "what did I try last"). Keyed by step index.
- **Update cycle** — after each action, both structures are updated: new entities inserted into the semantic graph, new episode entries appended to the episodic store.
- **Retrieval at decision time** — the agent's planner queries both memories: semantic graph for *facts about the world*, episodic memory for *what's already been attempted*. Retrieval fuses the two into the actor's context.
- **Domain** — text-adventure / planning environments. Not a QA system — the agent is solving long-horizon exploration tasks (TextWorld, ALFWorld).
- **Differentiation vs GSW:** AriGraph is a world-model built online from agent-environment observations. GSW is a query-time representation built from a static document corpus. Different problem regime entirely.

### 5.3 HyperMem

- [`arxiv 2604.08256`](https://arxiv.org/html/2604.08256v1)

**Architecture**
- Stores **hyperedges** (relations with > 2 arguments), not just binary edges. Natural for conversational memory where a single utterance introduces multiple entities simultaneously ("Alice told Bob about the Paris trip last Tuesday" — 4-ary: teller, listener, topic, time).
- **Three-tier retrieval hierarchy:**
  - **Topic level** — which broad subject is the query about (e.g. "travel plans").
  - **Episode level** — which specific conversation or session within the topic.
  - **Fact level** — which specific hyperedge within the episode.
- Retrieval cascades coarse-to-fine: topic filter → episode filter → fact lookup.
- **Domain** — long-term conversational memory (LoCoMo, MSC).
- **Differentiation vs GSW:** HyperMem targets *conversational* memory with dialogue-turn-level granularity. GSW is document-grounded, not turn-grounded. Orthogonal axis.

### 5.4 MAGMA

- [`arxiv 2601.03236`](https://arxiv.org/html/2601.03236v1)

**Architecture**
- "Multi-graph agentic memory" — the agent maintains **multiple co-existing graphs** over the same content, each capturing a different type of relation:
  - A **causal/mechanistic** graph — edges encode "A causes B" or "A is a precondition of B" rather than "A is associated with B."
  - An **associative** graph — edges encode topical proximity (closer to what classic triple-KGs capture).
  - Optionally a **temporal** graph — edges encode "A precedes B."
- Retrieval picks *which graph* to traverse based on the query type (causal questions route to the causal graph, etc.).
- **Headline argument** — triple-KGs conflate "these things co-occur" with "these things are causally connected." MAGMA separates them explicitly.
- **Differentiation vs GSW:** MAGMA's critique of triple-KGs ("associative proximity, not mechanistic dependency") is a problem GSW sidesteps by construction — GSW's verb-phrase nodes carry directional question-answer pairs that encode *how* entities participate in a relation (subject role, object role, with modifiers). The *direction* of a question and the roles of its participants are mechanistic by definition. **Positive positioning point for GSW:** MAGMA's own critique of the triple-KG lineage does not apply to us.

### 5.5 A-Mem

**Architecture**
- **Zettelkasten-style note-taking agent**, not a pre-built graph. On each new input:
  - LLM drafts a short **note** (one atomic insight).
  - LLM decides which *existing notes* the new one links to (by topic / analogy / contradiction) and writes **typed bidirectional links**.
  - Periodically, the agent runs **consolidation**: revisit notes, update links, merge near-duplicates, surface emergent patterns.
- Notes evolve over time — the memory is not a static extraction, it's an actively-maintained hypertext.
- **Retrieval** — link-walking from seed notes identified by query similarity.
- Reports +27.44 pp on multi-hop (likely LoCoMo).
- **Differentiation vs GSW:** A-Mem's notes are free-form prose; GSW's nodes are schema-typed entities + verb phrases + question-answer pairs. A-Mem optimizes for *open-ended accumulation*; GSW optimizes for *query-answerable structured lookup*. A-Mem's link types are informal; GSW's links are directional questions with verified answer entities.

### 5.6 GraphRAG / Zep / Graphiti

- [falkordb.com/blog/mcp-knowledge-graph-graphiti-falkordb](https://www.falkordb.com/blog/mcp-knowledge-graph-graphiti-falkordb/)

**Architecture (shared pattern)**
- **Offline indexing pass** over the entire corpus builds a **global knowledge graph**:
  - Entity extraction per document → deduplicate across documents.
  - Relation extraction → edges with typed predicates and source pointers.
  - Optional community detection (GraphRAG's Leiden-style clustering) to surface thematic communities across the graph.
  - **Bi-temporal layer (Zep / Graphiti)** — edges carry both `event_time` (when the fact occurred) and `ingestion_time` (when the system learned it), enabling "what did we believe at time T" queries.
- **Retrieval at query time** — combines dense vector search over node/edge summaries with graph traversal for multi-hop expansion.
- Zep hits 71.2% on LongMemEval (GPT-4o).

**Differentiation vs GSW**
- GraphRAG / Zep / Graphiti are **corpus-wide pre-indexed**. Indexing cost scales with the whole corpus; query cost is small.
- GSW's intended regime (query-focused, on-demand) is **query-scoped just-in-time**. Indexing cost ≈ 0; per-query cost scales with the sub-question's retrieval scope.
- Different position on the "when to build structure" axis. Both are defensible; the trade-off is corpus-size × query-rate — pre-indexed wins when corpus is small + query rate is high; on-demand wins when corpus is huge + most queries touch only a small slice.

### 5.7 Memory-R1 (2025)

- [`arxiv 2508.19828`](https://arxiv.org/pdf/2508.19828)

**Architecture**
- Base: **Qwen2.5** (7B-class), fine-tuned with **PPO**.
- The agent's task is not to *use* memory but to **manage** it — decide per incoming fact whether to `ADD`, `UPDATE`, `DELETE`, or `NOOP` against the existing memory store.
- **Reward** — downstream QA accuracy *after* the management actions are applied. The management policy is credit-assigned via end-to-end QA performance.
- Addresses the "lost-in-the-middle" failure mode by letting the model learn to *prune* stale or low-utility memories before they crowd the context window.
- **Differentiation vs GSW:** Memory-R1 is a **memory-management policy** (what to keep, what to drop) over any underlying memory store. GSW is a **memory structure** (schema + retrieval model). They are complementary: you could imagine a Memory-R1-style policy operating over a GSW substrate, deciding which entity subgraphs to retain vs evict.

---

## Part 6 — Where everyone lacks: attack surfaces for our research agent

Consolidated from §2–§5, the specific gaps a query-driven focused-GSW research agent can hit:

1. **No paper uses FRAMES + MoNaCo.** First dual submission is a contribution on coverage alone.
2. **No agentic baseline on MoNaCo.** Any reasonable system clears 61.2 F1 → automatic headline claim.
3. **No structured-representation system on FRAMES.** ASearcher / Context-1 / WebSailor / Search-o1 are all text trajectories. GSW is the unclaimed structured-representation axis.
4. **No small model (<20B) has beaten ASearcher-QwQ-32B.** Opens a clean "smaller model, less compute, beats 32B async RL" headline.
5. **No system uses QDMR as training supervision.** MoNaCo provides it natively; we can distill into the problem composer without synthesizing.
6. **No training pipeline targets list-recall specifically.** MoNaCo's list-question failure mode is documented but unaddressed. Structured aggregation substrate (GSW) is the right tool.
7. **No system has per-entity pruning.** Context-1 prunes *chunks*; we can prune *entity subgraphs* — higher precision, denser recall signal.
8. **No system combines problem-composer + on-demand structured memory + aggregator end-to-end.** The closest is Context-1 (retrieval subagent) but it outsources reasoning to a frontier caller. Ours is the full stack.
9. **No system exposes an interpretable intermediate state.** A failed ASearcher rollout is an opaque text blob; a failed GSW rollout is a visual graph with source spans. Publishable interpretability side-benefit on either FRAMES or MoNaCo.
10. **SMTL's "search more, think less"** is the head-on counter-thesis — our eval must include a SMTL-style ablation (no scratchpad, lighter thinking, more iterations) to hit it directly.

---

## Part 7 — Open questions for scoping

Still to decide before implementation starts:

1. **Benchmark priority.** MoNaCo-first (maximum headroom) vs FRAMES-first (visibility) vs both-simultaneously (ambitious).
2. **Scope of the agent.** Full stack (composer + focused-GSW generator + aggregator) vs retrieval-substrate-only drop-in for an existing agent loop.
3. **Base model.** Qwen3-8B/14B, Qwen2.5-7B/14B, gpt-oss-20B (MoE, matches Context-1), DeepSeek-V3-Distill-7B, or reuse QwQ-32B?
4. **Training regime.** SFT-only from teacher traces vs SFT+GRPO (reuse veRL) vs CISPO-style curriculum RL vs Search-R1-style DeepSeek-R1-zero.
5. **Synthetic data source.** MoNaCo-derived QDMR traces vs Context-1-style Explore→Verify→Distract chains vs WebShaper-style set-theoretic Knowledge Projections.

See also: [Run N+9 section of the plan file](file:///home/yigit/.claude/plans/eager-weaving-canyon.md) for the running design log.

---

## Appendix A — Papers + GitHub links (quick reference)

### FRAMES-side

| System | arXiv | GitHub |
|---|---|---|
| FRAMES (dataset) | [2409.12941](https://arxiv.org/abs/2409.12941) | [HF dataset](https://huggingface.co/datasets/google/frames-benchmark) |
| ASearcher / Beyond Ten Turns | [2508.07976](https://arxiv.org/abs/2508.07976) | [inclusionAI/ASearcher](https://github.com/inclusionAI/ASearcher) |
| AReaL (ASearcher's RL framework) | — | [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) |
| Context-1 (Chroma) | [tech report](https://www.trychroma.com/research/context-1) | [chroma-core/context-1-data-gen](https://github.com/chroma-core/context-1-data-gen) · [weights](https://huggingface.co/chromadb/context-1) |
| Search-o1 | [2501.05366](https://arxiv.org/abs/2501.05366) | [RUC-NLPIR/Search-o1](https://github.com/RUC-NLPIR/Search-o1) |
| Search-R1 | [2503.09516](https://arxiv.org/abs/2503.09516) | [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) |
| WebWalker | [2501.07572](https://arxiv.org/abs/2501.07572) | [Alibaba-NLP/WebAgent](https://github.com/Alibaba-NLP/WebAgent) |
| WebDancer | (see WebAgent) | (see WebAgent) |
| WebSailor | [2507.02592](https://arxiv.org/pdf/2507.02592) | (see WebAgent) |
| WebShaper | [2507.15061](https://arxiv.org/abs/2507.15061) | (see WebAgent) |
| Tongyi DeepResearch | [2510.24701](https://arxiv.org/html/2510.24701v2) | [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) · [weights](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B) |
| EigentSearch-Q+ | [2604.07927](https://arxiv.org/abs/2604.07927) | integrated into Eigent |
| SMTL / Search More Think Less | [2602.22675](https://arxiv.org/pdf/2602.22675) | (HF release referenced) |
| A-RAG | [2602.03442](https://arxiv.org/abs/2602.03442) | [Ayanami0730/arag](https://github.com/Ayanami0730/arag) |
| EVO-RAG | [2505.17391](https://arxiv.org/abs/2505.17391) | (no public URL surfaced) |
| InfoDeepSeek (benchmark) | [2505.15872](https://arxiv.org/abs/2505.15872) | [YunjiaXi/InfoDeepSeek](https://github.com/YunjiaXi/InfoDeepSeek) |
| RAGCap-Bench | [2510.13910](https://arxiv.org/html/2510.13910v1) | — |
| DeepResearch Bench | [2506.11763](https://arxiv.org/abs/2506.11763) | [Ayanami0730/deep_research_bench](https://github.com/Ayanami0730/deep_research_bench) |
| DR-Arena | [2601.10504](https://arxiv.org/html/2601.10504v1) | — |

### MoNaCo-side

| System | arXiv | GitHub |
|---|---|---|
| MoNaCo (dataset) | [2508.11133](https://arxiv.org/abs/2508.11133) | [tomerwolgithub/monaco](https://github.com/tomerwolgithub/monaco) · [HF dataset](https://huggingface.co/datasets/allenai/MoNaCo_Benchmark) · [project page](https://tomerwolgithub.github.io/monaco/) |

*All downstream work on MoNaCo as of April 16, 2026: none.*

### Agentic research agents with structured state (Part 4b — competitors)

| System | arXiv / ref | GitHub | Adapter status |
|---|---|---|---|
| WebResearcher / IterResearch | [2509.13309](https://arxiv.org/abs/2509.13309) · [2511.07327](https://arxiv.org/html/2511.07327v1) | [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) | covered via `tongyi_deep_research` (#17) |
| Graph-R1 | [2507.21892](https://arxiv.org/html/2507.21892v1) | [LHRLAB/Graph-R1](https://github.com/LHRLAB/Graph-R1) | **`graph_r1`** prompt-mode (new) · `graph_r1_trained` alias reserved |
| GAM (General Agentic Memory) | [2511.18423](https://arxiv.org/abs/2511.18423) | [VectorSpaceLab/general-agentic-memory](https://github.com/VectorSpaceLab/general-agentic-memory) | **`gam`** (new) |
| Agentic Reasoning (Mind-Map agent) | [2502.04644](https://arxiv.org/abs/2502.04644) · ACL'25 long | — | **`agentic_reasoning_mindmap`** (new) |
| Chain-of-Agents / AFM | [2508.13167](https://arxiv.org/abs/2508.13167) | [OPPO-PersonalAI/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models) · [AFM-MHQA-3B-rl](https://huggingface.co/PersonalAILab/AFM-MHQA-Agent-3B-rl) | `smtl` / `afm` (#20) |

### Surveyed but out of scope for FRAMES

| System | arXiv / ref | GitHub |
|---|---|---|
| Think-on-Graph 3.0 | [2509.21710](https://arxiv.org/abs/2509.21710) | — |
| DeepAgent (RUC-NLPIR) | [2510.21618](https://arxiv.org/abs/2510.21618) · WWW'26 Oral | [RUC-NLPIR/DeepAgent](https://github.com/RUC-NLPIR/DeepAgent) |
| Agentic-KGR | [2510.09156](https://arxiv.org/html/2510.09156v1) | — |
| GraphReader | [2406.14550](https://arxiv.org/abs/2406.14550) · EMNLP'24 Findings | — |
| AGENTiGraph | [2508.02999](https://arxiv.org/html/2508.02999) | — |

### Graph-RAG retrieval lineage (Part 4c — context only, not competitors)

| System | arXiv / ref | GitHub |
|---|---|---|
| HippoRAG / HippoRAG 2 | [2405.14831](https://arxiv.org/abs/2405.14831) · [ICML'25](https://icml.cc/virtual/2025/poster/45585) | [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) |
| GraphRAG (Microsoft) | [project](https://www.microsoft.com/en-us/research/project/graphrag/) | [microsoft/graphrag](https://github.com/microsoft/graphrag) |
| LightRAG | [2410.05779](https://arxiv.org/abs/2410.05779) | [lightrag.github.io](https://lightrag.github.io/) |
| PathRAG | [2502.14902](https://arxiv.org/abs/2502.14902) | — |
| StructRAG | ICLR'25 | — |
| ToG 1.0 / 2.0 | [2307.07697](https://arxiv.org/html/2307.07697v6) · [2407.10805](https://arxiv.org/html/2407.10805v1) | — |
| KG-Agent | [2402.11163](https://arxiv.org/html/2402.11163v1) | — |
| Agentic RAG + KGs (applied) | [2507.16507](https://arxiv.org/html/2507.16507v1) | — |
| Enterprise KG Crawling | [2604.14220](https://arxiv.org/abs/2604.14220) | — |
| Awesome-GraphRAG (curated list) | — | [DEEP-PolyU/Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG) |

### Structured-memory adjacent (Part 5)

| System | arXiv / ref | GitHub |
|---|---|---|
| Mem0 / Mem0ᵍ | [mem0.ai/research](https://mem0.ai/research) | — |
| AriGraph | [IJCAI 2025](https://www.ijcai.org/proceedings/2025/0002.pdf) | — |
| HyperMem | [2604.08256](https://arxiv.org/html/2604.08256v1) | — |
| MAGMA | [2601.03236](https://arxiv.org/html/2601.03236v1) | — |
| Memory-R1 | [2508.19828](https://arxiv.org/pdf/2508.19828) | — |
| Graph-based Agent Memory (survey) | [2602.05665](https://arxiv.org/html/2602.05665v1) | — |

---

## Appendix B — One-line takeaways

> FRAMES is the 2025–2026 agentic-RAG coliseum; MoNaCo is the 2026 research-agent frontier. Nobody has stepped into both. Nobody has tried a small-model structured-scratchpad approach on either. The first paper that does both with a <20B model + focused-GSW substrate has three clean contribution axes (coverage, structure, compute efficiency) without having to beat ASearcher's absolute number on FRAMES.
