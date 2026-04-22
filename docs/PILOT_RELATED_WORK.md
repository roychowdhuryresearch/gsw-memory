# Pilot-Scope Related Work — 13 Systems on FRAMES

> **Scope.** One compact entry per system that actually has an adapter in [`research_agent/`](../research_agent/) and a row in the 34-cell pilot grid described in [`EXPERIMENTS.md`](../research_agent/docs/EXPERIMENTS.md). This is **not** the full benchmark survey — for the broader landscape (systems we considered but excluded, adjacent memory-for-agents work, every agentic paper that reports FRAMES) see [`RESEARCH_AGENT_LITERATURE_REVIEW.md`](RESEARCH_AGENT_LITERATURE_REVIEW.md). This doc is what you'd hand to a reviewer of the pilot writeup.

---

## Table of contents

1. [The two benchmarks](#the-two-benchmarks)
2. [Systems in the pilot grid](#systems-in-the-pilot-grid)
   - [E1. Vanilla RAG + ReAct (baseline)](#e1-vanilla-rag--react-baseline)
   - [E2. Search-o1](#e2-search-o1)
   - [E3. Search-R1](#e3-search-r1)
   - [E4 / E5. ASearcher](#e4--e5-asearcher)
   - [E6. Context-1 + reasoner](#e6-context-1--reasoner)
   - [E7. Tongyi DeepResearch (incl. IterResearch)](#e7-tongyi-deepresearch-incl-iterresearch)
   - [E8. SMTL / AFM-MHQA (Chain-of-Agents)](#e8-smtl--afm-mhqa-chain-of-agents)
   - [E9. EigentSearch Q+](#e9-eigentsearch-q)
   - [E10. Ours — focused GSW v1](#e10-ours--focused-gsw-v1)
   - [E11. Graph-R1 (prompt-mode)](#e11-graph-r1-prompt-mode)
   - [E12. Agentic Reasoning + Mind-Map](#e12-agentic-reasoning--mind-map)
   - [E13. GAM — General Agentic Memory](#e13-gam--general-agentic-memory)
3. [Thesis positioning — who we actually compete with](#thesis-positioning--who-we-actually-compete-with)
4. [Not in the pilot](#not-in-the-pilot)

---

## The two benchmarks

### FRAMES (Google DeepMind, `arxiv 2409.12941`)

- 824 multi-hop Wikipedia questions, 2–15 docs per question, tests factuality + retrieval + reasoning + temporal disambiguation.
- HuggingFace: [`google/frames-benchmark`](https://huggingface.co/datasets/google/frames-benchmark).
- **Original baselines (Gemini-Pro-1.5):** closed-book 0.40 · BM25 retrieval 0.474 · oracle 0.729 · multi-step retrieval 0.66.
- **Open-source bar (2026):** ASearcher-QwQ-32B = **70.9 Avg@4 / 84.0 Pass@4** (async GRPO, 128-turn rollouts, ~7.6k H800-hours). Chroma Context-1 = **0.87 F1 / 0.96 at 4×** (retrieval subagent, gpt-oss-20B base).
- Our pilot subset: 30 questions stratified by hop count (10 × 2-hop, 10 × 3-hop, 8 × 4–5-hop, 2 × 6+-hop). Judge: GPT-4o LLM-judge with exact-match fallback.

### MoNaCo (AI2 + UPenn + Anthropic, `arxiv 2508.11133`, TACL 2025)

- 1,315 natural multi-hop questions, **avg 43.3 docs/Q** (median 12), 36K distinct Wikipedia pages.
- HuggingFace: [`allenai/MoNaCo_Benchmark`](https://huggingface.co/datasets/allenai/MoNaCo_Benchmark). Repo: [tomerwolgithub/monaco](https://github.com/tomerwolgithub/monaco).
- Ships **QDMR decomposition annotations** per question.
- **All 15 published baselines are zero-shot closed-book LLMs** (top F1: o3 @ 61.2%, GPT-5 @ 60.1%, Claude-4-Opus @ 55.0%, Qwen-2.5-72B @ 42.9%). **Zero downstream agentic papers** as of April 2026.
- **Not in this pilot** — targeted as the next-phase benchmark once the FRAMES pilot validates the agent.

---

## Systems in the pilot grid

### E1. Vanilla RAG + ReAct (baseline)

- **Adapter**: [`vanilla_rag_react.py`](../research_agent/src/research_agent/adapters/vanilla_rag_react.py) · `system_id = "vanilla_rag_react"`
- **Paper**: none — baseline built by us from scratch.
- **What it is.** Minimal 2-tool ReAct loop (`search`, `read`) over BM25. No decomposition, no structured state, no scratchpad. This is the reference point every other adapter is measured against at the same base model.
- **Grid rows**: GPT-5 · gpt-oss-20B · Qwen3.5-9B.
- **Why in the pilot.** **Frontier ceiling + all-small control.** Any other system that doesn't beat this at the same base has a contribution problem.

### E2. Search-o1

- **Adapter**: [`search_o1.py`](../research_agent/src/research_agent/adapters/search_o1.py) · `system_id = "search_o1"`
- **Paper**: *Search-o1: Agentic Search-Enhanced Large Reasoning Models*, [`arxiv 2501.05366`](https://arxiv.org/abs/2501.05366) · EMNLP 2025.
- **GitHub**: [RUC-NLPIR/Search-o1](https://github.com/RUC-NLPIR/Search-o1)
- **What it is.** Wraps a long-reasoning model (QwQ-32B target) with special-token search tags (`<|begin_search_query|>…<|end_search_query|>`) and a Reason-in-Documents distillation module that compresses retrieved chunks before re-injecting them into the reasoning stream.
- **Reported FRAMES**: 63.6 Avg@4 on QwQ-32B.
- **Grid rows**: QwQ-32B · gpt-oss-20B · Qwen3.5-9B.
- **Why in the pilot.** Prompt-only agentic wrapper that gets mid-tier FRAMES. Tests whether the search-tag-and-distill pattern portable across model scales.

### E3. Search-R1

- **Adapter**: [`search_r1.py`](../research_agent/src/research_agent/adapters/search_r1.py) · `system_id = "search_r1"`
- **Paper**: *Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*, [`arxiv 2503.09516`](https://arxiv.org/abs/2503.09516) · COLM 2025.
- **GitHub**: [PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) (built on veRL + RAGEN)
- **What it is.** PPO-trained small LLM (Qwen2.5-3B / 7B, Llama 3.2-3B) that emits inline tags `<think>/<search>/<information>/<answer>`. Training reward = rule-based outcome.
- **Published benchmarks**: NaturalQuestions, TriviaQA, HotpotQA. **FRAMES not reported** by the paper.
- **Grid rows**: Qwen2.5-7B (paper's baseline) · gpt-oss-20B · Qwen3.5-4B (tiny sanity).
- **Why in the pilot.** Reuses the same veRL infra we plan for our own training; prompt-mode run here establishes the "inline-tag ReAct, untrained base" floor.

### E4 / E5. ASearcher

- **Adapter**: [`asearcher.py`](../research_agent/src/research_agent/adapters/asearcher.py) · `system_id = "asearcher_prompt"` and `asearcher_trained`
- **Paper**: *Beyond Ten Turns: Unlocking Long-Horizon Agentic Search with Large-Scale Asynchronous RL*, [`arxiv 2508.07976`](https://arxiv.org/abs/2508.07976).
- **GitHub**: [inclusionAI/ASearcher](https://github.com/inclusionAI/ASearcher) · RL framework: [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL)
- **What it is.** Current open-source FRAMES SOTA. GRPO with fully async rollouts up to 128 turns. Training-data recipe: 14,107 seed Qs → 25,624 via Injection (+ Wikipedia facts) + Fuzzing (blur details). Pure text trajectories; **no structured intermediate state.**
- **Reported FRAMES**: **70.9 Avg@4 / 84.0 Pass@4** on ASearcher-Web-QwQ-32B. 7.6k H800-hours.
- **Grid rows (E4)**: QwQ-32B prompt-mode · gpt-oss-20B · Qwen3.5-9B.
- **Grid rows (E5)**: trained ASearcher-Web-7B · ASearcher-Web-14B (both on HuggingFace).
- **Why in the pilot.** The single most important competitor to position against on FRAMES. E4 isolates the prompt; E5 measures what the RL training actually bought.

### E6. Context-1 + reasoner

- **Adapter**: [`context1.py`](../research_agent/src/research_agent/adapters/context1.py) · `system_id = "context1_plus_reasoner"`
- **Paper / tech report**: [trychroma.com/research/context-1](https://www.trychroma.com/research/context-1) (March 2026).
- **GitHub (data-gen)**: [chroma-core/context-1-data-gen](https://github.com/chroma-core/context-1-data-gen) · Weights: [chromadb/context-1](https://huggingface.co/chromadb/context-1)
- **What it is.** A **retrieval subagent**, not a full-stack reasoner. gpt-oss-20B base + LoRA, SFT + CISPO RL, 4 reward components (F1, trajectory recall, answer-bonus, prune penalties). Self-editing context via `search / read / prune / done` tools. Designed to be paired with a frontier reasoner that reads the kept chunks and answers.
- **Reported FRAMES**: 0.87 F1 single-run / 0.96 at 4× parallel; 10× faster + 25× cheaper than GPT-5.4 frontier.
- **Grid rows**: Context-1 + GPT-5 · Context-1 + gpt-oss-20B · Context-1 + Qwen3.5-9B. All three swap only the downstream reasoner; the retrieval subagent is fixed.
- **Why in the pilot.** Tests the "separate retrieval subagent" design paradigm. Compare to our E10 (single model doing everything) at same total compute.

### E7. Tongyi DeepResearch (incl. IterResearch)

- **Adapter**: [`tongyi_deep_research.py`](../research_agent/src/research_agent/adapters/tongyi_deep_research.py) · `system_id = "tongyi_deep_research"`
- **Papers**: Tongyi DR tech report [`arxiv 2510.24701`](https://arxiv.org/html/2510.24701v2). Underlying paradigm: WebResearcher [`arxiv 2509.13309`](https://arxiv.org/abs/2509.13309) / IterResearch [`arxiv 2511.07327`](https://arxiv.org/html/2511.07327v1).
- **GitHub**: [Alibaba-NLP/DeepResearch](https://github.com/Alibaba-NLP/DeepResearch) · Weights: [Tongyi-DeepResearch-30B-A3B](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B)
- **What it is.** 30.5B total / 3.3B active MoE (from Qwen3-30B-A3B base), purpose-built for long-horizon research. Tools: Search, Visit, Python, Scholar, File Parser. Two inference paradigms: **ReAct mode** (our adapter) and **IterResearch "heavy" mode** where the Markovian state = (question, evolving report, last action/obs) and the agent resets context each round using the report as compressed memory.
- **Reported**: FRAMES in tech-report (exact number not extracted). IterResearch v2: +14.5pp avg across HLE / BrowseComp / GAIA / xBench / SEAL-0 vs prior open-source.
- **Grid rows**: Tongyi-30B-A3B (single row, no swap).
- **Why in the pilot.** **Most-aligned text-structured-state system.** IterResearch's "evolving report" is the text-rival to our GSW scratchpad. If Tongyi's single row is below our E10, structured > text state at small scale.

> Note: Our adapter currently runs Tongyi in ReAct mode. IterResearch-heavy-mode is a follow-up adapter if the plain run's results motivate it.

### E8. SMTL / AFM-MHQA (Chain-of-Agents)

- **Adapter**: [`smtl.py`](../research_agent/src/research_agent/adapters/smtl.py) · `system_id = "smtl"`
- **Papers**: *Search More, Think Less* (SMTL-30B), [`arxiv 2602.22675`](https://arxiv.org/pdf/2602.22675) · *Chain-of-Agents: End-to-End Agent Foundation Models via Multi-Agent Distillation and Agentic RL*, [`arxiv 2508.13167`](https://arxiv.org/abs/2508.13167).
- **GitHub**: [OPPO-PersonalAI/Agent_Foundation_Models](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models) · MHQA variant: [PersonalAILab/AFM-MHQA-Agent-3B-rl](https://huggingface.co/PersonalAILab/AFM-MHQA-Agent-3B-rl)
- **What it is.** 6-function prompt (`<think>/<plan>/<wiki_search>/<observation>/<reflection>/<answer>`). SMTL's thesis: "lighter per-step thinking + more exploration iterations beats deep deliberation." AFM distills multi-agent trajectories into a single end-to-end model, with chains spanning 5–20 hops in training data.
- **Reported (AFM-32B)**: GAIA 55.3% Pass@1 · BrowseComp 11.1% · WebWalker 63.0% · HLE 18.0%.
- **Grid rows**: SMTL-30B · gpt-oss-20B · Qwen3.5-9B.
- **Why in the pilot.** SMTL is the head-on philosophical counter-thesis to "structured state helps." AFM-MHQA specifically targets 5–20 hop QA, which overlaps FRAMES' harder bucket.

### E9. EigentSearch Q+

- **Adapter**: [`eigent_search_q_plus.py`](../research_agent/src/research_agent/adapters/eigent_search_q_plus.py) · `system_id = "eigent_search_q_plus"`
- **Paper**: *EigentSearch-Q+: Enhancing Deep Research Agents with Structured Reasoning Tools*, [`arxiv 2604.07927`](https://arxiv.org/abs/2604.07927) (2026).
- **What it is.** Tools-layer contribution on top of a frontier model (GPT-4.1 / GPT-5.1 / Minimax M2.5). Exposes 4 query-and-evidence-processing tools — `plan_next_searches`, `select_query_and_search`, `extract_relevant_details`, `analyze_search_progress` — that make the browser sub-agent's planning + progress-monitoring + extraction explicit.
- **Reported**: +0.6–3.8 pp absolute across SimpleQA-Verified / FRAMES / WebWalkerQA / xBench-DeepSearch (benchmark-size-weighted average).
- **Grid rows**: GPT-5 · gpt-oss-20B · Qwen3.5-9B.
- **Why in the pilot.** Closest prior art to "graph-manipulation tools" — they expose structured-reasoning tools at inference time, no training. Direct comparison point for our tools surface.

### E10. Ours — focused GSW v1

- **Adapter**: [`ours_gsw_v1.py`](../research_agent/src/research_agent/adapters/ours_gsw_v1.py) · `system_id = "ours_gsw_v1"`
- **Source**: This project. First instantiation of the query-driven research agent thesis.
- **What it is.** 5-stage pipeline per question: (1) problem composer LLM decomposes into ≤ 6 sub-Qs with entity focus + hop type; (2) per-sub-Q BM25 retrieval; (3) focused GSW extraction — LLM emits entities + binary (subject, verb, object) triples with `evidence_chunk_ids`; (4) per-sub-Q answer from triples; (5) aggregator combines.
- **Grid rows**: gpt-oss-20B (primary) · optional Qwen3.5-9B.
- **Headline test**: E10-gpt-oss-20B vs E1-gpt-oss-20B at equal base. If we win by ≥ 5 pp, the structured scratchpad is doing real work. If ±2 pp, we're burning tokens for no gain.

### E11. Graph-R1 (prompt-mode)

- **Adapter**: [`graph_r1.py`](../research_agent/src/research_agent/adapters/graph_r1.py) · `system_id = "graph_r1"` (+ reserved alias `graph_r1_trained`)
- **Paper**: *Graph-R1: Towards Agentic GraphRAG Framework via End-to-End RL*, [`arxiv 2507.21892`](https://arxiv.org/abs/2507.21892).
- **GitHub**: [LHRLAB/Graph-R1](https://github.com/LHRLAB/Graph-R1)
- **What it is (paper).** Multi-turn agent + knowledge hypergraph + GRPO. `<think>/<query>/<retrieve>/<answer>` loop. Base model = Qwen2.5-1.5B/3B/7B. Reward = format reward × F1. Paper's result: 57.82 avg F1 on HotpotQA / 2Wiki / MuSiQue / NQ / PopQA / TriviaQA, beats Search-R1 at same scale by ~11 pp.
- **What our adapter is.** **Prompt-mode reimplementation** of the paper's loop. Upstream ships training code + TeraBox-hosted hypergraph artifacts but **no HuggingFace checkpoints** — their headline 7B numbers are user-trained. Two pragmatic deviations: (a) hypergraph is built **on-the-fly per retrieval** via a cheap second LLM pass extracting `(subject, predicate, object, context)` tuples; (b) BM25 retrieval via our shared retriever.
- **Grid rows (E11 prompt-mode)**: Qwen2.5-7B-Instruct (paper's base) · gpt-oss-20B · Qwen3.5-9B.
- **Grid row (E11b trained)**: self-trained Graph-R1-Qwen2.5-7B on 2WikiMultiHopQA using upstream's GRPO recipe. **Gated on task #30** — LHRLAB ships zero HF checkpoints, so the operator has to clone the repo, download TeraBox data, and run `run_grpo.sh` on 4×48GB GPUs (~4–12h). `graph_r1_trained` adapter alias + `scripts/setup_graph_r1.sh` + `scripts/serve_graph_r1_trained.sh` are in place.
- **Why in the pilot.** **Single closest competitor to our GSW thesis by design shape** — multi-turn agent + graph intermediate state + RL-trainable on multi-hop QA. E11 (prompt-mode) vs E10 tests the loop design; E11b (trained) vs E11 isolates the contribution of their GRPO training on top of the prompt structure.

### E12. Agentic Reasoning + Mind-Map

- **Adapter**: [`agentic_reasoning_mindmap.py`](../research_agent/src/research_agent/adapters/agentic_reasoning_mindmap.py) · `system_id = "agentic_reasoning_mindmap"`
- **Paper**: *Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools*, [`arxiv 2502.04644`](https://arxiv.org/abs/2502.04644) · [ACL 2025 long](https://aclanthology.org/2025.acl-long.1383.pdf).
- **What it is (paper).** ReAct-style agent with three peer tools: Web-Search, Coding, and a **Mind-Map agent** that maintains a persistent structured KG across the reasoning chain. Deployed on DeepSeek-R1; claims comparable to OpenAI Deep Research at release.
- **What our adapter is.** Same tool-call shape but (a) Web-Search replaced with our BM25 retriever; (b) Coding dropped (FRAMES rarely needs arithmetic); (c) mind-map implemented as per-question `List[Tuple[subject, relation, object]]` with `mind_map_update(edges)` and `mind_map_query(focus)` tools.
- **Grid rows**: GPT-5 (closest frontier match to DeepSeek-R1) · gpt-oss-20B · Qwen3.5-9B.
- **Why in the pilot.** **Closest prompt-only spiritual ancestor to a query-focused GSW scratchpad.** A clean ablation against E1 at same model isolates the "persistent structured scratchpad" contribution without decomposition, without training.

### E13. GAM — General Agentic Memory

- **Adapter**: [`gam.py`](../research_agent/src/research_agent/adapters/gam.py) · `system_id = "gam"`
- **Paper**: *General Agentic Memory Via Deep Research*, [`arxiv 2511.18423`](https://arxiv.org/abs/2511.18423) (Nov 2025).
- **GitHub**: [VectorSpaceLab/general-agentic-memory](https://github.com/VectorSpaceLab/general-agentic-memory)
- **What it is (paper).** Names the **"JIT compilation"** paradigm explicitly. Duo design: (1) **Memorizer** — offline pass producing only lightweight hints over a raw page-store; (2) **Researcher** — online multi-turn agent that uses hints to target its fetches. Maximize storage fidelity, offload complexity to runtime.
- **What our adapter is.** Memorizer builds one hint per chunk (title + 160-char digest) at adapter init (~2k chunks, no LLM calls). Researcher uses two tools: cheap `browse_hints(query)` (title + digest only) and expensive `fetch_page(chunk_id)` (full article from page-store).
- **Grid rows**: GPT-5 · gpt-oss-20B · Qwen3.5-9B.
- **Why in the pilot.** GAM is the paper that names the paradigm our whole pivot is about. Tests whether the JIT-compile pattern — hints + on-demand fetch — holds up on FRAMES. Direct browse/fetch ratio is a measurable proxy for "did the agent internalize the cost asymmetry the paradigm claims."

---

## Thesis positioning — who we actually compete with

Among the 12 non-baseline cells:

**Direct competitors (must cite, must ablate against):**
- **E5 (ASearcher-trained)** — open-source FRAMES SOTA by a wide margin. Sets the "what's possible without structured state" ceiling. Our pitch: match or beat at smaller compute.
- **E7 (Tongyi DeepResearch / IterResearch)** — most-aligned text-structured-state competitor. "Evolving report" is the alternative thesis to our graph-structured scratchpad.
- **E11 (Graph-R1)** — single closest paper by design shape: multi-turn agent + graph + RL. Our differentiation: QA-pair primitives (not triples), query-focused (not pre-built hypergraph), QDMR-trainable.

**Head-on spiritual ancestors (prompt-only versions of our thesis):**
- **E12 (Mind-Map)** — persistent per-question KG scratchpad. No training. A clean isolation of "does structured state alone help?"
- **E13 (GAM)** — names the JIT-compilation paradigm our pivot is built around.

**Paradigm foils (our thesis explicitly disagrees):**
- **E8 (SMTL)** — "search more, think less." Claims light per-step thinking + many iterations beats structured planning. We must beat SMTL at same base to defend the structured-state contribution.
- **E6 (Context-1)** — "separate retrieval subagent + frontier reasoner." Alternative system decomposition to our single-model query-focused GSW.

**Sanity baselines:**
- **E1 (Vanilla RAG+ReAct)** — floor.
- **E2 (Search-o1), E3 (Search-R1), E4 (ASearcher-prompt)** — inline-tag ReAct variants without structured state. Establish the "agentic but unstructured" band.
- **E9 (EigentSearch Q+)** — tools-layer-only contribution. Measures the bump from externalized reasoning tools without any scratchpad.

---

## Not in the pilot

Systems we considered and excluded from this pilot (brief rationale in [`RESEARCH_AGENT_LITERATURE_REVIEW.md §4b — Surveyed but out of scope`](RESEARCH_AGENT_LITERATURE_REVIEW.md)):

- **ToG 3.0** — no released checkpoints; prompt-only; heterogeneous graph too complex.
- **DeepAgent** — 16k-RapidAPI tool-use regime, wrong use case.
- **Agentic-KGR** — offline KG construction, not a runtime agent.
- **GraphReader** — per-document graph, doesn't aggregate across docs.
- **AGENTiGraph** — chatbot framework.
- **HippoRAG 2 / GraphRAG / LightRAG / PathRAG / StructRAG** — single-shot graph-RAG retrieval, not multi-turn agents.
- **Mem0 / AriGraph / HyperMem / MAGMA / A-Mem / Zep / Memory-R1** — conversational / long-horizon-interaction memory, not research-agent memory.

All of these get a citation in the broader related-work section but no adapter + run budget.

---

## Cross-references

- **Full experiment spec (one section per cell, exact run commands)**: [`research_agent/docs/EXPERIMENTS.md`](../research_agent/docs/EXPERIMENTS.md)
- **Full benchmark + landscape survey**: [`RESEARCH_AGENT_LITERATURE_REVIEW.md`](RESEARCH_AGENT_LITERATURE_REVIEW.md)
- **Current pilot failure analysis**: [`FRAMES_PILOT_FAILURE_ANALYSIS.md`](FRAMES_PILOT_FAILURE_ANALYSIS.md)
- **Running iteration log (plan file, across runs)**: `/home/yigit/.claude/plans/eager-weaving-canyon.md`
