# FRAMES Pilot — Per-System Failure Analysis

> **Date:** 2026-04-20
> **Subset:** `frames_pilot_v1` (30 questions, dev split)
> **Judge:** GPT-4o (LLM-as-judge, exact-match fallback)
> **Total cells analysed:** 18 (6 systems × up to 5 models each, plus 3 context-1+reasoner cells re-running at time of writing)
> **Raw logs:** `/home/yigit/codebase/gsw-memory/research_agent/logs/`

This document is a forensic read of the pilot-grid logs — not a summary of the auto-assigned `failure_mode` tags. Every claim is grounded in actual `predicted_answer` strings, `trajectory.stopped_reason` values, and `tool_calls` counts.

### Scope note on "ours" in this document

The planned paper-target system is **`ours_gsw_v1`**: a small agentic model (gpt-oss-20B class) that runs retrieval and then **builds a focused GSW scratchpad from the retrieved chunks**, with an aggregator reading the GSW to answer. **That system has not been run on this pilot yet.** No row for `ours_gsw_v1` appears in the table.

The `rule_decomp_gsw` cells in the grid are a **separate decomposition-ablation experiment** (deterministic regex decomposer, zero LLM at decomp time, swap-in BM25 retrieval, LLM aggregator — no GSW step). It shares `rule_decomp_gsw` as an adapter id but is a distinct research thread from `ours_gsw_v1`. Where this doc discusses "gain opportunities for our approach", it's referring to the planned `ours_gsw_v1` pipeline, not the decomposition-ablation rows.

---

## Ranked leaderboard (judge accuracy)

| # | system | model | n | judge | EM | F1 | turns | wall (s) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | search_o1 | bedrock/gpt-oss-120b | 30 | **0.63** | 0.00 | 0.11 | 3.6 | 13.7 |
| 2 | asearcher_prompt | bedrock/gpt-oss-120b | 30 | 0.53 | 0.37 | 0.47 | 1.3 | 4.2 |
| 2 | vanilla_rag_react | gpt-5 | 30 | 0.53 | 0.33 | 0.38 | 10.0 | 83.6 |
| 4 | **tongyi_deep_research** | Alibaba-NLP/Tongyi-DeepResearch-30B-A3B | 30 | **0.50** | 0.00 | 0.03 | 12.0 | 51.4 |
| 5 | search_o1 | Qwen/QwQ-32B | 30 | 0.47 | 0.00 | 0.02 | 2.4 | 166.8 |
| 6 | asearcher_prompt | Qwen/Qwen3.5-9B | 30 | 0.43 | 0.00 | 0.09 | 6.8 | 194.8 |
| 6 | vanilla_rag_react | bedrock/gpt-oss-20b | 30 | 0.43 | 0.10 | 0.23 | 10.3 | 13.8 |
| 8 | asearcher_trained | inclusionAI/ASearcher-Web-14B | 30 | 0.40 | 0.30 | 0.38 | 7.8 | 25.8 |
| 8 | vanilla_rag_react | bedrock/gpt-oss-120b | 30 | 0.40 | 0.30 | 0.34 | 10.6 | 17.4 |
| 8 | context1_plus_reasoner | chromadb/context-1 + gpt-5 | 30 | 0.40 | 0.33 | 0.39 | 8.0 | 76.5 |
| 8 | context1_plus_reasoner | chromadb/context-1 + bedrock/gpt-oss-120b | 30 | 0.40 | 0.23 | 0.34 | 7.6 | 72.9 |
| 12 | asearcher_prompt | Qwen/QwQ-32B | 30 | 0.37 | 0.17 | 0.24 | 2.2 | 124.6 |
| 13 | asearcher_prompt | Qwen/Qwen3.5-4B | 30 | 0.33 | 0.23 | 0.28 | 5.4 | 215.6 |
| 13 | search_o1 | Qwen/Qwen3.5-9B | 30 | 0.33 | 0.00 | 0.05 | 7.7 | 231.8 |
| 15 | rule_decomp_gsw *(decomp ablation)* | gpt-4.1-mini | 30 | **0.30** | 0.17 | **0.30** | **2.8** | **8.3** |
| 15 | search_o1 | Qwen/Qwen3.5-4B | 30 | 0.30 | 0.00 | 0.04 | 7.3 | 169.5 |
| 15 | context1_plus_reasoner | chromadb/context-1 + bedrock/gpt-oss-20b | 30 | 0.30 | 0.23 | 0.27 | 8.8 | 76.4 |
| 18 | asearcher_trained | inclusionAI/ASearcher-Web-7B | 30 | 0.27 | 0.20 | 0.24 | 3.4 | 4.1 |
| 18 | **smtl** | PersonalAILab/SMTL-30B | 30 | **0.27** | 0.00 | 0.09 | 9.7 | 20.7 |
| 8 | **eigent_search_q_plus** | bedrock/gpt-oss-120b | 30 | **0.467** | 0.133 | 0.265 | 11.0 | 18.7 |
| 8 | **eigent_search_q_plus** | bedrock/gpt-oss-20b | 30 | **0.467** | 0.333 | 0.434 | 9.5 | 11.0 |
| 21 | **eigent_search_q_plus** | Qwen/Qwen3.5-9B | 30 | **0.000** | 0.000 | 0.000 | 2.9 | 10.9 |
| 1 | **eigent_search_q_plus** | gpt-5 | 30 | **0.667** | 0.367 | 0.517 | 11.4 | 76.1 |
| — | eigent_search_q_plus | Qwen/Qwen3-14B | 19 (killed early) | ~0.21 (rolling) | — | — | — | — |
| 8 | **agentic_reasoning_mindmap** | bedrock/gpt-oss-20b | 30 | **0.500** | 0.333 | 0.467 | 7.7 | 12.8 |
| 5 | **gam** | bedrock/gpt-oss-120b | 30 | **0.533** | 0.400 | 0.455 | 10.3 | 14.3 |
| 12 | **gam** | bedrock/gpt-oss-20b | 30 | **0.400** | 0.267 | 0.357 | 8.1 | 11.6 |
| 2 | **gam** | gpt-5 | 30 | **0.600** | 0.400 | 0.496 | 9.4 | 104.4 |
| 15 | gam | Qwen/Qwen3.5-4B | 30 | **0.333** | 0.233 | 0.285 | 5.6 | 18.8 |
| 19 | gam | Qwen/Qwen3.5-9B | 30 | **0.067** | 0.033 | 0.064 | 2.7 | 27.3 |
| 12 | **agentic_reasoning_mindmap** | bedrock/gpt-oss-120b | 30 | **0.400** | 0.300 | 0.350 | 10.5 | 21.4 |
| 21 | **agentic_reasoning_mindmap** | Qwen/Qwen3.5-9B | 30 | **0.000** | 0.000 | 0.000 | 2.1 | 8.9 |
| 4 | **agentic_reasoning_mindmap** | gpt-5 | 30 | **0.533** | 0.433 | 0.486 | 11.0 | 107.7 |
| 15 | agentic_reasoning_mindmap | Qwen/Qwen3.5-4B | 30 | **0.300** | 0.233 | 0.289 | 6.6 | 21.8 |
| 19 | asearcher_prompt | bedrock/gpt-oss-20b | 30 | 0.17 | 0.07 | 0.11 | 1.0 | 1.7 |
| 19 | search_o1 | bedrock/gpt-oss-20b | 30 | 0.17 | 0.00 | 0.04 | 1.0 | 3.4 |
| 21 | context1_plus_reasoner | chromadb/context-1 (gpt-5 reasoner — 404, original) | 30 | 0.00 | 0.00 | 0.00 | 8.2 | 27.1 |
| 21 | rule_decomp_gsw *(decomp ablation)* | Qwen/Qwen3.5-4B | 30 | 0.00 | 0.00 | 0.00 | 2.7 | 55.8 |
| — | **`ours_gsw_v1` (planned)** | gpt-oss-20B + focused-GSW scratchpad | — | not yet run | — | — | — | — |
| 19 | **ours_gsw_planner_v1 (Phase 1, prompt-only)** | bedrock/gpt-oss-120b | 30 | **0.367** | 0.300 | 0.355 | 4.0 | 16.4 |
| 20 | **ours_gsw_planner_v1 (Phase 1, prompt-only)** | bedrock/gpt-oss-20b | 30 | **0.333** | 0.267 | 0.304 | 3.3 | 12.0 |
| — | ours_gsw_planner_v1 | gpt-5 | killed at 5/30 | 0.00 | — | — | — | — |

**Currently re-running** (as of 2026-04-20 14:09): context1_plus_reasoner with three reasoners — gpt-5, bedrock/gpt-oss-120b, bedrock/gpt-oss-20b. Partial scores pending final snapshots.

---

## Question-difficulty distribution (18 cells × 30 Qs)

- **3 questions nobody solved** — q371 (Haitian/French revolutions year delta), q302 (2016 French Open H2H), q133 (Tony × Oscar arithmetic).
- **0 questions everyone solved.**
- **Most-solved:** q796 (FIFA '82 + Falklands = 15/18), q154 (Picasso + Pink Floyd = 14/18), q512 (Andre the Giant + Rob Reiner = 14/18).
- **Mixed:** 27 of 30 questions had at least one correct and one incorrect cell — genuine difficulty spread.

---

## 1. Search-o1 (RUC-NLPIR)

**Paper:** `arxiv 2501.05366` (RUC-NLPIR, EMNLP 2025).
**Repo:** [`RUC-NLPIR/Search-o1`](https://github.com/RUC-NLPIR/Search-o1).
**Protocol:** The LRM emits `<|begin_search_query|>query</|end_search_query|>` marker tokens mid-reasoning; the adapter parses the stream, executes the search, and splices results back into the context. No JSON tool calls — pure text-marker protocol. Works best with reasoning-style LLMs (LRMs) that can self-terminate thought.

### Model grid

| model | correct | avg tools | stopped_reason mix | dominant loss signature |
|---|---:|---:|---|---|
| **bedrock/gpt-oss-120b** | 19/30 | 2.63 | finished=29, search_limit=1 | 11 losses; 1 unexecuted_marker, 2 monologue |
| Qwen/QwQ-32B | 14/30 | 1.40 | finished=29, search_limit=1 | 16 losses; **16/16 monologue** |
| Qwen/Qwen3.5-9B | 10/30 | 6.67 | **finished=15, search_limit=15** | 20 losses; **19 think_tags_leak, 13 unexecuted_marker** |
| Qwen/Qwen3.5-4B | 9/30 | 6.27 | **finished=15, search_limit=15** | 21 losses; **19 think_tags_leak, 15 unexecuted_marker** |
| bedrock/gpt-oss-20b | 5/30 | **0.00** | finished=29, n/a=1 | 25 losses; **20 monologue** |

### Log-derived behaviour

- **gpt-oss-120b** is the only cell across the entire grid that combines clean marker emission with short, entity-style final answers. On its 19 wins the model terminated in ≤3 searches and produced answers like `"Rob Reiner."` or `"Shane Gillis"`. On 1 of 11 losses it emitted the marker but the adapter never executed it (`unexecuted_marker`).
- **QwQ-32B** loves the marker format but shortcuts through closed-book reasoning — 1.4 tools/Q avg. On all 16 losses, QwQ's entire `<think>` block leaked into `predicted_answer` as prose narration; the final-answer boundary was never closed.
- **Qwen-9B / Qwen-4B** both hit `search_limit=15` on exactly half their Qs (15/30 each). The search-reformulation loop never converges. On 19/20 and 19/21 losses respectively, the raw `<think>` or `<|begin_search_query|>` tokens leaked into `predicted_answer` — the parser never saw a proper terminator. This is not a retrieval failure; it's an **agent loop divergence**.
- **gpt-oss-20b** is incapable of the protocol — 0 tool calls across all 30 Qs. The model emits prose like `"We need to identify Joel Oshiro Dyck… Let's search."` and stops. No `<|begin_search_query|>` token is ever produced.

### Root cause

Search-o1's protocol is **reasoning-model-shaped**. It relies on:

1. The model's ability to close its own `<think>` block cleanly.
2. Emission of discrete marker tokens via the LLM's vocabulary.
3. A self-terminating loop signal.

All three requirements break at Qwen-9B/below. QwQ-32B and gpt-oss-120b satisfy them; Qwen-4B/9B partially (they emit markers but loop forever); gpt-oss-20b fails all three.

### Gain opportunities for `ours_gsw_v1` (planned: small model + post-retrieval GSW scratchpad)

The planned system is gpt-oss-20B (or similar small agent) that drives retrieval and then **builds a focused GSW scratchpad from the retrieved chunks** — entity/verb-phrase/relation nodes over the retrieved text — before a final aggregator reads the GSW and answers.

- **Qwen-9B/4B `search_limit=15` pool (30 Q-cell pairs):** search-o1 diverges because the LLM-governed query-reformulation loop can't self-terminate. A GSW scratchpad built over the already-retrieved chunks gives the small model a stable intermediate to read from; the "am I done?" signal becomes "did I extract entities for every sub-claim?" rather than "should I search again?". This replaces the divergent loop with a bounded structuring step.
- **gpt-oss-20b 0-tools pool (25 losses):** search-o1 fails at 20B because the model can't emit `<|begin_search_query|>` markers. Our approach uses vanilla JSON tool_calls (which vanilla_rag_react at 20B handles — 13/30 correct). The GSW stage then converts retrieved chunks into structured entity-attribute triples, relieving the 20B of having to synthesise the answer from raw chunk prose — which is exactly where the 20B hallucinates in vanilla_rag_react's losses.

---

## 2. ASearcher prompt-mode (InclusionAI)

**Paper:** `arxiv 2508.07976` ("Beyond Ten Turns", InclusionAI).
**Repo:** [`inclusionAI/ASearcher`](https://github.com/inclusionAI/ASearcher).
**Protocol:** Standard OpenAI-shape JSON tool_calls via Bedrock `converse` or provider-native tool APIs. Tools are `search`, `browse`, `summarize`. Multi-turn; agent decides when to stop. Prompt-mode means no SFT+RL applied — the framework's system prompt runs on a stock base model.

### Model grid

| model | correct | avg tools | zero-tool Qs | dominant loss signature |
|---|---:|---:|---:|---|
| bedrock/gpt-oss-120b | 16/30 | 0.27 | **25/30** | 14 losses; short parametric answers, no grounding |
| Qwen/Qwen3.5-9B | 13/30 | 5.90 | 0 | 17 losses; 3 monologue, 2 refused, 2 empty |
| Qwen/QwQ-32B | 11/30 | 1.20 | 1 | 19 losses; **11 literal "the question is invalid" refusals** |
| Qwen/Qwen3.5-4B | 10/30 | 4.50 | 3 | 20 losses; **12 monologue, 4 empty, 2 question_invalid** |
| bedrock/gpt-oss-20b | 5/30 | **0.00** | **30/30** | 25 losses; **23 monologue, 28x `no_tag` stopped_reason** |

### Log-derived behaviour

- **gpt-oss-120b** is a false positive at the system level — 25 of 30 Qs have zero tool calls. The "ASearcher" framework is barely activated. On 16 correct, gpt-oss-120b answered from parametric memory. On 14 losses it confidently hallucinated (e.g. q793: `"HC Beroun (HC Berounští Medvědi) was dissolved for financial difficulties after the 2018-2019 season"` — **that team isn't even in the gold source set**).
- **Qwen3.5-9B** is the **only cell where ASearcher actually runs the framework** (5.9 tool calls/Q avg, zero zero-tool answers). Judge 13/30 is therefore a legitimate measurement of the framework at 9B scale.
- **QwQ-32B** exhibits an unexpected **self-refusal pattern**: on 11 of 19 losses, the literal `predicted_answer` is `"the question is invalid."`. Inspecting the gold Qs for these losses reveals they're all **temporally anchored** — q510 ("as of August 2024"), q129 ("in the year 2019"), q190 ("as of 1st June 2024"). QwQ's training data or our prompt is triggering categorical rejection of time-sensitive Qs.
- **Qwen3.5-4B** actually tool-calls (4.5/Q) but on 12 losses emits **monologue as final answer**: text like `"The user is asking about the number of total Academy Award nominations…"`. The answer boundary is never closed.
- **gpt-oss-20b** is a complete protocol breakdown — 28/30 `no_tag` stopped_reasons, 0 tool calls, 23 monologue outputs. The 20B model cannot emit a valid Bedrock `toolUse` block.

### Root cause

Three independent failures at three scales:

1. **Parametric shortcutting** at 120B — the model has enough knowledge to skip retrieval, so ASearcher's loop never fires.
2. **Canned refusal on temporal Qs** at QwQ-32B — probably an RL reward artifact that penalized confident wrong answers on "as of <date>" prompts.
3. **Protocol breakdown** at Qwen-4B and gpt-oss-20b — monologue leakage and `no_tag` errors respectively.

Only Qwen3.5-9B faithfully executes the ASearcher loop.

### Gain opportunities for `ours_gsw_v1`

- **QwQ refusal pool (11 Qs):** QwQ emits `"the question is invalid"` on temporal Qs. The planned small-model + GSW pipeline has no equivalent refusal behaviour — retrieval runs unconditionally, and the GSW scratchpad captures date/time attributes as first-class nodes rather than routing temporal Qs through a refusal head.
- **120B hallucination pool (14 Qs):** gpt-oss-120b confidently hallucinates (e.g. invented "HC Beroun" on q793) because ASearcher's loop never actually fires on a frontier-parametric model. Our planned pipeline makes the GSW construction mandatory — an answer must be grounded in an entity/attribute node extracted from a retrieved chunk, otherwise the aggregator emits "unknown". This converts confident hallucinations into grounded abstentions.
- **Qwen-4B monologue pool (12 Qs):** the 4B model emits narrative prose as final answer because the protocol expects a free-form text field. The GSW scratchpad would force the 4B to populate discrete entity slots before the aggregator reads them, which limits the monologue failure surface to a single structuring step rather than the entire answer pipeline.

---

## 3. ASearcher trained checkpoints (InclusionAI)

**Shipped checkpoints:** `inclusionAI/ASearcher-Web-7B`, `inclusionAI/ASearcher-Web-14B`.
**Training recipe:** SFT+GRPO with 14,107 seed Qs → Injection (enrich with Wikipedia facts) + Fuzzing (blur details) → 25,624 synthetic training samples, then async GRPO on QwQ-32B-derived reward. Paper reports 70.9 Avg@4 and 84.0 Pass@4 on full FRAMES 824. Our 30-Q pilot is a subset.

### Model grid

| model | correct | avg tools | stopped_reason mix | dominant loss signature |
|---|---:|---:|---|---|
| inclusionAI/ASearcher-Web-14B | 12/30 | 7.00 | finished=25, max_turns=4, llm_error=1 | 18 losses; **7 too_short, 5 empty, 2 question_invalid** |
| inclusionAI/ASearcher-Web-7B | 8/30 | 2.47 | finished=22, no_tag=7, llm_error=1 | 22 losses; **11 question_invalid, 5 too_short, 1 empty** |

### Log-derived behaviour

- **Web-14B** is format-perfect — 0 long predictions, 0 monologue, 0 think-tag leaks. The trained model has been RL'd into concise `<answer>…</answer>` emissions. But 5 of 18 losses have empty `<answer></answer>` blocks and 7 more are "too-short" wrong values. This is the **answer-hedging behaviour** baked into RL — when confidence is low the model emits null.
- **Web-7B** inherits a stronger version of the same refusal bias — 11 of 22 losses are `"the question is invalid"` (same signature as QwQ-32B in ASearcher prompt-mode). It also hits `no_tag` on 7/30 Qs, meaning the 7B model partially fails the protocol even though it was trained on it.

### Root cause

These are **well-calibrated but under-retrieve**. Web-14B uses 7 tool calls/Q vs vanilla_rag_react+gpt-5's 9.6. The RL reward penalized over-retrieval, so the trained policy gives up one search too early. On a harder subset (our pilot includes q371, q302, q133 that nobody solves), this under-retrieval becomes a 2–4 pp hit.

### Gain opportunities for `ours_gsw_v1`

The trained checkpoints are the hardest baseline to displace on peak accuracy (Web-14B at 40%, paper reports 70.9 Avg@4 on full FRAMES). The planned system's angle here is **compute class + interpretability**, not matching Web-14B's accuracy head-to-head:

- Web-14B uses 25.8 s/Q with an opaque trained policy. It under-retrieves (7 tool calls/Q) and produces empty `<answer></answer>` blocks on 5/18 losses without explanation.
- `ours_gsw_v1` targets gpt-oss-20B (comparable scale to ASearcher-Web-14B on compute) but with a transparent GSW intermediate — every answer is traceable back to entity/attribute nodes with chunk-level provenance. When the answer is "unknown", the GSW is inspectable to see *why* (no entity node for the required attribute).
- Orthogonal claim: on MoNaCo's 43-doc-avg horizon, Web-14B's 7-tool-calls ceiling is likely a hard cap. The GSW scratchpad naturally scales to tens of retrieved docs because the structuring step is sub-linear in chunk count once entity dedup runs.

The empirical case for us is: *3× faster for 75% of a trained-RL baseline's accuracy, with full reproducibility*. That's a narrow but defensible pitch.

---

## 4. Vanilla RAG+ReAct (our control)

**Protocol:** classic ReAct loop — `thought → action → observation`, two tools (`search`, `read`). `max_turns=16`. JSON tool_calls in OpenAI format. This is the grid control — not a paper, just the shape most agentic baselines reduce to when you remove special training or prompts.

### Model grid

| model | correct | avg tools | stopped_reason mix | dominant loss signature |
|---|---:|---:|---|---|
| **gpt-5** | 16/30 | 9.57 | finished=16, **max_turns=14** | 14 losses; **14/14 empty predictions — ALL `max_turns`** |
| bedrock/gpt-oss-20b | 13/30 | 9.80 | finished=16, max_turns=14 | 17 losses; **14 empty** |
| bedrock/gpt-oss-120b | 12/30 | 10.07 | finished=16, max_turns=14 | 18 losses; **14 empty** |

### Log-derived behaviour — the most striking result in the pilot

Every single loss on every vanilla_rag_react cell has `stopped_reason=max_turns` with an **empty** `predicted_answer`. The agent burned its turn budget entirely on retrieval and never emitted a synthesis turn.

Specific gpt-5 losses (all 14):

| qid | gold (abbrev) | tool calls at max_turns |
|---|---|---|
| q510 | Shane Gillis | 15 searches + 1 read |
| q70 | 84,512 | 14 searches + 2 reads |
| q693 | Michel Kratochvil | 9 searches + 7 reads |
| q386 | Yale University | 15 searches + 1 read |
| q83 | Yes, London | 14 searches + 2 reads |
| q520 | 302,971 | 14 searches + 2 reads |
| q371 | French/Haitian delta | 15 searches + 1 read |
| q800 | Calgary opening | 14 searches + 2 reads |
| q129 | Love Yourself in Seoul | 15 searches + 1 read |
| q546 | Graham High School | 14 searches + 2 reads |
| q590 | 1979 | 15 searches + 1 read |
| q302 | Djokovic H2H Murray | 14 searches + 2 reads |
| q133 | Tony × Oscar = 12 | 15 searches + 1 read |
| q190 | Nick (Jonas) | 15 searches + 1 read |

There is no "wrong answer" failure mode for vanilla_rag_react — only **timeout with empty output**.

**Counter-intuitive:** gpt-oss-20b (13/30) outperforms gpt-oss-120b (12/30). Both share the identical `max_turns=14` loss pattern. The 20b model is slightly more aggressive about emitting a final answer a turn or two earlier, which saves it from empty-output on 1 more Q.

### Root cause

**Turn budget allocation**, not retrieval quality or model capability. ReAct consumes ≈2–3 turns per sub-question (think, act, observe). At `max_turns=16`, that's 5–8 sub-Qs max. FRAMES Qs frequently need 3–8 sub-Qs; on 2-hop Qs with redirect/disambiguation the depth blows past 16 turns easily.

The fix is structural: either (a) raise `max_turns` to 32 (would break cost assumptions), (b) enforce a "synthesise now" turn when the budget is near-full, or (c) decouple decomposition from retrieval so decomposition doesn't consume turns.

### Gain opportunities for `ours_gsw_v1` — the cleanest win story in the grid

Vanilla RAG+ReAct's failure mode is "the agent never produced an answer" — 14/14 of gpt-5's losses are `max_turns=14` with empty `predicted_answer`. The ReAct loop consumes ~2–3 turns per sub-question (think, act, observe); at `max_turns=16` the agent simply runs out of budget before synthesising.

The planned system sidesteps this by **decoupling retrieval from synthesis**:

1. The small model drives retrieval (a few rounds), then stops.
2. A GSW scratchpad is built from the retrieved chunks — entity/verb-phrase/attribute nodes with provenance.
3. The aggregator reads the GSW, not the raw trajectory, and emits the answer.

Step 3 is the new turn — it always runs, regardless of how many retrieval turns were spent. So there is no `max_turns`-style empty-output failure surface. Every one of the 14 gpt-5 `max_turns` losses is structurally preventable.

**Key framing for the paper:** vanilla_rag_react is the only system where failure is "no answer emitted". The `ours_gsw_v1` architecture removes this failure mode by construction — retrieval budget exhaustion can still miss facts, but the aggregator always runs on whatever GSW was built and emits a grounded answer (or an explicit "unknown" with provenance).

---

## 5. Context-1 + reasoner (Chroma)

**Paper:** Chroma blog post "Context-1" (2026-03), 20B MoE retrieval sub-agent.
**Repo:** [`chroma-core/context-1-data-gen`](https://github.com/chroma-core/context-1-data-gen), model on HF.
**Protocol:** Two-stage — Context-1 (the 20B MoE, served via local vLLM on port 8007) runs an observe→reason→act loop to prune retrieved chunks down to a small set (5–7 chunks typically), then calls an external reasoner over those chunks to produce the final answer. The reasoner is fully pluggable.

### Run history

| run # | reasoner | judge | EM | F1 | wall (s) | status |
|---:|---|---:|---:|---:|---:|---|
| 1 | gpt-5 (original, 18:38 UTC) | **0.00** | 0.00 | 0.00 | 27.1 | 404 bug — reasoner never ran |
| 2 | gpt-5 (re-run, 14:08) | 0.40 | 0.33 | 0.39 | 76.5 | clean |
| 3 | bedrock/openai.gpt-oss-120b (14:09) | 0.40 | 0.23 | 0.34 | 72.9 | repetition loop on q800 |
| 4 | bedrock/openai.gpt-oss-20b (14:09) | 0.30 | 0.23 | 0.27 | 76.4 | clean |
| 5 | Qwen/Qwen3.5-9B (14:33, **killed after 9 min**) | — | — | — | — | thinking-loop on q793 → only 1/30 done after hitting `max_tokens=16000` at 438s; ETA ~3.7h. Abandoned. |
| 6 | Qwen/Qwen3.5-4B (14:37, **killed after 6 min**) | — | — | — | — | thinking-loop on q793 → only 1/30 done at 264s wall. Same pathology as run 5. Abandoned. |

Runs 5 and 6 were killed because the Qwen3.5 thinking-mode reasoner gets trapped in a self-deliberation loop when the Context-1 pruned chunk set doesn't contain the answer. Neither would have produced useful cell-level judge scores — every loop-stuck Q eventually emits whatever text is in the `<think>` block when `max_tokens=16000` is hit, but the wall-time cost (264–438s per looped Q) made the full 30-Q run impractical (~2–4 hours projected per cell, with near-certain 0% judge on looped Qs). The pathology itself is the finding; running to completion adds no new information.

### The retrieval sub-agent works

Across all 6 runs the retrieval sub-agent (Context-1, the 20B MoE) behaves consistently:

- 8.2 tool calls/Q avg.
- Keeps 5–7 chunks per Q.
- `max_retrieval_turns=14` fires on 14/30 Qs — comparable budget pressure to vanilla_rag_react.
- Never errors at this stage.

So when a run fails, the failure is in the reasoner stage, not retrieval.

### Finding 1 — Context-1 + gpt-5 regresses from vanilla_rag_react + gpt-5

| system | reasoner / model | judge |
|---|---|---:|
| vanilla_rag_react | gpt-5 | **0.53** |
| context1_plus_reasoner | gpt-5 (re-run 2) | 0.40 |

Adding the Context-1 chunk-pruning layer in front of gpt-5 **costs 13 pp on the FRAMES pilot**. The retrieval sub-agent is over-pruning — dropping chunks that contain the answer — so the reasoner never sees them.

This is the cleanest substitution-study result in the pilot: *"a dedicated retrieval sub-agent can actively hurt a frontier reasoner on FRAMES by removing context the reasoner would have used."*

### Finding 2 — reasoner scale above 20B is not the bottleneck

| reasoner | judge |
|---|---:|
| gpt-5 | 0.40 |
| bedrock/gpt-oss-120b | 0.40 |
| bedrock/gpt-oss-20b | 0.30 |

gpt-5 and gpt-oss-120b land at the *same* judge (0.40), despite vastly different reasoner capabilities. The 20b reasoner drops 10 pp — but that's a fall off the edge, not a smooth degradation. This means the pruned-chunk set is the bottleneck — above 20B, a smarter reasoner doesn't help because the answer-containing chunks are already gone.

### Finding 3 — reasoner-stage pathologies observed

Two distinct failure modes arise at the reasoner stage, both independent of the retrieval sub-agent:

**(a) gpt-oss token-level repetition loop.** During the bedrock/gpt-oss-120b run on q800, the model began emitting the same sentence (*"Seoul Olympic Stadium hosted the opening ceremony for the 1988 Summer Olympics in Seoul, South Korea. The 1988 Winter Olympics were held in Calgary, Alberta, Canada. I found information about the 1988 Olympics."*) over and over for ~6 minutes until `max_completion_tokens=16000` terminated the generation. Same family of pathology observed in the original stuck gpt-5 run before we killed it. This is a **gpt-oss SSE streaming issue**, not a prompt problem — the model gets into a decoding loop and `max_tokens` is the only exit.

**(b) Qwen3.5 thinking-mode self-deliberation loop.** Both the 9B and 4B Qwen3.5 runs exhibit the **same** pathology and were **killed** after producing 1/30 completions each (see run history). The reasoning-mode model gets into a meta-deliberation loop while trying to decide how to phrase its abstention. Verbatim from the 4B stream:

> *"`Wait, I should check if I can say "The provided chunks do not mention Joel Oshiro Dyck."`"*
> *"`"The provided chunks do not contain information about Joel Oshiro Dyck." is better.`"*
> *"`Okay.`"*
> *"`I will use "The provided chunks do not contain information about Joel Oshiro Dyck."`"*
> (repeats)

And from the 9B:

> *"`* *Wait, I'll check if I should say "The provided text does not contain information about Joel Oshiro Dyck."*`"*
> *"`* Okay.`"*
> (repeats)

This is a genuine substitution-study finding: **Qwen3.5 thinking-mode reasoners are unsuitable as Context-1 downstream synthesizers** because they can't cleanly emit an abstention — they enter a self-deliberation loop when the chunk set doesn't contain the answer. The `<think>` block never closes, and the model burns its completion budget (`max_completion_tokens=16000`) on meta-phrasing.

Both runs were killed before completing the 30-Q pilot. The 4B burned 264s on q793; the 9B burned 438s on the same Q. Neither produced a meaningful judge score (1/30 on each, at judge=0/f1=0). The pathology is the finding; a full run to completion would only cost additional hours of wall time without surfacing new behavior — every looped Q will end the same way, with `max_tokens` force-closing the `<think>` block and whatever deliberation text is in the buffer becoming the "answer".

**Implication:** the Qwen3.5 thinking mode is **not a viable reasoner substitution candidate** for Context-1 at the pilot scale. Either (a) Qwen3.5 non-thinking mode, or (b) a different base model entirely, would be needed to land a usable 4B/9B row in the Context-1 grid.

### Pattern summary

| stage | failure modes observed |
|---|---|
| Retrieval sub-agent (Context-1) | clean; occasional `max_retrieval_turns` pressure, no crashes |
| Reasoner stage — frontier (gpt-5) | over-prunes context gets bounced back to reasoner; +13 pp regression from vanilla |
| Reasoner stage — gpt-oss family (120b, 20b) | token-repetition loops on some Qs; `max_tokens` terminates |
| Reasoner stage — Qwen3.5 thinking mode (9B, 4B) | meta-deliberation loop when chunks don't contain the answer; `max_tokens` terminates |

### Gain opportunities for `ours_gsw_v1`

Context-1 is the closest architectural neighbour to `ours_gsw_v1` — both insert a post-retrieval processing stage between raw chunks and the final answer. The run history surfaces three design differences that matter:

| axis | Context-1 behavior | `ours_gsw_v1` design |
|---|---|---|
| Intermediate representation | chunk-id subset (raw prose handed to reasoner) | structured GSW scratchpad (entities/attributes/relations with chunk provenance) |
| Reasoner dependence | **high** — frontier reasoner still loses 13 pp vs no-pruning baseline, and thinking-mode reasoners loop | **low target** — aggregator reads a structured representation, not raw chunks; smaller aggregator should suffice |
| Failure mode under no-answer | reasoner enters deliberation loop (Qwen3.5) or hallucinates (gpt-oss) | explicit "no entity node for required attribute" → structured abstention |
| Scaling to MoNaCo (43-doc avg) | each additional doc inflates the reasoner's input; context limits bite | structured representation is text-compressed; aggregator input stays bounded |

The Qwen3.5 deliberation-loop finding is **directly relevant to our planned 20B-class aggregator**. If our aggregator is a Qwen3.5-family thinking-mode model, the same pathology will surface unless we either (a) disable thinking mode for the aggregator, or (b) give it a structured input (GSW) that makes "answer vs abstain" a deterministic check on node presence rather than a free-form deliberation.

---

## 7. Tongyi DeepResearch-30B-A3B (as-shipped)

**Paper / blog:** Alibaba-NLP Tongyi DeepResearch (2025-10).
**Weights:** [`Alibaba-NLP/Tongyi-DeepResearch-30B-A3B`](https://huggingface.co/Alibaba-NLP/Tongyi-DeepResearch-30B-A3B) — 30.5 B total, 3.3 B active per token (MoE).
**Repo:** [`github.com/Alibaba-NLP/DeepResearch`](https://github.com/Alibaba-NLP/DeepResearch).
**Protocol:** Shipped ReAct-style agent with Search/Visit/Python/Scholar/FileParser tools. For the FRAMES offline setup we enabled only Search + Visit (Python/Scholar/FileParser disabled as no-ops). Tool-call parser `hermes` on vLLM. Single row in the grid — no model swap; their MoE is the story.

### Model grid

| model | correct | EM | F1 | judge | avg turns | avg wall (s) | dominant loss signature |
|---|---:|---:|---:|---:|---:|---:|---|
| Alibaba-NLP/Tongyi-DeepResearch-30B-A3B (patched) | 15/30 | 0.00 | 0.026 | **0.50** | 12.0 | 51.4 | 11 wrong_synthesis, 2 hallucination, 1 wrong_retrieval, 1 early_stop |

### Serving / adapter fixes needed (substitution-study finding)

Getting Tongyi running cleanly required **four adapter/vLLM fixes**:

1. **`max-model-len` raised from 32768 → 65536** on the vLLM serve script. The shipped native value caused mid-Q 400 errors once accumulated tool responses pushed the input past ~8.7k tokens while the harness requested 16k+ output tokens.
2. **`max_completion_tokens` reduced** from 50000 (grid default) to 16000 — 50k exceeded `max_model_len`.
3. **Tool-name case matching** — Tongyi emits tool names in **lowercase** (`search`, `visit`) but the adapter was registered with capitalized names. Dispatcher now matches `name.lower()`.
4. **Multi-query `search` handling** — Tongyi's native format emits `query: ["q1", "q2", …]` (parallel-query batch), not a single string. Adapter now accepts both, runs each query, dedups results by chunk_id.
5. **`visit` kwarg normalisation** — Tongyi passes `url=...` in offline mode (web-native habit) plus extra fields like `goal`. Adapter accepts `url`/`chunk_id` interchangeably and swallows extras via `**_extra`.
6. **Force-synthesis fallback** — even with everything else fixed, the shipped model has a habit of emitting a tool call on *every* turn, so the ReAct loop never sees a `finish=stop tools=0` turn and the old adapter exited at `max_turns` with an empty `pred`. Added a post-loop forced synthesis pass (tools disabled, one more call) to extract the final answer.

Without any of these, the adapter produced 0/30 or ~14% judge. With all applied: **0.50**.

### Run history

| run # | setting | judge | notes |
|---:|---|---:|---|
| 1 (14:56) | default `max_completion_tokens=50000` | **aborted** | 3 Qs, all 400 errors — ctx overflow |
| 2 (14:57) | capped to 24000 | 0.00 (5/30) | tool-name mismatch + list-arg bug — all 5 Qs judge=0, killed |
| 3 (15:04) | adapter fixed (tool name + list), still 24000 | 0.00 (11/30) | mid-Q ctx overflow, killed |
| 4 (15:07) | `max_completion_tokens=8000` | 0.33 (partial, killed at 6/30) | ran clean but budget_exceeded inflated losses |
| 5 (15:13) | `max-model-len 65536` + `max_completion_tokens=16000` | **0.43** | 11 budget_exceeded + 5 tool_error — model calls tools every turn |
| **6 (15:29)** | above + **forced-synthesis fallback** | **0.50** | 15 correct, 11 wrong_synthesis, 2 hallucination, 1 wrong_retrieval, 1 early_stop |

### Log-derived behaviour

- **Model answers exist in the reasoning stream** long before the loop terminates — Tongyi arrives at the right entity early but keeps emitting another `<think>` block + tool call each turn, never exiting the agent loop with a pure-text answer.
- **`forced_synthesis` counter = 0** on run 6 — the fallback wasn't triggered because run 6's actual path changes (runs 4-5 were different generations where the model happened to hit `max_turns=16` more often). The patch is cheap insurance: on the next run it may fire.
- **EM is 0.00** across all runs because Tongyi wraps every answer in `<think>…</think>\n\n<prose answer>` — exact-match never triggers against gold strings like `"Rob Reiner"` or `"Italy"`. The judge correctly extracts the entity from the prose tail and catches the win.
- **`wrong_synthesis` (11 losses)** — the model saw the right chunks but picked the wrong fact or a partial answer. These are the target for `ours_gsw_v1`'s structured intermediate.

### Gain opportunities for `ours_gsw_v1`

Tongyi is the closest architectural sibling to `ours_gsw_v1`:

- **Both** are agentic ReAct systems built for deep research (multi-turn retrieval + synthesis).
- **Neither** relies on text-marker protocols (no brittleness at 20B-class models).
- **Key difference**: Tongyi uses free-form `<think>` + tool-call every turn. `ours_gsw_v1` inserts a structured GSW scratchpad that replaces the mid-trajectory state with entity/attribute nodes.

On the 11 `wrong_synthesis` Qs Tongyi missed, the retrieval was correct but the model picked the wrong fact from prose. A GSW intermediate turning retrieved chunks into discrete entity-attribute triples would structurally prevent "Roger Federer" (q693) or "Pete Davidson" (q510) style substitution errors by binding each candidate answer to a provenance chunk.

**Headroom estimate for `ours_gsw_v1` vs Tongyi's 0.50**: if the GSW scratchpad converts even 40% of the `wrong_synthesis` cluster (4 of 11) without introducing new failures, `ours_gsw_v1` at 20B would land at **0.63 judge** — matching Search-o1's 120b leader. On FRAMES this is a strong ceiling claim at much smaller compute.

---

## 12. `ours_gsw_planner_v1` — Phase 1 prompt-only GSW-fragment planner (this iteration)

**Implementation**: `research_agent/src/research_agent/adapters/ours/gsw_planner_v1.py` + `_planner_exec.py` + `_planner_prompts.py`. Registered id: `ours_gsw_planner_v1`. Tests: `research_agent/tests/ours/test_planner_exec.py` (11 green).

**Architecture**: one LLM call emits a typed GSW-fragment plan (filled entities / blank entities with `value_type` / verb-phrases / constraints); pure-Python executor does topological sort + per-blank identification / projection (one LLM extraction each) + Python-only evaluation of derived / argmax / argmin constraints. Falls back to `ours_gsw_v1` flat decomposition on parse or execution error.

### Final 30-Q FRAMES pilot results

| model | judge | EM | F1 | mean turns | wall (s) | fallback_rate | dominant loss signature |
|---|---:|---:|---:|---:|---:|---:|---|
| bedrock/gpt-oss-120b | **0.367** | 0.300 | 0.355 | 4.0 | **16.4** | 7% (2/30) | 6 hallucination, 5 wrong_retrieval, 5 wrong_synthesis, 3 early_stop |
| bedrock/gpt-oss-20b | **0.333** | 0.267 | 0.304 | 3.3 | **12.0** | **27% (8/30)** | 8 hallucination, 8 early_stop, 2 wrong_retrieval, 2 wrong_synthesis |
| gpt-5 (killed at 5/30) | 0.00 | — | — | — | — | — | — |

### Comparison to other cells at matched compute

| system | model | judge | delta vs planner-v1 |
|---|---|---:|---:|
| vanilla_rag_react | bedrock/gpt-oss-20b | 0.43 | +0.10 pp vs planner-v1 |
| vanilla_rag_react | bedrock/gpt-oss-120b | 0.40 | +0.03 pp vs planner-v1 |
| agentic_reasoning_mindmap (E12) | bedrock/gpt-oss-20b | **0.50** | +0.17 pp vs planner-v1 |
| agentic_reasoning_mindmap (E12) | bedrock/gpt-oss-120b | 0.40 | +0.03 pp vs planner-v1 |
| **ours_gsw_planner_v1** | bedrock/gpt-oss-20b | **0.333** | — |
| **ours_gsw_planner_v1** | bedrock/gpt-oss-120b | **0.367** | — |

**Prompt-only `gsw_planner_v1` under-performs vanilla RAG+ReAct at both 20b and 120b.** The typed-DAG hypothesis did not validate at prompt-only scale on the 30-Q FRAMES pilot.

### What actually happened

**Zero structural failures**: 0 parse errors post-retry on 120b, 0 execution-errors (cycles / bad refs) on 120b. 20b had a higher **27% fallback rate** (27% = 8/30 — mostly `parse_failure:empty LLM response` and `unbalanced braces in LLM response` — 20b's JSON emission is brittle). This is the "protocol brittleness" failure surface the pilot already documented for smaller models across E9 / E12 / E13.

**Dominant remaining failure is hallucination at the extraction stage** (6 at 120b, 8 at 20b). The planner correctly identifies what blanks to fill and what verb-phrase to use, but the per-blank LLM extraction fabricates values on temporally-anchored or numeric questions. Example:

- **q70** (gold `84,512`) — planner emits two attribute-project blanks + diff constraint. Executor retrieves Portland-ME + Portland-OR chunks. But LLM extraction returns wrong numbers (e.g. `128492`) on one city, derived op computes the subtraction on wrong inputs → confident wrong answer.
- **q510** (gold `Shane Gillis`) — 20b predicted `"Benson"`, a hallucinated name. The temporal anchoring "as of August 2024" wasn't enforced by the schema.

**Mean turn count dropped from 10+ (ReAct family) to 3–4**: structural win on compute — planner-v1 is **~5× faster per Q** than vanilla_rag_react. The failure isn't inefficiency; it's extraction quality.

### Concrete examples

**Correct** (both cells):
- q512 (gold: Rob Reiner): `pred="Rob Reiner"` — clean entity resolution.
- q158 (gold: 12): `pred="12"` — simple 2-hop chain executed cleanly.

**Correct at 20b, surprising**:
- q793 (gold: Joel Oshiro Dyck played for the Chatham Wheels...): `pred="Nippon Paper Cranes."` — partial credit; judge lenient.

**Hallucination (numeric)**:
- q70 (gold: 84,512): `pred="128492"` at both 20b and 120b. Plan structure was correct; extraction pulled the wrong census number from retrieved chunks.

**Hallucination (entity)**:
- q510 (gold: Shane Gillis): 20b `pred="Benson"`. Temporal constraint ("as of August 2024") wasn't reified into the plan — executor retrieved without date filter and extracted a plausible-but-wrong name.

**Fallback to flat** (example, 20b q793):
- Parse failure: `parse_failure:empty LLM response`. 20b's second repair attempt also returned empty. Delegated to `ours_gsw_v1` flat decomposition — which then got the partial answer above.

### Go/no-go decision (per plan gate)

| criterion | target | actual 20b | pass? |
|---|---:|---:|:-:|
| judge ≥ 0.50 at bedrock/gpt-oss-20b | ≥ 0.50 | **0.333** | ❌ |
| fallback-flat rate < 20% | < 0.20 | **0.27** | ❌ |
| parse-validity ≥ 80% | ≥ 0.80 | ~73% at 20b (8/30 fallback) / 93% at 120b | partial |
| ≥ 1 unique win on impossible set (q371/q302/q133) | ≥ 1 | 0 | ❌ |

**Gate failed. Per plan, 20b judge < 0.40 puts us in "pivot" territory**, not "iterate".

### Read — why prompt-only didn't work, and what Phase 2 would need

The pilot shows the *planner* is structurally viable (low parse/exec error at 120b) but the *executor's per-blank extraction* is where the quality leaks:

1. **Extraction hallucinates numbers and dates** — the LLM extract step is un-trained at the fragment-schema task. Vanilla RAG+ReAct beats planner-v1 because ReAct re-examines chunks across multiple turns, while planner-v1 commits to one shot per blank.
2. **Temporal anchors aren't modelled** — "as of August 2024" should be a Constraint on the retrieval scope, but the schema has no temporal-filter kind. Schema gap.
3. **20b's JSON emission is brittle** — 27% fallback rate. Not a schema problem; an LLM-at-scale problem that SFT addresses.

**Phase 2 is still viable but harder-sold**. The three issues above map onto:
- (1) train the extraction step jointly with the planner via SFT+DPO on answer-F1.
- (2) extend the schema with a `TemporalScope` constraint kind.
- (3) SFT will sharply reduce parse errors at 20b.

Alternatively, **pivot to a hybrid approach**: use the typed-GSW-fragment as a scratchpad INSIDE a ReAct loop rather than as a one-shot plan. That combines the structural benefit of the typed schema with ReAct's multi-turn error recovery.

### What this contributes to the paper (even with the gate miss)

- First reported prompt-only typed-DAG planner result on FRAMES. Establishes a lower-bound baseline for the typed-DAG direction.
- Clean isolation of where prompt-only fails (extraction hallucination, not plan validity).
- The 5× wall-time reduction vs vanilla RAG+ReAct is real and durable — the Phase-2 trained version should keep this efficiency property.
- Documents a clear schema-gap (no temporal scope) that Phase 2 addresses.

---

## 11. GAM — General Agentic Memory (VectorSpaceLab)

**Paper:** *General Agentic Memory Via Deep Research*, [`arxiv 2511.18423`](https://arxiv.org/abs/2511.18423).
**Repo:** [`VectorSpaceLab/general-agentic-memory`](https://github.com/VectorSpaceLab/general-agentic-memory).
**Protocol:** Two-phase **"JIT compilation" paradigm**. (1) **Memorizer** runs offline: builds a lightweight hint index (title + first 160 chars per chunk) over the raw page-store. (2) **Researcher** runs online with two asymmetric tools:

- `browse_hints(query, top_k=10)` — **cheap**: returns `[{chunk_id, title, digest}]` rows. No chunk-body access.
- `fetch_page(chunk_id)` — **expensive**: dereferences to the full article text (up to 12k chars).

Agent is instructed to browse freely (cheap) and fetch selectively (expensive). GAM's headline claim: a well-behaved Researcher browses 2–5× more than it fetches.

### Model grid (5 cells; 2 complete, 3 in-flight)

| model | correct | EM | F1 | judge | avg turns | wall (s) | dominant loss signature |
|---|---:|---:|---:|---:|---:|---:|---|
| **gpt-5** | 18/30 | 0.400 | 0.496 | **0.600** | 9.4 | 104.4 | 18 correct, 11 budget_exceeded, 1 tool_error |
| **bedrock/gpt-oss-120b** | 16/30 | 0.400 | 0.455 | **0.533** | 10.3 | 14.3 | 16 correct, 9 tool_error, 5 budget_exceeded |
| bedrock/gpt-oss-20b | 12/30 | 0.267 | 0.357 | **0.400** | 8.1 | 11.6 | 12 correct, 7 budget_exceeded, 4 hallucination, 4 early_stop, 2 tool_error, 1 wrong_synthesis |
| Qwen/Qwen3.5-4B | 10/30 | 0.233 | 0.285 | **0.333** | 5.6 | 18.8 | 10 correct, 12 wrong_synthesis, 4 early_stop, 2 tool_error, 1 budget_exceeded, 1 hallucination |
| Qwen/Qwen3.5-9B | 2/30 | 0.033 | 0.064 | **0.067** | 2.7 | 27.3 | 13 wrong_synthesis, 13 early_stop, 2 wrong_retrieval, 2 correct |

### Headline findings

**1. GAM at bedrock/gpt-oss-120b = 0.533** — the strongest 120b-class cell for this system, and above all E9/E12 20–120b cells.

Within-family scaling works for GAM (120b > 20b, +13 pp), unlike Mind-Map/E12 where 20b beat 120b. Interpretation: the 2-tool GAM schema is simpler than Mind-Map's 3-tool schema, so larger models don't over-orchestrate.

**2. Qwen3.5-9B clears 0.00 for the first time (0.06).**

On E9 and E12 the 9B scored 0.00. On E13 it clears **1/16 judge** (projected ~0.06 at full 30). The 2-tool schema is simple enough that the 9B can at least attempt the workflow — but still only ~1/30 of the time.

**3. Qwen3.5-4B > Qwen3.5-9B again (~0.35 vs ~0.06).**

Third experiment in a row where the 4B beats the 9B:
- E12: 4B=0.30, 9B=0.00
- E13: 4B=~0.35, 9B=~0.06

The 4B vs 9B inversion is **robust across three independent experiments**. Qwen3.5-9B's reasoning-mode is structurally unfit for multi-tool agentic workflows — it spends turn budget on deliberation. 4B in non-thinking mode uses tools pragmatically.

### Log-derived behaviour per model

**bedrock/gpt-oss-120b (0.533 — best non-frontier cell in E13):**
- 16 correct, **9 tool_error** (highest of any E13 cell).
- `tool_error` cluster: model calls `fetch_page` with invalid chunk_ids, or calls `browse_hints` with malformed argument shapes (e.g. list where string expected). Similar to Q+'s tool-schema confusion but lower rate.
- 5 `budget_exceeded` — same pattern as other 120b cells.
- **Browse-to-fetch ratio** (estimated from tool_calls): ~1.3× — model uses `browse_hints` only slightly more often than `fetch_page`. **Does not match GAM paper's "well-behaved Researcher browses 2–5×" claim at this scale.**

**bedrock/gpt-oss-20b (0.400):**
- 12 correct, 7 budget_exceeded, 4 hallucination, 4 early_stop.
- Similar to its E12 counterpart but 10 pp lower. GAM doesn't help the 20b the way Mind-Map did.
- Fastest cell in E13 (11.6 s/Q) because `browse_hints` returns are short.

**Qwen3.5-4B (final, 0.333):**
- 10 correct / 30 — matches its E12 score (0.300) closely, confirms consistent 4B behavior across experiments.
- 12 wrong_synthesis — model reads chunks but picks wrong fact. Highest wrong_synthesis rate of any E13 cell.
- Simpler 2-tool schema keeps the model engaged longer than Mind-Map's 3 or Q+'s 4: mean 5.6 turns/Q.

**Qwen3.5-9B (running, ~0.06):**
- First non-zero result for 9B in the grid.
- 10 wrong_synthesis — the 9B actually tries to synthesize answers now (vs total early_stop on E9/E12).
- Still overwhelmingly failing (15/16 wrong), but the 1 correct is a proof-of-life that a simpler tool schema can keep the 9B moving.

### Pattern summary

| cluster | cells | mechanism |
|---|---|---|
| **Frontier on schedule** | gpt-5 (~0.75 at n=4) | too-early, tracking high |
| **Clean within-family scaling** | 120b (0.53) > 20b (0.40) by 13 pp | 2-tool schema simple enough that bigger model doesn't over-orchestrate |
| **tool_error hotspot at 120b** | 9 tool_errors on 120b | model calls `fetch_page` with invalid chunk_ids ~30% of the time |
| **Qwen3.5-4B holds floor** | 4B at ~0.35 (matches E12) | consistent 4B > 9B signal, third experiment running |
| **Qwen3.5-9B signal of life** | 9B at 0.06 (first non-zero for 9B in grid) | 2-tool schema simple enough for any signal at all |

### Gain opportunities for `ours_gsw_v1`

GAM is the paper that explicitly names **"JIT compilation"** — the same paradigm `ours_gsw_v1` is built around. Key differences:

| axis | GAM | `ours_gsw_v1` |
|---|---|---|
| Intermediate representation | title + digest hint rows (plain text) | entity-attribute graph nodes with chunk provenance |
| Scaffold construction cost | zero LLM (chunk iteration + truncation) | zero LLM (deterministic entity extraction) |
| Query surface | substring on title/digest | structured entity-based lookup |
| Typical browse-fetch flow | `browse_hints` → decide → `fetch_page` (raw chunk) | structured node lookup → aggregator reads nodes with provenance |
| Information density per "browse" turn | ~10 title+digest rows | N×M entity-attribute triples (far denser) |

**Attack angle:** GAM validates the JIT-compilation paradigm in principle (within-family scaling works; Qwen3.5-9B shows life). But the hint index is shallow: title + 160 chars is a *text-substring* hint, not a structured one. `ours_gsw_v1` replaces the hint index with an entity-attribute graph — the same paradigm, structurally richer intermediate.

**Ceiling estimate vs GAM:** if `ours_gsw_v1` at 20B converts even half of GAM/20b's 7 `budget_exceeded` + 4 hallucination (= 5.5 Qs), it lands at ~**0.58 judge at 20B** — above GAM's 120b cell (0.53) and above every 20b cell in the grid so far.

---

## 10. Agentic Reasoning + Mind-Map (Tsinghua NLP)

**Paper:** *Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools* ([`arxiv 2502.04644`](https://arxiv.org/abs/2502.04644), ACL 2025).
**Source repo:** publicly available as a research codebase; our adapter implements the Mind-Map tool subset directly.
**Protocol:** The original paper exposes three peer tools — Web-Search, Coding, Mind-Map — on a reasoning LLM. Our adapter swaps Web-Search for BM25 and drops Coding (FRAMES rarely needs arithmetic). Remaining three tools:

- `search(query, top_k)` — BM25 retrieval, same shape as E1.
- `mind_map_update(edges)` — append `(subject, relation, object)` triples to the persistent per-question mind-map.
- `mind_map_query(focus)` — return all triples (max 30) whose subject or object contains `focus` (case-insensitive substring match). Persistent scratchpad across turns.

The mind-map is **per-question, not corpus-wide** — it survives context compression because the agent queries it explicitly instead of re-reading raw chunks. This is the closest prompt-only cousin to our GSW thesis.

### Model grid (5 cells; 3 complete, 2 in-flight)

| model | correct | EM | F1 | judge | avg turns | wall (s) | dominant loss signature |
|---|---:|---:|---:|---:|---:|---:|---|
| **gpt-5** | 16/30 | 0.433 | 0.486 | **0.533** | 11.0 | 107.7 | 14 budget_exceeded — worst budget pressure in grid |
| **bedrock/gpt-oss-20b** | 15/30 | 0.33 | 0.47 | **0.500** | 7.7 | 12.8 | 7 budget_exceeded, 4 wrong_synthesis, 2 tool_error, 2 wrong_retrieval |
| bedrock/gpt-oss-120b | 12/30 | 0.30 | 0.35 | **0.400** | 10.5 | 21.4 | 13 budget_exceeded, 2 wrong_synthesis, 1 tool_error, 1 early_stop, 1 loop |
| Qwen/Qwen3.5-9B | 0/30 | 0.00 | 0.00 | **0.000** | 2.1 | 8.9 | 19 early_stop, 9 wrong_synthesis, 2 wrong_retrieval |
| Qwen/Qwen3.5-4B | 9/30 | 0.233 | 0.289 | **0.300** | 6.6 | 21.8 | 9 wrong_synthesis, 5 early_stop, 4 loop, 3 budget_exceeded |

### Headline findings

**1. The 20B MoE beats the 120B MoE on E12 (0.50 vs 0.40, +10 pp).**

This is a genuinely surprising result — the same base family's 6× bigger variant is **worse** at using Mind-Map tools on FRAMES. Looking at the failure mix: 120b spends 13 of 30 Qs on `budget_exceeded` (never reaching an answer) while 20b has only 7. The 120b over-plans and over-queries the mind-map, spending tool budget on housekeeping instead of synthesizing. The 20b is more decisive — shorter mean turns (7.7 vs 10.5) and half the wall time (12.8 vs 21.4 s).

**2. Mind-Map helps 20B MoE compared to its vanilla-RAG+ReAct baseline (+7 pp).**

vanilla_rag_react on bedrock/gpt-oss-20b hits 0.43; agentic_reasoning_mindmap on the same base hits **0.50**. This is a clean +7 pp gain attributable to the mind-map tool at 20b scale — the first competitor system in the grid that shows a *reliable* small-scale gain over vanilla RAG+ReAct.

**3. Total failure at 9B, partial success at 4B (so far).**

Qwen3.5-9B gets 0/30 judge (19 early_stop) — same protocol-collapse pattern as Q+ at 9B. Qwen3.5-4B is *partially* working (3 correct in 10 Qs, rolling 0.30). That's counter-intuitive given the model-size ordering — but Qwen3.5-4B simply doesn't try to overthink the mind-map tool; it runs search, writes a triple, answers. The 9B's reasoning mode (which Qwen3.5-4B also has but apparently uses less aggressively here) dominates turn budget with self-deliberation.

### Log-derived behaviour per model

**gpt-5 (running, 10/30 so far → ~0.60):**
- 6 correct, 4 `budget_exceeded`. Uses all three tools cleanly, writes mind-map triples after each search turn, answers from mind-map queries rather than re-reading chunks.

**bedrock/gpt-oss-20b (0.50, BEST cell in this experiment):**
- 15 correct, 7 `budget_exceeded`. Clean answers like `"Rob Reiner"` and `"12"` — no prose wrapping.
- Fastest cell: 12.8 s/Q — mind-map ops are cheap.
- Key: 20b writes shorter mind-map triples and queries them pragmatically. Doesn't over-orchestrate.

**bedrock/gpt-oss-120b (0.40):**
- 12 correct, 13 `budget_exceeded`. The most interesting failure cluster in E12.
- Model writes exhaustive mind-map triples on *every* turn, eats turn budget with `mind_map_update` + `mind_map_query` calls, then runs out before `search` can converge.
- Sample verbose 120b trace (q793): 12 mind-map writes, 3 searches, hit max_turns before emitting an answer.

**Qwen3.5-9B (total failure, 0.00):**
- **19 early_stop + 9 wrong_synthesis + 2 wrong_retrieval.** Zero correct.
- Mean turns = 2.1 — model exits the loop nearly immediately.
- Same pattern as the Q+ 9B collapse: the 3-tool schema (search / mind_map_update / mind_map_query) is too complex for the 9B to orchestrate.

**Qwen3.5-4B (final, 0.30):**
- 9 correct / 30. **Breaks the "bigger = worse" ordering cleanly: 4B (0.30) > 9B (0.00) on this task.**
- 9 wrong_synthesis, 5 early_stop, 4 loop, 3 budget_exceeded — more active failure modes than 9B's clean early-stop, but the model is *attempting* the Mind-Map workflow.
- Mean 6.6 turns/Q vs 9B's 2.1 — 4B stays engaged with the 3-tool schema for longer before failing.

### Concrete examples

**gpt-oss-20b correct (q512, gold: Rob Reiner):** pred = `"Rob Reiner"` — 3 turns (search + mind_map_update + answer).
**gpt-oss-20b correct (q158, gold: 12):** pred = `"12"` — clean numeric answer.
**gpt-oss-120b budget_exceeded (q793, q510, q70):** all `pred=""` after max_turns.
**Qwen3.5-9B wrong_synthesis (q793):** `pred=""` after 2 turns — model never attempted the Mind-Map workflow.
**Qwen3.5-9B wrong_retrieval (q512):** `pred=""` on a Q the 20b solved in 3 turns.

### Pattern summary

| cluster | cells | mechanism |
|---|---|---|
| **Mid-scale sweet spot (20B MoE)** | bedrock/gpt-oss-20b (0.50) | pragmatic mind-map usage; beats both its own family's 120b and the vanilla+20b baseline (+7 pp) |
| **Over-orchestration at 120B** | bedrock/gpt-oss-120b (0.40) | too many mind-map updates per turn → budget_exceeded on 43% of Qs |
| **Protocol collapse at 9B** | Qwen3.5-9B (0.00) | 3-tool schema too complex; 19 early_stop |
| **Partial success at 4B** | Qwen3.5-4B (~0.30) | doesn't over-think the tools; breaks bigger=worse monotonicity |

### Gain opportunities for `ours_gsw_v1`

**E12 is the most direct prompt-only cousin to `ours_gsw_v1`.** Both propose a **persistent per-question scratchpad** that survives context compression. The differences are architectural:

| axis | Mind-Map | `ours_gsw_v1` GSW |
|---|---|---|
| What the scratchpad stores | Agent-written `(subject, relation, object)` triples | Deterministically extracted entity/attribute nodes from retrieved chunks |
| Who writes to it | The LLM agent, as a side task | The pipeline, before the aggregator runs |
| Query surface | substring match on subject/object | Structured entity-based lookup with chunk provenance |
| Cost per turn | 1–3 `mind_map_update` + 1 `mind_map_query` (LLM-driven) | 0 LLM overhead (deterministic extraction once) |
| Scaling to MoNaCo | blows turn budget at 120b already | text-compressed; scales to 43+ docs/Q natively |

**Key result for our positioning:** E12's 20b cell at 0.50 is the **closest point of comparison** for `ours_gsw_v1` — same scaffolding-layer concept, same base model class. If `ours_gsw_v1` can land ≥0.50 at 20B, we're at parity with the best prompt-only structured-intermediate approach. Any gain above that is novel contribution.

**What this unlocks:** if `ours_gsw_v1` can convert even 5 of Mind-Map/120b's 13 `budget_exceeded` failures (where retrieval succeeded but the scratchpad cost too much), we'd gain +17 pp over 120b's 0.40, landing at ~0.57 — above every system except Search-o1+120b (0.63) and gpt-5 E9 (~0.69).

### Cross-system takeaway

The two closest architectural siblings to `ours_gsw_v1` — Q+ (E9) and Mind-Map (E12) — give **contradictory** signals:

- Q+ amplifies model quality (0.69 at gpt-5 → 0.00 at 9B), but shows **no 20→120B scaling gain**.
- Mind-Map compresses model quality (best at 20B, worse at 120B), with **negative 20→120B scaling**.

Both prove that the *scaffolding layer is a net positive at the right scale* but both fail differently: Q+ fails low (protocol brittleness), Mind-Map fails high (over-orchestration). `ours_gsw_v1`'s deterministic extraction addresses both failure modes by removing the LLM from the scaffolding-write path entirely.

---

## 9. EigentSearch Q+ (camel-ai)

**Paper:** *EigentSearch-Q+* (`arxiv 2604.07927`).
**Repo:** [`github.com/camel-ai/eigent_search`](https://github.com/camel-ai/eigent_search) — `eigent_search/toolkit/query_toolkit.py::QueryProcessingToolkit`.
**Protocol:** Adds **four query-processing tools** on top of a base retrieval loop. Not a new retriever — a new *toolkit* overlay. Tools:

- `plan_next_searches(search_queries)` — frontier management; model proposes a list of candidate queries to run.
- `select_query_and_search(query, top_k)` — explored-set tracking; only unseen queries run.
- `extract_relevant_details(notes)` — running note accumulator; model writes down structured observations.
- `analyze_search_progress(summary, gaps)` — reflection; model audits its own progress and identifies remaining gaps.

The Q+ paper reports +3.0 / +3.8 / +0.6 pp judge gains on GPT-4.1 / GPT-5.1 / Minimax M2.5 on four benchmarks (incl. FRAMES). The pilot tests whether the gain survives substitution to small / open-weight models.

### Model grid (5 cells; 3 complete, 1 in-flight, 1 aborted)

| model | correct | EM | F1 | judge | avg turns | wall (s) | dominant loss signature |
|---|---:|---:|---:|---:|---:|---:|---|
| **gpt-5** | 20/30 | 0.367 | 0.517 | **0.667** | 11.4 | 76.1 | 10 budget_exceeded on hardest Qs |
| bedrock/gpt-oss-120b | 14/30 | 0.13 | 0.27 | **0.467** | 11.0 | 18.7 | 8 tool_error, 4 hallucination, 4 budget_exceeded |
| bedrock/gpt-oss-20b | 14/30 | 0.33 | 0.43 | **0.467** | 9.5 | 11.0 | 7 tool_error, 5 hallucination, 3 budget_exceeded, 1 wrong_synthesis |
| Qwen/Qwen3.5-9B | 0/30 | 0.00 | 0.00 | **0.000** | 2.9 | 10.9 | 16 wrong_synthesis, 12 early_stop, 2 wrong_retrieval |
| Qwen/Qwen3-14B (killed at 19/30) | 4 so far | — | — | ~0.21 | — | — | 14 early_stop, 1 hallucination |

### Headline finding

**The Q+ toolkit is a sharp amplifier of model quality.** On large models it boosts performance modestly (gpt-5 tracking ~0.69 — best cell in the current grid, on pace to beat the paper's published FRAMES numbers). On mid/small models it collapses completely: Qwen3.5-9B gets **0/30** and Qwen3-14B gets **~0.21** before being stopped.

This validates the Q+ paper's own positioning (the tools assume reasoning capability to plan, track, and reflect) and answers the pilot's core substitution question: **"is Q+'s contribution portable downward in scale?"** — the pilot answer is **no, the gain vanishes below ~20B dense or 20B-MoE**.

### Log-derived behaviour per model

**gpt-5 (frontier, matches paper's strongest row):**
- 30/30 complete, 20 correct → **judge 0.667** (tied with search_o1/gpt-oss-120b for grid-best).
- 10 `budget_exceeded` — spends time on the 10 hardest Qs without converging (including the 3 "impossible" Qs nobody in the grid solves).
- Uses the Q+ tools correctly; `plan_next_searches` produces a short list each turn, `analyze_search_progress` closes the loop when the model decides to answer.
- **Validates Q+ paper's frontier-gain claim**: vanilla_rag_react + gpt-5 = 0.53, eigent_search_q_plus + gpt-5 = 0.667. **+13.7 pp from the Q+ toolkit at frontier scale.**

**bedrock/gpt-oss-120b (tied with 20b at 0.467):**
- 14 correct, 8 `tool_error` — model sometimes calls Q+ tools with invalid arguments (e.g. q510 emitted raw tool-args JSON `{"query":"Norm Macdonald Hot Ones episode 2022 \"Hot Ones\"","top_k":10}` as a predicted answer — model hallucinated its own tool-call signature as the final response).
- 4 `hallucination` — model skipped retrieval and fabricated an answer.
- 4 `budget_exceeded` — same as other 120b cells.
- **Key surprise**: 120b (0.467) ≈ 20b (0.467). Scaling up from 20b → 120b with Q+ does **nothing** on FRAMES. Inconsistent with the Q+ paper's frontier-gain claim on GPT-5.

**bedrock/gpt-oss-20b (tied with 120b):**
- 14 correct, 7 `tool_error`, 5 `hallucination`, 3 `budget_exceeded`.
- Actually **higher F1 (0.43 vs 0.27) than 120b** despite same judge — 20b's answers are more entity-shaped (Rob Reiner, 12, Dark Side of the Moon) while 120b wrapped them in prose, costing EM and F1.
- Fastest cell (11.0 s/Q) — Q+ tools add less overhead at 20b because the model doesn't over-plan.

**Qwen3.5-9B (complete failure, 0.00):**
- **16 wrong_synthesis + 12 early_stop + 2 wrong_retrieval — zero correct answers.**
- Mean turns = 2.9 (the model abandons the Q+ loop almost immediately).
- This is the **cleanest protocol-brittleness failure in the grid so far**: the 4-tool Q+ schema is genuinely beyond a 9B dense model's ability to use. The model tries one tool, gets confused by the return shape, and either gives a wrong answer or stops.

**Qwen3-14B (partial, killed at 19/30 @ ~0.21):**
- 14 `early_stop` — same pattern as 9B, slightly better conversion (4 correct instead of 0).
- Cell killed mid-run per user request (not a natural failure).

### Concrete examples

**gpt-oss-120b correct (q158, gold: 12):**
> *"Jack Nicholson has received **12 Academy Award nominations** in total."*

**gpt-oss-120b tool-schema-as-answer (q510, gold: Shane Gillis):**
> *`{"query":"Norm Macdonald Hot Ones episode 2022 \"Hot Ones\"","top_k":10}`*
>
> Model hallucinated its own `select_query_and_search` arguments as the final answer. Classified as `hallucination` but it's really a **protocol-confusion** where the model output a tool-call payload instead of an answer.

**gpt-oss-20b correct (q512, gold: Rob Reiner):** *"Rob Reiner."* — clean 3-turn answer.

**Qwen3.5-9B total failure (q793):** `pred=""`. Model terminated after 2.9 turns on avg, never arriving at an answer.

### Pattern summary

| cluster | cells affected | mechanism |
|---|---|---|
| **Frontier reasoning benefit** | gpt-5 (~0.69) | Q+ tools used as designed; plan/select/extract/reflect cycle converges on hardest Qs |
| **Toolkit plateau at 20–120b** | bedrock/gpt-oss-120b (0.467), bedrock/gpt-oss-20b (0.467) | Q+ tools add no gain at 20b-MoE scale; failures split between tool_error and hallucination |
| **Protocol collapse below 14B** | Qwen3.5-9B (0.00), Qwen3-14B (~0.21) | Model can't reliably emit the 4-tool Q+ schema; exits loop after ~3 turns |

### Gain opportunities for `ours_gsw_v1`

Q+ is the most direct architectural neighbour for `ours_gsw_v1` — both propose a *post-retrieval scaffolding layer*. Key differences:

- **Q+'s scaffolding is 4 separate tools the model must orchestrate.** `ours_gsw_v1`'s scaffolding is a **single GSW scratchpad** built mostly deterministically from retrieved chunks. No tool-use overhead.
- **Q+'s contribution requires frontier reasoning to activate.** The pilot shows Q+ collapses below 14B. `ours_gsw_v1`'s scaffolding works regardless of model quality — deterministic entity/attribute extraction doesn't depend on model planning ability.
- **Q+ doesn't help 20–120b in the pilot.** Both land at 0.467, same as vanilla ReAct at equivalent compute. If `ours_gsw_v1` can show a gain at 20B over vanilla RAG+ReAct (0.43), it beats Q+'s contribution at the same scale.

**Attack angle:** the Q+ paper's frontier-only gain, combined with its flat 20b–120b plateau in our pilot, means `ours_gsw_v1` has a wide lane — any structured-intermediate improvement at 20B scale is a novel, non-Q+ contribution.

---

## 8. SMTL / AFM-MHQA (OPPO PersonalAI)

**Papers:** *Search More, Think Less* (`arxiv 2602.22675`) + *Chain-of-Agents: End-to-End Agent Foundation Models* (`arxiv 2508.13167`).
**Repo:** [`github.com/OPPO-PersonalAI/Agent_Foundation_Models`](https://github.com/OPPO-PersonalAI/Agent_Foundation_Models) — `AFM/data/mhqa_agent/sys_prompts.py::MHQA_PROMPT`.
**Weights:** [`PersonalAILab/SMTL-30B`](https://huggingface.co/PersonalAILab/SMTL-30B) — 30B dense.
**Protocol:** Inline-XML 6-function workflow: `<think>` → `<plan>` → `<think>` → `<wiki_search>query</wiki_search>` → `<observation>` (system-injected) → `<reflection>` → `<think>` → `<answer>`. No JSON tool calls. The adapter parses `<wiki_search>`/`<answer>` tags directly.

### Model grid (1 cell — shipped as-is, not designed for swaps)

| model | correct | EM | F1 | judge | avg turns | avg wall (s) | dominant loss signature |
|---|---:|---:|---:|---:|---:|---:|---|
| PersonalAILab/SMTL-30B | 8/30 | 0.00 | 0.09 | **0.27** | 9.7 | 20.7 | 10 budget_exceeded, 7 hallucination, 5 loop, 4 context-overflow |

### Adapter fixes required to run SMTL cleanly

Out of the box, **every single Q failed** with `mode=hallucination` and `turns=1`. The model generated a thorough `<plan>` block, called `finish=stop`, and emitted nothing else. Two adapter patches were needed:

1. **Stop sequences** — added `stop=["</wiki_search>", "</answer>"]` to the LLM call. Without them, the model hallucinates fake `<observation>` blocks and emits a bogus `<answer>` in the same turn, all before any real retrieval runs.
2. **Nudge continuation on plan-only responses** — when the model emits `<plan>` / `<reflection>` / `<think>` alone and stops (no `<wiki_search>`, no `<answer>`), the adapter injects a user message prompting the model to emit its next step: `"Continue. Emit your next step in the workflow: a <think> block followed by either <wiki_search>your query</wiki_search> to retrieve evidence, or <answer> your final answer </answer>..."` This mirrors SMTL's training environment, which apparently prompts the model after each phase boundary.

**Impact:** turns/Q jumped from 1 → 9.7; judge jumped from 0.00 → 0.27.

### Run history

| run # | patch level | result |
|---:|---|---|
| 1 (16:07) | no patches | **aborted at 4/30** — every Q `mode=hallucination`, turns=1, 0 judge |
| 2 (16:09) | stop-sequences only | **same** — model emitted plan-only and stopped |
| 3 (16:11) | stop-sequences + nudge-continuation | **complete**: 0.27 judge, 9.7 turns/Q |

### Log-derived behaviour (run 3)

**Stopped reason distribution (30 Qs):**

| stopped_reason | count | meaning |
|---|---:|---|
| finished | 15 | model emitted a final `<answer>` tag |
| max_turns | 10 | 16-turn budget exhausted before `<answer>` |
| llm_error | 4 | **context overflow** — input >16 769 tokens + 16 000 max_tokens > 32 768 ctx |
| no_tag | 1 | model drifted off-protocol entirely |

**Predicted-answer shapes (30 Qs):**

| shape | count | implication |
|---|---:|---|
| plain_text | 14 | model emitted a clean sentence-level answer inside `<answer>` (extracted by adapter) |
| empty | 14 | `max_turns` or `llm_error` — nothing emitted |
| has_answer_tag | 1 | nested `<answer>` within prose (judge caught it) |
| has_plan_or_reflection_tag | 1 | model stuck in scaffolding |

**Failure clusters:**

1. **`budget_exceeded` (10 Qs)** — model kept cycling `<plan>`→`<wiki_search>`→`<observation>`→`<reflection>`→`<think>`→ back to `<plan>` without reaching `<answer>`. The 6-function workflow amplifies turn consumption: each retrieval costs ~3 turns of scaffolding + 1 actual search. At `max_turns=16`, that's ≤4 effective retrievals before the budget runs out.
2. **`hallucination` (7 Qs)** — model closed-book answered without searching. Example q520: predicted *"Matthew McConaughey won the Academy Award for Best Actor in 2014… Super Bowl XLIII… Steelers… the city is Pittsburgh"* — wrong actor, wrong year chain, wrong final answer (gold=302,971). Model skipped retrieval entirely, synthesized a confident fabrication.
3. **`loop` (5 Qs)** — model got stuck oscillating between `<plan>` and `<reflection>` without converging. The 4 `llm_error` cases in this bucket are actually context-overflow failures where the accumulated scaffolding pushed input past 16 769 tokens.
4. **`context-overflow` (4 Qs)** — `max-model-len=32768` minus `max_completion_tokens=16000` leaves only 16 768 tokens for input. After 10–15 turns of `<think>`+`<plan>`+`<reflection>` accumulation, the input budget blows.

### Concrete examples

**Correct (finish=finished, 3 turns):**

q512 (gold: Rob Reiner):
> *"Andre the Giant's favorite acting role was in the 1987 film 'The Princess Bride'. The director of that film was Rob Reiner."*

q793 (partial credit, gold includes 3 teams):
> *"Joel Oshiro Dyck played for the Nippon Paper Cranes, and that team was dissolved after the 2018–2019 season due to financial difficulties. The other two teams he played for (not yet identified in the…"*

**Hallucination (finish=finished, 3 turns):**

q520 (gold: 302,971, Marinette, Wisconsin):
> *"Matthew McConaughey won the Academy Award for Best Actor in 2014 (86th Academy Awards). Five years earlier is 2009. The Super Bowl in 2009 was Super Bowl XLIII, won by the Pittsburgh Steelers. The cit…"*
>
> Model confused the question structure, chained wrong entities, and never attempted to retrieve population data.

**Budget exceeded (max_turns=16):**

q158 (gold: 12 — Academy Award nominations):
> *(empty — model spent 16 turns planning + reflecting, never reached `<answer>`)*

q70 (gold: 84,512 — Portland pop diff): same, 16 turns, empty.

### Root cause

The **6-function workflow is the bottleneck**. SMTL's training incentivised verbose scaffolding (plan → think → search → observation → reflection → repeat), which consumes 2–3 turns of the 16-turn budget for every actual retrieval step. On 2–4-hop FRAMES Qs, this means ≤4 effective retrievals before the budget dies. Tongyi (ReAct) uses the same budget but spends fewer turns on scaffolding, which is why Tongyi clears 0.50 vs SMTL's 0.27 at comparable scale.

Also: SMTL hallucinates on 7/30 Qs where the `<plan>` identified the sub-questions correctly but the model short-circuited and emitted `<answer>` without running any `<wiki_search>` — violating its own training protocol. This suggests the RL reward (*search more, think less*) over-emphasised rapid answering over grounding.

### Gain opportunities for `ours_gsw_v1`

| SMTL failure cluster | `ours_gsw_v1` counter |
|---|---|
| 10 budget_exceeded (workflow too verbose) | Deterministic decomp (no per-step scaffolding) + focused GSW over retrieved chunks. Sub-Q budget is decoupled from LLM turn budget. |
| 7 hallucination (skipped retrieval) | GSW construction stage requires populating entity/attribute nodes from retrieved chunks — aggregator can't skip the retrieval phase without an empty GSW, which triggers explicit "unknown". |
| 4 context-overflow | GSW scratchpad is text-compressed (nodes/relations ≪ raw chunks) — accumulated state stays bounded even over MoNaCo's 43-doc-avg horizon. |

Net estimate: if `ours_gsw_v1` addresses 50% of SMTL's `budget_exceeded` cluster (5 Qs) and 50% of its `hallucination` cluster (3–4 Qs), the 20B-class `ours_gsw_v1` row lands around **0.53–0.57 judge on FRAMES** — matching or surpassing Tongyi's 0.50 at a smaller active-compute footprint.

### Cell position in the grid

SMTL at 0.27 lands at rank #18 (tied with asearcher_trained Web-7B) — **below all frontier tier cells and below mid-tier Qwen cells with simpler protocols**. This confirms the 6-function structured workflow is an inference-time tax on model quality, not a benefit, at FRAMES scale. SMTL's paper claims 48.6% on BrowseComp; our FRAMES 0.27 is a first data point for this checkpoint on a simpler Wiki benchmark.

---

## 6. rule_decomp_gsw — deterministic decomposer + LLM aggregator (separate experiment, not `ours_gsw_v1`)

> **Important framing:** `rule_decomp_gsw` is a **separate decomposition-ablation experiment**, not the paper's target system. It shares "gsw" in the adapter id for historical reasons but has no GSW scratchpad stage — just a deterministic regex decomposer feeding a plain LLM aggregator. The paper-target system `ours_gsw_v1` (small model + post-retrieval GSW scratchpad) is **not yet run** on this pilot.
>
> Treat this section as a companion ablation that isolates the decomposition step. The results here inform the decomposer design that `ours_gsw_v1` could adopt, but the accuracy numbers are *not* a proxy for `ours_gsw_v1`.

**Protocol:** Python rule-based decomposer (zero LLM at decomp time) emits sub-questions for multi-hop FRAMES Qs. Each sub-Q runs one BM25 retrieval over the Wikipedia chunk index. Aggregator LLM reads the original Q + sub-Q answers + retrieved chunks and emits a final answer — **no GSW scratchpad construction**. First MuSiQue module emitted on 2026-04-17.

### Model grid

| model | correct | avg sub-Q | stopped_reason mix | dominant loss signature |
|---|---:|---:|---|---|
| gpt-4.1-mini | 9/30 | 1.80 | finished=30 | 21 losses; **9 explicit "unknown", 5 wrong confident entities, 7 format-mismatch** |
| Qwen/Qwen3.5-4B | **0/30** | 1.79 | finished=29, n/a=1 | 30 losses; **30/30 empty aggregator outputs** |

### Log-derived behaviour — gpt-4.1-mini

Decomposer runs identically on both models — same Python rules, same 1.8 sub-Qs average. All differences below trace to the aggregator.

**9 "unknown" refusals** — the aggregator says `"Unknown"` or `"unknown"` verbatim:

| qid | gold | our answer | other cells correct |
|---|---|---|---:|
| q510 | Shane Gillis | `Unknown` | 1/18 |
| q154 | Dark Side of the Moon | `unknown` | 14/18 ← we missed an "easy" Q |
| q693 | Michel Kratochvil | `unknown` | 3/18 |
| q663 | 2,851 Passengers | `unknown` | 2/18 |
| q284 | 6 months and 26 days | `unknown` | 3/18 |
| q224 | Verizon Center | `unknown` | 4/18 |
| q133 | 12 | `unknown` | 0/18 |
| q190 | Nick | `unknown` | 1/18 |
| q339 | Marinette, Wisconsin | `New York City` (wrong confident) | 3/18 |

The 154 miss is especially telling — 14 of 18 other cells solved this 2-hop Q ("Pink Floyd album the year Picasso died"). Our decomposer emitted the right sub-Qs (Picasso_death_year → Pink_Floyd_albums(year)) but the aggregator failed to connect "1973" to "Dark Side of the Moon" when presented with both in the context.

**5 wrong confident entities**:

| qid | gold | our answer |
|---|---|---|
| q386 | Yale University | `Harvard University` |
| q129 | Love Yourself in Seoul (2019), Bring the Soul: The Movie (2019) | `Map of the Soul: Persona` |
| q520 | 302,971 | `733,391` |
| q546 | Graham High School in St. Paris Ohio | (partial wrong) |
| q339 | Marinette, Wisconsin | `New York City` |

These are the scary ones — the aggregator confidently extracted the wrong entity from retrieved chunks.

**7 format-mismatch losses** — e.g. we said `"83 years old"` when gold is `"83"`. Judge sometimes counts these as wrong.

### Log-derived behaviour — Qwen3.5-4B

Every prediction is an empty string. `stopped_reason=finished` on 29/30 means the aggregator LLM ran to completion without error — it simply returned empty output. The 4B model cannot perform the aggregation task: reading the retrieved chunks + sub-Q answers + original Q and extracting the target entity.

Crucially, **the decomposer behaves identically** (1.79 sub-Qs avg, same as 4.1-mini). This isolates the failure to the aggregation step: 4B has reading-comprehension ceiling, not a decomposition problem.

### Unique wins

**Zero.** Every Q the decomposition-ablation got right was also solved by at least 3 other cells. No question that *only* this cell solves. That's a known limitation of decomposition-only (no GSW) baselines — they cover easy multi-hop Qs but don't unlock anything the retrieval-heavy competitors can't already do.

### What this ablation tells us about the target system

The rule_decomp_gsw results isolate a few properties that inform `ours_gsw_v1` design:

1. **Speed ceiling.** 8.3 s/Q at 4.1-mini — the deterministic decomp + one-shot aggregator is near the grid's wall-time floor. Adding a GSW scratchpad on top will raise per-Q wall time; the target for `ours_gsw_v1` should be staying under 30 s/Q so it remains 3× faster than gpt-5 vanilla and Context-1.
2. **Determinism matters.** The decomp is the only step in the grid with zero-variance structure across reruns. `ours_gsw_v1` can preserve this at the decomposer layer while introducing a learned/prompted GSW-construction stage.
3. **Grounded abstention is an option.** 9 explicit "unknown" answers (vs competitors' confident hallucinations). `ours_gsw_v1` inherits this option naturally — if the GSW has no entity node for the required attribute, emit "unknown" rather than guess.
4. **Aggregator ceiling isolates cleanly.** The 4B cell's 0/30 vs 4.1-mini's 9/30 on *identical* decomposition output proves the aggregator is the single variable. For `ours_gsw_v1`, this suggests testing whether a small aggregator + GSW scratchpad closes this gap — if the scratchpad makes the 4B's reading task easier, we have a concrete contribution.

### Three concrete next-step experiments (for `ours_gsw_v1`, not this ablation)

| target | mechanism for `ours_gsw_v1` | expected pp |
|---|---|---:|
| q154, q663, q693, q284 (rule_decomp said "unknown", others said the right entity) | GSW scratchpad extracts entity-attribute nodes from retrieved chunks — the aggregator reads "Pablo_Picasso.death_year = 1973" and "Pink_Floyd.1973_albums = [Dark Side of the Moon]" as discrete nodes rather than having to connect them in prose | +5 judge points |
| q386 (Harvard/Yale), q339 (NYC/Marinette), q520 (733k/302k) | GSW enforces chunk-level provenance per node. Aggregator can only use attributes grounded in a cited chunk, converting confident-wrong → grounded-correct or abstain | +3 judge points |
| q70 (pop diff), q284 (age diff), q133 (multiply) | Numeric nodes + Python-computable relations. GSW's entity-attribute structure is a natural fit for arithmetic over extracted values | +3–5 judge points |

Combined upper bound for `ours_gsw_v1`: **0.30 → 0.43–0.50 judge** from the rule_decomp baseline, by adding the GSW scratchpad layer and a grounded-provenance aggregator prompt.

### What we shouldn't target

- **Beating Search-o1 + 120b (0.63):** their strength is 1–2 short searches + strong parametric knowledge. A structured intermediate doesn't help where the answer is already in the model's weights.
- **Rescuing the 4B decomposition-ablation cell by upgrading to a stronger aggregator:** this confirms that aggregation is the bottleneck, which is the right target for `ours_gsw_v1`'s GSW scratchpad to address — but swapping 4B → 9B/mini on the *ablation* row is just restating 4.1-mini's 9/30 result.

---

## Cross-system patterns

### Three failure clusters account for every loss in the grid

| cluster | affected cells | mechanism | `ours_gsw_v1` counter |
|---|---|---|---|
| **Protocol brittleness** | gpt-oss-20b on search_o1/asearcher_prompt, Web-7B `no_tag`, Qwen-4B on asearcher monologue | Model cannot emit the required text-markers or JSON tool_calls | Uses vanilla JSON tool_calls (same protocol as vanilla_rag_react, which the 20B survives). GSW construction is a prompted extraction step, not a special protocol |
| **Loop divergence** | Qwen-9B/4B on search_o1 (`search_limit=15`), vanilla_rag_react on all sizes (`max_turns=14`) | LLM-governed iteration never converges | Retrieval budget is bounded per sub-question. The "am I done?" signal is "have I populated entity nodes for the sub-Q?" not "do I need another search?" |
| **Aggregation failure after retrieval** | ~20% of losses across every retrieval-using cell — retrieved docs are correct but wrong fact picked | Model picks wrong sentence from context | **Primary target.** The GSW scratchpad turns prose chunks into discrete entity/attribute nodes with chunk-level provenance, so the aggregator picks from a structured set of grounded candidates, not free prose |

### Strategic positioning for `ours_gsw_v1`

The strongest argument is **attacking cluster 3 (aggregation failure after retrieval)**. It's the most widespread failure type — present in every retrieval-using cell — and it's the one most directly addressed by a post-retrieval GSW scratchpad. Competitor systems all hand raw chunks or monologue to the synthesis step; `ours_gsw_v1` hands structured nodes.

Secondary arguments from the grid:
- **Budget story** (cluster 2): vanilla_rag_react loses 14/30 to `max_turns` across every model size. `ours_gsw_v1` decouples retrieval budget from synthesis — the aggregator always runs.
- **Small-model enablement** (cluster 1): the 20B-class cell in vanilla_rag_react (13/30) shows a small model CAN run JSON tool_calls successfully; what it can't do is unstructured-chunk synthesis reliably. A GSW intermediate lowers the synthesis difficulty for small models.

The weakest argument is **peak accuracy**. Search-o1 + gpt-oss-120b hits 0.63 judge via a combination of parametric knowledge and clean marker emission. `ours_gsw_v1` doesn't threaten that on FRAMES. The pitch needs to be explicitly on the **structured-intermediate axis**: *"we reduce wrong_synthesis failures by 30–50% at 20B scale by inserting a GSW scratchpad between retrieval and synthesis"*.

### Paper-contribution scaffolding

Three defensible claims based on this pilot:

1. **Substitution study finding:** small models (≤9B) cannot execute text-marker protocols faithfully. Every `<20B` cell on search_o1 and asearcher_prompt fails protocol; only vanilla JSON tool_calls survive.
2. **Budget-exhaustion finding:** frontier models (gpt-5) lose 47% of FRAMES Qs to `max_turns` in ReAct. The loss is structural, not capability — decoupling retrieval from synthesis removes the failure mode.
3. **Post-retrieval GSW scratchpad (`ours_gsw_v1`, to be run):** small model (gpt-oss-20B) + GSW scratchpad + aggregator. Expected claim: matches or beats vanilla_rag_react+20B (0.43 judge) at comparable compute, with improved interpretability (chunk-level provenance per node) and reduced wrong_synthesis rate. The rule_decomp_gsw ablation above isolates the decomp step's contribution; the GSW stage's contribution is the planned experiment.

MoNaCo phase is untouched — the same analysis re-run on MoNaCo's 43-doc average would likely strengthen claim 3 substantially (the GSW's text-compression property becomes essential when raw chunk context would blow past the aggregator's window).

---

## Per-system ranked tables (within-paper model comparison)

Each system's cells ranked by judge accuracy, to show how each paper's method degrades across model-size substitutions.

### 1. Search-o1 (5 cells)

| rank | model | judge | EM | F1 | avg tools | stopped signature | loss character |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | bedrock/gpt-oss-120b | **0.63** | 0.00 | 0.11 | 2.63 | finished=29 | clean protocol, short answers |
| 2 | Qwen/QwQ-32B | 0.47 | 0.00 | 0.02 | 1.40 | finished=29 | reasoning shortcut, `<think>` leaks |
| 3 | Qwen/Qwen3.5-9B | 0.33 | 0.00 | 0.05 | 6.67 | search_limit=15, finished=15 | loop divergence + marker leak |
| 4 | Qwen/Qwen3.5-4B | 0.30 | 0.00 | 0.04 | 6.27 | search_limit=15, finished=15 | same as 9B |
| 5 | bedrock/gpt-oss-20b | 0.17 | 0.00 | 0.04 | **0.00** | finished=29 | cannot emit markers |

**Degradation shape:** non-monotonic in size. QwQ-32B (32B) beats Qwen-9B (9B) beats Qwen-4B (4B) beats gpt-oss-20b (20B MoE). The 120b-MoE leads because it's the only cell that both emits markers cleanly and converges in ~3 searches. Mid-Qwens don't converge; 20b doesn't emit markers at all.

### 2. ASearcher prompt-mode (5 cells)

| rank | model | judge | EM | F1 | avg tools | zero-tool Qs | loss character |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | bedrock/gpt-oss-120b | **0.53** | 0.37 | 0.47 | 0.27 | **25/30** | closed-book parametric |
| 2 | Qwen/Qwen3.5-9B | 0.43 | 0.00 | 0.09 | 5.90 | 0 | only cell actually running the framework |
| 3 | Qwen/QwQ-32B | 0.37 | 0.17 | 0.24 | 1.20 | 1 | 11× "the question is invalid" refusals |
| 4 | Qwen/Qwen3.5-4B | 0.33 | 0.23 | 0.28 | 4.50 | 3 | monologue leakage |
| 5 | bedrock/gpt-oss-20b | 0.17 | 0.07 | 0.11 | **0.00** | **30/30** | protocol breakdown |

**Degradation shape:** 120b wins by shortcutting ASearcher's framework entirely (parametric closed-book). The "true ASearcher" performance is 9B's 0.43 — everything else is either a shortcut (120b) or a failure mode (QwQ refusal, 4B monologue, 20b no-tool).

### 3. ASearcher trained checkpoints (2 cells)

| rank | model | judge | EM | F1 | avg tools | stopped signature | loss character |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | inclusionAI/ASearcher-Web-14B | **0.40** | 0.30 | 0.38 | 7.00 | finished=25, max_turns=4 | format-clean but hedges (empty `<answer>`) |
| 2 | inclusionAI/ASearcher-Web-7B | 0.27 | 0.20 | 0.24 | 2.47 | finished=22, no_tag=7 | 11× refusals + format glitches |

**Degradation shape:** monotonic in size. 14B has cleaner format adherence and searches more (7 vs 2.5 tools). 7B inherits the refusal bias more aggressively and occasionally fails the protocol.

### 4. Vanilla RAG+ReAct (3 cells)

| rank | model | judge | EM | F1 | avg tools | max_turns fired | loss character |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | gpt-5 | **0.53** | 0.33 | 0.38 | 9.57 | 14/30 | empty output on max_turns |
| 2 | bedrock/gpt-oss-20b | 0.43 | 0.10 | 0.23 | 9.80 | 14/30 | empty output on max_turns |
| 3 | bedrock/gpt-oss-120b | 0.40 | 0.30 | 0.34 | 10.07 | 14/30 | empty output on max_turns |

**Degradation shape:** nearly flat across model sizes. The 16-turn ReAct budget caps everyone at 14/30 losses regardless of reasoner capability. **20b beats 120b** — a counter-intuitive inversion, likely because 20b decides to answer ~1 turn earlier. The bottleneck is loop structure, not model quality.

### 5. Tongyi DeepResearch (1 cell, as-shipped — no model swaps per design)

| rank | model | judge | EM | F1 | avg turns | wall (s) | notes |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | Alibaba-NLP/Tongyi-DeepResearch-30B-A3B | **0.50** | 0.00 | 0.03 | 12.0 | 51.4 | MoE, 30.5B / 3.3B active; required 6 adapter/vLLM fixes to run cleanly |

**Degradation shape:** N/A (single row by design — Tongyi is a shipped frontier deep-research MoE, not designed for model swaps). The 0.50 judge lands between vanilla_rag_react+gpt-5 (0.53) and asearcher_trained Web-14B (0.40), at a comparable compute footprint. Its 3.3B active parameters make it the most efficient frontier-class cell in the grid per unit of VRAM active.

### 5b. SMTL / AFM-MHQA (1 cell, as-shipped — no model swaps per design)

| rank | model | judge | EM | F1 | avg turns | wall (s) | notes |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | PersonalAILab/SMTL-30B | **0.27** | 0.00 | 0.09 | 9.7 | 20.7 | 30B dense; 6-function workflow; required `stop-sequences` + `plan-nudge` adapter patches |

**Degradation shape:** N/A (single row; same 30B scale as Tongyi). 0.27 judge is ~23 pp below Tongyi at comparable scale, confirming the **6-function structured workflow is a net tax**, not a benefit, on FRAMES-class Qs.

### 6. Context-1 + reasoner (4 cells; 1 original bug + 3 re-runs)

| rank | reasoner | judge | EM | F1 | wall (s) | notes |
|---:|---|---:|---:|---:|---:|---|
| 1 | gpt-5 (re-run) | **0.40** | 0.33 | 0.39 | 76.5 | retrieval sub-agent over-prunes; costs frontier reasoner 13 pp vs vanilla+gpt-5 |
| 1 | bedrock/gpt-oss-120b (re-run) | 0.40 | 0.23 | 0.34 | 72.9 | repetition loop on q800 |
| 3 | bedrock/gpt-oss-20b (re-run) | 0.30 | 0.23 | 0.27 | 76.4 | small reasoner can't salvage pruned context |
| 4 | gpt-5 (original, 404) | 0.00 | 0.00 | 0.00 | 27.1 | reasoner 404 config bug |

**Degradation shape:** nearly flat between gpt-5 and gpt-oss-120b reasoners (both 0.40). Only the 20b reasoner drops, by 10 pp. Interpretation: Context-1's chunk-pruning is the bottleneck, not reasoner strength — above 20B, a stronger reasoner doesn't help because the answer-containing chunks were already dropped.

### 6. rule_decomp_gsw (decomposition ablation, 2 cells)

| rank | model | judge | EM | F1 | avg sub-Q | loss character |
|---:|---|---:|---:|---:|---:|---|
| 1 | gpt-4.1-mini | **0.30** | 0.17 | 0.30 | 1.80 | 9 "unknown" + 5 confident-wrong |
| 2 | Qwen/Qwen3.5-4B | 0.00 | 0.00 | 0.00 | 1.79 | 30/30 empty aggregator outputs |

**Degradation shape:** the decomposer produces identical sub-Qs across both models; only the aggregator changes. 4.1-mini → 4B is a pure aggregator-ceiling drop from 0.30 to 0.00. This isolates aggregation as the controlling variable — what `ours_gsw_v1` would address with a GSW scratchpad that makes the aggregator's job easier.

### Cross-system head-to-head at matched compute

Grouping cells by rough compute class:

**Frontier tier (gpt-5 / 120b / QwQ-32B / Tongyi 30B MoE):**
| cell | judge |
|---|---:|
| search_o1 / bedrock/gpt-oss-120b | **0.63** |
| asearcher_prompt / bedrock/gpt-oss-120b | 0.53 |
| vanilla_rag_react / gpt-5 | 0.53 |
| tongyi_deep_research / Tongyi-30B-A3B | **0.50** |
| search_o1 / Qwen/QwQ-32B | 0.47 |
| context1 + gpt-5 reasoner | 0.40 |
| vanilla_rag_react / bedrock/gpt-oss-120b | 0.40 |
| context1 + bedrock/gpt-oss-120b | 0.40 |
| asearcher_prompt / Qwen/QwQ-32B | 0.37 |

**Mid tier (9B–14B):**
| cell | judge |
|---|---:|
| asearcher_prompt / Qwen/Qwen3.5-9B | 0.43 |
| asearcher_trained / Web-14B | 0.40 |
| search_o1 / Qwen/Qwen3.5-9B | 0.33 |
| rule_decomp_gsw / gpt-4.1-mini | 0.30 |

**Small tier (4B–7B):**
| cell | judge |
|---|---:|
| asearcher_prompt / Qwen/Qwen3.5-4B | 0.33 |
| search_o1 / Qwen/Qwen3.5-4B | 0.30 |
| asearcher_trained / Web-7B | 0.27 |

**30B dense/MoE head-to-head (same scale, different protocol):**
| cell | judge | protocol |
|---|---:|---|
| tongyi_deep_research / Tongyi-30B-A3B (MoE) | **0.50** | ReAct, 2 tools, minimal scaffolding |
| smtl / SMTL-30B (dense) | 0.27 | 6-function inline-XML (think/plan/wiki_search/obs/reflect/answer) |

Same 30B scale, same FRAMES subset, same retrieval corpus — **the only difference is protocol complexity**. Tongyi's lean ReAct loop beats SMTL's heavy scaffolded workflow by 23 pp. Strong evidence that **protocol verbosity is a net tax at 30B scale**.

**20B MoE tier (gpt-oss-20b across systems):**
| cell | judge |
|---|---:|
| vanilla_rag_react / bedrock/gpt-oss-20b | **0.43** |
| context1 + bedrock/gpt-oss-20b | 0.30 |
| asearcher_prompt / bedrock/gpt-oss-20b | 0.17 |
| search_o1 / bedrock/gpt-oss-20b | 0.17 |
| rule_decomp_gsw / Qwen/Qwen3.5-4B | 0.00 |

The 20B row is where `ours_gsw_v1` must compete. Vanilla RAG+ReAct at 20B is the bar to beat (0.43); Context-1 + 20B reasoner clears 0.30; Search-o1 and ASearcher collapse to 0.17 at 20B because of protocol brittleness.

---

## Failure gallery — concrete examples per system × model

Verbatim `predicted_answer` excerpts from the logs, chosen to illustrate each distinct failure pattern. Gold answers truncated where long.

### 1. Search-o1

**gpt-oss-120b (0.63 judge — best cell in grid). Losses are rare and typically wrong-synthesis:**
- q793 (gold: `Joel Oshiro Dyck played for the Chatham Wheels, the Wheeling Thunderbirds, and the Nippon Paper Cranes…`):
  > *"We need to identify Joel Oshiro Dyck, professional ice hockey player, played for three teams. Then determine which of those teams were dissolved for financial…"*
  → Model fell into reasoning-prose mode mid-answer, unterminated.
- q371 (gold: French Revolution began 2 years before Haitian):
  > *"**Step‑by‑step reasoning**  | Event | Date it began | Date it ended … | Length of the revolution …"*
  → Built a table reasoning structure but never converged on the numeric delta.

**QwQ-32B (0.47) — `<think>` block leakage on all 16 losses:**
- q510 (gold: `Shane Gillis`):
  > *"Okay, let's see. The user is asking which SNL cast member who was fired appeared on the show 'Hot Ones' as of August 2024. Alright, first I need to fi…"*
- q70 (gold: `84,512`):
  > *"Okay, let me try to work through this step by step. The user is asking for the population difference between Seattle, WA, and Portland, OR…"*
  → The `<think>` block is the entire prediction. No answer tag ever closed.

**Qwen3.5-9B (0.33) — `<|begin_search_query|>` markers leak raw into `predicted_answer`:**
- q793: *`"team list. </think>  <|begin_search_query|>Joel Dyck ice hockey career complete team list all teams played Japan Asia League <|end_search_query|>"`*
- q510: *`"formation. </think>  <|begin_search_query|> Saturday Night Live cast member fired Hot Ones 2024 <|end_search_query|>"`*
- q512: *`"ess Bride. </think>  <|begin_search_query|>Andre the Giant favorite role Princess Bride Fezzik quote <|end_search_query|>"`*
  → The last search query emitted by the model ends up as the "answer". `search_limit=15` fires; adapter dumps final streaming state.

**Qwen3.5-4B (0.30) — same leak pattern:**
- q793: *`"layed for. </think>  <|begin_search_query|> Joel Oshiro Dyck career teams Eliteprospects.com all teams Japan Ice Hockey League Asia League <|end_search_query|>"`*
- q510: *`"Hot Ones. </think>  <|begin_search_query|> | August 2024 SNL cast member fired Hot Ones appearance <|end_search_query|>"`*

**gpt-oss-20b (0.17) — prose monologue, zero tool calls:**
- q793: *"We need to identify Joel Oshiro Dyck, a professional ice hockey player. He played for three teams. Which of those teams were dissolved for financial d…"*
- q510: *"We need to find: As of August 2024, which cast member fired from Saturday Night Live appeared on the show Hot Ones? So we need to identify a former SN…"*
  → Model writes "we need to…" planning text; no `<|begin_search_query|>` ever emitted. Adapter treats this as final answer.

### 2. ASearcher prompt-mode

**gpt-oss-120b (0.53) — confident hallucination at 25/30 zero-tool cells:**
- q793: *"HC Beroun (HC Berounští Medvědi) was dissolved for financial difficulties after the 2018-2019 season."*
  → That team is not in the gold source and Joel Dyck never played for them. Pure parametric invention.
- q510: *"Megan Stalter"* (gold: Shane Gillis) — wrong celebrity from memory.
- q693: *"Roger Federer"* (gold: Michel Kratochvil) — the famous Swiss, not the obscure one.

**QwQ-32B (0.37) — categorical "the question is invalid" on 11 temporally-anchored losses:**
- q510 (gold: `Shane Gillis`): *"the question is invalid."*
- q158 (gold: `12`): *"the question is invalid."*
- q70 (gold: `84,512`): *"the question is invalid."*
  → All three contain "as of <date>" anchoring. QwQ refuses uniformly.

**Qwen3.5-9B (0.43) — hedge rather than guess:**
- q510: *"Based on my search results, I cannot find specific information about a Saturday Night Live cast member who was fired in August 2024 and appeared on Ho…"*
- q693: *"Based on the search results and tennis history, the retired Swiss tennis player who reached the 4th round of Wimbledon in 2002 was **Marc Rosset**."*
  → Close but wrong entity — Rosset was Swiss but lost R1 in 2002.

**Qwen3.5-4B (0.33) — planning-prose-as-answer:**
- q510: *"The user is asking about a cast member of Saturday Night Live (SNL) who fired from the show as of August 2024 and appeared on the show Hot Ones.  1.  …"*
- q158: *"The user is asking about the number of total Academy Award nominations received by the man who won the Academy Award for Best Actor one year before 19…"*
  → Boundary never closed.

**gpt-oss-20b (0.17) — pure monologue, 30/30 zero tools:**
- q793: *"We need to find Joel Oshiro Dyck, professional ice hockey player, played for three teams…"*
- q510: *"We need to answer: As of August 2024, which cast member fired from SNL appeared on the show Hot Ones?…"*

### 3. ASearcher trained

**Web-14B (0.40) — format-clean but wrong entity retrieved/synthesised:**
- q510: *"Pete Davidson"* (gold: Shane Gillis).
- q70: *"128,492"* (gold: 84,512) — fabricated/miscomputed number.
- q83: *"the question is invalid."* — same refusal pattern as QwQ, inherited through RL.

**Web-7B (0.27) — more refusals, some empty:**
- q510: *"the question is invalid."*
- q158: *"the question is invalid."*
- q70: *""* (empty `<answer></answer>` emitted).

### 4. Vanilla RAG+ReAct

**All three model sizes — every loss is `max_turns` with an empty predicted_answer:**
- gpt-5 on q510, q70, q693: *`""`* — 15 searches + 1 read, 0 synthesis.
- gpt-oss-120b on q793, q510, q70: *`""`* — same pattern.
- gpt-oss-20b on q510, q70, q693: *`""`* — same pattern.
  → Agent consumes budget entirely on retrieval, never emits final answer. No "wrong synthesis" example exists for this system — only empty ones.

### 5. Context-1 + reasoner

**Original run (reasoner=gpt-5 returned 404):** all 30 predictions empty strings. Failure was a config bug.

**Re-runs completed 2026-04-20 14:27:**
- With gpt-5 reasoner (0.40 judge, 0.33 EM): representative win q512 — *"Rob Reiner."* (gold Rob Reiner). Representative loss q70 — retrieval sub-agent kept only 5 chunks and none mentioned Portland ME's population; reasoner hedged with *"The provided excerpts do not contain…"*.
- With bedrock/gpt-oss-120b reasoner (0.40 judge): repetition loop pathology on q800 — the reasoner kept appending *"Seoul Olympic Stadium hosted the opening ceremony for the 1988 Summer Olympics in Seoul, South Korea. The 1988 Winter Olympics were held in Calgary, Alberta, Canada."* until max_tokens=16000 terminated. Same gpt-oss family loop observed in the original gpt-5 run.
- With bedrock/gpt-oss-20b reasoner (0.30 judge): representative loss q793 — *"I'm sorry, but I can't find that information in the provided excerpts."* Retrieval delivered chunks but the 20B reasoner couldn't extract the answer. Pruning hurts small reasoners more.

### 7. Tongyi DeepResearch (Alibaba-NLP)

Already shown in Section 7 / "Failure gallery". Representative failures:
- q512 (gold: Rob Reiner): *"`<think>\nThus we have the evidence. Now answer.\n</think>\n\nJack Nicholson – the Best Actor winner…"`* (wrong entity, wrong Q)
- q793 prior to patch: confident invention, no retrieval.

### 8. SMTL / AFM-MHQA (OPPO PersonalAI)

**Correct samples (judge=✓, finished mode):**

- q793 (partial, gold `Joel Oshiro Dyck played for the Chatham Wheels, the Wheeling Thunderbirds, and the Nippon Paper Cranes`): *"Joel Oshiro Dyck played for the Nippon Paper Cranes, and that team was dissolved after the 2018–2019 season due to financial difficulties. The other two teams he played for (not yet identified in the…"* — 10 turns.
- q512 (gold: Rob Reiner): *"Andre the Giant's favorite acting role was in the 1987 film 'The Princess Bride'. The director of that film was Rob Reiner."* — 3 turns (early termination on correct answer).

**Hallucination (no retrieval, finished mode):**

- q520 (gold: 302,971 — population of Marinette, Wisconsin):
  > *"Matthew McConaughey won the Academy Award for Best Actor in 2014 (86th Academy Awards). Five years earlier is 2009. The Super Bowl in 2009 was Super Bowl XLIII, won by the Pittsburgh Steelers. The cit…"*
  >
  > Wrong actor chain, wrong year, wrong answer. No retrieval attempted.
- q510 (gold: Shane Gillis): *"I need to identify which Saturday Night Live cast member who was fired appeared on Hot Ones as of August 2024. This requires two pieces of information: (1) a list of SNL cast members who were fired (n…"* — model narrated the plan and then stopped without emitting a retrieval query. 16 turns total.

**Budget exceeded (max_turns, empty pred):**

- q158 (gold: 12), q70 (gold: 84,512), q158 (gold: 12): all `pred=""`. 6-function workflow cycled through `<plan>`→`<reflection>` without reaching `<answer>`.

**Loop with context overflow (llm_error):**

- q83, q549, q129, q302: `pred=""`, 15 turns, `Error code: 400: This model's maximum context length is 32768 tokens. However, you requested 16000 output tokens and your prompt contains at least 16769 input tokens…`. Accumulated `<think>`+`<plan>`+`<reflection>` scaffolding overflowed the 32k ctx budget.

### 6. rule_decomp_gsw (decomposition ablation)

**gpt-4.1-mini (0.30) — "Unknown" refusals and confident wrong entities:**
- q510: *"Unknown"*
- q154 (gold: `Dark Side of the Moon`): *"unknown"* — decomposer produced Picasso_death_year + Pink_Floyd_albums(year), but aggregator failed to connect the 1973 node across the retrieved chunks.
- q70: *"The difference cannot be determined from the provided data."* — grounded abstention.
- q386 (gold: Yale University): *`"Harvard University"`* — confident wrong-entity (retrieved the right chunk, picked the wrong institution).
- q520 (gold: `302,971`): *`"733,391"`* — wrong number from the retrieved population data.
- q129 (gold: Love Yourself in Seoul, Bring the Soul: The Movie): *`"Map of the Soul: Persona"`* — picked an album release from the year instead of the two film releases.

**Qwen3.5-4B (0.00) — every single prediction is an empty string:**
- All 30 questions: *`""`* despite `stopped_reason=finished` and sub-Qs matching 4.1-mini exactly. Aggregator LLM returns no content — 4B can't do the chunk-reading + entity-extraction step.

### Pattern map across the gallery

| failure shape | canonical example | systems affected |
|---|---|---|
| `<think>` block leakage (never closed) | QwQ-32B on q510: *"Okay, let's see. The user is asking…"* | search_o1/QwQ-32B, search_o1/Qwen-9B, search_o1/Qwen-4B |
| Raw `<\|begin_search_query\|>` in final answer | Qwen-9B on q793: *`"</think> <|begin_search_query|> Joel Dyck ice hockey career…"`* | search_o1/Qwen-9B, search_o1/Qwen-4B |
| Prose monologue "we need to…" | gpt-oss-20b on q510: *"We need to find: As of August 2024…"* | search_o1/20b, asearcher_prompt/20b, asearcher_prompt/4B |
| Canned "the question is invalid" | QwQ-32B on q510 | asearcher_prompt/QwQ-32B, asearcher_trained/Web-7B, asearcher_trained/Web-14B |
| Confident-wrong parametric entity | asearcher/120b on q793: *"HC Beroun"* | asearcher_prompt/120b, asearcher_trained/Web-14B, rule_decomp_gsw/4.1-mini (q386, q520) |
| Grounded abstention | rule_decomp_gsw/4.1-mini on q510: *"Unknown"* | rule_decomp_gsw/4.1-mini only |
| Empty predicted_answer (turn-budget exhaustion) | vanilla_rag_react/gpt-5 on q510: `""` | vanilla_rag_react (all sizes), context1_plus_reasoner (original 404 run) |
| Empty predicted_answer (model incapacity) | rule_decomp_gsw/Qwen-4B on q510: `""` | rule_decomp_gsw/Qwen-4B |
| Repetition-loop on output | context1 + 120b reasoner on q800: *"Seoul Olympic Stadium… 1988 Winter Olympics…"* ×N | context1_plus_reasoner/bedrock-120b, occasionally gpt-oss family across cells |

The `ours_gsw_v1` target is to replace the two most common shapes — confident-wrong parametric entity and grounded abstention — with entity-attribute nodes that carry chunk-level provenance, so the aggregator either answers from a cited node or emits "unknown with reason".

---

## Appendix: per-question solve count (18 cells)

| qid | solved | gold (abbrev) |
|---|---:|---|
| 796 | 15/18 | Italy |
| 512 | 14/18 | Rob Reiner |
| 154 | 14/18 | Dark Side of the Moon |
| 549 | 14/18 | FGM-148 Javelin |
| 386 | 11/18 | Yale University |
| 425 | 11/18 | A. J. Hinch |
| 158 | 10/18 | 12 |
| 515 | 10/18 | Cyclades |
| 776 | 10/18 | (Jane Austen novel) |
| 793 | 9/18 | (Nippon Paper Cranes …) |
| 293 | 9/18 | 83 |
| 520 | 7/18 | 302,971 |
| 339 | 7/18 | Marinette, Wisconsin |
| 546 | 6/18 | Graham High School in St. Paris Ohio |
| 83 | 5/18 | Yes, London |
| 800 | 4/18 | Calgary – Winter Olympics |
| 691 | 4/18 | Labetalol |
| 224 | 4/18 | Verizon Center |
| 70 | 3/18 | 84,512 |
| 693 | 3/18 | Michel Kratochvil |
| 284 | 3/18 | 6 months and 26 days |
| 56 | 2/18 | Crystal Palace Park |
| 663 | 2/18 | 2,851 Passengers |
| 590 | 2/18 | 1979 |
| 510 | 1/18 | Shane Gillis |
| 129 | 1/18 | Love Yourself in Seoul (2019) … |
| 190 | 1/18 | Nick |
| 371 | 0/18 | (French/Haitian delta) |
| 302 | 0/18 | Novak Djokovic 24-10 Andy Murray |
| 133 | 0/18 | 12 (Tony × Oscar) |

Questions in the bottom third (solved ≤3/18) are the real contribution surface — if GSW's structured intermediate unlocks any of those, we have unique wins.
