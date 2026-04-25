# Planner-ReAct FRAMES Pilot — Gains & Losses

> **Scope.** Results and interpretation for the planner-scaffolded ReAct adapter
> (`ours_gsw_planner_react_v1`) on the FRAMES dev-100 subset using
> `gpt-oss-120b` as the reasoner. Compared against flat ReAct
> (`vanilla_rag_react`) on the same 100 questions.
> No implementation detail — this is a results-and-reading document.

Date: 2026-04-24
Benchmark: FRAMES dev 100Q (stratified)
Reasoner: `bedrock/openai.gpt-oss-120b-1:0`, `max_turns=50`
Judge: `gpt-4o` LLM-judge, standard prompt.

---

## 1. Headline table

All runs completed (100/100). Hallucination % is the rate within each run's
judge-✓ wins where the predicted answer string is not locatable in any chunk
retrieved during the trajectory (after unicode / punctuation normalization +
manual triage for short-numeric and paraphrase answers).

| run | adapter | prompt | retriever | judge ✓ | judge % | halluc | halluc % | grounded ✓ | grounded % |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| **vanilla Dense** | flat ReAct | — | Dense (Qwen3-Embedding-8B) | **53** | 53.0% | ~4 | **7.5%** | ~49 | 49.0% |
| **v4 BM25** | planner + ReAct | v4 | BM25 | 45 | 45.0% | ~1 | **2.2%** | ~44 | 44.0% |
| **v5.1 Dense** | planner + ReAct | v5.1 | Dense | **53** | 53.0% | ~1 | **1.9%** | ~52 | **52.0%** |
| **v5.2 Dense** | planner + ReAct | v5.2 | Dense | 46 | 46.0% | ~1 | 2.2% | ~45 | 45.0% |
| **v5.2 Hybrid** | planner + ReAct | v5.2 | BM25+Dense RRF | 49 | 49.0% | ~1 | 2.0% | ~48 | 48.0% |

**v5.1 Dense is the winner** on both judge and grounding. The v5.2 softer-
protocol iteration regressed 7 points on Dense and 4 points on Hybrid vs
v5.1, for no compensating gain.

## 2. The headline reading

- **Raw judge**: planner matches vanilla (53 vs 53 with v5.1 Dense).
- **Grounded judge**: planner beats vanilla by ~3 points (52 vs 49).
- **The 3-point gap is the planner's real gain** — vanilla's 53 includes
  ~4 wins that come from the model's training priors rather than retrieved
  evidence (Paula Radcliffe, Carl Schuhmann, and similar famous-entity
  trivia). Planner's evidence-chunk-ids contract structurally blocks that
  behavior.

So the honest framing is: **the planner gives a safety win (fewer
hallucinations), not a capability win (more correct answers)**.

## 3. Paired-question breakdown (v5.1 Dense vs vanilla Dense, same 100Q)

| bucket | n | reading |
|---|---:|---|
| both ✓ | 45 | neither adapter differentiates |
| both ✗ | 39 | the real ceiling — both miss |
| planner WIN | 8 | plan scaffold unlocks a Q vanilla misses |
| planner LOST | 8 | planner refuses to commit, vanilla guesses and wins |

Net delta on judge: **+0**. But the planner trades 8 for 8, so half of the
both-matched total of 45 is hiding a real swap.

### Why planner loses its 8

| sub-pattern | count |
|---|---:|
| empty prediction (target never committed; budget exhausted) | 5 |
| target blank left unresolved after max_turns | 3 |
| target resolved but with wrong value | 0 |

The loss mode is almost entirely **"refuse to commit"**. When retrieval
fails, planner keeps trying and runs out of turns; vanilla commits something
and the something happens to be right on priors.

### Why planner wins its 8

Multi-hop chains where flat ReAct drifts after the first hop. The plan
carries hop-2 and hop-3 anchors explicitly, so the reasoner stays on-
mission.

## 4. The 39 both-failed bucket (the real ceiling)

| diagnostic tag within this bucket | count |
|---|---:|
| planner target_unresolved | 21 |
| planner empty pred | 18 |
| planner resolved ALL blanks but judge still ✗ | ~15 |
| vanilla empty pred | 15 |
| vanilla committed wrong | 24 |

Two sub-patterns inside this bucket matter:

1. **`blanks_resolved = N/N` AND judge ✗** (~15 Qs). Plan executed fully,
   target carries a committed value, but the value is wrong. Two causes
   suspected:
   - Plan semantically misaligned with the question (wrong target
     `value_type`, wrong VP predicate, constraint op chosen that doesn't
     match the ask).
   - Answer format mismatch: target holds an entity name, gold requires
     a natural-language response shape the judge reads more strictly.
2. **`target_unresolved` with ≥20% of other blanks resolved**. The chain
   got part-way through the plan then retrieval stopped surfacing gold
   chunks. These are retrieval-bottleneck Qs that need a different
   retriever or a corpus expansion — not a plan-prompt change.

## 5. Prompt-iteration history

| prompt | judge % on 100Q | key change | observed effect |
|---|---:|---|---|
| v3 | pre-pilot only | initial schema-aware prompt | plan emission quality confirmed |
| v4 | 45 (BM25) | emphasis on walking topological order | baseline for planner |
| v5 | ~53 estimated | harder protocol; target-blank hard stop; evidence required | matched vanilla, eliminated JSON-as-text bugs |
| **v5.1** | **53 (Dense)** | sentinel-as-value rejected server-side + prompt tightened | 0 sentinel commits; flipped q531 ✗→✓ |
| v5.2 | **46 (Dense) / 49 (Hybrid)** | softer sentinel message; "2-3 retrievals" relaxation; concat auto-compute | **Regression**: Dense −7, Hybrid −4 vs v5.1; bug relapses (JSON-as-text, literal 'None', blank_id-as-value, sentinel-as-value) |

**Reading**: the 7-point Dense regression is not noise; it is a real
behaviour shift caused by the softer protocol. Across v5.2 we observed four
distinct failure modes that v5.1 had eliminated:

- JSON-as-text (q229 "Willie Mays", q184 Bundesliga, q145 Hybrid)
- `"None"` literal emitted as the final answer (q308 Dense + Hybrid)
- Blank-identifier emitted as the value (q438 "b_climber")
- `"insufficient_evidence"` reappearing in the `value` slot (q145 Hybrid)

v5.1 wins on both correctness and token-efficiency. **Adopt v5.1 as the
canonical planner prompt.**

## 6. Retriever comparison

| retriever | judge on v5.1 | judge on v5.2 |
|---|---:|---:|
| BM25 | — (45 on v4) | — |
| Dense (Qwen3-8B) | **53** | 46 |
| Hybrid (RRF k=60) | *pending* | 49 |

- Dense gave roughly +8 points over BM25 with the same prompt family.
- Hybrid on v5.2 (49) landed between v5.2 Dense (46) and v5.1 Dense (53).
  v5.1 Hybrid has not been run yet — that is the open cell to fill the
  retriever × prompt grid.
- No retriever swap so far breaks through the ~53 ceiling.

## 7. New failure modes observed in v5.2

These are concrete, reproducible loss modes surfaced in the v5.2 runs:

1. **Invented tools** (q209 in both Dense and Hybrid). Model imagines a
   `find(pattern, chunk_id)` tool and burns 9+ turns on it. Directly
   addressed by adding a real `find` tool (ships separately; runs below
   pre-date that change).
2. **Literal-string predictions**. q308 v5.2 Dense produced
   `pred='None'` — the model wrote the English word "None" as its final
   answer instead of using the sentinel path or committing a real value.
   Indicates prompt-slot confusion on the finish rule.
3. **JSON-as-text regressions**. q229 ("Willie Mays"), q184 (Bundesliga)
   in v5.2 Hybrid: model emitted `{"query": "...", "top_k": N}` *as* the
   finish answer, i.e. a serialized tool call rendered as text. v5 had
   eliminated this; v5.2's softer protocol brought it back.
4. **Budget-exceeded chains on famous-entity Qs**. q63 Carl Schuhmann,
   q227 Matsuyama. Planner can't find the gold in the retrieved chunks,
   refuses to commit from priors, runs 50 turns, emits empty. Same Qs
   that vanilla wins by guessing from training.

## 8. Where the planner *actually* gives value

- **Hallucination suppression**. Planner runs at ~2% hallucination vs
  vanilla's ~8% (substring-in-retrieved-chunks heuristic after manual
  triage). This is a 4× reduction.
- **Evidence auditability**. Every committed blank carries
  `evidence_chunk_ids`, making post-hoc verification straightforward.
- **Multi-hop carry**. On Qs with ≥2 hops the plan explicitly, flat ReAct
  is observed to drift; planner stays on-script.

## 9. Where the planner does not help

- **When vanilla's priors happen to be correct.** Famous names, common
  numbers, well-known dates. Priors-arbitrage costs the planner 3-5
  raw judge points it cannot recover without allowing priors-based
  commits (which would break the grounding story).
- **When retrieval doesn't surface the gold chunk.** The plan is a
  scaffold for reasoning over retrieved evidence, not a replacement for
  retrieval. Dense and Hybrid don't break the ceiling either.
- **When the plan itself is semantically wrong.** ~15 Qs in the
  both-failed bucket show `N/N` blanks resolved and still ✗, which is
  evidence that plan-emission is not always faithful to the question's
  real structure. Plan-semantic audit is not yet in the pipeline.

## 10. Cost accounting

Model pricing: `bedrock/openai.gpt-oss-120b-1:0` at **$0.15/M input, $0.60/M
output**. Judge pricing: `gpt-4o` at $2.50/M in, $10/M out (≈$0.18 per run,
negligible).

| run | tok_in | tok_out | turns | wall | $ model | $ per grounded ✓ |
|---|---:|---:|---:|---:|---:|---:|
| vanilla Dense | 74.9M | 446K | 2094 | 1.8h | **$11.50** | $0.23 |
| v4 BM25 | 126.5M | 1.04M | 3079 | 3.1h | **$19.60** | $0.45 |
| **v5.1 Dense** | 91.6M | 916K | 2528 | 2.7h | **$14.28** | **$0.27** |
| v5.2 Dense | ~123M | 949K | 2744 | 2.9h | **$19.02** | $0.42 |
| v5.2 Hybrid | ~119M | 912K | 2592 | 2.9h | **$18.37** | $0.38 |

Total for the pilot series: **~$82.80** of 120b compute.

Cheapest cost-per-grounded-answer: **v5.1 Dense at $0.27/✓**. v5.2 Dense
burned +30% more tokens per question than v5.1 and produced fewer correct
answers — the softer protocol widens the retrieval loop without adding
discipline to commit.

## 11. Implications for next iterations

1. **Prompt iteration is tapped out.** Further v5.x wording changes
   swing results inside the noise band. Do not ship a v5.3 for marginal
   gains on 100Q.
2. **Bigger sample** (500Q+) is the only way to separate genuine prompt
   gains from prompt-lottery at this accuracy level.
3. **Plan-semantic audit** is the highest-leverage next step: ~15 Qs in
   the both-failed bucket currently execute the plan fully and still
   fail. If even half of those are plan-error rather than judge-mismatch,
   that's a latent +7 on 100Q.
4. **Reasoner upgrade** (to a non-120b-class model) likely breaks the
   ceiling more than any prompt tuning at this level.
5. **Priors-aware finish** — letting the model commit a priors-based
   value with an explicit flag — would close most of the 8-Q
   planner-LOST bucket. Trade-off: it breaks the current
   grounding-correctness story. Worth evaluating on cross-domain
   benchmarks where vanilla's priors are *not* the gold.

---

## Appendix A — Per-question failure-analysis JSON

Streamlit-ready per-question data for v5.1 Dense vs vanilla Dense:

```
logs/failure_analysis_v5_1_dense_vs_vanilla_dense.json
```

Fields: `qid`, `question`, `gold`, `planner_pred`, `vanilla_pred`,
`planner_judge`, `vanilla_judge`, `planner_f1`, `vanilla_f1`, `category`
(`both_✓` / `both_✗` / `planner_WIN` / `planner_LOST` /
`planner_LOST_vanilla_hallucinated`), `diag` (list of failure-mode tags),
`planner_mode`, `planner_turns`, `planner_finish_rejections`,
`planner_target_status`, `planner_blanks_resolved`, `planner_grounded`,
`vanilla_grounded`.

Filter suggestions:
- `category == "planner_LOST"` — the 8 Qs we gave away. Mostly budget-
  exceeded.
- `category == "both_✗" AND planner_blanks_resolved matches "N/N"` — the
  ~15 Qs where the plan ran to completion and still failed. Highest
  leverage bucket for plan-semantic audit.
- `category == "planner_WIN"` — the 8 Qs the scaffold unlocked vs flat
  ReAct. Useful for understanding when the plan genuinely carries
  multi-hop state.

## Appendix B — Source-run pointers

| tag | cell_result.json |
|---|---|
| vanilla Dense 100Q | `logs/vanilla_dense_100q/` |
| v4 BM25 100Q | `logs/planner_v4_bm25_100q/` |
| v5.1 Dense 100Q | `logs/planner_v5.1_dense_100q/` |
| v5.2 Dense 100Q | `logs/planner_v5.2_dense_100q/` |
| v5.2 Hybrid 100Q | `logs/planner_v5.2_hybrid_100q/` |
| v5.1 Hybrid 100Q (in-flight) | raw ts dir; will rename to `logs/planner_v5.1_hybrid_100q/` on completion |
