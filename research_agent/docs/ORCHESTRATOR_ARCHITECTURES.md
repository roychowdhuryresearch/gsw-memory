# Orchestrator architectures — deterministic vs LLM

> **Scope.** Side-by-side description of the two modes supported by the
> `ours_gsw_planner_orchestrator_v1` adapter, with concrete examples
> pulled from the first 10-question FRAMES probe.
> Both modes share the same per-blank ReAct researcher class, the
> same retrieval tools, and the same `emit_plan` + topology
> infrastructure. They differ only in **who controls the outer loop**.

Date: 2026-04-24
Benchmark: FRAMES dev 10Q (first 10 in `configs/frames_dev_100.json`)
Reasoner: `bedrock/openai.gpt-oss-120b-1:0`, hybrid retriever.

---

## 1. The two modes in one diagram

### Mode A — deterministic orchestrator (Phase-1 default)

```
 emit_plan(Q) ─► GSWPlan ─► topo sort + dep graph ─► levels (dict[blank_id, int])
                                                          │
                                                          ▼
                        ┌─────────────────────────────────────────────────┐
                        │  Python for-loop over sorted levels             │
                        │    level_blanks = blanks at this level,         │
                        │                   minus constraint outputs      │
                        │                                                 │
                        │    one ReActResearcher per level with           │
                        │    allowed_blank_ids = all level_blanks         │
                        │    (may be 1..N blanks handled sequentially     │
                        │     inside one researcher conversation)         │
                        │                                                 │
                        │    cascade_auto_compute(plan, state)            │
                        └─────────────────────────────────────────────────┘
                                          │
                                          ▼
                   final_answer = state[target.id].value (or "")
```

No LLM runs at the orchestrator layer. Control flow is fully
deterministic. Researchers use the usual ReAct tools (`search`,
`read`, `find`, `update_blank`, `get_state`) and stop when every
blank in their slice is resolved or when their turn budget is
exhausted.

### Mode B — LLM orchestrator (Phase-2)

```
 emit_plan(Q) ─► GSWPlan ─► topo + levels (hints, not enforced)
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────┐
 │  ORCHESTRATOR (LLM, ReAct loop, max 12 turns)                   │
 │                                                                 │
 │  Tools:                                                         │
 │    - dispatch_subplan(blank_ids, hints?) ───► ThreadPool of     │
 │        per-blank researchers (cap 8 concurrent). Each blank     │
 │        gets its OWN researcher with a single-element slice.     │
 │    - request_plan_update(reason, evidence) ──► plan-updater     │
 │        LLM (one-shot). Returns a revised GSWPlan;               │
 │        resolved blanks carry over by id.                        │
 │    - get_state() ─► current state snapshot                      │
 │    - submit_answer(answer) ─► terminal                          │
 │                                                                 │
 │  Cascade_auto_compute runs after every dispatch.                │
 └─────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                    final_answer = args["answer"] from submit_answer
                                   (fallback: state[target.id].value)
```

The LLM makes every strategic choice: WHICH blanks to resolve next,
whether to re-dispatch after weak retrieval, whether to revise the
plan, when to finalize. Researchers are still ReAct agents and handle
retrieval on their assigned blank.

---

## 2. Side-by-side — what each mode produces

| field | deterministic | llm |
|---|---|---|
| `extra["orchestrator_mode"]` | `"deterministic"` | `"llm"` |
| `extra["researcher_traces"]` | one per topological level | one per blank per dispatch |
| `extra["plan_json_versions"]` | absent | list; length = 1 + number of plan revisions |
| `extra["plan_updates"]` | absent | list of `{turn, reason, evidence, diff_summary, ...}` |
| `extra["dispatches"]` | absent | list of `{dispatch_idx, blank_ids, hints, partial}` |
| tool_calls with `level=-1` | none | orchestrator-level tool calls (dispatch / plan_update / submit_answer) |
| researcher tool_calls `.level` | topological level int | dispatch-index int (0, 1, 2, …) |
| `messages._level` | topological level | -1 for orchestrator, dispatch-idx for researchers |

---

## 3. Head-to-head on the 10Q probe

Reasoner model / retriever / subset identical across both columns.
Same 10 questions, same judge, same retriever cache.

| qid | gold | **deterministic** pred (turns) | **llm** pred (turns) |
|---|---|---|---|
| q174 | "The Boeing 777 was first flown on…" | ✗ `'Boeing 777'` (18) | ✗ `'Boeing 777'` (6) |
| q70 | `'84,512'` | **✓ `'84512'` (20)** | **✓ `'84512'` (22)** |
| q445 | `'14'` | ✗ `'15'` (17) | ✗ `'8.0'` (17) |
| q401 | `'100'` | ✓ `'100'` (11) | ✓ `'100'` (9) |
| q376 | `'Ornithologie'` | ✗ empty (20, max_turns) | **✓ `'Ornithologie'` (52)** |
| q196 | `'5'` | ✓ `'5'` (6) | ✓ `'5'` (5) |
| q167 | `'Daniela Lavender'` | ✓ `'Daniela Lavender'` (8) | ✓ `'Daniela Lavender'` (8) |
| q742 | `'LeBron James, Diana Taurasi, Tina Charles, DeWanna Bonner'` | ✗ 3/4 correct, wrong 4th (20) | ✗ empty (orchestrator LLM error, 0 turns) |
| q154 | `'Dark Side of the Moon'` | ✓ `'The Dark Side of the Moon'` (5) | ✓ `'The Dark Side of the Moon'` (4) |
| q796 | `'Italy'` | ✓ `'Italy'` (5) | ✓ `'Italy'` (5) |

**Finals**: deterministic **6/10 (60%)**, llm **7/10 (70%)**.

Both modes strongly exceed flat v5.1 Hybrid's **5/10 (50%)** on the
same 10-Q slice. LLM mode beats deterministic by +1 on this sample
— winning q376 (via multi-dispatch recovery) while losing q742 (an
orchestrator LLM call errored at turn 0, treated as
`target_unresolved`; a deterministic-mode researcher would have
produced a partial commit here).

### Key divergence: q376

This is the single Q that differentiates the two modes so far.

- **Deterministic** walks the topological plan once, sees that its
  single-researcher's 20-turn budget is exhausted without resolving
  the target, and stamps `stopped_reason = target_unresolved` →
  `predicted_answer = ""`.
- **LLM orchestrator** receives the same "weak retrieval" signal
  from its first dispatch, decides to re-dispatch (and/or revise
  hints), and eventually commits the correct answer after 52 total
  turns.

This is exactly the failure mode the LLM mode was built for: when
the plan + first-pass retrieval is insufficient, the orchestrator
can replan or retry; the deterministic mode cannot.

Cost of that recovery: 13k output tokens vs 5k on the deterministic
fail. The LLM mode trades tokens for correctness on hard questions.

---

## 4. Worked trajectory — q70 (both modes ✓, simplest case)

> Question: *What was the daily average passenger count in 2011 of
> the first station on the train line that serves Hiraka Train
> Station in Japan?*
> Gold: `'84,512'`

The emitted plan has one bridge blank (`b_line`, the line serving
Hiraka) and one target (`b_count`, the passenger count at the line's
first station).

### Deterministic trajectory (20 turns, 2.5k out-tokens)

1. Topology: one level (all blanks have no mutual deps, because the
   emitted plan has t ← line via a VP edge).

   Actually: `b_line` at level 0, `b_count` at level 1 via
   projection edge. Two researchers.

2. Researcher 1 (level 0, slice=[b_line]): 8 turns of search/find,
   commits `b_line = "Kōnan Railway"`.
3. Orchestrator cascades constraints (none to fire).
4. Researcher 2 (level 1, slice=[b_count]) starts with the now-
   resolved `b_line` in its prior-values table: searches
   "Kōnan Railway" + "first station" + "passenger count 2011",
   reads `Kōnan Line__0`, commits `b_count = 84512`.
5. Target status = resolved; final_answer = "84512".

### LLM-orchestrator trajectory (22 turns, 3.4k out-tokens)

Orchestrator turns:
1. `dispatch_subplan(blank_ids=["b_line"])` — kick off researcher A
   on just the line blank.
2. Researcher A resolves `b_line = "Kōnan Railway"`.
3. Orchestrator sees `b_line` resolved in the state table;
   dispatches `dispatch_subplan(blank_ids=["b_count"])`.
4. Researcher B (single-blank slice) retrieves with the resolved
   line as an anchor, commits `b_count = 84512`.
5. Orchestrator calls `submit_answer("84512")`.

Both succeed, but the LLM orchestrator's dispatches are explicit
entries in `extra["dispatches"]` (2 dispatches), while the
deterministic mode just has 2 entries in `extra["researcher_traces"]`
indexed by topological level.

---

## 5. Worked trajectory — q376 (only LLM recovers)

> Question: *What is the French (not Latin) term for the branch of
> science known as ornithology?*
> Gold: `'Ornithologie'`

The question is trivially a retrieval-and-translate, but the emitted
plan often gets confused between multiple candidate French terms
(the article likely mentions several). Flat v5.1 Hybrid committed
`'Nouveau dictionnaire d'histoire naturelle'` (wrong); orch-det
ran out its budget without committing.

### Deterministic trajectory (20 turns, empty pred)

- One researcher with allowed_blank_ids=[t].
- 20 turns of search + read on variations of "ornithology French".
- No commit (the v5.1 commit rule requires confidence or insufficient-evidence fallback; the model kept re-querying instead of bailing).
- stopped = max_turns; researcher.unresolved = [t].
- Orchestrator has no replay mechanism. Final answer = "".

### LLM-orchestrator trajectory (52 turns, 13k out-tokens)

Conceptually:
1. `dispatch_subplan(blank_ids=["t"])` — first try.
2. Researcher A: 20 turns, no commit. Orchestrator sees
   `partial: true`.
3. Orchestrator inspects state, notices target still unresolved,
   issues a second `dispatch_subplan(["t"], hints=<targeted>)`.
4. Researcher B (second dispatch, same blank) gets a different
   slice of retrieval context (or the refined hint), commits
   `t = "Ornithologie"`.
5. `submit_answer("Ornithologie")`.

Two dispatches, both `extra["dispatches"]` entries, both researchers
tagged with different `level` values in tool_calls. The
orchestrator's own turn count is 3 (dispatch, dispatch, submit); the
total turns metric aggregates researcher turns.

---

## 6. When to prefer each mode

| situation | prefer |
|---|---|
| Budget-sensitive workflows (one-shot, cheap) | **deterministic** |
| Plans that are known to be brittle / plan-error is the dominant failure | **llm** |
| Multi-blank parallel levels where you want maximum concurrency | **llm** (per-blank researchers run parallel via threadpool) |
| Reproducibility / no randomness required | **deterministic** (no orchestrator LLM) |
| You want plan-revision signal in the trajectory for debugging | **llm** |

---

## 7. Current results snapshot

FRAMES dev 100Q reference results (gpt-oss-120b, hybrid retriever):

| system | judge % | notes |
|---|---:|---|
| vanilla_rag_react (flat ReAct) | 53% | but ~7.5% of wins are hallucinated |
| planner_react_v1 (flat planner + reasoner, v5.1 prompt, Dense) | 53% | hallucination ~2% |
| planner_react_v1 (v5.1 prompt, Hybrid) | ~49-50% (100Q in flight) | same hallucination |
| orchestrator_v1 deterministic (10Q) | **60%** | +10 pts over flat hybrid |
| orchestrator_v1 llm (10Q) | **70%** | +20 pts over flat hybrid |

10Q is a thin sample — the real confirmation requires a 100Q pilot
on both modes. But the 10Q probe recovers one Q that every prior
pipeline missed (q376 on llm), demonstrating the "replan on weak
retrieval" contract does what it's supposed to.

---

## 8. Inspector / Streamlit

The inspector at `playground/planner_orchestrator_run_inspector.py`
detects `extra["orchestrator_mode"]` per question and shows
different tabs:

- **deterministic mode**: Plan / Researchers / Executed blanks /
  Messages / Reasoning
- **llm mode**: Plan / **Orchestrator** (new) / **Researchers** /
  **Plan versions** (new) / Executed blanks / Messages / Reasoning

The Orchestrator tab lists every orchestrator tool call with a
colored chip (red for `dispatch_subplan`, pink for
`request_plan_update`, orange for `submit_answer`), plus a
plan-update events list if the orchestrator invoked the plan-updater.
Plan-versions tab shows each plan revision side-by-side with its
own graphviz graph.

Run:
```bash
.venv/bin/streamlit run playground/planner_orchestrator_run_inspector.py
```

---

## 9. Log pointers (for streamlit)

| cell | directory |
|---|---|
| Orch 10Q probe — deterministic | `logs/ours_gsw_planner_orchestrator_v1__bedrock_openai.gpt-oss-120b-1:0__20260424_235207/` |
| Orch 10Q probe — LLM | `logs/ours_gsw_planner_orchestrator_v1__bedrock_openai.gpt-oss-120b-1:0__20260424_235206/` |

Both dirs will contain `cell_result.json` once the LLM run finishes
(deterministic already has one).
