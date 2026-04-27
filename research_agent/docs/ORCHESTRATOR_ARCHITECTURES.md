# Orchestrator architectures — deterministic vs LLM (Phase-3.2, current)

The `ours_gsw_planner_orchestrator_v1` adapter ships two modes that share
the same plan emission, retriever, researcher class, and constraint-cascade
machinery. They differ in **who runs the outer loop** that decides which
blanks to resolve next and when to finalize.

- **deterministic** — Python for-loop walks the topological levels.
- **llm** — an LLM orchestrator chooses dispatches, can request a plan
  revision mid-run, listens for researcher escalations, and decides when
  to submit the answer.

Both produce the same `Trajectory` shape (with extra fields specific to llm
mode); both are inspectable via the same Streamlit UI.

### Recent phases (chronological)

- **Phase-2** (commit `e7d3a7a`, tag `orch-v2-shipped`) — original llm
  mode shipped. Orchestrator can dispatch in parallel and call
  `request_plan_update` mid-run.
- **Phase-3** — `suggest_plan_revision` researcher tool added. The
  researcher can signal "the plan is missing a step" and end its loop;
  the orchestrator sees `revision_requests` in the dispatch result.
- **Phase-3.1** — researcher prompt tightened: mandatory escalate when
  retrieval returns no plausible value; explicit anti-priors language
  (forbid `"likely X"` / `"probably X"` commits). Closed the loophole
  where the researcher would commit a priors-pulled value with the
  `insufficient_evidence` sentinel.
- **Phase-3.2** (current) — orchestrator-side loop control:
  - **Plan-update cap**: `request_plan_update` is rejected after
    `MAX_PLAN_UPDATES_PER_RUN = 2` successful calls per question.
  - **Give-up path**: `submit_answer("")` is accepted (with
    `stopped_reason="give_up_unanswerable"`) when the cap is reached
    AND the target is still unresolved. Without the cap reached,
    submit_answer still requires a resolved target.
  - **Hint enrichment**: after a successful `request_plan_update`, the
    next `dispatch_subplan` auto-prepends `[after plan revision: …]`
    to the orchestrator-supplied `hints`, so the new researcher knows
    the prior context and doesn't rediscover the same gap.

> Recovery: `git checkout orch-v2-shipped` reverts to Phase-2; Phase-3+
> is on the active branch and can be reset to the tag if needed.

---

## 1. Shared pieces (used by both modes)

| component | file | role |
|---|---|---|
| `emit_plan` | `_planner_emit.py` | one LLM call → validated `GSWPlan` (with one repair retry) |
| `GSWPlan`, `BlankResult` | `_planner_exec.py` | typed plan + blank-fill state |
| `topological_sort_blanks`, `build_dependency_graph`, `_compute_levels` | `_planner_exec.py` + `gsw_planner_react_v1.py` | dependency DAG → `dict[blank_id, level]` |
| `ReActResearcher` | `gsw_planner_orchestrator_v1.py` | per-blank ReAct loop, **6 tools** (search / read / find / update_blank / get_state / suggest_plan_revision) |
| `_cascade_auto_compute` | `gsw_planner_react_v1.py` | fires constraint outputs once their inputs are resolved |
| `_PlannerFallbackMixin` | `_planner_emit.py` | falls back to flat `ours_gsw_v1` on planner-side errors |

The researcher class is identical in both modes — only its `allowed_blank_ids`
differs (one researcher gets the level's whole slice in deterministic mode;
each researcher gets a single-blank slice in llm mode).

---

## 2. Mode A — deterministic orchestrator

```
 emit_plan(Q) ──► GSWPlan
                    │
                    ▼
   topological_sort_blanks + build_dependency_graph
                    │
                    ▼
   levels: dict[blank_id, int]    ← grouped by level number
                    │
                    ▼
   for lvl in sorted(levels):
       slice = blanks_at(lvl) - constraint_outputs
       if not slice:
           cascade_auto_compute(plan, state); continue

       researcher = ReActResearcher(
           allowed_blank_ids = slice,        # whole level at once
           state = shared_state,
           max_turns = level_max_turns,      # default 20
       )
       trace = researcher.solve()
       cascade_auto_compute(plan, state)

   final_answer = state[target.id].value if resolved else ""
```

- **No LLM at the outer layer.** Control flow is fully deterministic.
- **One researcher per topological level.** If a level has K blanks,
  the researcher juggles all K inside one conversation.
- **Sequential across levels** — strict topological order.
- **Stops** on `state[target.id].status == "resolved"` or when a researcher
  hits its turn budget without resolving the target.

`Trajectory.extra` keys (deterministic mode):

| key | shape | meaning |
|---|---|---|
| `orchestrator_mode` | `"deterministic"` | mode tag |
| `plan_json` | `dict` | the emitted plan |
| `topological_order` | `list[str]` | blanks in solve order |
| `topology_levels` | `dict[str, int]` | blank → level |
| `researcher_traces` | `list[dict]` | one entry per topological level |
| `executed_blanks` | `list[dict]` | final state per blank |
| `stopped_reason` | `str` | `"finished"` / `"target_unresolved"` / `"llm_error"` |

---

## 3. Mode B — LLM orchestrator

```
 emit_plan(Q) ──► GSWPlan ──► topo + levels (used as hints only)
                                       │
                                       ▼
 ┌─────────────────────────────────────────────────────────────┐
 │  ORCHESTRATOR (LLM ReAct loop, max 12 turns)                │
 │                                                             │
 │  Tools (all strict, tool_choice=required):                  │
 │    • dispatch_subplan(blank_ids, hints?) ─────► fan out     │
 │       one researcher per id (cap 8 concurrent threads).     │
 │       Each gets a fresh state snapshot; merged back on      │
 │       completion. cascade_auto_compute fires after.         │
 │    • request_plan_update(reason, evidence) ───► one-shot    │
 │       plan-updater LLM revises the GSWPlan. State is        │
 │       reconciled by blank_id (keep, add, drop).             │
 │    • get_state() ─► snapshot                                │
 │    • submit_answer(answer) ─► terminate                     │
 │                                                             │
 │  Each turn the system prompt is REBUILT from the current    │
 │  plan + state, so the orchestrator always sees fresh        │
 │  context after every dispatch / plan revision.              │
 └─────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                   submit_answer or max_turns → final_answer
```

- **The LLM owns the outer loop.** It picks which blanks to dispatch and
  when to revise the plan or submit.
- **Per-blank parallelism.** A `dispatch_subplan` with K ids spawns K
  researchers concurrently, each with a single-element `allowed_blank_ids`.
- **Plan revision is a tool call.** `request_plan_update` invokes a
  separate one-shot LLM (`update_plan` in `_plan_updater.py`) whose
  output is a revised `GSWPlan`. State carries by blank_id; new blanks
  start `unknown`, dropped blanks are removed.
- **Stops** on `submit_answer` (success), max_turns (fail), or LLM error.

`Trajectory.extra` keys (llm mode — superset of deterministic):

| key | shape | meaning |
|---|---|---|
| `orchestrator_mode` | `"llm"` | mode tag |
| `plan_json` | `dict` | the FINAL plan after any revisions |
| `plan_json_versions` | `list[dict]` | every plan revision (length = 1 + #revisions) |
| `plan_updates` | `list[dict]` | `{turn, reason, evidence, diff_summary, preserved_ids, added_ids, dropped_ids}` per `request_plan_update` call |
| `dispatches` | `list[dict]` | `{dispatch_idx, blank_ids, hints, partial}` per `dispatch_subplan` call |
| `researcher_traces` | `list[dict]` | one entry per researcher (one per blank, per dispatch) |
| `executed_blanks` | `list[dict]` | final state per blank |
| `stopped_reason` | `str` | `"finished"` / `"max_turns"` / `"llm_error"` |

---

## 4. Tool surface — side by side

### Orchestrator tools (llm mode only)

| tool | inputs | returns |
|---|---|---|
| `dispatch_subplan` | `blank_ids: list[str]`, `hints?: str` | `{ok, resolved: {bid: {value, evidence_chunk_ids}}, partial}` |
| `request_plan_update` | `reason: str`, `evidence: str` | `{ok, diff_summary, preserved_ids, added_ids, dropped_ids}` |
| `get_state` | — | `{blanks: [...], target_blank_id, ...}` |
| `submit_answer` | `answer: str` | `{ok}` (terminal) |

### Researcher tools (both modes — identical)

| tool | inputs | returns |
|---|---|---|
| `search` | `query: str`, `top_k?: int` | `{results: [{chunk_id, title, score, text}]}` |
| `read` | `chunk_id: str` | `{chunk_id, title, article_text}` |
| `find` | `chunk_id: str`, `pattern: str` | `{match_count, snippets: [{offset, snippet}]}` |
| `update_blank` | `blank_id: str`, `value`, `evidence_chunk_ids: list[str]` | `{ok, state}` (rejects ids outside `allowed_blank_ids`; rejects values pulled from priors per Phase-3.1 prompt) |
| `get_state` | — | `{blanks: [{...writable}]}` |
| **`suggest_plan_revision`** | `reason: str`, `hint: str` | `{ok, escalated}` — ends the researcher's loop with `stopped="plan_revision_requested"`; the assigned blank stays unresolved; the orchestrator sees the request in its dispatch result |

Researchers do NOT have `finish` — the deterministic loop or the
orchestrator handles termination.

#### Researcher escalation contract (Phase-3.1)

The researcher MUST call `suggest_plan_revision` when:

- After 2 retrieval calls (`search` + optional `read`/`find`), no
  retrieved chunk contains a plausible value for the assigned blank.
- The researcher's reasoning starts reaching for training-data
  priors (`"likely X"`, `"probably X"`, `"might need external
  knowledge"`, `"based on what I know..."`).
- Resolving the blank requires a **new intermediate blank** that
  doesn't exist in the plan yet.

The researcher MAY call `update_blank` with `evidence_chunk_ids=
["insufficient_evidence"]` ONLY when a chunk did surface a plausible
value but its precision/confidence is low. A value that does not
appear in any retrieved chunk is NOT a valid sentinel-commit; that
case must escalate.

---

## 5. Researcher contract (both modes)

A researcher's `solve()` ends when:

| stop reason | trigger |
|---|---|
| `all_resolved` | every id in `allowed_blank_ids` has `status="resolved"` in shared state |
| `max_turns` | turn budget exhausted |
| `llm_error` | LLM call raised |
| `no_tool_call` | model returned text-only with no tool call |

The researcher prompt requires it to commit a value for every assigned
blank before stopping (using the `insufficient_evidence` sentinel in
`evidence_chunk_ids` as a fallback for weak retrieval). This rule is the
load-bearing piece behind the 10Q probe results below.

---

## 6. Plan-updater contract (llm mode only)

When the orchestrator calls `request_plan_update`, a separate LLM call
(`update_plan` in `_plan_updater.py`) takes:

- the question
- the current `GSWPlan` (full JSON)
- the current state (which blanks are resolved + their values)
- the orchestrator's `reason` and `evidence`

…and emits a revised `GSWPlan`. The updater MUST preserve existing
`blank_id`s wherever the slot semantics are unchanged (the orchestrator
relies on this for state carryover), MAY add or drop blanks, and MAY
change which blank is the target.

Validation pipeline reuses the planner's pydantic + grounding checks
from `_planner_emit.py`. On parse failure the updater retries once;
terminal failure is logged as `plan_update_rejected` and the orchestrator
continues with the OLD plan.

State reconciliation:

| diff class | action |
|---|---|
| blank_id in old AND new plan | preserve `BlankResult` (resolved values carry) |
| blank_id only in new plan | initialize `BlankResult(status="unknown")` |
| blank_id only in old plan | drop from state (logged in `dropped_ids`) |

---

## 7. 10Q FRAMES probe — head-to-head

Same 10 questions, same retriever (hybrid), same reasoner
(`bedrock/openai.gpt-oss-120b-1:0`), same judge (`gpt-4o`).

| qid | gold | deterministic | llm |
|---|---|---|---|
| q174 | "The Boeing 777 was first flown on…" | ✗ `'Boeing 777'` (18 turns) | ✗ `'Boeing 777'` (6 turns) |
| q70 | `'84,512'` | **✓ `'84512'` (20)** | **✓ `'84512'` (22)** |
| q445 | `'14'` | ✗ `'15'` (17) | ✗ `'8.0'` (17) |
| q401 | `'100'` | ✓ `'100'` (11) | ✓ `'100'` (9) |
| q376 | `'Ornithologie'` | ✗ empty (20, max_turns) | **✓ `'Ornithologie'` (52)** |
| q196 | `'5'` | ✓ `'5'` (6) | ✓ `'5'` (5) |
| q167 | `'Daniela Lavender'` | ✓ (8) | ✓ (8) |
| q742 | "LeBron James, Diana Taurasi, Tina Charles, DeWanna Bonner" | ✗ 3/4 names, wrong 4th (20) | ✗ orchestrator LLM error at turn 0 |
| q154 | `'Dark Side of the Moon'` | ✓ (5) | ✓ (4) |
| q796 | `'Italy'` | ✓ (5) | ✓ (5) |

**Totals**: deterministic **6/10 (60%)**, llm **7/10 (70%)**.
Reference: flat `planner_react` with v5.1 prompt + Hybrid retriever scored
**5/10 (50%)** on the same 10-question slice.

The single divergent Q is **q376**: deterministic exhausts its researcher's
turn budget without committing; llm re-dispatches after the first researcher
returns weak evidence and eventually commits the correct answer. This is
exactly the behavior the LLM mode was designed to enable.

The single regression is **q742**: the orchestrator LLM call errored at the
first turn (model returned no tool call), short-circuiting the run before
any researcher fired. Deterministic mode produces a partial-list commit
instead.

---

## 8. When to prefer each mode

| situation | prefer |
|---|---|
| Budget-sensitive, one-shot workflows | **deterministic** |
| Reproducibility (no orchestrator-side LLM noise) | **deterministic** |
| Plan errors are the dominant failure mode | **llm** |
| Multi-blank levels where parallel retrieval matters | **llm** (true thread-pool concurrency) |
| You want plan-revision events visible in the trajectory for debugging | **llm** |

---

## 9. Streamlit inspector

Both modes are inspectable via:

```bash
.venv/bin/streamlit run playground/planner_orchestrator_run_inspector.py
```

The inspector auto-detects `extra["orchestrator_mode"]` per question:

- **deterministic** mode → 5 tabs: Plan / Researchers / Executed blanks /
  Messages / Reasoning.
- **llm** mode → 7 tabs: adds **Orchestrator** (per-turn orchestrator
  tool calls + plan-update events + dispatches list) and **Plan versions**
  (graphviz of every plan revision side-by-side).

Inside each researcher card the **Turn-by-turn narrative** shows each
assistant turn with its hidden reasoning inlined above the tool calls
(JSON args + JSON tool results, all collapsible).

---

## 10. CLI

```bash
.venv/bin/python playground/run_substitution.py \
  --system ours_gsw_planner_orchestrator_v1 \
  --model bedrock/openai.gpt-oss-120b-1:0 \
  --retriever hybrid \
  --subset configs/frames_dev_100.json --limit 10 \
  --max-turns 60 \
  --orchestrator-mode {deterministic|llm} \
  --orchestrator-max-turns 12 \
  --llm-judge --judge-model gpt-4o
```

`--orchestrator-mode` is the only knob that switches between the two
modes; `--orchestrator-max-turns` is honored only by `llm` and bounds
the orchestrator-side ReAct loop.

---

## 11. Source pointers

| component | file:lines |
|---|---|
| Adapter entry point | `src/research_agent/adapters/ours/gsw_planner_orchestrator_v1.py` |
| Deterministic loop | `_run_deterministic` in same file |
| LLM-orchestrator loop | `_run_llm_orchestrator` in same file |
| Parallel dispatcher | `_dispatch_subplan_parallel` in same file |
| Researcher class | `ReActResearcher` in same file |
| Researcher prompt | `_researcher_prompt.py` |
| Orchestrator prompt | `_orchestrator_prompt.py` |
| Plan-updater | `_plan_updater.py` |
| Probe configs | `configs/frames_dev_100.json` |

| log dir | n | judge | mode |
|---|---:|---:|---|
| `logs/orch_phase1v1_10q/` | 10 | 0.30 | deterministic (v1 probe) |
| `logs/orch_phase1v2_10q/` | 10 | 0.50 | deterministic (commit-fix probe) |
| `logs/orch_phase2_det_10q/` | 10 | 0.60 | deterministic |
| `logs/orch_phase2_llm_10q/` | 10 | 0.70 | llm (Phase-2) |
| `logs/orch_phase3_llm_10q/` | 10 | 0.60 | llm (Phase-3 with escalation) |
