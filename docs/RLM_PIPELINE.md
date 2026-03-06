# RLM Pipeline: Deterministic Bridge QA Generation

## 1. Overview & Motivation

The **RLM (Recursive Language Models) pipeline** replaces the legacy agentic chat loop for sleep-time bridge QA generation. Inspired by the [RLM framework](https://alexzhang13.github.io/blog/2025/rlm/), the pipeline uses a **deterministic Python state machine** (the RootScheduler) that recursively dispatches compact, stateless LLM calls (the Edge Workers) rather than running an unbounded tool-calling conversation.

**Key benefits over the legacy pipeline:**

| Property | Legacy (agentic) | RLM |
|----------|------------------|-----|
| Token growth | O(conversation length) — history accumulates | O(edge packet size) — constant per call |
| Budget control | Soft (max_iterations) | Hard per-edge token + call caps |
| Exploration order | LLM-decided, non-deterministic | Python-sorted by cross-doc overlap, reproducible |
| Observability | Tool call traces in conversation | Structured events (edge_summary, doc_summary) |
| Model calls per edge | Variable (5-50+) | Bounded (1-4 typical, configurable max) |

**Source file:** `src/gsw_memory/sleep_time/rlm_pipeline.py`

---

## 2. Architecture

```
AgenticReconciler
  │
  ├─ pipeline_mode == "rlm"
  │   └─ _create_rlm_scheduler()
  │       ├─ RecursiveEdgeWorker  (stateless LLM caller)
  │       └─ RootScheduler        (deterministic state machine)
  │           ├─ run_corpus()     → iterates documents
  │           ├─ run_document()   → iterates edges within a doc
  │           └─ _run_edge()      → the core edge processing loop
  │
  └─ pipeline_mode == "legacy"
      └─ explore_entity()  → tool-calling chat loop
```

### Two-Component Design

**RootScheduler** — Python code that:
- Owns all exploration state (which edges are pending, explored, accepted)
- Calls GSWTools directly via `_call_tool()` (no LLM intermediary)
- Manages per-edge and per-document token budgets
- Mines PathProofs deterministically from QA pair data
- Decides when to recurse, retry, or stop

**RecursiveEdgeWorker** — Stateless LLM wrapper that:
- Receives a compact `EdgePacket` or `RenderInput` payload
- Returns JSON with `{status, candidates, need_recursion, notes}`
- Has zero memory between calls — no conversation history
- Includes JSON repair retry on parse failure

### Entry Points

The `AgenticReconciler.explore_entity()` method dispatches to RLM:

```python
if self.pipeline_mode == "rlm":
    if doc_id:
        return self.run_rlm_document(doc_id)    # single document
    if entity_name is None:
        return self.run_rlm_corpus()             # full corpus
```

Both methods create a fresh `RootScheduler` via `_create_rlm_scheduler()`.

### Note on "Root" Naming

"Root" in `RootScheduler` means **top-level orchestrator** — it sits at the root of the scheduling hierarchy and dispatches workers. It is almost entirely pure Python with no LLM calls. The one exception is `_select_optional_doc_ids()`: when there are more fuzzy-matched optional docs than `max_optional_docs_per_edge`, it makes a small LLM call (stage=`"root"`, using `root_model`) to pick the best ones. If that call fails, it falls back to deterministic top-k by score. This is the **only** place the `root_model` / `"root"` stage is used.

---

## 3. Data Structures

### EdgeKey
Identifies a single entity-neighbor edge within a document.

```python
@dataclass
class EdgeKey:
    doc_id: str            # e.g. "doc_3"
    entity_name: str       # source entity
    neighbor_name: str     # target neighbor entity
    relationship: str      # verb phrase connecting them
```

### EdgePacket
Compact payload sent to the worker LLM. Contains everything needed for one edge — no external lookups required.

```python
@dataclass
class EdgePacket:
    edge: EdgeKey
    source_docs: List[str]              # all docs involved (source + mandatory + optional)
    mandatory_docs: List[str]           # docs where neighbor exists by exact match
    optional_docs: List[str]            # docs from fuzzy matching
    source_context: Dict[str, Any]      # compacted QA pairs, relationships, roles for source entity
    mandatory_contexts: List[Dict]      # compacted contexts from mandatory docs
    optional_contexts: List[Dict]       # compacted contexts from optional/fuzzy docs
    constraints: Dict[str, Any]         # e.g. {"must_be_chain": True, "reverse_required": True}
    budget: Dict[str, int]              # {"max_depth", "max_calls", "max_tokens"}
```

### PathProof
A deterministic 2-hop path mined from QA pairs without any model call.

```python
@dataclass
class PathProof:
    edge: EdgeKey
    source_fact: Dict[str, Any]    # {doc_id, question, answer, answer_refs} from source entity
    neighbor_fact: Dict[str, Any]  # {doc_id, question, answer, answer_refs} from neighbor entity
    path_docs: List[str]           # sorted union of docs spanned (must be >= 2)
    target_refs: List[str]         # doc_id::entity_id refs from the neighbor fact
```

### RenderInput
Strict renderer payload — worker can ONLY use provided proofs, no free context search.

```python
@dataclass
class RenderInput:
    edge: EdgeKey
    proofs: List[PathProof]
    constraints: Dict[str, Any]
```

### WorkerOutput
Normalized output from one worker invocation.

```python
@dataclass
class WorkerOutput:
    status: str                      # "ok", "parse_error", etc.
    candidates: List[Dict[str, Any]] # normalized bridge candidates
    need_recursion: bool             # worker requests deeper exploration
    notes: str                       # free-form notes from worker
    parse_stage: str                 # "initial" or "repair_retry"
    raw_preview: str                 # truncated raw response for debugging
```

### EdgeRunResult
Execution summary for one edge, used for event emission and aggregation.

```python
@dataclass
class EdgeRunResult:
    edge: EdgeKey
    accepted: int           # bridges successfully created
    attempted: int          # total candidates proposed
    rejected: int           # candidates that failed sanitization or verification
    budget_exhausted: bool  # hit token/call cap
    edge_tokens: int        # tokens consumed for this edge
    property_attempted: bool # backward compat field (always False in current code)
```

---

## 4. Flow: Corpus → Document → Edge

### 4.1 Corpus Level — `run_corpus()`

```
while budget allows:
    plan = plan_corpus_exploration(strategy="max_pending_neighbors", limit=20)
    next_doc = plan["next_doc"]
    if not next_doc or next_doc.status == "completed": break
    run_document(next_doc.doc_id)
```

Budget checks: `max_documents` and `max_tokens` (from `reconciler.budget`).

### 4.2 Document Level — `run_document(doc_id)`

```
1. plan_document_exploration(doc_id)  → builds entity-neighbor checklist
2. edges = _iter_pending_edges(doc_id)  → sorted by cross-doc overlap (desc)
3. while edges exist (safety cap = max(10, len(edges)*3)):
       _run_edge(edges[0])
       edges = _iter_pending_edges(doc_id)  # re-fetch (edges get marked explored)
4. If all edges complete: mark_document_explored(doc_id)
```

Edge ordering: `_iter_pending_edges` sorts by `(-other_docs_count, entity_name, neighbor_name, relationship)` — edges with the most cross-document evidence are explored first.

### 4.3 Edge Level — `_run_edge(edge)` (the core loop)

This is the central processing loop. Each step in order:

```
 1. get_entity_context(entity_name, doc_id)     → source context
 2. begin_neighbor_focus(edge)                   → lock neighbor focus
 3. plan_neighbor_doc_coverage(edge)             → mandatory + optional fuzzy docs
 4. _select_optional_doc_ids(edge, fuzzy_docs)   → root model picks best optional docs
 5. _collect_contexts_for_docs(neighbor, docs)   → fetch contexts for mandatory + optional
 6. _build_edge_packet(edge, coverage, contexts) → compact EdgePacket
 7. _score_edge_signal(coverage, contexts)       → tier + budget allocation
 8. LOOP (bounded by call_limit + token_limit):
    a. _mine_path_proofs(packet)                 → deterministic 2-hop paths
    b. If proofs exist → RenderInput (constrained), else → free EdgePacket
    c. worker.generate(packet/render_input)      → LLM call → WorkerOutput
    d. _apply_candidates(packet, candidates)     → sanitize → dedupe → create_bridge_qa
    e. Loop control: check budget, recursion, repeated failure
 9. mark_neighbor_explored(edge, bridges_created)
```

---

## 5. Edge Signal Scoring & Budget Tiers

Before any LLM call, the scheduler scores each edge to allocate resources proportionally.

### Formula

```
signal_score = min(2.0, mandatory_docs * 0.8)
             + min(2.0, qa_count * 0.15)
             + min(1.0, best_fuzzy_score / 2.0)
```

Where:
- `mandatory_docs` = number of docs where neighbor exists by exact match
- `qa_count` = total QA pairs across mandatory contexts
- `best_fuzzy_score` = highest fuzzy match score among optional docs

### Tier Thresholds

| Tier | Score | Call Limit | Token Limit | Behavior |
|------|-------|------------|-------------|----------|
| **high** | ≥ 2.0 | `edge_max_calls` (default 2) | `edge_max_tokens` (default 3000) | Full budget |
| **medium** | ≥ 1.0 | min(edge_max_calls, 3) | min(edge_max_tokens, 7000) | Moderate budget |
| **low** | < 1.0 | min(edge_max_calls, 1) | min(edge_max_tokens, 2500) | One cheap probe |

### probe_only Flag

Set when `tier == "low"` AND `mandatory_docs == 0` AND `qa_count == 0`. In this case, the edge gets a single probe call and stops regardless of output.

---

## 6. PathProof Mining

`_mine_path_proofs(packet, include_optional, max_paths=6)` deterministically extracts 2-hop paths from QA pairs without any model call.

### Algorithm

```
1. Find source_links: source QAs that mention the neighbor entity name
2. For each source_link:
   a. Extract answer_refs → source_docs
   b. For each neighbor context (mandatory + optional):
      For each neighbor QA:
        - Must have valid answer_refs (doc_id::entity_id format)
        - Answer must not be the neighbor or source entity name (prevents trivial loops)
        - path_docs = source_docs ∪ target_ref_docs ∪ context_doc
        - Must span ≥ 2 documents
        - Deduplicate by (source_question, neighbor_question, target_refs, path_docs)
3. Return up to max_paths proofs
```

### When Proofs Exist vs. Don't

- **Proofs found** → Worker receives `RenderInput` (constrained renderer prompt). The worker can ONLY use facts from the provided proofs — no free exploration.
- **No proofs** → Worker receives full `EdgePacket` (free-form prompt). The worker searches the contexts freely to propose candidates.

---

## 7. Worker Prompts

### Two Prompt Modes

**`_build_messages(packet, depth, include_optional)`** — Free-form mode:
- System prompt: "You are an edge-local bridge generator..."
- Includes `WORKER_LEGACY_EXAMPLES` (4 good + 4 bad examples)
- Includes `WORKER_PRE_OUTPUT_CHECKLIST` (5-point validation)
- User prompt: full EdgePacket as JSON

**`_build_render_messages(render_input)`** — Constrained renderer mode:
- System prompt: "You are a bridge QA renderer. Use ONLY the provided path_proofs..."
- Same examples and checklist
- Grounding rule: "every answer ref and reverse answer ref must come from path_proofs"
- User prompt: RenderInput as JSON (proofs only, no free context)

### Prompt Constants

| Constant | Purpose |
|----------|---------|
| `WORKER_PROMPT_VERSION` | `"legacy_rich_4x4_v1"` — version tag for tracking prompt changes |
| `WORKER_LEGACY_EXAMPLES` | 4 good + 4 bad bridge examples showing proper 2-hop reasoning |
| `WORKER_PRE_OUTPUT_CHECKLIST` | 5-point checklist: forward chain, reverse validity, reasoning format, grounded refs, no leakage |

### JSON Repair

If the initial parse fails, the worker sends a repair message:
```
[original messages] + [assistant: raw_text] + [user: "Re-output ONLY one valid JSON object..."]
```
Temperature drops to 0.0 for the repair call. If repair also fails, returns `WorkerOutput(status="parse_error", candidates=[])`.

---

## 8. Candidate Sanitization & Entity Resolution

Every candidate from the worker passes through `_sanitize_candidate_refs(packet, candidate)` before being submitted to `create_bridge_qa`. This is a multi-stage pipeline:

### Stage 1: Source Doc Validation
- Candidate's `source_docs` are filtered to only those in `packet.source_docs`
- Must have ≥ 2 docs after filtering

### Stage 2: Context Ref Map
`_build_context_answer_ref_map(packet)` builds a lookup from normalized answer text → `doc_id::entity_id` refs across all contexts (source + mandatory + optional).

### Stage 3: Answer Ref Resolution
For each answer/reverse_answer in the candidate:
1. **Explicit ref** (`doc_id::entity_id`): validate doc is in source_docs, check ref exists in allowed_refs, or fuzzy-resolve the entity part
2. **Plain text**: try each source doc, resolve via:
   - Context ref map (exact normalized match)
   - Doc entity index (exact normalized name match)
   - Fuzzy token matching (Jaccard-like, threshold ≥ 0.8, must be unambiguous)

### Stage 4: Multi-Doc Enforcement
- Resolved refs must span ≥ 2 distinct documents
- `source_docs` is canonicalized to only include docs that have resolved refs

### Stage 5: In-Response Deduplication
`_exact_candidate_signature(candidate)` creates a tuple of (question, reverse_question, answers, reverse_answers, source_docs, reasoning) to dedupe within a single worker response.

---

## 9. Loop Control & Failure Detection

The `_run_edge` loop has several control mechanisms:

### Recursion Triggers
1. **Optional context expansion**: If no candidates accepted on first call AND optional contexts exist → enable `include_optional`, increment depth
2. **Worker-requested recursion**: If `worker_output.need_recursion == True` AND depth < max → increment depth

### Repeated Failure Detection
`_candidate_signature()` normalizes candidates into a comparable tuple:
```python
(normalized_question, normalized_reverse_question,
 sorted_normalized_answers, sorted_normalized_reverse_answers, sorted_source_docs)
```

If the same signature appears at the same state `(depth, include_optional, has_render_input, proof_count)` in consecutive failed calls → **stop immediately** (tracked as `rlm_metrics["repeated_failure_stops"]`).

### Budget Enforcement
- **Token budget**: `edge_token_limit - (reconciler.tokens_used - start_tokens)` checked before each call AND after each call
- **Call budget**: `call_count < edge_call_limit`
- **Early success exit**: If `accepted_total > 0` → break immediately (one good bridge per edge is enough)

### probe_only Shortcut
For low-signal edges with `probe_only=True`: if first call produces zero accepted → stop (no retry).

---

## 10. Thinking Token Extraction

For reasoning models (e.g., Bedrock/OpenAI o-series, Qwen thinking models), the worker extracts thinking content:

```python
# Check message attributes (set by some providers)
for attr in ("reasoning", "reasoning_content"):
    val = getattr(message, attr, None)
    if val: thinking_text = val

# Fallback: <think> tags in raw content
if not thinking_text and "<think>" in raw_text:
    match = re.search(r"<think>(.*?)</think>", raw_text)
    thinking_text = match.group(1)
```

- Logged via `logger.info("Worker thinking [edge=...]: ...")`
- Emitted as `rlm_worker_thinking` callback event for the Streamlit trace analyzer
- `<think>` tags are **stripped** from `raw_text` before JSON parsing to prevent parse failures

---

## 11. Token Counting & Metrics

### Token Counting Path

All model calls go through `AgenticReconciler._call_model_for_stage()`, which calls `_consume_usage(response, stage)`:

```python
def _consume_usage(self, response, stage):
    usage = response.usage
    self.tokens_used   += total_tokens     # global
    self.input_tokens  += prompt_tokens    # global
    self.output_tokens += completion_tokens # global

    # Stage-level breakdown:
    self.rlm_metrics[f"{stage}_input_tokens"]  += prompt_tokens
    self.rlm_metrics[f"{stage}_output_tokens"] += completion_tokens
```

Stages: `root` (only used for optional doc selection in `_select_optional_doc_ids` — typically very few calls), `worker` (bridge generation — the bulk of token usage), `verifier` (bridge verification in `create_bridge_qa`).

### Per-Edge Token Tracking

```python
start_tokens = self.reconciler.tokens_used
# ... all edge work ...
edge_tokens = self.reconciler.tokens_used - start_tokens
```

### rlm_metrics Dictionary

| Key | Type | Description |
|-----|------|-------------|
| `edges_explored` | int | Total edges processed |
| `edges_with_bridges` | int | Edges that produced ≥1 accepted bridge |
| `docs_attempted` | int | Documents entered |
| `docs_completed` | int | Documents fully explored (all edges done) |
| `recursive_invocations` | int | Times depth was incremented |
| `low_signal_edges` | int | Edges scored as "low" tier |
| `repeated_failure_stops` | int | Edges stopped due to repeated identical failures |
| `root_input_tokens` | int | Input tokens for root-stage calls |
| `root_output_tokens` | int | Output tokens for root-stage calls |
| `worker_input_tokens` | int | Input tokens for worker-stage calls |
| `worker_output_tokens` | int | Output tokens for worker-stage calls |
| `verifier_input_tokens` | int | Input tokens for verifier-stage calls |
| `verifier_output_tokens` | int | Output tokens for verifier-stage calls |

---

## 12. Events & Callbacks

All events are emitted via `AgenticReconciler.output_callback(event_type, data)` if set.

| Event | When | Key Fields |
|-------|------|------------|
| `tool_call` | Before each GSWTools call | `tool`, `arguments` |
| `tool_result` | After each GSWTools call | `tool`, `result`, `is_error` |
| `rlm_edge_summary` | After each edge completes | `edge`, `accepted`, `attempted`, `rejected`, `budget_exhausted`, `edge_tokens`, `signal_tier`, `signal_score`, `worker_prompt_version` |
| `rlm_doc_summary` | After each document completes | `entity` (doc_id), `iterations`, `bridges_created`, `edge_results[]` |
| `rlm_worker_thinking` | When thinking content is extracted | `edge`, `thinking`, `depth` |

---

## 13. GSWTools Used by RLM

The RootScheduler calls these tools directly (no LLM intermediary):

| Tool | Called In | Purpose |
|------|-----------|---------|
| `plan_corpus_exploration` | `run_corpus()` | Pick next document to explore |
| `plan_document_exploration` | `run_document()` | Build entity-neighbor checklist for a doc |
| `get_entity_context` | `_run_edge()`, `_collect_contexts_for_docs()` | Fetch QA pairs, relationships, roles for an entity |
| `begin_neighbor_focus` | `_run_edge()` | Lock focus on a specific neighbor |
| `plan_neighbor_doc_coverage` | `_run_edge()` | Get mandatory + optional fuzzy docs for a neighbor |
| `create_bridge_qa` | `_apply_candidates()` | Submit sanitized bridge candidate for verification + storage |
| `mark_neighbor_explored` | `_run_edge()` | Mark edge as done with bridge count |
| `mark_document_explored` | `run_document()` | Mark doc as fully explored |
| `get_doc_exploration_status` | `run_document()` | Check remaining pending edges |

---

## 14. Configuration Reference

All parameters are set on `AgenticReconciler.__init__()` and propagated to the scheduler/worker:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `pipeline_mode` | `"legacy"` | `"rlm"` to use the deterministic pipeline |
| `root_model` | same as `model_name` | Model for root-stage calls (optional doc selection) |
| `worker_model` | same as `model_name` | Model for edge worker calls |
| `edge_max_depth` | `1` | Max recursion depth per edge (0 = no recursion) |
| `edge_max_calls` | `2` | Max worker LLM calls per edge |
| `edge_max_tokens` | `3000` | Token budget per edge (input + output combined) |
| `max_optional_docs_per_edge` | `2` | Max fuzzy-matched docs to include per edge |
| `reasoning_effort` | `"medium"` | Passed to bedrock/gpt-oss models |
| `budget["max_tokens"]` | — | Global token cap for corpus run |
| `budget["max_documents"]` | — | Max documents to explore in corpus run |

### Worker Internal Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `WORKER_PROMPT_VERSION` | `"legacy_rich_4x4_v1"` | Tracked in events for prompt A/B testing |
| Max path proofs | `6` | `_mine_path_proofs(max_paths=6)` |
| Max QA pairs per context | `12` | `_compact_context_payload` truncates to 12 |
| Worker temperature | `0.2` | For initial call |
| Repair temperature | `0.0` | For JSON repair retry |
| Max response tokens | `min(1400, remaining_edge_tokens)` | Per worker call |
| Entity fuzzy match threshold | `0.8` | In `_resolve_entity_for_doc` |

---

## 15. CLI Usage

### Bridge Test (per-question evaluation)

```bash
python playground/sleep_time/run_bridge_test.py \
    --gsw_path /path/to/gsw/networks \
    --start 0 --end 10 \
    --model bedrock/openai.gpt-oss-120b-1:0 \
    --pipeline_mode rlm \
    --output_dir logs/bridge_test \
    --show-thinking --verbose
```

### Full Corpus Exploration

```bash
python playground/sleep_time/run_sleep_time.py \
    --gsw_path /path/to/gsw/networks \
    --num_docs 20 \
    --model bedrock/openai.gpt-oss-120b-1:0 \
    --pipeline_mode rlm \
    --output_dir logs/sleep_time_rlm \
    --verbose
```

### Key CLI Args for RLM

| Arg | Default | Effect |
|-----|---------|--------|
| `--pipeline_mode rlm` | `legacy` | Enable RLM pipeline |
| `--edge_max_depth` | `1` | Recursion depth per edge |
| `--edge_max_calls` | `2` | Worker calls per edge |
| `--edge_max_tokens` | `3000` | Token budget per edge |
| `--max_optional_docs` | `2` | Fuzzy docs per edge |
| `--show-thinking` | off | Log thinking tokens |
| `--reasoning_effort` | `medium` | For thinking models |

---

## 16. Comparison: Legacy vs RLM (Same Question Set)

From actual runs on MuSiQue question 0:

| Metric | Legacy | RLM |
|--------|--------|-----|
| Bridges | 12 | 52 |
| Total tokens | 826K | 574K |
| Worker calls | 661 | 210 |
| Token efficiency | 68K tok/bridge | 11K tok/bridge |

The RLM pipeline produces **4x more bridges** at **30% fewer tokens** by eliminating conversation history growth and applying targeted edge-level budgets.

---

## 17. Comparison with Original RLM Framework

Our pipeline is inspired by the [Recursive Language Models (RLM) framework](https://alexzhang13.github.io/blog/2025/rlm/) by Alex Zhang et al. Here is how the two relate.

### Similarities

| Concept | Original RLM | Our Pipeline |
|---------|-------------|--------------|
| Core idea | Don't feed everything into one giant context — decompose and recurse | Same: edge-local packets instead of growing conversation |
| Context rot prevention | Root LM never sees full context; sub-LMs get slices | Worker never sees conversation history; gets compact EdgePacket per call |
| Recursive depth | Root (depth=0) spawns sub-LM calls (depth=1) | RootScheduler spawns RecursiveEdgeWorker calls, with `edge_max_depth` controlling recursion |
| Deterministic orchestration | Python REPL environment controls flow | Python RootScheduler controls flow |
| Model decides candidates | LM chooses how to partition/map context | Worker decides bridge candidates; scheduler decides whether to recurse or expand optional docs |

### Differences

| Aspect | Original RLM | Our Pipeline |
|--------|-------------|--------------|
| Environment | General Python REPL — model writes arbitrary code | Fixed tool set (GSWTools) — scheduler calls tools directly, model only generates JSON |
| Who controls recursion | The LM itself decides when to spawn sub-calls | The Python scheduler decides (signal scoring, budget tiers, repeated failure detection) |
| Model autonomy | High — model chooses decomposition strategy, writes code, determines when to stop | Low — model is a stateless JSON generator; all state and strategy lives in Python |
| Budget guarantees | None — "no strong guarantees about controlling cost or runtime" | Hard per-edge token + call caps, tier-based budgets |
| Recursion mechanism | `RLM_M(q̂, Ĉ)` — sub-model gets a new query + context slice | `worker.generate(packet)` — worker gets a new EdgePacket, optionally with PathProofs |
| Context access | Model accesses context programmatically via Python variable manipulation | Context is pre-fetched by scheduler and embedded in the packet — no runtime access |
| Task type | General (QA over long docs, summarization, code review) | Specific (bridge QA generation across GSW documents) |
| Depth | Currently depth=1, designed for arbitrary depth | Configurable `edge_max_depth`, but "recursion" means retry-with-more-context, not spawning sub-RLMs |

### Key Insight

The original RLM philosophy is **"let the model decide everything"** — the model writes code, chooses decomposition strategy, and spawns sub-calls autonomously. Our pipeline takes the opposite stance: **the Python scheduler decides everything** (which edges to explore, what budget to allocate, when to recurse, when to stop), and the model is a narrow JSON-in/JSON-out bridge candidate generator. We borrowed the *structural pattern* (root orchestrator + recursive sub-calls to avoid context growth) but not the *autonomy principle*.
