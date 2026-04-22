## Complete Step-by-Step Workflow

### Phase 0: Discovery (Start Here!)

Before diving into entity exploration, understand the document landscape:

**Iteration 1:**

1. **Map document structure**: Call `get_document_overlap_matrix()` to see which document pairs share entities and where clusters form.

   **Example output:**
   ```json
   {
     "pairs": [
       {"doc_a": "doc_0", "doc_b": "doc_4", "shared_entities": ["Lothair II", "Teutberga"], "count": 2},
       {"doc_a": "doc_0", "doc_b": "doc_6", "shared_entities": ["Lothair II"], "count": 1}
     ],
     "total_pairs": 15
   }
   ```

**Iterations 2-4:**

2. **Scan promising entities**: For entities shared between interesting doc pairs, call `preview_cross_doc_connections(entity_name)` to quickly see what role an entity plays in each document. Look for entities with `bridge_potential: "high"` — they have diverse relationships across documents.

3. **Pick your targets**: Queue all promising entities where different documents provide different facts. An entity in 2 docs with diverse relationships is better than one in 5 docs saying the same thing.

4. **Lock in your queue**: Call `set_exploration_targets([...])` with your chosen entities and reasons. This persists your plan so you won't forget targets during deep exploration.

   **Example:**
   ```python
   set_exploration_targets(targets=[
     {"entity_name": "Lothair II", "reason": "high bridge_potential across 4 docs", "priority": "high"},
     {"entity_name": "Teutberga", "reason": "diverse family relationships across docs", "priority": "medium"}
   ])
   ```

---

### Phase 1: Entity Reconciliation

**Iteration 5-6:**

5. **Get next target**: Call `get_exploration_targets()` to see what's next in your queue.

6. **Call `reconcile_entity_across_docs(entity_name)`**
   - Purpose: Get unified view of entity across ALL documents
   - Returns: Which docs mention this entity, all QA pairs, all relationships

   **Example output:**
   ```json
   {
     "entity": "Lothair II",
     "docs": ["doc_0", "doc_4", "doc_6", "doc_9"],
     "merged_relationships": {
       "married to": ["Teutberga"],
       "son of": ["Emperor Lothair I", "Ermengarde of Tours"],
       "daughter of": ["Bertha"]
     }
   }
   ```

---

### Phase 2: Create Exploration Plan (MANDATORY)

**Iteration 7:**

7. **Call `plan_entity_exploration(entity_name, merged_relationships)`**
   - Purpose: Create explicit TODO list of ALL relationships to explore
   - Relationships are **deduplicated by entity name** — if the same entity appears under multiple relationship types, it appears only once in the plan
   - **This prevents the agent from forgetting relationships!**

   **Example output:**
   ```json
   {
     "entity": "Lothair II",
     "relationships_to_explore": [
       {"name": "Teutberga", "type": "married to", "status": "pending"},
       {"name": "Emperor Lothair I", "type": "son of", "status": "pending"},
       {"name": "Ermengarde of Tours", "type": "son of", "status": "pending"}
     ],
     "total_relationships": 3,
     "pending_count": 3
   }
   ```

---

### Phase 3: Systematic Relationship Exploration

**For EACH relationship in the plan:**

#### Step 3.1: Get Documents for Related Entity

8. **Call `get_entity_documents(related_entity_name)`**
   - Purpose: Find which docs mention this related entity

   **Example:**
   ```python
   get_entity_documents("Teutberga")
   # Returns: ["doc_0", "doc_4"]
   ```

#### Step 3.2: Batch Retrieve Contexts

9. **Call `get_entity_context(related_entity, [list_of_doc_ids])`**
   - Purpose: Get context from ALL documents in ONE call
   - **Batch mode saves 1 iteration per extra document!**

   **Example:**
   ```python
   get_entity_context("Teutberga", ["doc_0", "doc_4"])
   # Returns: [
   #   {"doc_id": "doc_0", "qa_pairs": [{"question": "Who was queen of Lotharingia?", "answer": "Teutberga"}], ...},
   #   {"doc_id": "doc_4", "qa_pairs": [{"question": "Who is Teutberga's father?", "answer": "Boso the Elder"}], ...}
   # ]
   ```

#### Step 3.3: Identify Multi-Doc Connections (Decomposition Test)

10. **Analyze contexts to find bridge opportunities**
    - Look for facts that combine information from different docs
    - **Apply the decomposition test**: Can you write Sub-Q1 and Sub-Q2 where Sub-Q2 uses Sub-Q1's answer?

    **Example findings from Teutberga:**
    - doc_0: "Teutberga is queen of Lotharingia"
    - doc_4: "Teutberga's father is Boso the Elder"
    - **Bridge opportunity**: "Who was Lotharingia's queen's father?" → "Boso the Elder" (doc_0 + doc_4)
      - Sub-Q1: Lotharingia >> queen → Teutberga (doc_0)
      - Sub-Q2: Teutberga >> father → Boso the Elder (doc_4)
      - ✓ Sub-Q2 uses #1 (Teutberga) — valid chain!

#### Step 3.4: Create Bridges in Batch Mode

11. **Call `create_bridge_qa(bridges=[{bridge1}, {bridge2}, ...])`**
    - Can create up to **20 bridges** in a single call
    - Use batch mode when you found multiple connections
    - **Duplicate detection**: Bridges with identical questions are automatically rejected (MD5-based dedup)
    - Each bridge must include: `question`, `answers`, `reverse_question`, `reverse_answers`, `source_docs`, `reasoning`

    **Example:**
    ```python
    create_bridge_qa(bridges=[
      {
        "question": "Who was Lotharingia's queen's father?",
        "answers": ["doc_4::e3"],
        "reverse_question": "Whose daughter was queen of Lotharingia?",
        "reverse_answers": ["doc_0::e2"],
        "source_docs": ["doc_0", "doc_4"],
        "reasoning": "Teutberga was queen of Lotharingia (doc_0). Her father was Boso the Elder (doc_4)."
      }
    ])
    # Returns: [{"success": True, "bridge_id": "bridge_abc123", "validation": {"valid": True, "confidence": 0.85}}]
    ```

#### Step 3.5: Mark Relationship as Explored

12. **Call `mark_relationship_explored(entity, relationship_name, bridges_created)`**
    - Purpose: Check off this relationship from TODO list
    - Supports batch mode for marking multiple at once
    - Returns updated checklist showing remaining relationships

    **Example:**
    ```python
    mark_relationship_explored("Lothair II", "Teutberga", bridges_created=2)
    # Returns: {
    #   "relationship_marked": "Teutberga",
    #   "remaining": ["Emperor Lothair I", "Ermengarde of Tours"],
    #   "explored_count": 1,
    #   "pending_count": 2,
    #   "completion_percentage": 33.3
    # }
    ```

#### Step 3.6: Continue to Next Relationship

13. **Repeat steps 8-12** for next relationship from the plan
    - Pick next pending relationship from the "remaining" list
    - Continue until all relationships explored

---

### Phase 4: Verification & Completion

**Final iterations:**

14. **Call `get_exploration_status(entity_name)`**
    - Purpose: Verify all relationships have been checked
    - Check: `ready_to_complete = true`

    **Example:**
    ```json
    {
      "entity": "Lothair II",
      "explored": [
        {"name": "Teutberga", "type": "married to", "bridges": 2},
        {"name": "Emperor Lothair I", "type": "son of", "bridges": 1},
        {"name": "Ermengarde of Tours", "type": "son of", "bridges": 2}
      ],
      "pending": [],
      "ready_to_complete": true
    }
    ```

    **If `ready_to_complete = false`**: Continue exploring remaining relationships!

15. **Call `mark_entity_explored(entity_name, total_bridges_created)`**
    - Purpose: Mark entity as fully explored
    - Entity won't be selected again
    - Exploration targets queue auto-updates

    **Example:**
    ```python
    mark_entity_explored("Lothair II", num_bridges_created=5)
    ```

16. **Call `get_exploration_targets()`** and repeat from Phase 1 for the next entity

---

## Bridge Quality: Decomposition Test

Every bridge must decompose into chained sub-questions where hop 1's answer feeds hop 2:

**Template**:
```
Sub-Q1: [Entity A] >> [relationship] → #1        (from doc X)
Sub-Q2: #1 >> [property] → Answer                 (from doc Y)
Bridge: "What is [property] of [Entity A's relationship]?"
```

**Good bridge** — decomposes cleanly:
- "What trade route did Merchant Giovanni's **patron** control?"
  - Sub-Q1: Giovanni >> patron → Baron Heinrich (doc_12)
  - Sub-Q2: Baron Heinrich >> trade route → Amber Road (doc_28)
  - ✓ You MUST resolve Sub-Q1 before answering Sub-Q2.

**Bad bridge** — does NOT decompose (fact conjunction):
- "Which NATO member country lacks a traditional army but participated in the ideological conflict with the Warsaw Pact?"
  - Fact 1: Iceland lacks army (doc_20)
  - Fact 2: NATO fought Warsaw Pact (doc_23)
  - ✗ These are independent facts — neither feeds the other.

**Decomposition test**: Try writing Sub-Q1 and Sub-Q2. If Sub-Q2 doesn't use #1 (the answer from Sub-Q1), it's a conjunction, not a chain.

**Flip the perspective**: When exploring entity A across docs, find its related entities and start the question from a "leaf" entity, not the "hub":
- Exploring "Guild of Weavers": appears in doc_7 (established by Werner) and doc_14 (traded with Flanders 1220)
- Bad: "What guild was established in 1215 and traded with Flanders?" ← conjunction about Guild
- Good: "When did **Master Craftsman Werner's guild** trade with Flanders?" ← start from leaf entity Werner

**Also avoid**:
- **Circular bridges**: Q and reverse Q restate the same fact
- **Answer-in-question**: Description gives away the answer

---

## Tools Reference (14 tools)

### Discovery Tools (4)

**`get_entity_documents(entity_name)`**
- Returns: List of document IDs mentioning this entity
- Example: `["doc_0", "doc_4", "doc_6"]`

**`get_document_entities(doc_id)`**
- Returns: List of entities mentioned in this document
- Example: `["Lothair II", "Teutberga", "Lotharingia"]`

**`get_document_overlap_matrix(min_shared=1)`**
- Returns: Document pairs sorted by shared entity count
- Use this FIRST to understand document structure and identify where bridges are likely

**`preview_cross_doc_connections(entity_name)`**
- Returns: Quick preview of entity's role in each document with `bridge_potential` rating
- Much cheaper than `reconcile_entity_across_docs` — use for scanning candidates

### Context Tools (2)

**`get_entity_context(entity_name, doc_id=None)`**
- If `doc_id` is string: Returns context from that doc
- If `doc_id` is list: Returns list of contexts (BATCH MODE)
- If `doc_id` is None: Returns merged context from all docs
- Example: `get_entity_context("Lothair II", ["doc_0", "doc_4"])` → batch retrieval

**`reconcile_entity_across_docs(entity_name)`**
- Returns: Merged view of entity across ALL documents
- Includes: docs list, merged QA pairs, merged relationships

### Bridge Tools (2)

**`create_bridge_qa(...)`**
- Single mode: Pass question, answers, reverse_question, reverse_answers, source_docs, reasoning
- Batch mode: Pass `bridges=[...]` with up to **20** bridge objects
- **Duplicate detection**: Rejects bridges with identical questions (MD5-based dedup)
- Returns: Success status, bridge IDs, validation results
- Example: `create_bridge_qa(bridges=[{...}, {...}])` → batch creation

**`get_bridge_statistics()`**
- Returns: Stats on bridges created so far
- Includes: total count, coverage, quality metrics

### Planning Tools (2)

**`set_exploration_targets(targets=[...])`**
- Purpose: Lock in entity exploration queue after discovery phase
- Input: List of `{entity_name, reason, priority}` objects
- Replaces queue if called again

**`get_exploration_targets()`**
- Returns: Queue showing completed, in-progress, and pending targets
- Key field: `next_target` — the next entity to explore

### Tracking Tools (3)

**`plan_entity_exploration(entity_name, relationships)`**
- Purpose: Create explicit TODO list after reconcile
- Input: Entity name + merged_relationships from reconcile
- **Deduplicates by entity name** — same entity under different relationship types appears only once
- Returns: Plan with all relationships marked "pending"

**`mark_relationship_explored(entity_name, relationship_name, bridges_created=0)`**
- Single mode: Mark one relationship
- Batch mode: Pass lists for relationship_name and bridges_created
- Returns: Updated checklist with completion percentage
- Example: `mark_relationship_explored(entity, ["rel1", "rel2"], [2, 1])` → batch marking

**`get_exploration_status(entity_name)`**
- Purpose: Verify all relationships checked before completion
- Returns: Lists of explored vs pending relationships
- Key field: `ready_to_complete` (true/false)

### Strategy Tools (1)

**`mark_entity_explored(entity_name, num_bridges_created=0)`**
- Purpose: Mark entity as fully explored
- Entity won't be selected again
- Call after `get_exploration_status` shows `ready_to_complete: true`

---

## Running Modes

### Autonomous Mode
Agent discovers entities itself using discovery tools (`get_document_overlap_matrix` → `preview_cross_doc_connections` → `set_exploration_targets`).

```bash
python playground/sleep_time/run_sleep_time.py \
    --gsw_path /path/to/networks \
    --num_docs 20 \
    --model Qwen/Qwen3-32B \
    --base_url http://127.0.0.1:6379/v1
```

### Seed Entity Mode
Pre-computed seed entities are provided via JSON file. Agent explores entities one-by-one without discovery phase.

```bash
python playground/sleep_time/run_sleep_time.py \
    --gsw_path /path/to/networks \
    --seed_entities_file data/sleep_time/musique/doc_entities.json \
    --model Qwen/Qwen3-32B \
    --base_url http://127.0.0.1:6379/v1
```

Seed entity mode is ~2.8x more token-efficient but lacks cross-entity context.

### Bridge Test (MuSiQue)
```bash
python playground/sleep_time/run_bridge_test.py \
    --gsw_path /path/to/networks \
    --start 0 --end 10 \
    --model Qwen/Qwen3-235B-A22B-Thinking-2507 \
    --output_dir logs/bridge_test
```

### Local vLLM GPT-OSS on 2x RTX A6000
Use the helper script:

```bash
playground/sleep_time/serve_vllm_gpt_oss_120b_a6000_tp2.sh
```

Then point sleep-time runs at:

```bash
--base_url http://127.0.0.1:6379/v1 \
--model openai/gpt-oss-120b
```

### Local vLLM Qwen 3.5 on 1x RTX A6000
Use the helper script:

```bash
playground/sleep_time/serve_vllm_qwen3_5_35b_a3b_a6000.sh
```

Then point sleep-time runs at:

```bash
--base_url http://127.0.0.1:6379/v1 \
--model Qwen/Qwen3.5-35B-A3B \
--root_model Qwen/Qwen3.5-35B-A3B \
--worker_model Qwen/Qwen3.5-35B-A3B
```

If the exact base model is unstable at `32768` context on a single A6000, retry the serve command with:

```bash
--max-model-len 16384
```

---

**Last Updated**: 2026-03-02
**Version**: 4.0 (decomposition test, discovery phase, 14 tools, duplicate detection)
