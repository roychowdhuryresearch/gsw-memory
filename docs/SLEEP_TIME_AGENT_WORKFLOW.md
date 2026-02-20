## Complete Step-by-Step Workflow

### Phase 1: Entity Selection & Reconciliation

**Iteration 1-2:**

1. **Start with an entity** (e.g., "Lothair II")

2. **Call `reconcile_entity_across_docs(entity_name)`**
   - Purpose: Get unified view of entity across ALL documents
   - Returns:
     - Which docs mention this entity
     - All QA pairs about the entity
     - All relationships (spouse, parent, children, etc.)

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

**Iteration 3:**

3. **Call `plan_entity_exploration(entity_name, merged_relationships)`**
   - Purpose: Create explicit TODO list of ALL relationships to explore
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

4. **Call `get_entity_documents(related_entity_name)`**
   - Purpose: Find which docs mention this related entity

   **Example:**
   ```python
   get_entity_documents("Teutberga")
   # Returns: ["doc_0", "doc_4"]
   ```

#### Step 3.2: Batch Retrieve Contexts (⚡ OPTIMIZATION)

5. **Call `get_entity_context(related_entity, [list_of_doc_ids])`**
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

#### Step 3.3: Identify Multi-Doc Connections

6. **Analyze contexts to find bridge opportunities**
   - Look for facts that combine information from different docs

   **Example findings from Teutberga:**
   - doc_0: "Teutberga is queen of Lotharingia"
   - doc_4: "Teutberga's father is Boso the Elder"
   - **Bridge opportunity**: "Who was Lotharingia's queen's father?" → "Boso the Elder" (doc_0 + doc_4)

#### Step 3.4: Create Bridges in Batch Mode

7. **Call `create_bridge_qa(bridges=[{bridge1}, {bridge2}, ...])`**
   - Can create 1-5 bridges in a single call
   - Use batch mode when you found multiple connections

   **Example:**
   ```python
   create_bridge_qa(bridges=[
     {
       "question": "Who was Lotharingia's queen's father?",
       "answer": "Boso the Elder",
       "source_docs": ["doc_0", "doc_4"],
       "reasoning": "Teutberga was queen of Lotharingia (doc_0). Her father was Boso the Elder (doc_4)."
     },
     {
       "question": "Who was Lotharingia's queen married to?",
       "answer": "Lothair II",
       "source_docs": ["doc_0", "doc_4"],
       "reasoning": "Teutberga was queen of Lotharingia (doc_0). She was married to Lothair II (doc_4)."
     }
   ])
   # Returns: [{"success": True, "bridge_id": "bridge_abc123"}, {"success": True, "bridge_id": "bridge_def456"}]
   ```

#### Step 3.5: Mark Relationship as Explored

8. **Call `mark_relationship_explored(entity, relationship_name, bridges_created)`**
   - Purpose: Check off this relationship from TODO list
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

9. **Repeat steps 4-8** for next relationship from the plan
   - Pick next pending relationship from the "remaining" list
   - Continue until all relationships explored

---

### Phase 4: Verification & Completion

**Final iterations:**

10. **Call `get_exploration_status(entity_name)`**
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

11. **Call `mark_entity_explored(entity_name, total_bridges_created)`**
    - Purpose: Mark entity as fully explored
    - Entity won't be selected again

    **Example:**
    ```python
    mark_entity_explored("Lothair II", num_bridges_created=5)
    ```

12. **Move to next entity** and repeat from Phase 1



## 🛠️ Tools Reference

### Discovery Tools (2)

**`get_entity_documents(entity_name)`**
- Returns: List of document IDs mentioning this entity
- Example: `["doc_0", "doc_4", "doc_6"]`

**`get_document_entities(doc_id)`**
- Returns: List of entities mentioned in this document
- Example: `["Lothair II", "Teutberga", "Lotharingia"]`

### Context Tools (2)

**`get_entity_context(entity_name, doc_id=None)`**
- If `doc_id` is string: Returns context from that doc
- If `doc_id` is list: Returns list of contexts (BATCH MODE ⚡)
- If `doc_id` is None: Returns merged context from all docs
- Example: `get_entity_context("Lothair II", ["doc_0", "doc_4"])` → batch retrieval

**`reconcile_entity_across_docs(entity_name)`**
- Returns: Merged view of entity across ALL documents
- Includes: docs list, merged QA pairs, merged relationships

### Bridge Tools (2)

**`create_bridge_qa(...)`**
- Single mode: Pass question, answer, source_docs, reasoning
- Batch mode: Pass `bridges=[...]` with 1-5 bridge objects
- Returns: Success status and bridge IDs
- Example: `create_bridge_qa(bridges=[{...}, {...}])` → batch creation

**`get_bridge_statistics()`**
- Returns: Stats on bridges created so far
- Includes: total count, coverage, quality metrics

### Tracking Tools (3) 🆕

**`plan_entity_exploration(entity_name, relationships)`**
- Purpose: Create explicit TODO list after reconcile
- Input: Entity name + merged_relationships from reconcile
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


**Last Updated**: 2026-02-12
**Version**: 3.0 (with batch optimizations)
