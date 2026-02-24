# Personal Memory GSW — Implementation Reference

## What Was Built

This document describes the personal conversational memory subsystem added to `src/gsw_memory/personal_memory/`. It processes two-person, multi-session conversations (LoCoMo format) into a three-layer hierarchical memory structure with speaker attribution, enabling adversarial question handling and cross-session reasoning.

---

## Module Layout

```
src/gsw_memory/personal_memory/
├── __init__.py                  # Public exports
├── models.py                    # ConversationMemory, PersonMemory containers
├── chunker.py                   # TopicBoundaryChunker
├── processor.py                 # PersonalMemoryProcessor (orchestrator)
├── reconciler.py                # ConversationReconciler (Layer 2 & 3)
├── qa_agent.py                  # PersonalMemoryQAAgent
└── data_ingestion/
    ├── __init__.py
    └── locomo.py                # LoCoMoLoader, Session, Turn, Conversation, QAPair
```

### Modified existing files
| File | Change |
|------|--------|
| `src/gsw_memory/memory/models.py` | Added `speaker_id`, `conversation_id`, `evidence_turn_ids` to `Role`, `EntityNode`, `Question` |
| `src/gsw_memory/prompts/operator_prompts.py` | Added `PromptType.CONVERSATIONAL` + `ConversationalOperatorPrompts` class |
| `src/gsw_memory/memory/operator_utils/gsw_operator.py` | Handle `CONVERSATIONAL` prompt type; pass `speaker_context` field |

---

## Three-Layer Architecture

```
Input: Conversation (N sessions, each with turns)
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — Session GSW (one per session)                        │
│  CorefOperator (full session) → TopicBoundaryChunker →          │
│  GSWOperator(CONVERSATIONAL) → Reconciler(exact)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │  List[GSWStructure]
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2 — Conversation GSW                                     │
│  ConversationReconciler.reconcile_sessions()                    │
│  — Speaker-filtered: only same-speaker entities are candidates  │
│  — Two passes: speaker_a entities, then speaker_b entities      │
└──────────────────────────┬──────────────────────────────────────┘
                           │  GSWStructure
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3 — Person GSW (cross-conversation, always runs)         │
│  ConversationReconciler.reconcile_conversations_agentic()       │
│  — LLM agent with get_entity_timeline / compare_entities /      │
│    detect_contradictions / finish_reconciliation tools          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Pipeline Detail (per conversation)

### Step 1 — Coreference Resolution
- **Class**: `CorefOperator` (existing, `memory/operator_utils/coref.py`)
- **Input**: Full session text from `session.to_document()` for all sessions
- **Why before chunking**: Pronouns need full-session context for resolution
- **Output**: Resolved text per session

### Step 2 — Topic-Boundary Chunking
- **Class**: `TopicBoundaryChunker` (`personal_memory/chunker.py`)
- **Method**: `chunk_session_from_text(resolved_text, session)`
  - Topic boundaries detected from **original** `session.turns` (more accurate than resolved text)
  - Chunk strings assembled from **resolved** text lines (distributed proportionally by turn count)
- **Fallback**: Fixed-size splitting every `max_turns_per_chunk=15` turns if LLM fails
- **Output**: `List[str]` chunk strings per session, each formatted as:
  ```
  [Session 2 — 2023-04-15]
  [D2:1] Caroline: text...
  [D2:2] Melanie: text...
  ```

### Step 3 — GSW Extraction (batch)
- **Class**: `GSWOperator(PromptType.CONVERSATIONAL)` (existing, extended)
- All chunks from all sessions batched into one curator call for efficiency
- Each input includes `speaker_context` field: `"Speaker A: Caroline, Speaker B: Melanie"`
- **Prompt**: `ConversationalOperatorPrompts` — full 6-task structure (same as episodic) plus:
  - Speaker attribution: every `Role` and `Question` gets `speaker_id` and `evidence_turn_ids`
  - Example uses Yigit/Mark (not LoCoMo test-set names)
- `response_format=GSWStructure` for constrained decoding

### Step 4 — Layer 1: Per-Session Reconciliation
- Fresh `Reconciler("exact")` per session
- Chunks reconciled in order; `conversation_id` stamped on all entities
- Uses existing `EntityVerifier` LLM inside `ExactMatchStrategy` for ambiguous merges

### Step 5 — Layer 2: Cross-Session Reconciliation
- `ConversationReconciler.reconcile_sessions(session_gsws, speaker_a, speaker_b)`
- **Speaker filter**: `_SpeakerFilteredIndex` wraps `EntityIndex` — returns empty candidates for cross-speaker queries
- Two reconciliation passes (one per speaker); results combined with `_merge_two_gsws()`

### Step 6 — Layer 3: Agentic Cross-Conversation Reconciliation
- `ConversationReconciler.reconcile_conversations_agentic(person_id, conversation_memories, model_name)`
- **Always runs** (not optional)
- LLM agent (Responses API tool calling) with 4 tools:
  | Tool | Purpose |
  |------|---------|
  | `get_entity_timeline` | Full roles/states for one entity by index |
  | `compare_entities` | Name similarity + shared roles between two entities |
  | `detect_contradictions` | Conflicting states for same role across entities |
  | `finish_reconciliation` | Submit merge pairs `[[keep_idx, discard_idx], ...]` |
- Agent absorbs discard entity's roles into keep entity, then drops discard

---

## Data Models

### Extended models (`memory/models.py`)
```python
class Role(BaseModel):
    ...
    speaker_id: Optional[str] = None          # who said this
    conversation_id: Optional[str] = None
    evidence_turn_ids: List[str] = []         # ["D1:3", "D4:7"]

class EntityNode(BaseModel):
    ...
    speaker_id: Optional[str] = None
    conversation_id: Optional[str] = None

class Question(BaseModel):
    ...
    speaker_id: Optional[str] = None
    conversation_id: Optional[str] = None
    evidence_turn_ids: List[str] = []
```
All fields optional → fully backward compatible.

### Container models (`personal_memory/models.py`)
```python
@dataclass
class ConversationMemory:
    conversation_id: str
    speaker_a: str
    speaker_b: str
    gsw: GSWStructure           # Layer 2 output
    session_gsws: List[GSWStructure]  # Layer 1 outputs

@dataclass
class PersonMemory:
    person_id: str
    conversation_memories: Dict[str, ConversationMemory]
    global_gsw: Optional[GSWStructure] = None   # Layer 3 output
```

---

## QA Agent

`PersonalMemoryQAAgent` (`personal_memory/qa_agent.py`):

### Question routing
1. `target_conversation_id` set → use `ConversationMemory.gsw` for that conversation
2. Question names a known speaker → use their `ConversationMemory.gsw`
3. Default → `PersonMemory.global_gsw` (Layer 3)

### In-memory tools (`_InMemoryGSWTools`)
Works directly on a `GSWStructure` object (no file I/O):
- `search_entities(query, limit)` — keyword search over names and roles
- `get_entity_context(entity_id)` — full roles, states, evidence_turn_ids
- `get_all_entities()` — summary listing

### Speaker verification (adversarial filter)
After retrieval, checks: does the question name speaker A but the retrieved `speaker_id` is speaker B? If mismatch → `abstain=True` in the return dict.

Return shape:
```python
{
    "answer": str,
    "reasoning": str,
    "speaker_id": Optional[str],
    "evidence_turn_ids": List[str],
    "abstain": bool,
    "confidence": "high" | "low",
    "conversation_id": Optional[str],
}
```

---

## Usage Examples

### Process one conversation (Layers 1 + 2)
```python
from gsw_memory.personal_memory import LoCoMoLoader, PersonalMemoryProcessor

convs = LoCoMoLoader("playground_data/locomo.json").load()
processor = PersonalMemoryProcessor(model_name="gpt-4o-mini")
conv_memory = processor.process_conversation(convs[0])

print(f"Sessions: {len(conv_memory.session_gsws)}")
print(f"Entities: {len(conv_memory.gsw.entity_nodes)}")
```

### Process a person across all conversations (Layers 1 + 2 + 3)
```python
person_memory = processor.process_person(
    person_id="Caroline",
    conversations=convs,
)
print(f"Global entities: {len(person_memory.global_gsw.entity_nodes)}")
```

### Answer questions
```python
from gsw_memory.personal_memory import PersonalMemoryQAAgent

agent = PersonalMemoryQAAgent(model_name="gpt-4o")
result = agent.answer(
    question="What job does Caroline have?",
    person_memory=person_memory,
)
print(result["answer"])
print("Abstain:", result["abstain"])   # True for adversarial mismatches
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Extend existing models, not separate hierarchy | Single source of truth; backward compatible; all new fields optional |
| Coref before chunking | Pronouns span full sessions; need full context for resolution |
| `chunk_session_from_text()` separates boundary detection from text assembly | Boundaries detected on original turns (cleaner), text assembled from resolved (accurate) |
| Speaker filter in Layer 2 | Prevents Caroline's "nurse" role from merging with Melanie's "nurse" role |
| Layer 3 always mandatory | Cross-conversation entity dedup is essential for person-level memory quality |
| New `CONVERSATIONAL` prompt type (not modifying `EPISODIC`) | Keeps the existing pipeline unchanged; opt-in only for conversational use |
| In-memory QA tools (not file-based GSWTools) | Personal memory is built in-memory; no need to write/read files for QA |
