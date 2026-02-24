# Personal Memory GSW - Design Document

## Executive Summary

This document outlines the design for extending GSW to handle personal conversational memory benchmarks (LoCoMo, LongMemEval). The key innovation is **hierarchical cross-conversation memory** with **speaker-attributed QA networks** and **agentic reconciliation**.

### Novel Contributions

1. **Speaker-Attributed QA Networks**: Questions and answers track who said what, when, in which conversation
2. **Hierarchical Cross-Conversation Structure**: Separate conversation-level and person-level memory
3. **Agentic Cross-Conversation Reconciliation**: Agent decides when to merge entities across different conversations

### Backward Compatibility

- All changes are backward compatible (optional fields, new files, parallel pipeline)
- Existing GSW pipeline continues to work unchanged
- Personal memory is a new opt-in system

---

## Problem Definition

### The John Challenge

In LoCoMo benchmark, **John appears in 3 different conversations**:
- conv-41: John ↔ Maria
- conv-43: Tim ↔ John
- conv-47: James ↔ John

**Challenge**: How do we reconcile John's memory across these independent conversations?

**Example scenario**:
- In conv-41 (May 2023), John tells Maria: "I'm a software engineer at TechCorp"
- In conv-43 (July 2023), John tells Tim: "I just became team lead at my company"
- In conv-47 (Sept 2023), John tells James: "I left my job to start a startup"

**Questions**:
1. Are these three different jobs or one job evolving?
2. Is "my company" in conv-43 the same as "TechCorp" from conv-41?
3. Should these be merged into a single "John's career" entity or kept separate?

### The Caroline-Melanie Challenge (Adversarial Questions)

In conv-26, Melanie ran a charity race, not Caroline.

**Adversarial question**: "What did Caroline realize after her charity race?"

**Expected answer**: Should abstain - Caroline never ran a race, MELANIE did.

**How GSW should handle this**:
- Track who said/did what (speaker attribution)
- Verify entity-action associations before answering
- Return provenance: "This was Melanie's action, not Caroline's"

---

## Architecture Overview

### Three-Layer Hierarchy

```
Layer 1: Session GSW
  - Raw extraction from one session's dialogue
  - Example: conv-41, session 5 (John-Maria conversation on May 15)

Layer 2: Conversation GSW
  - Sessions reconciled within one conversation
  - Example: All John-Maria sessions (1-19) merged into single GSW

Layer 3: Person GSW (Global)
  - Conversations reconciled across all of person's conversations
  - Example: John's unified memory from conv-41, conv-43, conv-47
```

### Data Flow Example

**Input**: John appears in 3 conversations

**Processing**:

1. **Layer 1**: Extract session-level GSWs
   ```
   conv-41 sessions: [s1_gsw, s2_gsw, ..., s19_gsw]
   conv-43 sessions: [s1_gsw, s2_gsw, ..., s19_gsw]
   conv-47 sessions: [s1_gsw, s2_gsw, ..., s19_gsw]
   ```

2. **Layer 2**: Reconcile within conversations
   ```
   conv-41: Merge all John-Maria sessions → john_maria_gsw
   conv-43: Merge all John-Tim sessions → john_tim_gsw
   conv-47: Merge all John-James sessions → john_james_gsw
   ```

3. **Layer 3**: Agentic cross-conversation reconciliation
   ```
   Input: john_maria_gsw, john_tim_gsw, john_james_gsw
   Agent: Decides which entities to merge across conversations
   Output: john_global_gsw (unified memory)
   ```

**Final Structure**:
```
PersonMemory(person_id="John")
  ├── conversation_memories
  │   ├── conv-41: ConversationMemory(gsw=john_maria_gsw, partner="Maria")
  │   ├── conv-43: ConversationMemory(gsw=john_tim_gsw, partner="Tim")
  │   └── conv-47: ConversationMemory(gsw=john_james_gsw, partner="James")
  └── global_gsw: john_global_gsw (unified across all conversations)
```

---

## Speaker Attribution System

### What Gets Attributed

Every entity role, question, and answer tracks:
- **speaker_id**: Who said/claimed this
- **conversation_id**: Which conversation
- **evidence_turn_ids**: Which dialogue turns (e.g., ["D1:3", "D2:5"])

### Example: Caroline's Career Decision

**Input dialogue** (conv-26):
```
[Session 1 — May 8, 2023]
[D1:3] Caroline: I went to a LGBTQ support group yesterday...
[D1:9] Caroline: I'm keen on counseling or working in mental health
[D1:11] Caroline: I'd need psychology courses and counseling certification

[Session 4 — June 27, 2023]
[D4:7] Caroline: I'm thinking of working with trans people, helping them accept themselves
```

**Extracted GSW with speaker attribution**:

**Entity**: Caroline
- Role: "aspiring_counselor"
  - States: ["interested_in_mental_health", "considering_trans_counseling"]
  - speaker_id: "Caroline" (self-reported)
  - conversation_id: "conv-26"
  - mentioned_by: ["Caroline", "Melanie"]

**Question**: "What career path is Caroline considering?"
- Answers: ["counseling", "mental_health", "trans_counseling"]
- speaker_id: "Caroline" (Caroline said this)
- conversation_id: "conv-26"
- evidence_turn_ids: ["D1:9", "D1:11", "D4:7"]

**Verification for adversarial questions**:
```
Question: "What did Caroline realize after her charity race?"
System checks: Does Caroline have role="runner" with speaker_id="Caroline"?
Answer: NO - only Melanie has that role
Response: ABSTAIN - "This action was attributed to Melanie, not Caroline"
```

---

## Cross-Conversation Reconciliation

### Agentic Decision Making

**Scenario**: John's "job" entity from two conversations

**conv-41 (May 2023) - John with Maria**:
- Entity: "job"
  - Role: "software_engineer"
  - States: ["works_at_TechCorp", "senior_position"]
  - speaker_id: "John"
  - conversation_id: "conv-41"

**conv-43 (July 2023) - John with Tim**:
- Entity: "job"
  - Role: "team_lead"
  - States: ["managing_team", "promotion"]
  - speaker_id: "John"
  - conversation_id: "conv-43"

**Agent reasoning process**:

1. **Candidate detection**: Same entity name ("job") from same person (John) in different conversations

2. **Tool-based investigation**:
   - Tool: `get_entity_timeline("job", "John")`
     - Result: May 2023 (conv-41) → "software engineer"
                July 2023 (conv-43) → "team lead"

   - Tool: `compare_entities_across_conversations("job", "conv-41", "conv-43")`
     - Result: Both mention work/employment, timeline is sequential (May → July)

   - Tool: `get_conversation_context("conv-41", "job")`
     - Result: John said "TechCorp" in conv-41

   - Tool: `get_conversation_context("conv-43", "job")`
     - Result: John said "my company" in conv-43 (no company name)

3. **Agent decision**:
   ```
   Decision: MERGE
   Reason: "Both conversations refer to John's employment. Timeline is sequential
           (May → July). The role evolution from 'software_engineer' to 'team_lead'
           suggests a promotion at the same company. 'My company' in conv-43 likely
           refers to TechCorp from conv-41."

   Action: Merge into unified "job" entity with temporal role evolution
   ```

4. **Resulting unified entity**:
   - Entity: "job"
     - Roles:
       - [May 2023] "software_engineer" (states: ["works_at_TechCorp", "senior_position"])
       - [July 2023] "team_lead" (states: ["managing_team", "promotion"])
     - conversation_contexts: ["conv-41", "conv-43"]
     - Timeline: Tracks evolution across conversations

### Counter-Example: Separate Entities

**conv-41 (May 2023)**:
- Entity: "relationship"
  - Role: "dating_partner"
  - States: ["dating_Sarah", "serious_relationship"]

**conv-47 (Sept 2023)**:
- Entity: "relationship"
  - Role: "married"
  - States: ["married_to_Lisa", "newlywed"]

**Agent reasoning**:
```
Decision: SEPARATE
Reason: "Different partners (Sarah vs Lisa). Timeline gap (May → Sept) suggests
        the first relationship ended. These are two distinct relationships,
        not evolution of the same one."

Action: Keep as separate entities (relationship_1, relationship_2)
```

---

## Conversational Chunking

### Why Chunking Matters

LoCoMo sessions have ~20 turns each. A single session might cover multiple topics:
- Turns 1-8: Discussing weekend plans
- Turns 9-15: Career decisions
- Turns 16-20: Family matters

**Problem**: Fixed sentence chunking might split a coherent topic discussion.

### Chunking Strategies

**Strategy 1: Fixed Turn Count**
- Split every N speaker turns (e.g., N=10)
- Simple, predictable
- May split coherent topics

**Strategy 2: Time Gap Detection** (if timestamps available)
- Detect large gaps between turns
- Natural conversation breaks
- Requires timestamp metadata

**Strategy 3: Hybrid**
- Combine turn count + topic shifts
- Most context-aware
- More complex

### Example: Hybrid Chunking

**Input session** (20 turns):
```
Turns 1-8: [Topic: Weekend] Caroline and Melanie discuss a support group event
Turns 9-15: [Topic: Career] Caroline talks about counseling career
Turns 16-20: [Topic: Family] Caroline mentions adoption plans
```

**Chunking output**:
```
Chunk 1: Turns 1-8 (Weekend/Support Group topic)
Chunk 2: Turns 9-15 (Career topic)
Chunk 3: Turns 16-20 (Family topic)
```

**Benefit**: Each chunk focuses on one coherent topic, leading to cleaner GSW extraction.

---

## Reconciliation Strategies by Layer

### Layer 1: Within-Session (Chunks → Session GSW)

**Method**: Standard reconciliation (exact matching or embedding-based)

**Example**:
- Chunk 1 mentions "Caroline's career"
- Chunk 2 mentions "Caroline's career plan"
- Reconcile: Same entity, merge states

**Characteristics**:
- Same conversation, same session
- High confidence merges (short time span)
- Use existing GSW reconciler

### Layer 2: Cross-Session (Sessions → Conversation GSW)

**Method**: Conversation-aware reconciliation with person filtering

**Example**:
- Session 1 (May 8): Caroline mentions "considering counseling"
- Session 4 (June 27): Caroline mentions "trans counseling focus"
- Reconcile: Same career entity, states evolving over time

**Person filtering**:
- In John-Maria conversation, only reconcile John's entities and Maria's entities
- Don't merge John's entities with Maria's entities (different people)

**Characteristics**:
- Same conversation, multiple sessions
- Medium confidence merges (weeks apart)
- Filter by speaker before reconciling

### Layer 3: Cross-Conversation (Conversations → Person Global GSW)

**Method**: AGENTIC reconciliation with cross-conversation tools

**Example**:
- conv-41: John's "job" with Maria
- conv-43: John's "job" with Tim
- Agent decides: Merge or separate?

**Tools**:
- Timeline comparison: Are dates sequential or overlapping?
- Context inspection: Same company mentioned?
- Contradiction detection: Conflicting information?

**Characteristics**:
- Different conversations, different partners
- LOW confidence merges (requires reasoning)
- Agent-driven with tool use

---

## Agentic Tools for Cross-Conversation Reconciliation

### Tool 1: `get_entity_timeline(entity_name, person_id)`

**Purpose**: Show entity evolution across all conversations

**Example**:
```
Input: get_entity_timeline("job", "John")

Output:
  conv-41 (May 2023, with Maria):
    - Role: software_engineer
    - States: [works_at_TechCorp, senior_position]

  conv-43 (July 2023, with Tim):
    - Role: team_lead
    - States: [managing_team, promotion]

  conv-47 (Sept 2023, with James):
    - Role: startup_founder
    - States: [left_job, raising_funding]
```

**Agent use**: Detect temporal patterns (sequential evolution vs independent entities)

### Tool 2: `get_conversation_context(conv_id, entity_id)`

**Purpose**: Get detailed context of entity from specific conversation

**Example**:
```
Input: get_conversation_context("conv-41", "john_job")

Output:
  Entity: job
  Partner: Maria
  Mentions:
    - [D3:5] John: "I work at TechCorp as a software engineer"
    - [D7:10] John: "I've been there for 3 years, senior role now"

  Questions involving this entity:
    - Q: Where does John work? A: TechCorp
    - Q: What is John's role? A: software_engineer
```

**Agent use**: Verify details before merging (company names, specific states)

### Tool 3: `compare_entities_across_conversations(entity_name, conv_A, conv_B)`

**Purpose**: Side-by-side comparison

**Example**:
```
Input: compare_entities_across_conversations("job", "conv-41", "conv-43")

Output:
  conv-41 (May 2023):
    - Roles: [software_engineer]
    - Company: TechCorp

  conv-43 (July 2023):
    - Roles: [team_lead]
    - Company: [not specified, said "my company"]

  Overlap: Both involve employment, sequential timeline, possible promotion
```

**Agent use**: Identify similarities/differences to inform merge decision

### Tool 4: `detect_contradictions(entity_id)`

**Purpose**: Find conflicting states across conversations

**Example**:
```
Input: detect_contradictions("john_location")

Output:
  Contradiction detected:
    - conv-41 (May): John says "I live in Seattle"
    - conv-47 (Sept): John says "I moved to Austin for the startup"

  Resolution suggestion: Temporal evolution (moved between May and Sept)
```

**Agent use**: Distinguish evolution (moved cities) from errors (conflicting info)

---

## QA System Integration

### Question Routing

**Question analysis**: Identify target person and scope

**Example 1: Single conversation question**
```
Question: "What did Caroline tell Melanie about her career?"
Analysis:
  - Person: Caroline
  - Conversation: conv-26 (Caroline-Melanie)
  - Scope: Conversation-level GSW

Routing: Query ConversationMemory["conv-26"].gsw
```

**Example 2: Cross-conversation question**
```
Question: "What job does John have?"
Analysis:
  - Person: John
  - Conversation: Not specified
  - Scope: Global across all conversations

Routing: Query PersonMemory["John"].global_gsw
```

**Example 3: Adversarial question**
```
Question: "What did Caroline realize after her charity race?"
Analysis:
  - Person: Caroline
  - Action: "charity race"
  - Verification needed: Speaker attribution

Process:
  1. Search for "charity race" event in Caroline's memory
  2. Check speaker_id of the role
  3. Result: speaker_id="Melanie", not Caroline
  4. Response: ABSTAIN - "This action was performed by Melanie, not Caroline"
```

### Provenance Tracking

Every answer includes:
- **Evidence turn IDs**: ["D1:3", "D4:7"]
- **Speaker**: Who said this
- **Conversation**: Which conversation
- **Confidence**: Based on speaker attribution

**Example provenance**:
```
Question: "What career is Caroline pursuing?"
Answer: "Counseling and mental health work with transgender people"
Provenance:
  - Speaker: Caroline (self-reported)
  - Conversation: conv-26 (Caroline-Melanie)
  - Evidence: D1:9, D1:11, D4:7
  - Confidence: High (direct self-report from same person)
```

---

## Benchmark-Specific Strengths

### LoCoMo Strengths

**Category 1: Adversarial Questions (47 questions)**
- **GSW advantage**: Speaker attribution enables verification
- **Example**: "What did X do?" → Check if speaker_id matches X
- **Expected improvement**: Significant (baselines struggle here)

**Category 2: Cross-Session Questions (30/199 questions)**
- **GSW advantage**: Structured cross-session reconciliation
- **Example**: Multi-hop reasoning across sessions 1, 4, 7
- **Expected improvement**: Moderate (some baselines handle this)

**Category 3: Temporal Reasoning (37 questions)**
- **GSW advantage**: Temporal grounding in roles + SpaceTime linking
- **Example**: "When did Caroline go to support group?" (May 7, inferred from "yesterday")
- **Expected improvement**: Moderate (Zep's bi-temporal KG also strong here)

### LongMemEval Strengths

**Type: Multi-session reasoning (133 questions)**
- **GSW advantage**: Hierarchical structure preserves conversation context
- **Expected improvement**: Moderate

**Type: Knowledge update (78 questions)**
- **GSW advantage**: Temporal role evolution tracks changes
- **Example**: "What car did user buy?" (Answer changed from session 5 to session 20)
- **Expected improvement**: Moderate to high

---

## Implementation Phases (High-Level)

### Phase 1: Foundation (Data Models)
- Extend existing models with optional speaker/conversation fields
- Create PersonMemory and ConversationMemory abstractions
- Ensure backward compatibility

### Phase 2: Speaker-Aware Extraction
- Create personal memory prompts (separate from existing)
- Build operator that populates speaker fields
- Test on single conversation

### Phase 3: Hierarchical Reconciliation
- Session-level reconciliation (chunks → session GSW)
- Conversation-level reconciliation (sessions → conversation GSW)
- Test on single person with one conversation

### Phase 4: Agentic Cross-Conversation
- Build cross-conversation tools
- Implement agentic reconciler
- Test on John's 3 conversations

### Phase 5: QA Integration
- Build question routing logic
- Add speaker verification for adversarial questions
- Add provenance tracking

### Phase 6: Evaluation
- Run on full LoCoMo (10 conversations, 1990 QA pairs)
- Focus areas: adversarial, cross-session, temporal
- Compare to baselines

---

## Success Metrics

### Quantitative
- **LoCoMo adversarial F1**: Target >70% (baseline: ~50%)
- **LoCoMo cross-session F1**: Target >60%
- **Overall LoCoMo LLM-Judge**: Target >80%

### Qualitative
- Correct speaker attribution in 95%+ of extractions
- Agentic reconciliation decisions are interpretable (agent explains reasoning)
- No breaking changes to existing GSW pipeline

---

## Risk Mitigation

### Risk 1: Speaker Attribution Errors
- **Mitigation**: Extensive testing on dialogue parsing
- **Fallback**: If speaker unclear, mark as "unknown" rather than guessing

### Risk 2: Agentic Reconciliation Too Expensive
- **Mitigation**: Only use for cross-conversation (not cross-session)
- **Fallback**: Simple embedding-based merge for single-conversation scenarios

### Risk 3: Breaking Existing Pipeline
- **Mitigation**: All new fields optional, separate prompts, parallel pipeline
- **Validation**: Comprehensive backward compatibility tests

---

## Future Extensions

### Extension 1: Topic Nodes (Deferred)
- Add topic extraction and linking
- Enable topic-based retrieval across conversations
- **Rationale for deferring**: Speaker attribution is more critical for benchmarks

### Extension 2: Multi-Person Reconciliation
- Currently: Each person's memory is independent
- Future: Link shared entities (e.g., Caroline and Melanie both know "the support group")
- Enables: "What did Caroline and Melanie discuss together about X?"

### Extension 3: Temporal Conflict Resolution
- Currently: Agent detects contradictions
- Future: LLM-based temporal reasoning to resolve conflicts
- Example: "John said X in May, Y in July" → infer change over time

---

## Comparison to Baselines

### vs Mem0 (Atomic Facts)
- **Mem0**: Stores "John is a software engineer" (fact)
- **GSW**: Stores "Caroline claimed John is a software engineer in conv-26, session 3" (speaker-attributed fact)
- **Advantage**: Provenance, multi-perspective tracking

### vs Zep (Knowledge Graph Triples)
- **Zep**: (John, works_at, TechCorp) with bi-temporal tracking
- **GSW**: Same triple + speaker attribution + conversation context + pre-structured QA
- **Advantage**: Speaker awareness, QA networks

### vs A-MEM (Zettelkasten Notes)
- **A-MEM**: Notes evolve with dynamic links
- **GSW**: Entities evolve with temporal role tracking + cross-conversation reconciliation
- **Advantage**: Structured evolution, hierarchical memory

### vs MemoryOS (Hierarchical Tiers)
- **MemoryOS**: STM (raw dialogue) → MTM (segments) → LTM (profiles)
- **GSW**: Session GSW → Conversation GSW → Person GSW (all structured)
- **Advantage**: Everything is structured (no raw dialogue in memory)

### vs Associa (Event Graph + PCST)
- **Associa**: Event-centric graph with Prize-Collecting Steiner Tree retrieval
- **GSW**: Entity-centric QA networks with agentic retrieval
- **Advantage**: Speaker attribution, pre-structured QA

---

## Key Takeaways

1. **Core innovation**: Speaker-attributed QA networks with hierarchical cross-conversation memory
2. **Unique capability**: Agentic reasoning about entity identity across independent conversations
3. **Benchmark fit**: Directly addresses LoCoMo adversarial questions (speaker verification)
4. **Safety**: Fully backward compatible, parallel pipeline, no breaking changes
5. **Scope**: Well-defined, implementable in 3-5 days of core development

---

## Next Steps

1. Review this document - confirm approach alignment
2. Begin Phase 1: Extend data models with optional fields
3. Iterative development: Test each phase before moving to next
4. Continuous validation: Ensure backward compatibility at every step

---

**Document version**: 1.0
**Date**: 2026-02-20
**Status**: Design approved, ready for implementation
