# GSW for Personal Memory — Exploration Notes

## 1. Benchmarks

### 1.1 LoCoMo (ACL 2024)

**Paper:** Maharana et al., "Evaluating Very Long-Term Conversational Memory of LLM Agents"
**Data:** [snap-research/locomo](https://github.com/snap-research/locomo)

**Format:** 10 long two-person conversations, each with ~19 sessions spanning weeks/months. Each session is a timestamped dialogue (e.g., "1:56 pm on 8 May, 2023") with ~20 turns. Total ~420 turns per conversation, 199 QA pairs per conversation.

**Data structure:**
```json
{
  "sample_id": "conv-26",
  "conversation": {
    "speaker_a": "Caroline",
    "speaker_b": "Melanie",
    "session_1_date_time": "1:56 pm on 8 May, 2023",
    "session_1": [
      {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel! Good to see you!"},
      ...
    ],
    "session_2_date_time": "1:14 pm on 25 May, 2023",
    "session_2": [...],
    ...
  },
  "qa": [...],
  "event_summary": {...},
  "session_summary": {...}
}
```

**QA categories (conv-26 breakdown):**

| Category | Name | Count | Description |
|----------|------|-------|-------------|
| 1 | Single-hop | 32 | Answer in one turn |
| 2 | Temporal | 37 | Requires date/time reasoning |
| 3 | Multi-hop | 13 | Requires combining info across turns/sessions |
| 4 | Open-ended | 70 | Broader questions with longer answers |
| 5 | Adversarial | 47 | False premise — should abstain |

**Cross-session reasoning:** 30/199 QA pairs require evidence from multiple sessions. This is where structured memory should shine.

**Example — cross-session multi-hop:**
```
Q: What career path has Caroline decided to pursue?
A: Counseling or mental health for transgender people

Evidence:
  [Session 1, May 8]  Caroline: "I'm keen on counseling or working in mental health"
  [Session 4, Jun 27] Caroline: "I'm thinking of working with trans people,
                       helping them accept themselves..."
```

**Example — temporal reasoning:**
```
Q: When did Caroline go to the LGBTQ support group?
A: 7 May 2023

Evidence:
  [Session 1, May 8] Caroline: "I went to a LGBTQ support group yesterday..."
  → Must infer: session date is May 8, "yesterday" = May 7
```

**Example — adversarial (false premise):**
```
Q: What did Caroline realize after her charity race?
A: Should abstain — Caroline never ran a race, MELANIE did.

Evidence:
  [Session 2] Melanie: "I ran a charity race... self-care is really important."
  → Question wrongly attributes Melanie's action to Caroline
```

**Evaluation metrics:** F1, BLEU-1, LLM-as-Judge

---

### 1.2 LongMemEval (ICLR 2025)

**Paper:** Wu et al., "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory"
**Data:** [xiaowu0162/LongMemEval](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)

**Format:** 500 questions over user-assistant chat histories. Each question has a compiled chat history (haystack) of timestamped sessions. Available in three sizes: S (~115K tokens, ~40 sessions), M (~500 sessions), Oracle (evidence sessions only).

**Data structure:**
```json
{
  "question_id": "gpt4_2655b836",
  "question_type": "temporal-reasoning",
  "question": "What was the first issue I had with my new car after its first service?",
  "answer": "GPS system not functioning correctly",
  "question_date": "2023/04/10 (Mon) 23:07",
  "haystack_sessions": [[{role, content, has_answer?}, ...], ...],
  "haystack_dates": ["2023/04/10 (Mon) 17:50", ...],
  "answer_session_ids": [...]
}
```

**Question type breakdown (500 total):**

| Type | Count | Description |
|------|-------|-------------|
| temporal-reasoning | 133 | Temporal ordering, duration, recency |
| multi-session | 133 | Reasoning across multiple sessions |
| knowledge-update | 78 | Information that changes over time |
| single-session-user | 70 | Info from a single user message |
| single-session-assistant | 56 | Info from a single assistant message |
| single-session-preference | 30 | User preference extraction |

Questions ending with `_abs` are **abstention** questions (30 total) — system should recognize them as unanswerable.

**Key difference from LoCoMo:** User-assistant format (not two-person). Larger scale. Explicit temporal reasoning and knowledge-update categories.

**Evaluation:** LLM-as-Judge (GPT-4o) with type-specific prompts. Also supports retrieval metrics (Recall@k, NDCG@k).

---

### 1.3 LoCoMo-Plus (arXiv Feb 2026)

**Data:** [xjtuleeyf/Locomo-Plus](https://github.com/xjtuleeyf/Locomo-Plus)

Extends LoCoMo with harder "cognitive memory" tasks that test semantic disconnect between cue and target. No system has published results on this yet — potential differentiator.

### 1.4 ConvoMem (Salesforce, Nov 2025)

**Data:** [SalesforceAIResearch/ConvoMem](https://github.com/SalesforceAIResearch/ConvoMem)

75K QA pairs at scale. Good for efficiency testing. No system has published comprehensive results yet.

---

## 2. Prominent Methods

### 2.1 Associa (ACL 2025 Findings)

Zhang et al., "Bridging Intuitive Associations and Deliberate Recall"

**Core idea:** Event-centric memory graph with two-stage cognitive-inspired retrieval.

**Write time:** Conversations are parsed into an event-centric graph. Nodes = entities and events, edges = relationships. LLM-based extraction.

**Read time:** Two stages:
1. **Intuitive Association** — Prize-Collecting Steiner Tree (PCST) algorithm extracts evidence-rich subgraphs. PCST assigns "prizes" to relevant nodes and "costs" to edges, finding the optimal connected subgraph.
2. **Deliberating Recall** — Iterative LLM-driven query refinement for evidence the initial association may have missed.

**Benchmarks:** LoCoMo. Specific numbers not publicly available.

**Code:** Not released.

---

### 2.2 Mem0

Choudhary et al., "Building Production-Ready AI Agents with Scalable Long-Term Memory" (arXiv 2025)

**Core idea:** Extract atomic facts from conversations into a vector store with LLM-driven deduplication. Enhanced variant (Mem0^g) adds a directed labeled knowledge graph.

**Write time:** Two phases:
1. **Extraction** — LLM processes message pairs + conversation summary, outputs salient facts.
2. **Update** — For each fact, retrieve top-10 similar existing memories. LLM selects: ADD / UPDATE / DELETE / NOOP.
3. **(Mem0^g)** Additionally extracts (subject, relation, object) triples. Conflict detection + LLM resolution.

**Read time:**
- **Mem0:** Vector similarity search over stored facts.
- **Mem0^g:** Dual strategy — entity-centric graph traversal + semantic triple matching.

**LoCoMo results (GPT-4o-mini, F1 / BLEU-1 / LLM-Judge):**

| | Single-Hop | Multi-Hop | Open-Domain | Temporal |
|---|---|---|---|---|
| Mem0 | 38.72 / 27.13 / 67.1 | 28.64 / 21.58 / 51.2 | 47.65 / 38.72 / 72.9 | 48.93 / 40.51 / 55.5 |
| Mem0^g | 38.09 / 26.03 / 65.7 | 24.32 / 18.82 / 47.2 | 49.27 / 40.30 / 75.7 | 51.55 / 40.28 / 58.1 |

**Token efficiency:** ~1.8K context tokens vs 26K for full-context. p95 latency 1.44s vs 17.12s.

**Code:** [mem0ai/mem0](https://github.com/mem0ai/mem0) (Apache-2.0)

---

### 2.3 A-MEM (NeurIPS 2025)

Xu et al., "Agentic Memory for LLM Agents"

**Core idea:** Zettelkasten-inspired notes with LLM-determined links. Each memory is a structured note with keywords, tags, contextual description, and embeddings. Notes are dynamically interconnected and evolve as new information arrives.

**Write time:** Each new memory generates a note with 7 fields (content, timestamp, keywords, tags, context, embedding, links). LLM analyzes top-k similar existing notes to determine connections. Old memories evolve — their descriptions, keywords, and tags update as new information arrives.

**Read time:** Cosine similarity retrieves top-k notes. Linked notes from the same "box" (emergent cluster) are automatically included.

**LoCoMo results (GPT-4o-mini, F1 / BLEU-1):**

| | Single-Hop | Multi-Hop | Temporal | Open-Domain | Adversarial |
|---|---|---|---|---|---|
| A-MEM | 44.65 / 37.06 | 45.85 / 36.67 | 12.14 / 12.00 | 27.02 / 20.09 | 50.03 / 49.47 |

Strong on temporal reasoning. Particularly effective on smaller models (Qwen2.5-3b, Llama-3.2).

**Code:** [agiresearch/A-mem](https://github.com/agiresearch/A-mem) (MIT)

---

### 2.4 HippoRAG 2 (ICML 2025)

Gutierrez et al., "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models"

**Core idea:** Hippocampal-inspired indexing. LLM = neocortex, schema-less KG = hippocampal index, retrieval encoder = parahippocampal region. V2 adds passage nodes, query-to-triple linking, and LLM-based recognition memory.

**Write time:** LLM extracts OpenIE triples (subject-relation-object) from passages. Subjects/objects become phrase nodes; retrieval encoder identifies synonym pairs via vector similarity, adding synonym edges. V2 adds passage nodes connected to their derived phrases.

**Read time:**
1. Query-to-triple linking via text embeddings
2. LLM filters top-k triples (recognition memory)
3. Personalized PageRank from seed nodes through the graph
4. Top-ranked passages become reader context

**Evaluated on:** NQ, PopQA, MuSiQue, 2WikiMultiHop, HotpotQA, LV-Eval — document QA, **not conversational memory**. Not evaluated on LoCoMo or LongMemEval.

**Code:** [OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG)

---

### 2.5 Zep / Graphiti

Rasmussen et al., "Zep: A Temporal Knowledge Graph Architecture for Agent Memory" (arXiv 2025)

**Core idea:** Temporally-aware knowledge graph with bi-temporal tracking. Every edge carries four timestamps: when facts held true in the real world AND when the system learned about them. Old facts are invalidated, not deleted.

**Write time:** LLM extracts entities + facts from messages (with 4-turn context window). Entities matched via cosine + BM25; LLM resolves duplicates. Facts undergo temporal conflict resolution — contradictions invalidate the old edge's timestamp rather than deleting it.

**Read time:** Three parallel search methods combined:
1. Cosine similarity on embeddings
2. BM25 full-text search
3. BFS graph traversal from seed nodes

Reranked via Reciprocal Rank Fusion.

**LongMemEval results:** 71.2% accuracy (GPT-4o), 63.8% (GPT-4o-mini).

**Code:** [getzep/graphiti](https://github.com/getzep/graphiti) (Apache-2.0). Uses Neo4j backend.

---

### 2.6 MemoryOS (EMNLP 2025 Oral)

Kang et al., "Memory OS of AI Agent"

**Core idea:** Three-tier hierarchical memory inspired by OS memory management. Short-term (raw dialogue pages, FIFO queue), mid-term (topical segments, similarity-clustered), long-term (user/agent profiles + knowledge bases, heat-based promotion).

**Write time:**
- STM: Stores dialogue pages in fixed-length queue (7).
- STM -> MTM: FIFO transfer when STM overflows.
- MTM: Pages grouped into segments via cosine + Jaccard similarity. LLM summarizes each segment.
- MTM -> LTM: Heat score (visit frequency + interaction length + recency decay). Segments exceeding threshold transfer to LTM.

**Read time:** Multi-tier retrieval — all STM + top-k MTM segments + top-10 LTM entries by semantic relevance. Integrated into final prompt.

**LoCoMo results (GPT-4o-mini, F1 / BLEU-1):**

| | Single-Hop | Multi-Hop | Temporal | Open-Domain |
|---|---|---|---|---|
| MemoryOS | 35.27 / 25.22 | 41.15 / 30.76 | 20.02 / 16.52 | 48.62 / 42.99 |

Rank 1 across all categories on both GPT-4o-mini and Qwen2.5-3B.

**Code:** [BAI-LAB/MemoryOS](https://github.com/BAI-LAB/MemoryOS) (Apache-2.0)

---

### 2.7 Other Recent Systems (2025-2026)

| System | Key claim | LoCoMo | LongMemEval |
|--------|-----------|--------|-------------|
| **Hindsight** | Entity-temporal memory + retrospective reflection | 89.61% (Gemini-3 Pro) | 91.4% |
| **MemBuilder** | RL-trained memory construction model | 84.23% (Qwen3-4B) | 85.75% |
| **EverMemOS** | Self-organizing memory OS | 93% reasoning accuracy | — |
| **LiCoMemory** | Outperforms Mem0, Zep, A-Mem | — | 73.8% (GPT-4o-mini) |
| **Neuromem** | Scalable testbed for memory modules | Benchmark | Benchmark |

---

## 3. Where GSW Fits

### What GSW already does well:
- **Structured entity-event extraction** — entities with roles, states, and QA pairs capture richer semantics than flat facts (Mem0) or Zettelkasten notes (A-MEM)
- **Cross-document reconciliation** — the LOCAL reconciler merges entities across chunks/documents, directly applicable to cross-session entity linking
- **SpaceTime linking** — temporal and spatial grounding is built-in, addressing temporal reasoning questions
- **QA-pair memory** — information is pre-structured as QA pairs at write time, so the reader doesn't need to reason over raw text
- **Agentic retrieval** — tool-calling agent navigates the workspace via BM25 + embedding search

### Key differentiators vs competitors:

| Capability | GSW | Mem0 | A-MEM | Zep | MemoryOS | Associa |
|-----------|-----|------|-------|-----|----------|---------|
| Structured QA-pair memory | Yes | No (atomic facts) | No (notes) | No (triples) | No (raw dialogue) | No (event graph) |
| Entity roles + states | Yes | No | No | No | No | Partial |
| Cross-session reconciliation | Yes (LOCAL) | Partial (dedup) | Partial (evolution) | Yes (entity merge) | No | Unknown |
| Temporal grounding | Yes (SpaceTime) | No | No | Yes (bi-temporal) | No | No |
| Write-time QA generation | Yes | No | No | No | No | No |
| Agentic retrieval | Yes | No | No | No | No | Yes (2-stage) |

### GSW's unique angle:
Most competitors either (a) store flat facts/notes and retrieve by similarity, or (b) build knowledge graphs of triples. GSW is the only system that structures experiences as **QA-pair networks** with entity-event awareness. This means the read-time agent navigates pre-structured reasoning chains rather than assembling answers from scattered facts or triples.

### Potential weaknesses to watch:
- **Conversational language** — GSW was designed for documents (Wikipedia, narratives). Informal dialogue ("Gonna continue my edu") may trip up entity/role extraction.
- **Speaker attribution** — GSW extracts entities from text but doesn't natively track who said what. Need to ensure speaker labels propagate into entity states.
- **Scale** — LoCoMo is small (10 conversations). ConvoMem (75K QA pairs) will test efficiency.
- **Adversarial questions** — GSW's entity-speaker mapping should help, but needs explicit testing.

---

## 4. Benchmark Comparability Warning

Results across papers are **not directly comparable** because:
1. Different papers use different LoCoMo evaluation subsets (some include adversarial, some exclude it)
2. Different LLM backbones (GPT-4o vs GPT-4o-mini vs Qwen)
3. Mem0 introduced an LLM-as-Judge metric not used by earlier papers
4. Reproducibility disputes exist (e.g., [Zep's LoCoMo numbers](https://github.com/getzep/zep-papers/issues/5))

We should run all baselines ourselves under the same conditions for fair comparison.
