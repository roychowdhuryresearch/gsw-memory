---
title: "PANINI: Continual Learning in Token Space via Structured Memory"
author: "Shreyas Rajesh, Pavan Holur, Mehmet Yigit Turali, Chenda Duan, Vwani Roychowdhury"
date: "ICML 2026"
---

# PANINI: Continual Learning in Token Space via Structured Memory

Five-minute ICML paper talk.

Core message: structured write-time memory can make read-time reasoning cheaper and more reliable.

---

## Where This Fits

- LLMs need to reason over new documents and evolving knowledge.
- Parametric continual learning updates weights, but is expensive and risks forgetting.
- Standard RAG keeps the model fixed, but repeatedly feeds raw chunks to the LLM.
- PANINI keeps the model fixed and retrieves compact reasoning chains from structured memory.

---

## PANINI Writes Structured Memory

Use paper Figure 1.

Documents are encoded as Generative Semantic Workspaces: entity- and event-aware networks of question-answer pairs.

---

## PANINI Reads Memory with RICR

Use paper Figure 2.

1. Decompose the query.
2. Retrieve chains hop-by-hop from GSW space.
3. Answer from compact top-ranked evidence chains.

---

## Main Result

Use paper Table 2, simplified.

PANINI reaches 56.06 average F1 across six QA benchmarks, above HippoRAG 2 at 53.3 and dense retrieval with reranking at 50.5.

---

## Efficiency

Use paper Table 3, simplified.

PANINI uses 319.79 average answer-context tokens, compared with 705.27 for standard retrieval and 2457.7 for Search-R1.

---

## Reliability

Use paper Table 4, simplified.

PANINI improves answerable accuracy while keeping strong refusal accuracy on curated unanswerable questions.

---

## Takeaway

- Chunk RAG is a weak form of memory.
- PANINI writes semantic structure at encoding time.
- RICR reads that memory as inference chains.
- This improves accuracy, efficiency, and abstention behavior.
