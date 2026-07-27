# Project: Structured Memory Networks and Reasoning-Chain Retrieval

## Learning objectives

By the end of this project, students should be able to:

1. Construct and characterize a heterogeneous network from GSW JSON records.
2. Compare lexical, sparse, dense, and hybrid information retrieval methods.
3. Use a supplied neural decomposer and reranker in a reproducible pipeline.
4. Implement beam-search reasoning over entity-linked QA chains.
5. Evaluate retrieval quality, end-to-end accuracy, efficiency, and scaling.

## Part 1 — GSW network construction and analysis (25%)

Construct the native entity/verb-phrase network and a reconciled corpus-level
entity projection. Report node and edge counts, degree distributions,
components, GCC statistics, centrality, PageRank, clustering, and
assortativity. Visualize at least one complete multi-hop reasoning path.

## Part 2 — Question decomposition (10%)

Run the supplied fine-tuned Qwen decomposer with deterministic decoding. Parse
`<ENTITY_Qn>` references into a dependency graph, identify parallel retrieval
chains, validate the generated plan, and compare a subset against
instructor-reviewed decompositions.

## Part 3 — Retrieval methods (25%)

Implement and compare:

1. TF-IDF over QA-pair text.
2. BM25 over QA-pair text.
3. BM25 over entity names, roles, and states, followed by attached-QA
   expansion.
4. Dense search over supplied QA embeddings.
5. Hybrid retrieval using normalized score fusion and Reciprocal Rank Fusion.
6. Paper-style dual retrieval followed by the supplied Qwen reranker.

Students load the provided embedding tables and indices; embedding generation
is outside the project scope.

## Part 4 — RICR (25%)

Implement Reasoning Inference Chain Retrieval:

1. Retrieve candidates for the first atomic sub-question.
2. Retain the top `B` chains with unique current answers.
3. Substitute each answer into the next sub-question.
4. Expand every beam with the top `k` reranked candidates.
5. Score each chain with the geometric mean of hop scores.
6. Prune to the top `B` chains with unique current answers.
7. Combine and deduplicate evidence from parallel sub-question sequences.

Required ablations include beam width, unique-answer pruning, last-hop versus
geometric-mean scoring, and retrieval-backend choice.

## Part 5 — Evaluation and analysis (15%)

Report per-hop Recall@k, MRR, supporting-document recall, supporting-QA recall,
complete-chain recovery, answer Exact Match/F1, latency, and evidence size.
Compare gold and predicted decompositions and evaluate performance as
distractor documents are progressively added.

## Execution constraints

- The required workflow must run in Google Colab.
- Models are loaded sequentially rather than simultaneously.
- Every expensive stage writes a restartable artifact.
- Inference uses deterministic settings unless an experiment explicitly
  studies decoding.
- No paid API is required.

## Submission

Students submit:

1. Completed Colab notebooks or equivalent Python modules.
2. A report organized by the numbered project questions.
3. `ricr_results.jsonl` in the instructor-provided schema.
4. A short reproducibility manifest containing package and model versions.
