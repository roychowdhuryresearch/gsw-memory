# End-to-end validation report

Run date: 2026-07-27

Question:

> What is the cause of death of Sophie Gurney's father?

Gold answer: `multiple sclerosis`

## Models

- Decomposer: `yigitturali/GSW-QA-Decomposer-Qwen3-4B`
- Query encoder: `Qwen/Qwen3-Embedding-8B`
- Reranker: `Qwen/Qwen3-Reranker-8B`

The 4B decomposer produced the correct two-hop plan:

1. `Who was Sophie Gurney's father?`
2. `What was <ENTITY_Q1>'s cause of death?`

## Baseline finding

Using reranker probability alone failed at both `k=8, B=3` and the paper-scale
`k=15, B=5` setting. Dual retrieval ranked the correct first-hop QA first, but
the reranker assigned the paraphrase “Who are the parents of Sophie Jane
Gurney?” only `0.0903`. Less relevant, more lexically aligned records received
higher reranker scores and displaced the correct chain.

This is an instructive course ablation: a reranker is not guaranteed to improve
a strong structured retriever, especially for singular/plural relations and
entity aliases.

## Hybrid result

The validated run combined:

```text
0.5 * Qwen reranker probability + 0.5 * reciprocal retrieval rank
```

with `k=15`, `B=5`, and a 60-QA pre-rerank pool. It recovered:

```text
Who are the parents of Sophie Jane Gurney? -> Jacques Raverat
What did Jacques Pierre Paul Raverat die from? -> multiple sclerosis
```

Result:

- Exact Match: `1.0`
- Token F1: `1.0`
- Decomposition: `11.99 s`
- Retrieval plus RICR: `9.91 s`

The complete successful trace is in
`e2e_results/compositional_example_hybrid.json`. Failed baseline traces are
retained for the retrieval-fusion ablation.
