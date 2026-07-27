# Panini MuSiQue-100 Course Dataset Card

## Source and license

The source is
[MuSiQue: Multi-hop Questions via Single-hop Question Composition](https://github.com/stonybrooknlp/musique),
distributed under CC BY 4.0. This teaching package retains source question IDs
and corpus indices. The Panini course code is distributed under Apache 2.0.

## Selection

The deterministic selection seed is 232. Questions are sampled only when:

- the question is answerable;
- every supporting paragraph has a normalized GSW;
- all supporting paragraphs fit in the packaged context; and
- every annotated atomic answer appears as a named GSW node in the selected
  context.

Each question keeps all supporting paragraphs and deterministic distractors,
capped at ten context documents.

| Hop count | Development | Held out | Total |
|---|---:|---:|---:|
| 2 | 40 | 10 | 50 |
| 3 | 24 | 6 | 30 |
| 4 | 16 | 4 | 20 |
| **Total** | **80** | **20** | **100** |

## Package statistics

- 841 documents and normalized GSW files.
- 8,260 entity-index records.
- 9,991 QA-index records.
- 280 supplied fixed-query embeddings.

The development split includes answers, aliases, supporting document IDs, and
atomic evidence steps. Held-out records include only the question, hop count,
type, and context document IDs.

## Embeddings and indices

Entity, QA, and fixed-query embeddings use `Qwen/Qwen3-Embedding-8B`,
L2 normalization, 4,096 dimensions, and float16 storage. The package includes
matching scalar-quantized float16 FAISS inner-product indices, TF-IDF indices,
and BM25 indices. Students load these artifacts rather than regenerating them.

## Intended use and limitations

This is a small instructional transfer subset, not a replacement for the full
MuSiQue benchmark. It is designed for cross-dataset comparison after
hyperparameters have been selected on the course 2Wiki subset. Results should
not be reported as full-benchmark MuSiQue performance.
