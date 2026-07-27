# Panini 2Wiki Course Dataset Card

This teaching subset is derived from
[2WikiMultiHopQA](https://github.com/Alab-NII/2wikimultihop), distributed under
the Apache License 2.0. Please cite the original dataset paper:

> Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa.
> “Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of
> Reasoning Steps.” COLING 2020.

## Purpose

This is a compact, reproducible teaching subset of 2WikiMultiHopQA paired with
Panini Generative Semantic Workspace (GSW) networks. It supports network
analysis, retrieval comparisons, question decomposition, and RICR experiments
on free-tier Google Colab.

## Composition

The release contains 100 questions selected with seed 232:

| 2Wiki question type | Development | Held out | Total |
| --- | ---: | ---: | ---: |
| bridge comparison | 20 | 5 | 25 |
| comparison | 20 | 5 | 25 |
| compositional | 20 | 5 | 25 |
| inference | 20 | 5 | 25 |
| **Total** | **80** | **20** | **100** |

All ten context documents for each question are retained, including
distractors. Deduplication leaves 765 documents and 765 document-level GSWs.
The flattened search corpus has 6,805 entity records and 8,887 QA-pair
records.

## Labels and leakage controls

- Development questions include answers, supporting facts, and evidence.
- Held-out questions expose only ID, type, question, and context document IDs.
- Five reviewed decomposition examples per question type are public.
- Held-out answers and the complete reviewed decomposition set live in a
  separate instructor-only key and are not part of the student release.

The GSW corpus is itself the retrieval knowledge base, so answer strings may
occur in entity and QA records. This is intentional. Systems must retrieve and
chain the correct evidence rather than treat corpus occurrence as label
leakage.

## Embeddings and indexes

Entity, QA, and fixed-query vectors were generated with
`Qwen/Qwen3-Embedding-8B`, are L2-normalized, stored as float16, and have 4,096
dimensions. The release also provides aligned FAISS inner-product, TF-IDF, and
BM25 indexes. Fixed queries include original questions and reviewed
decomposition templates.

Answer substitution can produce a query string not present in the fixed table.
For that case only, students may use the supplied query-encoder wrapper and
cache the result. Corpus embeddings never need to be regenerated.

Question decomposition uses the instructor checkpoints
[Qwen3-4B](https://huggingface.co/yigitturali/GSW-QA-Decomposer-Qwen3-4B)
or [Qwen3-8B](https://huggingface.co/yigitturali/GSW-QA-Decomposer-Qwen3-8B).
The 4B variant is the default for free-tier Colab.

## Intended and out-of-scope uses

This package is for coursework and small controlled experiments. It is not a
representative benchmark for broad claims about open-domain QA, not a
production knowledge base, and not suitable for evaluating memorization or
factual freshness.

## Reproducibility

`manifest.json` records selection parameters and source checksums.
`release_manifest.json` records a SHA-256 digest for each distributed file.
Run `python verify_release.py .` after transfer.
