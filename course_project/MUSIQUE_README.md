# ECE 232E — Panini MuSiQue Companion Package

[Read the complete 100-point project handout](PROJECT_HANDOUT.pdf).

This folder is the self-contained 100-question MuSiQue companion to the
[2Wiki starter repository](https://github.com/YigitTurali/panini-course-project).
It contains the selected questions, source documents, normalized GSW JSON
networks, flattened entity/QA metadata, supplied Qwen3-Embedding-8B vectors,
dense/TF-IDF/BM25 indices, model configuration, and the `panini_course`
Python package.

## Fixed subset

- 50 two-hop questions: 40 development and 10 held out.
- 30 three-hop questions: 24 development and 6 held out.
- 20 four-hop questions: 16 development and 4 held out.
- 841 source documents and corresponding GSWs.
- 8,260 flattened entities and 9,991 flattened QA records.

Every supporting paragraph is included. Each question also receives
deterministically sampled distractors for a total of ten context documents.
All annotated atomic answers were checked for named-node coverage in the
packaged GSWs.

## Setup

Set the notebook's `PACKAGE_ROOT` to this directory and install:

```python
!pip install -q -r requirements-colab.txt
```

Verify every artifact:

```bash
python verify_release.py .
python quickstart.py .
```

## Supplied models

The default free-tier pipeline uses:

1. [GSW-QA-Decomposer-Qwen3-4B](https://huggingface.co/yigitturali/GSW-QA-Decomposer-Qwen3-4B)
2. supplied Qwen3-Embedding-8B corpus/fixed-query embeddings;
3. [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B); and
4. [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) for evidence-grounded
   final answers.

Load models sequentially in 4-bit mode. The optional
[8B decomposer](https://huggingface.co/yigitturali/GSW-QA-Decomposer-Qwen3-8B)
is not required.

## Student boundary

Corpus GSW and embedding generation is not a student task. Students analyze
the network, compare retrieval methods, run decomposition, and implement
unique-answer beam pruning plus the linear RICR loop. Held-out answers and
supporting evidence are stored separately from this package.

MuSiQue is distributed under
[CC BY 4.0](https://github.com/stonybrooknlp/musique).
