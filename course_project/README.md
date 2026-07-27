# ECE 232E Panini Course Project

This directory contains the in-progress course project based on:

> Panini: Continual Learning in Token Space via Structured Memory

The project studies Generative Semantic Workspaces (GSWs) as typed networks and
then implements Reasoning Inference Chain Retrieval (RICR) over a small,
controlled 2WikiMultiHopQA corpus.

## Course-scale pipeline

```text
question
  -> supplied fine-tuned Qwen decomposer
  -> atomic sub-question dependency graph
  -> TF-IDF / BM25 / supplied dense-index retrieval
  -> supplied Qwen reranker
  -> RICR beam search
  -> evidence and answer evaluation
```

Students do **not** generate corpus embeddings. The instructors provide aligned
entity, QA-pair, and query/sub-question embedding artifacts. Student code loads
these artifacts and implements search, fusion, reranking, and chain traversal.

The supplied decomposition checkpoints are:

- [GSW-QA-Decomposer-Qwen3-4B](https://huggingface.co/yigitturali/GSW-QA-Decomposer-Qwen3-4B)
  for free-tier Colab.
- [GSW-QA-Decomposer-Qwen3-8B](https://huggingface.co/yigitturali/GSW-QA-Decomposer-Qwen3-8B)
  for the higher-capacity comparison.

## Built dataset

The default build selects 100 questions from 2WikiMultiHopQA:

| Question type | Count |
| --- | ---: |
| compositional | 25 |
| inference | 25 |
| comparison | 25 |
| bridge comparison | 25 |

Every selected question retains its ten context documents, including
distractors. Documents and GSWs are deduplicated across questions. The default
split exposes 80 development questions and reserves 20 questions for held-out
evaluation.

The materialized release contains 80 development questions, 20 held-out
questions, 765 deduplicated documents/GSWs, 6,805 entity records, and 8,887
QA-pair records.

## Current layout

```text
course_project/
├── PROJECT_SPEC.md
├── DATA_CARD.md
├── README.md
├── Panini_Course_Project.ipynb
├── requirements-colab.txt
├── quickstart.py
├── release/
│   └── panini_2wiki_100/
├── scripts/
│   ├── build_dataset.py
│   ├── export_embeddings.py
│   ├── build_indices.py
│   └── finalize_release.py
├── src/
│   └── panini_course/
├── student_overrides/
│   └── panini_course/
└── instructor/
    └── panini_2wiki_100_gold/
```

The release is self-contained and does not import from the main Panini
repository. Instructor-only labels are stored outside the release.

## Build the dataset

From the repository root:

```bash
python course_project/scripts/build_dataset.py \
  --questions playground_data/2wikimultihopqa.json \
  --corpus playground_data/2wikimultihopqa_corpus.json \
  --gsw-root /path/to/2wiki/networks \
  --output course_project/release/panini_2wiki_100
```

To attach instructor-reviewed decompositions from an evaluation result:

```bash
python course_project/scripts/build_dataset.py \
  --questions playground_data/2wikimultihopqa.json \
  --corpus playground_data/2wikimultihopqa_corpus.json \
  --gsw-root /path/to/2wiki/networks \
  --decompositions-log logs/multihop_qa_chains_batched_results.json \
  --output course_project/release/panini_2wiki_100
```

The builder writes `embedding_contract.json`, which defines the IDs, ordering,
shapes, and filenames consumed by the separate instructor embedding export.

When a decomposition log is supplied, the complete set is kept under the
instructor-only directory. The public package exposes five reviewed
decompositions per question type for model evaluation.

## Rebuild the remaining artifacts

```bash
python course_project/scripts/export_embeddings.py \
  --package course_project/release/panini_2wiki_100 \
  --device cuda:0

python course_project/scripts/build_indices.py \
  --package course_project/release/panini_2wiki_100

python course_project/scripts/finalize_release.py \
  --package course_project/release/panini_2wiki_100 \
  --decomposition-prompt Panini/decomposition_prompt.txt
```

The committed release already contains those outputs. Verify it with:

```bash
python course_project/release/panini_2wiki_100/verify_release.py \
  course_project/release/panini_2wiki_100
```
