# ECE 232E — Panini Structured-Memory Project

[Open the student notebook in Google Colab](https://colab.research.google.com/github/YigitTurali/panini-course-project/blob/main/Panini_Course_Project.ipynb).

[Read the complete 150-point project handout](PROJECT_HANDOUT.pdf). The
handout uses this 100-question 2Wiki package and the companion 100-question
MuSiQue package at `packages/panini_musique_100`.
Its editable LaTeX source is included as `PROJECT_HANDOUT.tex`.

This folder is self-contained. It includes the selected 2WikiMultiHopQA
questions, source documents, GSW JSON networks, flattened entity/QA metadata,
Qwen3-Embedding-8B vectors, dense/TF-IDF/BM25 indices, model configuration,
and the `panini_course` Python package.

No access to the original GSW-Memory repository is required.

## Package contents

```text
.
├── documents.jsonl
├── questions/
├── gsws/
├── metadata/
├── embeddings/
├── indices/
├── models/
├── panini_course/
├── packages/
│   └── panini_musique_100/
├── Panini_Course_Project.ipynb
├── DATA_CARD.md
├── PROJECT_HANDOUT.tex
├── PROJECT_SPEC.md
├── quickstart.py
├── requirements-colab.txt
├── tests/
├── TESTING.md
└── manifest.json
```

## Colab setup

The notebook clones this repository automatically. For a manual Colab setup:

```python
!git clone --depth 1 https://github.com/YigitTurali/panini-course-project.git
%cd panini-course-project
!pip install -q -r requirements-colab.txt
```

```python
from panini_course import CoursePackage

package = CoursePackage(".")
print(package.manifest["counts"])

musique_package = CoursePackage("packages/panini_musique_100")
print(musique_package.manifest["counts"])
```

Verify the supplied retrieval artifacts without loading any Qwen model:

```bash
python quickstart.py .
```

`Panini_Course_Project.ipynb` is the 48-cell Colab-ready student workspace. It
mirrors the complete Questions 1–12 workflow but leaves the graded algorithms,
tables, conclusions, and twelve written responses as TODOs. Its run controls
separate decomposition, reranking/RICR, and answering so only one Qwen model is
resident at a time. JSONL checkpoints, the student's `ricr.py`, and custom
tests are kept under the selected Drive work directory and survive a runtime
disconnect. Start with `QUESTION_LIMIT = 2`; use `None` only for final runs.

## Supplied embeddings

Corpus embedding generation is complete and is not a student task:

- `embeddings/entity_embeddings.npy`
- `embeddings/qa_embeddings.npy`
- `embeddings/query_embeddings.npy`
- matching stable-ID JSON files
- float16 FAISS indices under `indices/`

The query table covers original questions, decomposition templates, and the
answer-instantiated queries reached by the required default and ablation
runs. Students do not generate embeddings. A missing query in a required run
means the plan or RICR trace diverged from the specified deterministic path.

## Supplied model checkpoints

See `models/model_config.json` for immutable Hugging Face model IDs and the
decomposition prompt. Models should be loaded sequentially in Colab:

1. Fine-tuned Qwen decomposer:
   - [Qwen3-4B decomposer](https://huggingface.co/yigitturali/GSW-QA-Decomposer-Qwen3-4B)
     is the free-tier Colab default.
   - [Qwen3-8B decomposer](https://huggingface.co/yigitturali/GSW-QA-Decomposer-Qwen3-8B)
     is the higher-capacity option.
2. [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B) in 4-bit
   mode, loaded by itself on a 15 GiB T4, or the
   [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
   4B fallback after an actual OOM. Use batch size 1 and 256-token inputs on T4.
3. [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) for evidence-grounded
   final answers.

Do not keep multiple neural models resident simultaneously.

## Student implementation boundary

The package supplies artifact loaders, retrieval backends, graph parsing
helpers, model wrappers, metrics, and RICR data structures. Students implement
the TODOs in `panini_course/ricr.py`: retrieval-DAG identification and the full
PANINI RICR executor, including multi-parent beam combination, distinct
intermediate/final-hop pruning, singleton fallback, and all-final-beam
evidence. The instructor solution and held-out labels are not included.

Cross-document entity reconciliation is a separate network-analysis exercise.
Students compare an unreconciled projection, the supplied exact-surface
baseline, and their own conservative reconciliation rule. The resulting
global graphs must not be used by the retrieval or RICR pipeline: Panini keeps
document-local entity nodes and forms cross-document chains dynamically.

## Starter tests

Run the supplied tests before editing the scaffold:

```bash
pytest -q tests
```

The retrieval, metrics, and package tests should pass immediately. RICR
implementation tests are skipped while the scaffold functions still raise
`NotImplementedError`; they turn on automatically as those functions are
implemented. See [TESTING.md](TESTING.md) for the tested behavior and the
additional tests students must write.
