# ECE 232E — Panini Structured-Memory Project

[Open the student notebook in Google Colab](https://colab.research.google.com/github/YigitTurali/panini-course-project/blob/main/Panini_Course_Project.ipynb).

[Read the complete 100-point project handout](PROJECT_HANDOUT.pdf). The
handout uses this 100-question 2Wiki package and the companion 100-question
MuSiQue package at `packages/panini_musique_100`.

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

`Panini_Course_Project.ipynb` is the Colab-ready student workspace. Its setup
cell uses `/content/panini-course-project`; change `PACKAGE_ROOT` only when
using a Drive copy or another directory.

## Supplied embeddings

Corpus embedding generation is complete and is not a student task:

- `embeddings/entity_embeddings.npy`
- `embeddings/qa_embeddings.npy`
- `embeddings/query_embeddings.npy`
- matching stable-ID JSON files
- float16 FAISS indices under `indices/`

The fixed query table covers all original questions and decomposition
templates. RICR may create new answer-instantiated sub-questions at runtime;
use `QwenQueryEncoder` only for those previously unseen query strings and
cache the resulting vectors.

## Supplied model checkpoints

See `models/model_config.json` for immutable Hugging Face model IDs and the
decomposition prompt. Models should be loaded sequentially in Colab:

1. Fine-tuned Qwen decomposer:
   - [Qwen3-4B decomposer](https://huggingface.co/yigitturali/GSW-QA-Decomposer-Qwen3-4B)
     is the free-tier Colab default.
   - [Qwen3-8B decomposer](https://huggingface.co/yigitturali/GSW-QA-Decomposer-Qwen3-8B)
     is the higher-capacity option.
2. 4-bit Qwen query encoder for uncached instantiated queries.
3. [Qwen3-Reranker-8B](https://huggingface.co/Qwen/Qwen3-Reranker-8B).
4. [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) for evidence-grounded
   final answers.

Do not keep multiple neural models resident simultaneously.

## Student implementation boundary

The package supplies artifact loaders, retrieval backends, graph parsing
helpers, model wrappers, metrics, and RICR data structures. Students implement
the two TODOs in `panini_course/ricr.py`: unique-answer beam pruning and the
linear RICR search loop. The instructor solution and held-out labels are not
included.

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
implementation tests are skipped while the two scaffold functions still raise
`NotImplementedError`; they turn on automatically as those functions are
implemented. See [TESTING.md](TESTING.md) for the tested behavior and the
additional tests students must write.
