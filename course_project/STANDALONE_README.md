# ECE 232E — Panini Structured-Memory Project

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
├── Panini_Course_Project.ipynb
├── DATA_CARD.md
├── PROJECT_SPEC.md
├── quickstart.py
├── requirements-colab.txt
├── manifest.json
└── release_manifest.json
```

## Colab setup

```python
!pip install -q -r requirements-colab.txt
```

```python
from panini_course import CoursePackage

package = CoursePackage(".")
print(package.manifest["counts"])
```

Verify the supplied retrieval artifacts without loading any Qwen model:

```bash
python quickstart.py .
```

`Panini_Course_Project.ipynb` is the Colab-ready student workspace. Set its
`PACKAGE_ROOT` cell to the copied release directory and work through Parts
1–5.

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

Do not keep all three models resident simultaneously.

## Student implementation boundary

The package supplies artifact loaders, retrieval backends, graph parsing
helpers, model wrappers, metrics, and RICR data structures. Students implement
the two TODOs in `panini_course/ricr.py`: unique-answer beam pruning and the
linear RICR search loop. The instructor solution and held-out labels are not
included.

## Integrity

`release_manifest.json` contains SHA-256 hashes for every release artifact.
Use `python verify_release.py .` to verify a downloaded copy.
