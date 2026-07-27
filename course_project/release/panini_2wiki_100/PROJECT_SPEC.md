# Course Project: Structured Memory Networks and RICR

The canonical specification is [PROJECT_HANDOUT.pdf](PROJECT_HANDOUT.pdf).
The handout contains the precise algorithms, metrics, output schema,
deliverables, and point allocation.

## Required datasets

Students run the same frozen pipeline on two independently packaged subsets:

- **2WikiMultiHopQA:** 100 questions, with 25 each from bridge-comparison,
  comparison, compositional, and inference.
- **MuSiQue:** 100 questions, stratified as 50 two-hop, 30 three-hop, and
  20 four-hop questions.

Each dataset has 80 labeled development questions and 20 held-out questions.
Algorithm choices and hyperparameters are selected on 2Wiki development and
then frozen before the MuSiQue transfer evaluation.

The course supplies documents, GSWs, metadata, corpus/fixed-query embeddings,
FAISS indices, TF-IDF/BM25 indices, and model configuration for each package.
Students do not generate GSWs or corpus embeddings.

## Grading

| Question | Component | Points |
|---|---|---:|
| Q1 | Understand the two data packages | 4 |
| Q2 | Native GSW network and entity projection | 8 |
| Q3 | Structured-memory network analysis | 8 |
| Q4 | Question decomposition and dependency graphs | 10 |
| Q5 | Sparse retrieval baselines | 8 |
| Q6 | Dense, hybrid, and paper-style dual retrieval | 12 |
| Q7 | Reranking analysis | 5 |
| Q8 | RICR implementation | 15 |
| Q9 | RICR ablations | 10 |
| Q10 | 2Wiki end-to-end evaluation | 7 |
| Q11 | MuSiQue transfer and scaling | 8 |
| Q12 | Reproducibility and submission | 5 |
|  | **Total** | **100** |

## Required models

- Default decomposer:
  [`yigitturali/GSW-QA-Decomposer-Qwen3-4B`](https://huggingface.co/yigitturali/GSW-QA-Decomposer-Qwen3-4B)
- Optional decomposer:
  [`yigitturali/GSW-QA-Decomposer-Qwen3-8B`](https://huggingface.co/yigitturali/GSW-QA-Decomposer-Qwen3-8B)
- Query encoder for uncached instantiated questions:
  `Qwen/Qwen3-Embedding-8B`
- Reranker: `Qwen/Qwen3-Reranker-8B`
- Evidence-grounded answer model: `Qwen/Qwen3-4B`

The required workflow must run in free-tier Google Colab. Models are loaded
sequentially, expensive stages produce restartable JSONL files, and no paid
API is required.
