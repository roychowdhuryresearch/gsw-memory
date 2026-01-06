# GSW Pipeline Usage

## Public Data

All related files are available at `/mnt/SSD3/chenda/gsw/public_data/`:

```
/mnt/SSD3/chenda/gsw/public_data/
├── 2wikimultihopqa_corpus.json                        # Source corpus
├── 2wiki_qwen8b_t0.1_p0.9_k20/                        # Flattened GSW output
│   ├── gsw_results_combined.json
│   ├── networks/                                       # GSW network files
│   └── networks_raw/                                   # Raw network files
├── full_2wiki_corpus_20260102_145007/                 # Original batched GSW output
└── multihop_qa_chains_batched_results_20260105_024502.json  # Evaluation results
```

## 1. Parameter Search (Optional)

Use `gsw_param_search.py` to find optimal GSW generation parameters:

```bash
uv run python playground/gsw_creation_local/gsw_param_search.py
```

## 2. Generate GSW Representations

Use `process_full_2wiki_corpus.py` to generate GSW representations for the corpus:

```bash
uv run python playground/process_full_2wiki_corpus.py
```

## 3. Flatten Batch Output

Use `flatten_batch_output.py` to consolidate batched GSW output into a single folder:

```bash
uv run python playground/flatten_batch_output.py \
    /mnt/SSD3/chenda/gsw/gsw-memory/logs/full_2wiki_corpus_20260102_145007/gsw_output/corpus_20260102_145007 \
    /mnt/SSD3/chenda/gsw/gsw-memory/data/networks/2wiki_qwen8b_t0.1_p0.9_k20
```

## 4. Build Entity Search Cache

Use `multi_hop_qa_chains.py` to build the entity search cache (requires 1 GPU for embeddings):

```bash
HF_HOME=/mnt/SSD3/chenda/gsw/cache \
uv run python playground/multi_hop_qa_chains.py \
    --gsw-path /mnt/SSD3/chenda/gsw/gsw-memory/data/networks/2wiki_qwen8b_t0.1_p0.9_k20/networks \
    --cache-dir /mnt/SSD3/chenda/gsw/gsw-memory/data/cache/.gsw_cache_2wiki_qwen8b_t0.1_p0.9_k20 \
    --cuda-devices 3 >> logs/build_chain_2wiki.log 2>&1 &
```

## 5. Start vLLM Servers

Start the question decomposition server (GPU 1):

```bash
HF_HOME=/mnt/SSD3/chenda/hf_cache \
CUDA_VISIBLE_DEVICES=1 \
uv run vllm serve yigitturali/qwen3-8b-qa-decomp-gsw-rank-256-gpt5-golden-large \
    --host 127.0.0.1 --port 8989 --api-key "token-abc123" \
    --dtype auto --trust-remote-code >> logs/question_decomp_server.log 2>&1 &
```

Start the question answering server (GPU 2):

```bash
HF_HOME=/mnt/SSD3/chenda/gsw/cache \
CUDA_VISIBLE_DEVICES=2 \
uv run vllm serve Qwen/Qwen3-8B \
    --chat-template /mnt/SSD3/chenda/gsw/cache/qwen3_nonthinking.jinja \
    --host 127.0.0.1 --port 6379 \
    --dtype auto --trust-remote-code >> logs/question_answerer_server.log 2>&1 &
```

## 6. Run Evaluation

Run the multi-hop QA evaluation (requires embedding GPU + 2 vLLM servers):

```bash
HF_HOME=/mnt/SSD3/chenda/gsw/cache \
uv run python playground/evaluate_multi_hop_qa_chains_batched.py \
    --gsw-path /mnt/SSD3/chenda/gsw/gsw-memory/data/networks/2wiki_qwen8b_t0.1_p0.9_k20/networks \
    --cache-dir /mnt/SSD3/chenda/gsw/gsw-memory/data/cache/.gsw_cache_2wiki_qwen8b_t0.1_p0.9_k20 \
    --questions-file /mnt/SSD3/chenda/gsw/gsw-memory/playground_data/2wikimultihopqa.json \
    --output-dir /mnt/SSD3/chenda/gsw/gsw-memory/data/output/2wiki_qwen8b_t0.1_p0.9_k20/ \
    --cuda-devices 0 >> logs/2wiki_eval.log 2>&1 &
```

## GPU Requirements

- GPU 0: Embedding model (Qwen3-Embedding-8B)
- GPU 1: Question decomposition vLLM server
- GPU 2: Question answering vLLM server
