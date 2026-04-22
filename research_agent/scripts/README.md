# Serve scripts

Each `serve_*.sh` launches a vLLM OpenAI-compatible endpoint for one
model used in the substitution grid. All scripts source
`serve_vllm_common.sh`, which:

- Exports `HF_HOME=/mnt/SSD3/yigit/hf_cache` + related cache paths.
- Prefers a local-downloaded copy under `/mnt/SSD3/yigit/models/<name>` if
  present; otherwise vLLM auto-downloads from HuggingFace on first run.
- Pins `CUDA_VISIBLE_DEVICES` and TP size per model.
- Announces resolved paths + ports before exec'ing vLLM.

## Ports + GPU plan

| Script | Model | HF id | Port | TP | Default GPU | VRAM |
|---|---|---|---|---|---|---|
| `serve_gpt_oss_20b.sh` | gpt-oss-20B MoE (reasoning) | `openai/gpt-oss-20b` | 8001 | 1 | 3 | ~40G |
| `serve_qwen3_5_9b.sh` | **Qwen3.5-9B (reasoning)** — thinking enabled via chat-template flag | `Qwen/Qwen3.5-9B` | 8002 | 1 | 3 | ~19G |
| `serve_qwen3_5_4b.sh` | **Qwen3.5-4B (reasoning)** — thinking enabled via chat-template flag | `Qwen/Qwen3.5-4B` | 8003 | 1 | 3 | ~9G |
| `serve_qwen25_7b.sh` | Qwen2.5-7B-Instruct (non-reasoning — deprecated for grid, keep for ablation) | `Qwen/Qwen2.5-7B-Instruct` | 8002 | 1 | 3 | ~15G |
| `serve_qwen25_3b.sh` | Qwen2.5-3B-Instruct (non-reasoning — deprecated for grid) | `Qwen/Qwen2.5-3B-Instruct` | 8003 | 1 | 3 | ~6G |
| `serve_qwq_32b.sh` | QwQ-32B | `Qwen/QwQ-32B` | 8004 | 2 | 2,3 | ~64G |
| `serve_asearcher_7b.sh` | ASearcher-Web-7B | `inclusionAI/ASearcher-Web-7B` | 8005 | 1 | 3 | ~15G |
| `serve_asearcher_14b.sh` | ASearcher-Web-14B | `inclusionAI/ASearcher-Web-14B` | 8006 | 1 | 3 | ~28G |
| `serve_context1.sh` | Context-1 | `chromadb/context-1` | 8007 | 1 | 3 | ~40G |
| `serve_tongyi_30b_a3b.sh` | Tongyi DeepResearch | `Alibaba-NLP/Tongyi-DeepResearch-30B-A3B` | 8008 | 2 | 2,3 | ~60G |
| `serve_smtl_30b.sh` | SMTL-30B (OPPO AFM) | `PersonalAILab/SMTL-30B` | 8009 | 2 | 2,3 | ~60G |

Weights already on-disk (as of 2026-04-17): gpt-oss-20b, Qwen2.5-3B, Qwen2.5-7B, bge-large-en-v1.5. All others auto-download to `/mnt/SSD3/yigit/hf_cache/` on first serve.

## Typical run

```bash
# Override the GPU if needed:
GPUS=3 ./scripts/serve_qwen25_7b.sh

# Or with different TP:
GPUS=2,3 TP=2 ./scripts/serve_asearcher_14b.sh

# Extra vLLM flags can be passed as trailing args:
./scripts/serve_gpt_oss_20b.sh --disable-log-stats
```

## Typical substitution-grid invocation

After the endpoint is up (say Qwen3.5-9B on 8002):

```bash
python playground/run_substitution.py \
    --system vanilla_rag_react \
    --model Qwen/Qwen3.5-9B \
    --base-url http://127.0.0.1:8002/v1 \
    --api-key dummy \
    --subset configs/pilot_subset.json
```

For GPT-5 (no local serve, uses OpenAI directly):

```bash
export OPENAI_API_KEY=sk-...
python playground/run_substitution.py \
    --system vanilla_rag_react --model gpt-5 \
    --subset configs/pilot_subset.json
```
