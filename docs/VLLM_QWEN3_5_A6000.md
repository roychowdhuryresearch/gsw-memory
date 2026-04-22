# vLLM Qwen 3.5 on 1x RTX A6000

Recommended local serve command for `Qwen/Qwen3.5-35B-A3B` on `1x RTX A6000 (48GB)`:

```bash
playground/sleep_time/serve_vllm_qwen3_5_35b_a3b_a6000.sh
```

Equivalent raw command:

```bash
vllm serve Qwen/Qwen3.5-35B-A3B \
  --port 6379 \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --language-model-only \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder
```

Notes:
- This is the single-GPU conservative serve shape for the exact base model.
- `--language-model-only` is intentional for the text-only sleep-time pipeline.
- Use `http://127.0.0.1:6379/v1` as the repo `base_url`.
- Set model, root model, and worker model to `Qwen/Qwen3.5-35B-A3B`.
- If the exact base model is unstable or OOMs at `32768`, retry with `--max-model-len 16384`.

Repo-side model settings for the first smoke run:

```bash
--base_url http://127.0.0.1:6379/v1 \
--model Qwen/Qwen3.5-35B-A3B \
--root_model Qwen/Qwen3.5-35B-A3B \
--worker_model Qwen/Qwen3.5-35B-A3B
```

Validation:

```bash
curl -s http://127.0.0.1:6379/v1/models
```

Expected model id:
- `Qwen/Qwen3.5-35B-A3B`

Basic API probe:

```bash
curl -s http://127.0.0.1:6379/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3.5-35B-A3B","messages":[{"role":"user","content":"say ok"}]}'
```
