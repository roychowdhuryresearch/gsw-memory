# vLLM GPT-OSS on 2x RTX A6000

Recommended local serve command for `openai/gpt-oss-120b` on `2x RTX A6000 (48GB)`:

```bash
playground/sleep_time/serve_vllm_gpt_oss_120b_a6000_tp2.sh
```

Equivalent raw command:

```bash
vllm serve openai/gpt-oss-120b \
  --tensor-parallel-size 2 \
  --port 6379 \
  --tool-call-parser openai \
  --enable-auto-tool-choice
```

Notes:
- This is the Ampere path. Do not use the Hopper or Blackwell GPT-OSS YAML recipes on RTX A6000.
- For this repo, point clients at `http://127.0.0.1:6379/v1`.
- Model, root model, and worker model should be `openai/gpt-oss-120b` when using this local vLLM server.
- For GPT-OSS structured stages, prefer the Responses API path over Chat Completions.

Validation:

```bash
curl -s http://127.0.0.1:6379/v1/models
```

Expected model id:
- `openai/gpt-oss-120b`

Responses API probe:

```bash
curl -s http://127.0.0.1:6379/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{"model":"openai/gpt-oss-120b","input":"say ok"}'
```
