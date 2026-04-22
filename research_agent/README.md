# research_agent

Pilot study: **substitute small open-source models into existing agentic-search frameworks** (Vanilla RAG+ReAct / Search-o1 / Search-R1 / ASearcher / Context-1 / Tongyi DeepResearch / SMTL / EigentSearch Q+) and observe what happens on a small FRAMES subset. Inference-only, no training. Plus our own focused-GSW system as the all-small contribution claim.

Lives inside `gsw-memory/` as a subdirectory — **not** a separate git repo. Shares the parent `gsw-memory` git history. The `.venv/` and `third_party/` are gitignored locally.

## Layout

```
research_agent/
├── src/research_agent/
│   ├── adapters/       # one per framework (10 total), shared Adapter ABC + registry
│   ├── eval/           # harness: (system, model, subset) -> CellResult (scores + traces)
│   ├── retrieval/      # BM25 + FRAMES corpus chunker
│   ├── models/         # Pydantic schemas (Trajectory, FailureMode, LLMClient)
│   └── utils/
├── playground/         # CLI drivers (select_pilot_subset, run_substitution, aggregate_grid)
├── scripts/            # vLLM serve_*.sh + shared bootstrap
├── configs/            # pilot_subset.json (30-Q stratified FRAMES)
├── docs/               # EXPERIMENTS.md (full per-cell spec)
├── third_party/        # cloned competitor repos (gitignored, read-only)
├── tests/              # 12 passing unit tests
└── logs/               # per-run trace dumps (gitignored)
```

## Setup

```bash
cd research_agent
uv venv && source .venv/bin/activate
uv pip install pydantic typer pytest python-dotenv openai huggingface-hub pandas tabulate
cp .env.example .env  # edit: HF_HOME, OPENAI_API_KEY, vLLM URLs
```

Model weights live on `/mnt/SSD3/yigit/models/`; HF cache on `/mnt/SSD3/yigit/hf_cache/`. Set `HF_HOME` in `.env` before any `hf download` or `vllm serve`.

## Quick run

```bash
# Build the 30-Q pilot subset (already done — configs/pilot_subset.json is checked in):
PYTHONPATH=src python playground/select_pilot_subset.py

# Run one cell (example: Vanilla RAG+ReAct + GPT-5, needs OPENAI_API_KEY):
PYTHONPATH=src python playground/run_substitution.py \
    --system vanilla_rag_react --model gpt-5 \
    --subset configs/pilot_subset.json --limit 5

# Aggregate after multiple cells ran:
PYTHONPATH=src python playground/aggregate_grid.py --out logs/grid_summary.md
```

See `docs/EXPERIMENTS.md` for the full per-cell spec, plus `/home/yigit/.claude/plans/eager-weaving-canyon.md` section `Run N+10` for the ongoing iteration log.

## Imports from the parent gsw-memory package

The adapter code is self-contained, but where it helps we import from the parent repo (`gsw-memory/src/gsw_memory/...`) via the `GSW_MEMORY_ROOT` env var in `.env` (defaults to `/home/yigit/codebase/gsw-memory`). Specifically, `eval/frames_dataset.py` reads the pre-cached FRAMES article cache at `$GSW_MEMORY_ROOT/data/sleep_time/frames/`.
