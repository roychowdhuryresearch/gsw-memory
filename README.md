# GSW Memory

Official repository for the **Generative Semantic Workspaces (GSW)** framework. This codebase supports two papers:

1. **[Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces](https://arxiv.org/abs/2511.07587)** (AAAI-26 Oral)
2. **[Panini: Continual Learning in Token Space via Structured Memory](https://arxiv.org/abs/2602.15156)** (Under review, ICML 2026)

A Python package for building structured memory systems using Generative Semantic Workspaces (GSW). The first paper introduces the GSW framework for episodic memory and entity-centric reasoning over long narratives. The second paper extends GSW to non-parametric continual learning, introducing Panini --- a chain-following retrieval system (RICR) for multi-hop question answering over structured QA-pair memories built at write time.

## Features

- **Document Processing**: Convert text documents into structured semantic workspaces
- **Entity Reconciliation**: Merge and reconcile entities across documents using multiple strategies
- **Question-Answering**: Answer questions using semantic memory with entity extraction and matching
- **Evaluation System**: Built-in evaluation tools for benchmarking Q&A performance


## Package Structure

```
gsw_memory/
├── memory/                    # Core GSW processing
│   ├── processors.py         # Document → GSW conversion
│   ├── reconciliation.py     # Entity reconciliation across documents
│   ├── aggregators.py        # Entity summary generation
│   └── models.py            # Data structures (EntityNode, GSWStructure, etc.)
├── qa/                       # Question-answering system
│   ├── qa_system.py         # Main Q&A orchestrator
│   ├── entity_extractor.py  # Extract entities from questions
│   ├── entity_matcher.py    # Match entities to GSW nodes
│   ├── summary_reranker.py  # Rerank summaries by relevance
│   └── answering_agent.py   # Generate final answers
├── evaluation/               # Evaluation framework
│   ├── judges/              # Base evaluation interfaces
│   └── benchmarks/          # Benchmark-specific evaluators
│       └── tulving_bench/   # Tulving Bench evaluation
└── benchmarks/              # Benchmark datasets
    └── tulvingbench/        # Tulving Bench data
```

## Installation

### For Users
```bash
pip install gsw-memory
```

### For Development
```bash
git clone <repository-url>
cd gsw-memory
uv sync --group dev
```

## Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
VOYAGE_API_KEY=your_voyage_api_key_here  # For embeddings
```

## Quick Start

Try our comprehensive end-to-end example that demonstrates the complete GSW pipeline:

```bash
cd gsw-memory
python playground/test_tulving_bench_e2e.py
```

This example shows:
1. **Document Processing** → GSW structures
2. **Entity Reconciliation** (LOCAL strategy, chapter-by-chapter)
3. **Entity Summary Generation** for each chapter
4. **Multi-Chapter Q&A System** that searches across chapters
5. **LLM-as-a-Judge Evaluation** using Tulving Bench
6. **Performance Comparison** against baseline

## Usage Examples

### Multi-Document Q&A (Recommended)
For processing multiple documents separately and answering questions across all of them:

```python
from gsw_memory import GSWProcessor, reconcile_gsw_outputs, GSWQuestionAnswerer
from gsw_memory.memory.aggregators import EntitySummaryAggregator

# Process documents
processor = GSWProcessor(model_name="gpt-4o")
gsw_structures = processor.process_documents(documents)

# Reconcile with local strategy (keeps documents separate)
reconciled_gsws = reconcile_gsw_outputs(gsw_structures, strategy="local")

# Generate entity summaries for each document
llm_config = {"model_name": "gpt-4o", "generation_params": {"temperature": 0.0}}
aggregators = []
for gsw in reconciled_gsws:
    aggregator = EntitySummaryAggregator(gsw, llm_config)
    aggregators.append(aggregator)

# Create Q&A system that searches across all documents
qa_system = GSWQuestionAnswerer(reconciled_gsws, aggregators, llm_config)
answer = qa_system.ask("Who is the main character?")
```

### Single Unified Q&A
For merging all documents into one unified GSW:

```python
# Reconcile with global strategy (merges all documents)
unified_gsw = reconcile_gsw_outputs(gsw_structures, strategy="global")

# Generate entity summaries for unified GSW
aggregator = EntitySummaryAggregator(unified_gsw, llm_config)

# Create Q&A system with single GSW (backward compatible)
qa_system = GSWQuestionAnswerer(unified_gsw, aggregator, llm_config)
answer = qa_system.ask("Who is the main character?")
```

### Evaluation

```python
from gsw_memory import TulvingBenchEvaluator

# Evaluate Q&A results
evaluator = TulvingBenchEvaluator(model_name="gpt-4o")
results = evaluator.evaluate(qa_results=qa_results, ground_truth=ground_truth)

print(f"Precision: {results['system_metrics']['precision']:.3f}")
print(f"Recall: {results['system_metrics']['recall']:.3f}")
print(f"F1 Score: {results['system_metrics']['f1']:.3f}")
```

## When to Use Each Strategy

**Local Strategy**: Use when you want to:
- Preserve document boundaries and sources
- Answer questions that may span multiple documents
- Maintain separate entity contexts per document
- Scale to many documents efficiently

**Global Strategy**: Use when you want to:
- Merge all information into one unified memory
- Simplify entity reconciliation across documents
- Have a single comprehensive knowledge base
- Work with smaller document sets

## Core Dependencies

- `bespokelabs-curator`: LLM orchestration and parallel processing
- `pydantic`: Data validation and serialization
- `openai`: LLM API access
- `langchain-voyageai`: Embeddings for entity matching and reranking
- `faiss-cpu`: Vector similarity search
- `rank-bm25`: BM25 retrieval for question answering

## Examples & Testing

The `playground/` directory contains comprehensive examples:

```bash
# Complete end-to-end pipeline with evaluation
python playground/test_tulving_bench_e2e.py

# Basic GSW processing functionality
python playground/test_operator.py

# Complete Q&A pipeline integration test
python playground/test_qa_complete.py
```

> **Note:** The code for **Panini** (multi-hop QA with RICR retrieval) is currently in `playground/`. We are actively working on integrating it into the main package API for easier access.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python playground/test_tulving_bench_e2e.py`)
6. Submit a pull request

## Citation

If you use this codebase, please cite the relevant paper(s) and consider starring the repo to help others find it!

```bibtex
@misc{rajesh2025factretrievalepisodicmemory,
      title={Beyond Fact Retrieval: Episodic Memory for RAG with Generative Semantic Workspaces},
      author={Shreyas Rajesh and Pavan Holur and Chenda Duan and David Chong and Vwani Roychowdhury},
      year={2025},
      eprint={2511.07587},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2511.07587},
}

@misc{rajesh2026paninicontinuallearningtoken,
      title={Panini: Continual Learning in Token Space via Structured Memory},
      author={Shreyas Rajesh and Pavan Holur and Mehmet Yigit Turali and Chenda Duan and Vwani Roychowdhury},
      year={2026},
      eprint={2602.15156},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2602.15156},
}
```
