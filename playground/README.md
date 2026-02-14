# Playground

This directory contains scripts for testing, experimentation, and evaluation.

## Core GSW Tests

These scripts test the main GSW pipeline from the AAAI-26 paper:

- `test_operator.py` - Basic GSW processing functionality
- `test_qa_complete.py` - Complete Q&A pipeline integration test
- `test_tulving_bench_e2e.py` - End-to-end Tulving Bench evaluation

## Multi-Hop QA (Preprint in Progress)

The following scripts are for upcoming work on multi-hop question answering with GSW. Preprint coming soon.

- `multi_hop_qa*.py` - Multi-hop QA implementations
- `evaluate_multi_hop_qa*.py` - Evaluation scripts for multi-hop QA
- `test_2wiki_e2e.py` - 2WikiMultiHopQA dataset evaluation
- `test_agentic_2wiki.py` - Agentic approach for 2wiki
- `process_full_2wiki_corpus.py` - Corpus processing for 2wiki
- `hypernode_*.py` - Hypernode clustering and querying
- `generate_hypernode_summaries.py` - Summary generation for hypernodes
