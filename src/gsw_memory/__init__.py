"""
GSW Memory Package

A clean, packageable implementation of Generative Semantic Workspaces (GSW)
for building semantic memory systems from text.

This package provides:
- GSW generation from text documents
- Entity reconciliation across documents  
- Entity summary aggregation
- Question-answering system using GSW structures

Example usage:

## Multi-Document Q&A (Local Strategy)
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

## Single Unified Q&A (Global Strategy)  
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
"""

# Core memory components
from .memory import GSWProcessor, reconcile_gsw_outputs
from .memory.models import GSWStructure, EntityNode, VerbPhraseNode, Role, Question
from .memory.aggregators import EntitySummaryAggregator

# Q&A system
from .qa import GSWQuestionAnswerer, QuestionEntityExtractor, EntityMatcher, SummaryReranker

# Evaluation system
from .evaluation.benchmarks.tulving_bench.evaluator import TulvingBenchEvaluator

__version__ = "0.1.0"

__all__ = [
    # Core processing
    "GSWProcessor",
    "reconcile_gsw_outputs", 
    
    # Data models
    "GSWStructure",
    "EntityNode", 
    "VerbPhraseNode",
    "Role",
    "Question",
    
    # Aggregators
    "EntitySummaryAggregator",
    
    # Q&A system
    "GSWQuestionAnswerer",
    "QuestionEntityExtractor",
    "EntityMatcher", 
    "SummaryReranker",
    
    # Evaluation
    "TulvingBenchEvaluator",
]


def main() -> None:
    print("Hello from gsw-memory!")
