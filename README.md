# GSW Memory

A Python package for generating Generative Semantic Workspaces (GSW) from text documents. This package provides tools for processing documents through coreference resolution, chunking, context generation, and semantic workspace creation with entity reconciliation capabilities.

## Features

- **Multi-document Processing**: Process multiple documents simultaneously with full parallelization
- **Coreference Resolution**: Resolve pronouns and references for clearer semantic understanding
- **Smart Chunking**: Split documents into overlapping chunks with configurable size and overlap
- **Context Generation**: Generate contextual information for each chunk within its document
- **GSW Generation**: Create semantic workspaces with entities, roles, and question-answer pairs
- **Entity Reconciliation**: Merge and reconcile entities across different chunks and documents
- **Comprehensive Saving**: Organized output structure with intermediates and visualizations

## Installation

### For Users
```bash
pip install gsw-memory
```

### For Development
```bash
git clone https://github.com/shreyasrajesh0308/gsw-memory.git
cd gsw-memory

# Option 1: Using uv (recommended for contributors)
uv pip install -e .

# Option 2: Using pip
pip install -e .
```

## Quick Start

### Complete GSW Memory Creation

```python
from gsw_memory.memory import GSWProcessor, reconcile_gsw_outputs

# Initialize the processor
processor = GSWProcessor(
    model_name="gpt-4o",
    enable_coref=True,
    enable_chunking=True,
    enable_context=True,
    chunk_size=3,
    overlap=0,
    enable_spacetime=True,
)

# Process documents
documents = [
    "John walked to the coffee shop. He ordered a latte. The drink was expensive.",
    "The barista prepared his drink carefully. She smiled at John. He thanked her."
]

# Step 1: Generate GSW structures from documents
gsw_structures = processor.process_documents(documents, output_dir="output")
print(f"Generated GSW structures for {len(gsw_structures)} documents")

# Step 2: Reconcile entities across chunks/documents
reconciled_chapters = reconcile_gsw_outputs(
    gsw_structures, 
    strategy="local",           # "local" or "global"
    matching_approach="exact",  # "exact" or "embedding"
    output_dir="reconciled_output",
    save_statistics=True,
    enable_visualization=False  # Set to True if you have NetworkX installed
)

print(f"Reconciled {len(reconciled_chapters)} chapters:")
for i, chapter_gsw in enumerate(reconciled_chapters):
    print(f"  Chapter {i}: {len(chapter_gsw.entity_nodes)} entities, "
          f"{len(chapter_gsw.verb_phrase_nodes)} verb phrases")
```

This creates organized output directories:
```
output/                    # GSWProcessor outputs
├── networks/             # Parsed GSW structures
├── networks_raw/         # Raw LLM responses  
├── coref/               # Coreference resolved texts
├── chunks/              # Individual chunks
└── context/             # Generated contexts

reconciled_output/         # Reconciliation outputs
├── reconciled/           # Final reconciled GSW structures
├── statistics/           # Reconciliation statistics
└── visualizations/       # Network visualizations (optional)
```

## Entity Reconciliation

The package provides flexible reconciliation strategies:

### Local vs Global Reconciliation
```python
# Local: Reconcile entities within each document separately
reconciled_local = reconcile_gsw_outputs(
    gsw_structures, 
    strategy="local"
)

# Global: Reconcile entities across all documents together
reconciled_global = reconcile_gsw_outputs(
    gsw_structures, 
    strategy="global"
)
```

### Matching Approaches
```python
# Exact matching: Entities with identical names
reconciled_exact = reconcile_gsw_outputs(
    gsw_structures,
    matching_approach="exact"
)

# Embedding matching: Semantically similar entities (requires additional dependencies)
reconciled_embedding = reconcile_gsw_outputs(
    gsw_structures,
    matching_approach="embedding",
    k=5  # Top-k similar entities to consider
)
```

## Configuration

```python
processor = GSWProcessor(
    model_name="gpt-4o",           # LLM model to use
    enable_coref=True,             # Enable coreference resolution
    enable_chunking=True,          # Enable text chunking
    enable_context=True,           # Enable context generation
    enable_spacetime=True,         # Enable spacetime linking
    chunk_size=3,                  # Sentences per chunk
    overlap=0,                     # Sentence overlap between chunks
    generation_params={            # LLM generation parameters
        "temperature": 0.0,
        "max_tokens": 2000
    }
)
```

## Dependencies

### Core Dependencies
- `bespokelabs-curator`: LLM orchestration and parallel processing
- `pydantic`: Data validation and serialization
- `openai`: LLM API access
- `numpy`: Numerical computations

### Optional Dependencies
- `networkx`: For network visualizations
- `faiss-cpu`: For embedding-based entity matching
- `langchain-voyageai`: For generating embeddings

## Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
VOYAGE_API_KEY=your_voyage_api_key_here  # For embedding features
```

## Examples & Testing

See the `playground/` directory for comprehensive examples:

```bash
# Basic functionality test
python playground/test_operator.py

# Interactive testing (Jupyter notebook)
jupyter notebook playground/operator_tests.ipynb
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`python playground/test_operator.py`)
6. Submit a pull request

## Citation

If you use this package in your research, please cite:

```bibtex
@inproceedings{rajesh2025gsw,
  title={Generative Semantic Workspaces: An Episodic Memory Framework for Large Language Models},
  author={Rajesh, Shreyas and Holur, Pavan and Duan, Chenda and Chong, David and Roychowdhury, Vwani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025},
  note={Under review}
}
```