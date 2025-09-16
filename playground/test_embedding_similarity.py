#!/usr/bin/env python3
"""
Simple test to check embedding similarity between text strings.
Uses the same Qwen3-Embedding-8B model and approach as hypernode clustering.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(str(Path(__file__).parent.parent))

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    print("Error: VLLM not available. Install with: pip install vllm>=0.8.5")
    sys.exit(1)


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Create instruction for Qwen embedding model."""
    return f'Instruct: {task_description}\nQuery: {query}'


def test_similarity(text1: str, text2: str, threshold: float = 0.85):
    """Test similarity between two text strings."""
    
    # Same task description as in hypernode clustering
    task = 'Given a comprehensive entity summary, create an embedding optimized for semantic search and question answering retrieval'
    
    print("Initializing Qwen3-Embedding-8B model...")
    model = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")
    
    # Prepare inputs
    input1 = get_detailed_instruct(task, text1)
    input2 = get_detailed_instruct(task, text2)
    
    print(f"\nGenerating embeddings...")
    outputs = model.embed([input1, input2])
    
    # Extract embeddings
    embedding1 = np.array([outputs[0].outputs.embedding])
    embedding2 = np.array([outputs[1].outputs.embedding])
    
    # Compute similarity
    similarity = cosine_similarity(embedding1, embedding2)[0][0]
    
    print(f"\n{'='*50}")
    print(f"Text 1: {text1}")
    print(f"Text 2: {text2}")
    print(f"{'='*50}")
    print(f"Similarity: {similarity:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Above threshold: {'✓ YES (would cluster)' if similarity >= threshold else '✗ NO (would not cluster)'}")
    
    return similarity


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test embedding similarity between text strings")
    parser.add_argument("--text1", type=str, default="Nicki Minaj person: Trinidadian-American, rapper, singer")
    parser.add_argument("--text2", type=str, default="Nicki Minaj person: stage name, musician, actress, model")
    parser.add_argument("--threshold", type=float, default=0.85, help="Similarity threshold (default: 0.85)")
    
    args = parser.parse_args()
    
    test_similarity(args.text1, args.text2, args.threshold)