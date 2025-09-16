#!/usr/bin/env python3
"""
Test script for embedding cache system.

Tests the caching functionality for entity and Q&A embeddings.
"""

import time
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from playground.simple_entity_search import EntitySearcher
from rich.console import Console

console = Console()


def test_cache_creation():
    """Test initial cache creation."""
    console.print("\n[bold cyan]Test 1: Initial Cache Creation[/bold cyan]")
    console.print("Loading with cache rebuild (force regenerate)...")
    
    start = time.time()
    searcher = EntitySearcher(num_documents=10, rebuild_cache=True)
    elapsed = time.time() - start
    
    console.print(f"Time taken (with embedding generation): {elapsed:.2f} seconds")
    console.print(f"Entity embeddings: {searcher.embeddings.shape if searcher.embeddings is not None else 'None'}")
    console.print(f"Q&A embeddings cached: {len(searcher.qa_embedding_cache)}")
    
    # Run a test query to ensure everything works
    results = searcher.search("king", top_k=3, verbose=False, generate_answer=False)
    console.print(f"Test query returned {len(results)} results")
    
    return searcher


def test_cache_loading():
    """Test loading from cache."""
    console.print("\n[bold cyan]Test 2: Cache Loading[/bold cyan]")
    console.print("Loading from cache (should be much faster)...")
    
    start = time.time()
    searcher = EntitySearcher(num_documents=10, rebuild_cache=False)
    elapsed = time.time() - start
    
    console.print(f"Time taken (loading from cache): {elapsed:.2f} seconds")
    console.print(f"Entity embeddings: {searcher.embeddings.shape if searcher.embeddings is not None else 'None'}")
    console.print(f"Q&A embeddings cached: {len(searcher.qa_embedding_cache)}")
    
    # Run a test query
    results = searcher.search("king", top_k=3, verbose=False, generate_answer=False)
    console.print(f"Test query returned {len(results)} results")
    
    return searcher


def test_cache_performance():
    """Test performance improvement from caching."""
    console.print("\n[bold cyan]Test 3: Cache Performance[/bold cyan]")
    
    # Create searcher (should use cache from previous tests)
    searcher = EntitySearcher(num_documents=10, rebuild_cache=False)
    
    test_queries = ["king", "battle", "prince", "castle", "war"]
    
    console.print(f"Running {len(test_queries)} test queries...")
    
    total_time = 0
    for query in test_queries:
        start = time.time()
        results = searcher.search(query, top_k=5, verbose=False, generate_answer=False)
        elapsed = time.time() - start
        total_time += elapsed
        console.print(f"  Query '{query}': {elapsed:.3f}s, {len(results)} results")
    
    console.print(f"Total time for {len(test_queries)} queries: {total_time:.2f}s")
    console.print(f"Average time per query: {total_time/len(test_queries):.3f}s")
    console.print(f"Cache stats - Hits: {searcher.cache_hits}, Misses: {searcher.cache_misses}")


def test_multihop_with_cache():
    """Test multi-hop QA with caching."""
    console.print("\n[bold cyan]Test 4: Multi-hop QA with Cache[/bold cyan]")
    
    from playground.multi_hop_qa import MultiHopQA
    
    # Initialize with small dataset for testing
    multihop = MultiHopQA(num_documents=10)
    
    # Test question
    test_question = "Who was the king of the country that fought in the battle?"
    
    console.print(f"Test question: {test_question}")
    
    start = time.time()
    # Decompose and process (without full answer generation for speed)
    decomposed = multihop.decompose_question(test_question)
    console.print(f"Decomposed into {len(decomposed)} questions")
    elapsed = time.time() - start
    
    console.print(f"Time taken: {elapsed:.2f}s")
    console.print(f"Entity searcher cache stats - Hits: {multihop.entity_searcher.cache_hits}, Misses: {multihop.entity_searcher.cache_misses}")


def main():
    """Run all cache tests."""
    console.print("\n[bold magenta]🧪 Embedding Cache System Tests[/bold magenta]")
    
    # Test 1: Create cache
    searcher1 = test_cache_creation()
    
    # Test 2: Load from cache
    searcher2 = test_cache_loading()
    
    # Test 3: Performance with cache
    test_cache_performance()
    
    # Test 4: Multi-hop with cache
    test_multihop_with_cache()
    
    console.print("\n[bold green]✅ All tests completed![/bold green]")
    
    # Clean up cache files for testing (optional)
    # Path(".").glob("*embeddings*.npz")


if __name__ == "__main__":
    main()