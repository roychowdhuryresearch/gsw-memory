#!/usr/bin/env python3
"""
Quick smoke test for query agent tools: search_entities, search_qa_pairs, get_bridge_evidence.

Tests against real GSW data (doc_4027=Sergio Gobbi, doc_5300=Daytime Wives, doc_2793=Émile Chautard).
Verifies that search_entities returns QA pairs inline so the agent doesn't need search_qa_pairs.

Usage:
    python tests/sleep_time/test_query_tools.py
    # or with a specific GSW path:
    python tests/sleep_time/test_query_tools.py --gsw_path /path/to/networks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from playground.simple_entity_search import EntitySearcher
from gsw_memory.sleep_time.tools import GSWTools
from gsw_memory.sleep_time.curriculum import BridgeRegistry

DEFAULT_GSW_PATH = "/mnt/SSD1/shreyas/SM_GSW/2wiki/networks"
# Docs for: Sergio Gobbi (4027), Daytime Wives film (5300), Émile Chautard (2793)
TEST_DOC_IDS = ["doc_4027", "doc_5300", "doc_2793"]


def setup_tools(gsw_path: str, doc_ids: list[str]) -> GSWTools:
    """Build EntitySearcher + GSWTools for the given docs."""
    import tempfile, shutil, json

    temp_dir = tempfile.mkdtemp(prefix="test_query_tools_")
    # Copy only the test docs
    for doc_id in doc_ids:
        src = Path(gsw_path) / doc_id
        dst = Path(temp_dir) / doc_id
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            print(f"  WARNING: {src} not found, skipping")

    entity_searcher = EntitySearcher(
        num_documents=len(doc_ids),
        path_to_gsw_files=temp_dir,
        cache_dir=temp_dir,
        rebuild_cache=True,
        verbose=False,
        use_bm25=True,
        use_gpu_for_qa_index=False,
        embedding_gpu_memory_utilization=0.5,
    )
    tools = GSWTools(entity_searcher)
    return tools, temp_dir


def test_search_entities_returns_qa_pairs(tools: GSWTools):
    """search_entities must return qa_pairs for each entity."""
    print("\n=== Test: search_entities returns QA pairs ===")
    results = tools.search_entities("Sergio Gobbi", top_k=3)

    assert len(results) > 0, "FAIL: No results for 'Sergio Gobbi'"
    top = results[0]
    assert top["name"] == "Sergio Gobbi", f"FAIL: Expected 'Sergio Gobbi', got '{top['name']}'"

    qa_pairs = top.get("qa_pairs", [])
    assert len(qa_pairs) > 0, "FAIL: search_entities returned 0 qa_pairs — the fix didn't work!"

    print(f"  Entity: {top['name']} (doc={top['doc_id']})")
    print(f"  QA pairs returned: {len(qa_pairs)}")
    for qa in qa_pairs:
        q = qa.get("question", "")
        a = ", ".join(qa.get("answer_names", []))
        print(f"    Q: {q}")
        print(f"    A: {a}")

    # Check specific facts exist
    questions = {qa["question"] for qa in qa_pairs}
    assert any("nationality" in q.lower() for q in questions), (
        f"FAIL: No nationality question found. Questions: {questions}"
    )
    assert any("born" in q.lower() for q in questions), (
        f"FAIL: No birth question found. Questions: {questions}"
    )
    print("  PASS: search_entities returns QA pairs with nationality and birth facts")


def test_search_entities_multi_hop(tools: GSWTools):
    """Test the 2-hop chain: Daytime Wives → director → nationality."""
    print("\n=== Test: 2-hop chain via search_entities ===")

    # Step 1: Find the film
    results = tools.search_entities("Daytime Wives", top_k=3)
    assert len(results) > 0, "FAIL: No results for 'Daytime Wives'"
    film = results[0]
    qa_pairs = film.get("qa_pairs", [])
    print(f"  Step 1: Found '{film['name']}' with {len(qa_pairs)} QA pairs")

    # Find director from QA pairs
    director = None
    for qa in qa_pairs:
        q = qa.get("question", "").lower()
        if "direct" in q or "director" in q:
            names = qa.get("answer_names", [])
            if names:
                director = names[0]
                print(f"    Director found: {director} (from Q: '{qa['question']}')")
                break
    assert director is not None, f"FAIL: No director found in QA pairs: {[qa['question'] for qa in qa_pairs]}"

    # Step 2: Find the director entity
    results2 = tools.search_entities(director, top_k=3)
    assert len(results2) > 0, f"FAIL: No results for director '{director}'"
    director_entity = results2[0]
    qa_pairs2 = director_entity.get("qa_pairs", [])
    print(f"  Step 2: Found '{director_entity['name']}' with {len(qa_pairs2)} QA pairs")

    # Find nationality from director's QA pairs
    nationality = None
    for qa in qa_pairs2:
        q = qa.get("question", "").lower()
        if "nationality" in q:
            names = qa.get("answer_names", [])
            if names:
                nationality = names[0]
                print(f"    Nationality found: {nationality} (from Q: '{qa['question']}')")
                break
    assert nationality is not None, f"FAIL: No nationality in director QA pairs: {[qa['question'] for qa in qa_pairs2]}"
    print(f"  PASS: 2-hop chain resolved in 2 search_entities calls → '{nationality}'")


def test_search_qa_pairs(tools: GSWTools):
    """Check what search_qa_pairs returns — expect poor results."""
    print("\n=== Test: search_qa_pairs quality check ===")

    queries = [
        "What is the nationality of Sergio Gobbi",
        "Who directed Daytime Wives",
        "Where was Sergio Gobbi born",
    ]
    for query in queries:
        results = tools.search_qa_pairs(query, top_k=3)
        if results:
            top = results[0]
            q = top.get("question", "")
            a = ", ".join(top.get("answer_names", []))
            doc = top.get("doc_id", "")
            # Check if result is relevant
            query_words = set(query.lower().split()) - {"the", "of", "is", "a", "was", "what", "who", "where"}
            result_words = set(q.lower().split())
            overlap = query_words & result_words
            relevant = len(overlap) >= 2
            icon = "✅" if relevant else "❌"
            print(f"  {icon} search_qa_pairs(\"{query}\")")
            print(f"     → Q: {q}")
            print(f"     → A: {a} (doc={doc}, overlap={overlap})")
        else:
            print(f"  ⚠️  search_qa_pairs(\"{query}\") → empty")


def test_faiss_qa_index(tools: GSWTools):
    """Diagnose FAISS QA index — check what's indexed and why search fails."""
    print("\n=== Test: FAISS QA index diagnosis ===")
    es = tools.entity_searcher

    # Check QA index stats
    qa_count = len(es.qa_metadata_cache)
    has_faiss = es.qa_faiss_index is not None
    has_matrix = es.qa_embeddings_matrix is not None
    matrix_shape = es.qa_embeddings_matrix.shape if has_matrix else None

    print(f"  QA metadata cache: {qa_count} pairs")
    print(f"  FAISS index exists: {has_faiss}")
    print(f"  Embeddings matrix: {matrix_shape}")

    if has_faiss:
        print(f"  FAISS index vectors: {es.qa_faiss_index.ntotal}")

    # Show what's actually in the index
    print(f"\n  All indexed QA pairs:")
    for qa_hash, meta in es.qa_metadata_cache.items():
        q = meta.get("question", "")
        a = ", ".join(meta.get("answer_names", []))
        doc = meta.get("doc_id", "")
        print(f"    [{doc}] Q: {q} → A: {a}")

    # Now search and check scores
    import numpy as np
    test_query = "What is the nationality of Sergio Gobbi"
    results = es.search_qa_pairs_direct(test_query, top_k=5, verbose=False)
    print(f"\n  Search results for '{test_query}':")
    for i, r in enumerate(results):
        q = r.get("question", "")
        a = ", ".join(r.get("answer_names", []))
        score = r.get("similarity_score", 0)
        doc = r.get("doc_id", "")
        print(f"    [{i}] score={score:.4f} [{doc}] Q: {q} → A: {a}")

    # Check if the correct QA pair exists and what its score would be
    target_hash = None
    for qa_hash, meta in es.qa_metadata_cache.items():
        if "nationality" in meta.get("question", "").lower() and "sergio" in meta.get("question", "").lower():
            target_hash = qa_hash
            print(f"\n  Target QA found in index: '{meta['question']}'")
            break

    if target_hash and target_hash in es.qa_hash_to_idx:
        idx = es.qa_hash_to_idx[target_hash]
        print(f"  Target index position: {idx}")
        # Get the embedding and compute similarity with query
        if has_matrix:
            target_emb = es.qa_embeddings_matrix[idx:idx+1].copy()
            import faiss
            faiss.normalize_L2(target_emb)
            query_emb = es._embed_query(test_query)
            if query_emb is not None:
                query_emb_np = query_emb[0].reshape(1, -1).astype(np.float32)
                faiss.normalize_L2(query_emb_np)
                cosine_sim = float(np.dot(query_emb_np[0], target_emb[0]))
                print(f"  Direct cosine similarity (query vs target): {cosine_sim:.4f}")
                # Compare with top result score
                if results:
                    print(f"  Top result score: {results[0].get('similarity_score', 0):.4f}")
                    print(f"  Gap: {results[0].get('similarity_score', 0) - cosine_sim:.4f}")
    else:
        print("  WARNING: Target QA pair not found in index!")


def test_bridge_evidence(tools: GSWTools):
    """Check that get_bridge_evidence works when registry is empty."""
    print("\n=== Test: get_bridge_evidence (empty registry) ===")
    from gsw_memory.query_then_sleep.query_agent import QueryToolAdapter

    query_tools = QueryToolAdapter(
        entity_searcher=tools.entity_searcher,
        bridge_registry=BridgeRegistry(),
        bridge_top_k=5,
    )
    results = query_tools.get_bridge_evidence("What nationality is the director of Daytime Wives")
    print(f"  Results with empty registry: {len(results)} (expected 0)")
    assert len(results) == 0, "FAIL: Should return 0 with empty registry"
    print("  PASS: Empty registry returns no results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gsw_path", default=DEFAULT_GSW_PATH)
    args = parser.parse_args()

    print(f"Setting up tools with docs: {TEST_DOC_IDS}")
    tools, temp_dir = setup_tools(args.gsw_path, TEST_DOC_IDS)
    print(f"  Loaded {len(tools.entity_searcher.gsw_by_doc_id)} docs, {len(tools.entity_searcher.entities)} entities")

    try:
        test_search_entities_returns_qa_pairs(tools)
        test_search_entities_multi_hop(tools)
        test_search_qa_pairs(tools)
        test_faiss_qa_index(tools)
        test_bridge_evidence(tools)
        print("\n✅ All tests passed!")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    finally:
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
