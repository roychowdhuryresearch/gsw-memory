#!/usr/bin/env python3
"""
Complete Q&A Pipeline Integration Test

This script tests the entire GSW Q&A pipeline:
1. Process documents → GSW structures
2. Reconcile GSW structures  
3. Generate entity summaries
4. Extract entities from questions
5. Match entities to GSW nodes
6. Retrieve and rerank summaries
7. Generate answers

This demonstrates the complete end-to-end functionality.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
from gsw_memory.memory import GSWProcessor, reconcile_gsw_outputs
from gsw_memory.memory.aggregators import EntitySummaryAggregator
from gsw_memory.qa import GSWQuestionAnswerer

# Disable cache for testing
os.environ["CURATOR_DISABLE_CACHE"] = "true"

# Load environment variables
load_dotenv()


def test_complete_qa_pipeline():
    """Test the complete Q&A pipeline from documents to answers."""
    
    print("=== Complete GSW Q&A Pipeline Test ===")
    
    # Test documents with overlapping entities for reconciliation
    test_documents = [
        """
        John Smith walked into the coffee shop on Main Street. He was wearing a blue jacket.
        The barista, Sarah, greeted him warmly. John ordered his usual large latte.
        The coffee shop was busy with morning customers.
        """,
        """
        At the coffee shop, John Smith sat down at his favorite table by the window.
        He opened his laptop and began working on his presentation. Sarah brought him his latte.
        The coffee was perfectly hot and had great foam art. John was impressed by Sarah's skill.
        """,
        """
        Later that day, John Smith met his colleague Mary at the same coffee shop.
        They discussed the upcoming project while Sarah served them both.
        Mary was impressed by the coffee shop's atmosphere and Sarah's friendly service.
        The meeting went well and they made good progress on their project.
        """
    ]
    
    test_questions = [
        "Who is John Smith?",
        "What did John order?", 
        "Who served the coffee?",
        "Where did John meet Mary?",
        "What did John and Mary discuss?"
    ]
    
    print(f"Processing {len(test_documents)} documents...")
    
    # Step 1: Process documents with GSWProcessor
    processor = GSWProcessor(
        model_name="gpt-4o",
        enable_coref=True,
        enable_chunking=False,  # Keep simple for testing
        enable_context=False,   # Skip context for this test
        enable_spacetime=False  # Skip spacetime for this test
    )
    
    # Process all documents
    gsw_structures = processor.process_documents(test_documents)
    print(f"Generated GSW structures for {len(gsw_structures)} documents")
    
    # Step 2: Test both reconciliation strategies
    print("\nTesting LOCAL reconciliation strategy (multiple GSWs)...")
    local_reconciled_gsws = reconcile_gsw_outputs(
        gsw_structures, 
        strategy="local",  # Keep each document separate
        matching_approach="exact"
    )
    
    print(f"Local strategy: Reconciled into {len(local_reconciled_gsws)} GSW structures")
    for i, gsw in enumerate(local_reconciled_gsws):
        print(f"  Document {i}: {len(gsw.entity_nodes)} entities, {len(gsw.verb_phrase_nodes)} verb phrases")
    
    print("\nTesting GLOBAL reconciliation strategy (unified GSW)...")
    global_reconciled_gsw = reconcile_gsw_outputs(
        gsw_structures,
        strategy="global",  # Merge all documents into one GSW
        matching_approach="exact"
    )
    
    print(f"Global strategy: Unified GSW with {len(global_reconciled_gsw.entity_nodes)} entities, {len(global_reconciled_gsw.verb_phrase_nodes)} verb phrases")
    
    # Step 3: Generate entity summaries for all GSWs (local strategy)
    print("\n=== Testing Multi-GSW Q&A System (Local Strategy) ===")
    print("Generating entity summaries for all documents...")
    
    llm_config = {
        "model_name": "gpt-4o",
        "generation_params": {"temperature": 0.0, "max_tokens": 500},
    }
    
    # Create aggregators for each GSW
    local_aggregators = []
    for i, gsw in enumerate(local_reconciled_gsws):
        print(f"  Creating aggregator for document {i}...")
        aggregator = EntitySummaryAggregator(gsw, llm_config)
        summaries = aggregator.precompute_summaries(include_space_time=False)
        print(f"    Generated summaries for {len(summaries)} entities")
        local_aggregators.append(aggregator)
    
    # Step 4a: Initialize multi-GSW Q&A system (local strategy)
    print("\nInitializing multi-GSW Q&A system...")
    multi_qa_system = GSWQuestionAnswerer(
        gsw=local_reconciled_gsws,  # List of GSWs
        entity_aggregator=local_aggregators,  # List of aggregators
        llm_config=llm_config,
        embedding_model="voyage-3"
    )
    
    # Step 4b: Initialize single-GSW Q&A system (global strategy)
    print("\n=== Testing Single-GSW Q&A System (Global Strategy) ===")
    print("Generating entity summaries for unified GSW...")
    
    global_aggregator = EntitySummaryAggregator(global_reconciled_gsw, llm_config)
    global_summaries = global_aggregator.precompute_summaries(include_space_time=False)
    print(f"Generated summaries for {len(global_summaries)} entities")
    
    single_qa_system = GSWQuestionAnswerer(
        gsw=global_reconciled_gsw,  # Single GSW (backward compatibility)
        entity_aggregator=global_aggregator,  # Single aggregator 
        llm_config=llm_config,
        embedding_model="voyage-3"
    )
    
    # Step 5: Test individual question processing steps with multi-GSW system
    print("\n=== Testing Individual Q&A Steps (Multi-GSW) ===")
    
    test_question = test_questions[0]  # "Who is John Smith?"
    print(f"\nTesting with question: '{test_question}' across multiple documents")
    
    # Step 5a: Entity extraction
    print("\n1. Extracting entities...")
    extracted_entities = multi_qa_system.extract_entities(test_question)
    print(f"   Extracted entities: {extracted_entities}")
    
    # Step 5b: Entity matching across multiple GSWs
    print("\n2. Matching entities across all GSWs...")
    matched_entities_with_source = multi_qa_system.find_matching_entities(extracted_entities)
    print(f"   Matched {len(matched_entities_with_source)} entities:")
    for entity, gsw_index in matched_entities_with_source:
        print(f"     - {entity.name} (ID: {entity.id}, from Document {gsw_index})")
    
    # Step 5c: Summary retrieval from multiple aggregators
    print("\n3. Retrieving entity summaries from multiple sources...")
    entity_summaries = multi_qa_system.get_entity_summaries(matched_entities_with_source)
    print(f"   Retrieved {len(entity_summaries)} summaries:")
    for name, summary in entity_summaries:
        print(f"     - {name}: {summary[:100]}...")
    
    # Step 5d: Summary reranking
    print("\n4. Reranking summaries by relevance...")
    ranked_summaries = multi_qa_system.rerank_summaries(entity_summaries, test_question, max_summaries=3)
    print(f"   Reranked to top {len(ranked_summaries)} summaries:")
    for name, summary, score in ranked_summaries:
        print(f"     - {name} (score: {score:.3f}): {summary[:100]}...")
    
    # Step 5e: Compare with single-GSW system
    print("\n=== Comparing with Single-GSW System ===")
    
    single_extracted = single_qa_system.extract_entities(test_question)
    single_matched = single_qa_system.find_matching_entities(single_extracted)
    single_summaries = single_qa_system.get_entity_summaries(single_matched)
    single_ranked = single_qa_system.rerank_summaries(single_summaries, test_question, max_summaries=3)
    
    print(f"Single-GSW system found {len(single_ranked)} relevant summaries:")
    for name, summary, score in single_ranked:
        print(f"     - {name} (score: {score:.3f}): {summary[:100]}...")
    
    # Step 6: Test batch processing with both systems
    print("\n=== Testing Batch Q&A Processing ===")
    
    print(f"\nProcessing {len(test_questions)} questions in batch with multi-GSW system...")
    multi_batch_results = multi_qa_system.ask_batch(test_questions, max_summaries=3)
    
    print(f"\nMulti-GSW batch processing results:")
    for i, result in enumerate(multi_batch_results):
        print(f"\nQuestion {i+1}: {result['question']}")
        print(f"  Extracted entities: {result['extracted_entities']}")
        print(f"  Matched entities: {result['matched_entities']}")
        print(f"  Summaries used: {result['num_summaries_used']}")
        print(f"  Answer: {result['answer']}")
    
    print(f"\nProcessing same questions with single-GSW system...")
    single_batch_results = single_qa_system.ask_batch(test_questions, max_summaries=3)
    
    print(f"\nSingle-GSW batch processing results:")
    for i, result in enumerate(single_batch_results):
        print(f"\nQuestion {i+1}: {result['question']}")
        print(f"  Extracted entities: {result['extracted_entities']}")
        print(f"  Matched entities: {result['matched_entities']}")
        print(f"  Summaries used: {result['num_summaries_used']}")
        print(f"  Answer: {result['answer']}")
    
    # Step 7: Test single question interface with cross-document question
    print("\n=== Testing Cross-Document Question Answering ===")
    
    cross_doc_question = "Where did John meet Mary?"  # This spans multiple documents
    print(f"\nTesting cross-document question: '{cross_doc_question}'")
    
    multi_result = multi_qa_system.ask(cross_doc_question, max_summaries=5)
    print(f"\nMulti-GSW result:")
    print(f"  Question: {multi_result['question']}")
    print(f"  Extracted entities: {multi_result['extracted_entities']}")
    print(f"  Matched entities: {multi_result['matched_entities']}")
    print(f"  Summaries used: {multi_result['num_summaries_used']}")
    print(f"  Answer: {multi_result['answer']}")
    
    single_result = single_qa_system.ask(cross_doc_question, max_summaries=5)
    print(f"\nSingle-GSW result:")
    print(f"  Question: {single_result['question']}")
    print(f"  Extracted entities: {single_result['extracted_entities']}")
    print(f"  Matched entities: {single_result['matched_entities']}")
    print(f"  Summaries used: {single_result['num_summaries_used']}")
    print(f"  Answer: {single_result['answer']}")
    
    print("\n✅ Complete Q&A pipeline test completed successfully!")
    print("   All components are working and integrated properly.")


if __name__ == "__main__":
    test_complete_qa_pipeline()