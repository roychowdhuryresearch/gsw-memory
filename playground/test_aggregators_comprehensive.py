#!/usr/bin/env python3
"""
Comprehensive test of GSW Memory Aggregators with caching and similarity features.

This script demonstrates:
1. Loading a reconciled GSW structure from JSON
2. Testing all three aggregators (Entity, Verb Phrase, Conversation)
3. File caching functionality 
4. Similarity-based aggregate functions
5. Event-focused verb phrase summaries

Usage:
    python examples/test_aggregators_comprehensive.py
"""

import json
import os
import sys
from pathlib import Path

# Add the src directory to the path so we can import gsw_memory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gsw_memory.memory.models import GSWStructure
from gsw_memory.memory.aggregators import (
    EntitySummaryAggregator,
    VerbSummaryAggregator, 
    ConversationSummaryAggregator
)


def load_gsw_from_file(file_path: str) -> GSWStructure:
    """Load GSW structure from JSON file."""
    print(f"Loading GSW from: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        gsw_data = json.load(f)
    
    gsw = GSWStructure.from_json(gsw_data)
    
    print(f"✅ Loaded GSW with:")
    print(f"   - {len(gsw.entity_nodes)} entities")
    print(f"   - {len(gsw.verb_phrase_nodes)} verb phrases") 
    print(f"   - {len(gsw.conversation_nodes)} conversations")
    print(f"   - {len(gsw.space_nodes)} space nodes")
    print(f"   - {len(gsw.time_nodes)} time nodes")
    
    return gsw


def test_entity_aggregator(gsw: GSWStructure, output_dir: str):
    """Test Entity Summary Aggregator with caching."""
    print("\n" + "="*60)
    print("🧑 TESTING ENTITY SUMMARY AGGREGATOR")
    print("="*60)
    
    llm_config = {
        "model_name": "gpt-4o",
        "generation_params": {"temperature": 0.0, "max_tokens": 500}
    }
    
    # Initialize aggregator
    entity_agg = EntitySummaryAggregator(gsw, llm_config)
    
    # Setup cache file path
    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, "entity_summaries.json")
    
    print(f"Cache file: {cache_file}")
    
    # Test 1: Generate summaries with caching
    print("\n📝 Generating entity summaries (first run)...")
    entity_summaries = entity_agg.precompute_summaries(
        cache_file=cache_file,
        include_space_time=True
    )
    
    print(f"✅ Generated {len(entity_summaries)} entity summaries")
    
    # Show a few examples
    print("\n📋 Sample entity summaries:")
    for i, (entity_id, summary_data) in enumerate(list(entity_summaries.items())[:3]):
        entity_name = summary_data["entity_name"]
        summary = summary_data["summary"][:200] + "..." if len(summary_data["summary"]) > 200 else summary_data["summary"]
        print(f"\n{i+1}. {entity_name} ({entity_id}):")
        print(f"   {summary}")
    
    # Test 2: Load from cache (should be fast)
    print(f"\n🚀 Loading entity summaries from cache...")
    cached_summaries = entity_agg.precompute_summaries(cache_file=cache_file)
    
    if len(cached_summaries) == len(entity_summaries):
        print("✅ Cache loading successful!")
    else:
        print("❌ Cache loading failed")
    
    # Test 3: Test aggregate function (debugging interface)
    print(f"\n🔍 Testing aggregate function for specific entities...")
    try:
        # Try to find an entity to test with
        if entity_summaries:
            sample_entity = list(entity_summaries.values())[0]
            entity_name = sample_entity["entity_name"]
            
            # Test query-based aggregation
            entity_view = entity_agg.aggregate(f"Tell me about {entity_name}")
            
            print(f"✅ Aggregate function found {len(entity_view.content.get('entity_summaries', {}))} entities for query: '{entity_name}'")
        else:
            print("⚠️  No entities to test aggregate function with")
    except Exception as e:
        print(f"⚠️  Aggregate function test failed: {e}")


def test_verb_aggregator(gsw: GSWStructure, output_dir: str):
    """Test Verb Phrase Summary Aggregator with similarity ranking."""
    print("\n" + "="*60)
    print("🎬 TESTING VERB PHRASE SUMMARY AGGREGATOR")
    print("="*60)
    
    llm_config = {
        "model_name": "gpt-4o", 
        "generation_params": {"temperature": 0.0, "max_tokens": 500}
    }
    
    # Initialize aggregator with embedding model
    verb_agg = VerbSummaryAggregator(gsw, llm_config, embedding_model="voyage-3")
    
    # Setup cache file path
    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, "verb_summaries.json")
    
    print(f"Cache file: {cache_file}")
    
    # Test 1: Generate verb phrase summaries with caching
    print(f"\n📝 Generating verb phrase summaries (first run)...")
    verb_summaries = verb_agg.precompute_summaries(cache_file=cache_file)
    
    print(f"✅ Generated {len(verb_summaries)} verb phrase summaries")
    
    # Show a few examples of event-focused summaries
    print(f"\n📋 Sample verb phrase event summaries:")
    for i, (vp_id, summary_data) in enumerate(list(verb_summaries.items())[:3]):
        verb_phrase = summary_data["verb_phrase"]
        summary = summary_data["summary"][:300] + "..." if len(summary_data["summary"]) > 300 else summary_data["summary"]
        print(f"\n{i+1}. '{verb_phrase}' ({vp_id}):")
        print(f"   Event Summary: {summary}")
    
    # Test 2: Load from cache
    print(f"\n🚀 Loading verb phrase summaries from cache...")
    cached_summaries = verb_agg.precompute_summaries(cache_file=cache_file)
    
    if len(cached_summaries) == len(verb_summaries):
        print("✅ Cache loading successful!")
    else:
        print("❌ Cache loading failed")
    
    # Test 3: Test similarity-based aggregate function
    print(f"\n🔍 Testing similarity-based verb phrase ranking...")
    test_queries = [
        "Who went somewhere?",
        "What was given to someone?", 
        "Who spoke or talked?",
        "What actions happened?"
    ]
    
    for query in test_queries:
        try:
            verb_view = verb_agg.aggregate(query, max_verb_phrases=3)
            matched_verbs = verb_view.content.get("verb_phrase_summaries", {})
            
            print(f"\n📍 Query: '{query}'")
            print(f"   Found {len(matched_verbs)} relevant verb phrases:")
            
            for vp_id, summary_data in matched_verbs.items():
                verb_phrase = summary_data["verb_phrase"]
                print(f"     - '{verb_phrase}' ({vp_id})")
                
        except Exception as e:
            print(f"   ⚠️  Query failed: {e}")


def test_conversation_aggregator(gsw: GSWStructure, output_dir: str):
    """Test Conversation Summary Aggregator with similarity ranking.""" 
    print("\n" + "="*60)
    print("💬 TESTING CONVERSATION SUMMARY AGGREGATOR")
    print("="*60)
    
    if not gsw.conversation_nodes:
        print("⚠️  No conversations found in GSW - skipping conversation tests")
        return
    
    # Initialize aggregator with embedding model
    conv_agg = ConversationSummaryAggregator(gsw, embedding_model="voyage-3")
    
    # Setup cache file path
    os.makedirs(output_dir, exist_ok=True)
    cache_file = os.path.join(output_dir, "conversation_summaries.json")
    
    print(f"Cache file: {cache_file}")
    
    # Test 1: Extract conversation summaries with caching
    print(f"\n📝 Extracting conversation summaries (first run)...")
    conv_summaries = conv_agg.precompute_summaries(cache_file=cache_file)
    
    print(f"✅ Extracted {len(conv_summaries)} conversation summaries")
    
    # Show examples
    print(f"\n📋 Sample conversation summaries:")
    for i, (conv_id, summary_data) in enumerate(list(conv_summaries.items())[:3]):
        participants = ", ".join(summary_data["participants"])
        summary = summary_data["summary"][:200] + "..." if len(summary_data["summary"]) > 200 else summary_data["summary"]
        print(f"\n{i+1}. Conversation {conv_id}:")
        print(f"   Participants: {participants}")
        print(f"   Summary: {summary}")
    
    # Test 2: Load from cache
    print(f"\n🚀 Loading conversation summaries from cache...")
    cached_summaries = conv_agg.precompute_summaries(cache_file=cache_file)
    
    if len(cached_summaries) == len(conv_summaries):
        print("✅ Cache loading successful!")
    else:
        print("❌ Cache loading failed")
    
    # Test 3: Test similarity-based aggregate function
    print(f"\n🔍 Testing similarity-based conversation ranking...")
    test_queries = [
        "What conversations happened?",
        "Who talked to whom?",
        "What was discussed?",
        "Any dialogue or speech?"
    ]
    
    for query in test_queries:
        try:
            conv_view = conv_agg.aggregate(query, max_conversations=3)
            matched_convs = conv_view.content.get("conversation_summaries", {})
            
            print(f"\n📍 Query: '{query}'")
            print(f"   Found {len(matched_convs)} relevant conversations:")
            
            for conv_id, summary_data in matched_convs.items():
                participants = ", ".join(summary_data["participants"])
                print(f"     - {conv_id}: {participants}")
                
        except Exception as e:
            print(f"   ⚠️  Query failed: {e}")


def main():
    """Main test function."""
    print("🚀 GSW Memory Aggregators Comprehensive Test")
    print("="*60)
    
    # File paths
    gsw_file = "/mnt/SSD1/chenda/gsw-memory/test_output/reconciled_local/reconciled/doc_0_reconciled.json"
    base_output_dir = "/mnt/SSD1/chenda/gsw-memory/test_output/summary"
    
    # Check if GSW file exists
    if not os.path.exists(gsw_file):
        print(f"❌ GSW file not found: {gsw_file}")
        print("Please ensure the file exists before running this test.")
        return
    
    try:
        # Load GSW structure
        gsw = load_gsw_from_file(gsw_file)
        
        # Test all aggregators
        test_entity_aggregator(gsw, os.path.join(base_output_dir, "entity"))
        test_verb_aggregator(gsw, os.path.join(base_output_dir, "verbphrase"))
        test_conversation_aggregator(gsw, os.path.join(base_output_dir, "conversation"))
        
        print("\n" + "="*60)
        print("🎉 ALL TESTS COMPLETED!")
        print("="*60)
        print(f"📁 Summary files saved to: {base_output_dir}")
        print("   - entity/entity_summaries.json")
        print("   - verbphrase/verb_summaries.json") 
        print("   - conversation/conversation_summaries.json")
        print("\n✨ Features tested:")
        print("   ✅ File caching (save/load)")
        print("   ✅ Event-focused verb phrase summaries")
        print("   ✅ Similarity-based ranking for verb phrases")
        print("   ✅ Similarity-based ranking for conversations")
        print("   ✅ Entity summaries with space-time information")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 