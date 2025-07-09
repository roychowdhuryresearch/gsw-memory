#!/usr/bin/env python3
"""
Interactive tool caller for the GSW Memory Narrative QA system.

Usage:
    python examples/call_tools.py
"""

import json
import os
import sys
import warnings
import contextlib
from pathlib import Path
from typing import Optional, Dict, Any

# Suppress all verbose output
warnings.filterwarnings("ignore")
os.environ["CURATOR_DISABLE_RICH_DISPLAY"] = "True"

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress stdout."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gsw_memory.memory.models import GSWStructure
from gsw_memory.memory.aggregators import (
    EntitySummaryAggregator,
    VerbSummaryAggregator,
    ConversationSummaryAggregator
)
from gsw_memory.qa.qa_system_narrative import GSWQuestionAnswerer_Narrative


class ToolCaller:
    """Interactive tool caller for testing QA system tools directly."""
    
    def __init__(self, reconciled_file_path: str, cache_base_dir: str, chunks_folder_path: Optional[str] = None):
        """Initialize the QA system for tool testing."""
        print("🛠️  Initializing GSW Memory Tool Caller...")
        
        # Load reconciled GSW structure
        with suppress_stdout():
            with open(reconciled_file_path, 'r', encoding='utf-8') as f:
                gsw_data = json.load(f)
            self.gsw = GSWStructure(**gsw_data)
        
        print(f"📚 Loaded data: {len(self.gsw.entity_nodes)} entities, {len(self.gsw.verb_phrase_nodes)} verb phrases")
        
        # Setup LLM config
        llm_config = {"model_name": "gpt-4o", "generation_params": {"temperature": 0.0}}
        llm_config_qa = {"model_name": "gpt-4.1", "generation_params": {"temperature": 0.0}}
        
        # Load aggregators from cache
        with suppress_stdout():
            # Entity aggregator
            entity_cache_file = os.path.join(cache_base_dir, "entity", "entity_summaries.json")
            self.entity_aggregator = EntitySummaryAggregator(self.gsw, llm_config)
            self.entity_aggregator.precompute_summaries(cache_file=entity_cache_file)
            
            # Verb phrase aggregator
            verb_cache_file = os.path.join(cache_base_dir, "verbphrase", "verb_summaries.json")
            self.verb_aggregator = VerbSummaryAggregator(self.gsw, llm_config, embedding_model="voyage-3")
            self.verb_aggregator.precompute_summaries(cache_file=verb_cache_file)
            
            # Conversation aggregator
            self.conversation_aggregator = None
            if self.gsw.conversation_nodes:
                conv_cache_file = os.path.join(cache_base_dir, "conversation", "conversation_summaries.json")
                self.conversation_aggregator = ConversationSummaryAggregator(self.gsw, embedding_model="voyage-3")
                self.conversation_aggregator.precompute_summaries(cache_file=conv_cache_file)
        
        # Initialize QA system to get tools
        with suppress_stdout():
            self.qa_system = GSWQuestionAnswerer_Narrative(
                gsw=self.gsw,
                entity_aggregator=self.entity_aggregator,
                verb_aggregator=self.verb_aggregator,
                conversation_aggregator=self.conversation_aggregator,
                llm_config=llm_config_qa,
                include_verb_phrases=True,
                include_conversations=bool(self.conversation_aggregator),
                embedding_model="voyage-3",
                chunks_folder_path=chunks_folder_path,
                initial_context_size=10,
                reranker_size=5,
                context_retrieval_size=10,
                max_turns=10,
                verbose=True  # Enable verbose for tool output
            )
        
        # Get tools instance
        self.tools = self.qa_system.tools
        
        print("✅ Tool caller ready! You can now test tools directly.\n")
    
    def show_available_tools(self):
        """Display available tools and their usage."""
        print("🛠️  Available Tools:")
        print()
        
        print("1️⃣  search_entity_by_name")
        print("   Usage: search_entity_by_name <query> [k]")
        print("   Example: search_entity_by_name Chris 5")
        print()
        
        print("2️⃣  search_entity_summaries") 
        print("   Usage: search_entity_summaries <query> [k]")
        print("   Example: search_entity_summaries doctor 3")
        print()
        
        print("3️⃣  search_vp_summaries")
        print("   Usage: search_vp_summaries <query> [k]")
        print("   Example: search_vp_summaries conversation 5")
        print()
        
        print("4️⃣  search_conversation_summaries")
        print("   Usage: search_conversation_summaries <query> [k]")
        print("   Example: search_conversation_summaries doctor patient 3")
        print()
        
        print("5️⃣  get_detailed_conversation")
        print("   Usage: get_detailed_conversation <conversation_id>")
        print("   Example: get_detailed_conversation cv_0")
        print()
        
        print("6️⃣  search_chunks")
        print("   Usage: search_chunks <chunk_id1> [chunk_id2] [chunk_id3] ...")
        print("   Example: search_chunks 0_33 0_38")
        print()
        
        print("🔧 Commands:")
        print("   • 'tools' or 't' - Show this tools list")
        print("   • 'quit' or 'q' - Exit the program")
        print()
    
    def parse_and_call_tool(self, command: str):
        """Parse user command and call the appropriate tool."""
        parts = command.strip().split()
        if not parts:
            print("❌ Empty command. Type 'tools' to see available tools.")
            return
        
        tool_name = parts[0]
        args = parts[1:]
        
        try:
            if tool_name == "search_entity_by_name":
                if len(args) < 1:
                    print("❌ Usage: search_entity_by_name <query> [k]")
                    return
                query = " ".join(args[:-1]) if len(args) > 1 and args[-1].isdigit() else " ".join(args)
                k = int(args[-1]) if len(args) > 1 and args[-1].isdigit() else 5
                result = self.tools.search_entity_by_name(query, k, f"Manual test: {query}")
                
            elif tool_name == "search_entity_summaries":
                if len(args) < 1:
                    print("❌ Usage: search_entity_summaries <query> [k]")
                    return
                query = " ".join(args[:-1]) if len(args) > 1 and args[-1].isdigit() else " ".join(args)
                k = int(args[-1]) if len(args) > 1 and args[-1].isdigit() else 5
                result = self.tools.search_entity_summaries(query, k, f"Manual test: {query}")
                
            elif tool_name == "search_vp_summaries":
                if len(args) < 1:
                    print("❌ Usage: search_vp_summaries <query> [k]")
                    return
                query = " ".join(args[:-1]) if len(args) > 1 and args[-1].isdigit() else " ".join(args)
                k = int(args[-1]) if len(args) > 1 and args[-1].isdigit() else 5
                result = self.tools.search_vp_summaries(query, k, f"Manual test: {query}")
                
            elif tool_name == "search_conversation_summaries":
                if len(args) < 1:
                    print("❌ Usage: search_conversation_summaries <query> [k]")
                    return
                query = " ".join(args[:-1]) if len(args) > 1 and args[-1].isdigit() else " ".join(args)
                k = int(args[-1]) if len(args) > 1 and args[-1].isdigit() else 5
                result = self.tools.search_conversation_summaries(query, k, f"Manual test: {query}")
                
            elif tool_name == "get_detailed_conversation":
                if len(args) != 1:
                    print("❌ Usage: get_detailed_conversation <conversation_id>")
                    return
                conversation_id = args[0]
                result = self.tools.get_detailed_conversation(conversation_id, f"Manual test: {conversation_id}")
                
            elif tool_name == "search_chunks":
                if len(args) < 1:
                    print("❌ Usage: search_chunks <chunk_id1> [chunk_id2] [chunk_id3] ...")
                    return
                chunk_ids = args
                result = self.tools.search_chunks(chunk_ids, f"Manual test: {chunk_ids}")
                
            else:
                print(f"❌ Unknown tool: {tool_name}")
                print("Type 'tools' to see available tools.")
                return
            
            # Display results
            self.display_results(tool_name, result)
            
        except Exception as e:
            print(f"❌ Error calling {tool_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def display_results(self, tool_name: str, results):
        """Display tool results in a formatted way."""
        print(f"\n🔍 Results from {tool_name}:")
        print("=" * 60)
        
        if not results:
            print("   ❌ No results found")
            return
        
        for i, result in enumerate(results, 1):
            print(f"\n📄 Result {i}:")
            
            if isinstance(result, dict):
                # Display key fields prominently
                if "entity_name" in result:
                    print(f"   🏷️  Entity: {result['entity_name']} ({result.get('entity_id', 'unknown')})")
                    if "score" in result:
                        print(f"   📊 Score: {result['score']:.3f}")
                    if "summary" in result:
                        summary = result['summary'][:200] + "..." if len(result['summary']) > 200 else result['summary']
                        print(f"   📝 Summary: {summary}")
                    if "chunk_ids" in result:
                        print(f"   📁 Chunks: {result['chunk_ids']}")
                
                elif "vp_id" in result:
                    print(f"   🎬 VP: {result['vp_id']}")
                    if "score" in result:
                        print(f"   📊 Score: {result['score']:.3f}")
                    if "summary" in result:
                        summary = result['summary'][:200] + "..." if len(result['summary']) > 200 else result['summary']
                        print(f"   📝 Summary: {summary}")
                    if "chunk_ids" in result:
                        print(f"   📁 Chunks: {result['chunk_ids']}")
                
                elif "conversation_id" in result and "participants" in result:
                    # Detailed conversation
                    print(f"   💬 Conversation: {result['conversation_id']}")
                    if "participants" in result:
                        participants = [p['entity_name'] for p in result['participants']]
                        print(f"   👥 Participants: {', '.join(participants)}")
                    if "summary" in result:
                        summary = result['summary'][:200] + "..." if len(result['summary']) > 200 else result['summary']
                        print(f"   📝 Summary: {summary}")
                    if "topics_general" in result:
                        print(f"   🏷️  Topics: {result['topics_general']}")
                    if "chunk_ids" in result:
                        print(f"   📁 Chunks: {result['chunk_ids']}")
                
                elif "conversation_id" in result:
                    # Simple conversation summary
                    print(f"   💬 Conversation: {result['conversation_id']}")
                    if "score" in result:
                        print(f"   📊 Score: {result['score']:.3f}")
                    if "summary" in result:
                        summary = result['summary'][:200] + "..." if len(result['summary']) > 200 else result['summary']
                        print(f"   📝 Summary: {summary}")
                    if "chunk_ids" in result:
                        print(f"   📁 Chunks: {result['chunk_ids']}")
                
                elif "chunk_id" in result:
                    print(f"   📄 Chunk: {result['chunk_id']}")
                    if "text" in result:
                        text = result['text'][:300] + "..." if len(result['text']) > 300 else result['text']
                        print(f"   📝 Text: {text}")
                
                # Show full JSON for detailed inspection if requested
                print(f"   🔧 Full JSON: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}...")
            else:
                print(f"   📋 Raw result: {result}")
        
        print(f"\n📊 Total results: {len(results)}")
        print("=" * 60)


def main():
    """Main interactive function."""
    # Default paths (same as ask_questions.py)
    reconciled_file = "/mnt/SSD1/chenda/gsw-memory/test_output/reconciled_local/reconciled/doc_0_reconciled.json"
    cache_dir = "/mnt/SSD1/chenda/gsw-memory/test_output/summary"
    chunks_folder_path = "/mnt/SSD1/chenda/gsw-memory/test_output/gsw_output/chunks"
    
    # Check if files exist
    if not os.path.exists(reconciled_file):
        print(f"❌ Reconciled file not found: {reconciled_file}")
        print("Please run the reconciler first.")
        return
    
    # Initialize the tool caller
    try:
        tool_caller = ToolCaller(reconciled_file, cache_dir, chunks_folder_path)
    except Exception as e:
        print(f"❌ Failed to initialize tool caller: {e}")
        return
    
    # Show available tools
    tool_caller.show_available_tools()
    
    # Interactive loop
    while True:
        try:
            command = input("🛠️  Enter tool command (or 'tools'/'quit'): ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if command.lower() in ['tools', 't', 'help']:
                tool_caller.show_available_tools()
                continue
            
            if not command:
                print("Please enter a command. Type 'tools' to see available tools.")
                continue
            
            tool_caller.parse_and_call_tool(command)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()


if __name__ == "__main__":
    main() 