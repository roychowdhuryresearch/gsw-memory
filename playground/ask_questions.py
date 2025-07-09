#!/usr/bin/env python3
"""
Clean interface for asking questions with the GSW Memory Narrative QA system.

Usage:
    python examples/ask_questions.py
"""

import json
import os
import sys
import warnings
import contextlib
from pathlib import Path
from typing import Optional

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


class NarrativeQA:
    """Clean interface for narrative question answering."""
    
    def __init__(self, reconciled_file_path: str, cache_base_dir: str, verbose: bool = True, debug_log_dir: Optional[str] = None, chunks_folder_path: Optional[str] = None):
        """Initialize the QA system with reconciled data and cached summaries."""
        print("🚀 Initializing GSW Memory Narrative QA System...")
        
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
        
        # Initialize QA system
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
                verbose=verbose
            )
        
        self.verbose = verbose
        self.debug_log_dir = debug_log_dir
        
        print("✅ System ready! You can now ask questions.\n")
    
    def ask(self, question: str) -> dict:
        """Ask a single question and return the result."""
        print(f"❓ Question: {question}")
        if not self.verbose:
            print("🔄 Processing...")
        
        # Set up debug log file if debug logging is enabled
        if self.debug_log_dir:
            import time
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            safe_question = "".join(c for c in question if c.isalnum() or c in (' ', '-', '_'))[:50]
            debug_filename = f"debug_{timestamp}_{safe_question.replace(' ', '_')}.json"
            debug_log_file = os.path.join(self.debug_log_dir, debug_filename)
            
            # Update QA system debug log file
            self.qa_system.debug_log_file = debug_log_file
        else:
            self.qa_system.debug_log_file = None
        
        if self.verbose:
            result = self.qa_system.ask_narrative(question)
        else:
            with suppress_stdout():
                result = self.qa_system.ask_narrative(question)
        
        return result
    
    def ask_and_display(self, question: str):
        """Ask a question and display the answer cleanly."""
        result = self.ask(question)
        
        print(f"✅ Completed in {result.get('total_turns', 0)} agent turns")
        
        final_answer = result.get('final_answer')
        if final_answer:
            if isinstance(final_answer, dict):
                answer = final_answer.get('answer', 'No answer provided')
                sources = final_answer.get('sources', [])
                reasoning = final_answer.get('reasoning', '')
                
                print(f"\n📝 Answer:")
                print(f"   {answer}")
                
                if sources:
                    print(f"\n📚 Sources: {', '.join(sources)}")
                    
                if reasoning and reasoning.strip():
                    print(f"\n🧠 Reasoning:")
                    print(f"   {reasoning}")
            else:
                print(f"\n📝 Answer: {final_answer}")
        else:
            print("❌ No answer received")
        
        print("\n" + "="*50 + "\n")


def main():
    """Main interactive function."""
    # Default paths
    reconciled_file = "/mnt/SSD1/chenda/gsw-memory/test_output/reconciled_local/reconciled/doc_0_reconciled.json"
    cache_dir = "/mnt/SSD1/chenda/gsw-memory/test_output/summary"
    log_dir = "/mnt/SSD1/chenda/gsw-memory/test_output/log"
    chunks_folder_path = "/mnt/SSD1/chenda/gsw-memory/test_output/gsw_output/chunks"
    
    # Check if files exist
    if not os.path.exists(reconciled_file):
        print(f"❌ Reconciled file not found: {reconciled_file}")
        print("Please run the reconciler first.")
        return
    
    # Initialize the QA system
    try:
        qa = NarrativeQA(reconciled_file, cache_dir, verbose=True, debug_log_dir=log_dir, chunks_folder_path=chunks_folder_path)
    except Exception as e:
        print(f"❌ Failed to initialize QA system: {e}")
        return
    
    example_questions = [
        "The psychoanalyst gives Margaret some advice to help her husband, what is that advice?",
    ]
    
    print("💡 Example questions you can ask:")
    for i, q in enumerate(example_questions, 1):
        print(f"   {i}. {q}")
    print()
    print("🔧 Commands:")
    print("   • 'verbose' or 'v' - Toggle verbose mode to see agent's thinking process")
    print("   • 'quiet' - Disable verbose mode")
    print("   • 'debug on' - Enable debug logging (saves detailed agent steps to JSON files)")
    print("   • 'debug off' - Disable debug logging") 
    print("   • 'quit' or 'q' - Exit the program")
    print()
    
    # Interactive loop
    while True:
        try:
            question = input("🎯 Enter your question (or commands: 'quit'/'verbose'/'debug on'): ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() in ['verbose', 'v']:
                qa.verbose = not qa.verbose
                qa.qa_system.verbose = qa.verbose
                mode = "enabled" if qa.verbose else "disabled"
                print(f"🔧 Verbose mode {mode}")
                continue
            
            if question.lower() in ['quiet', 'silent']:
                qa.verbose = False
                qa.qa_system.verbose = False
                print("🔇 Quiet mode enabled")
                continue
            
            if question.lower() == 'debug on':
                if not qa.debug_log_dir:
                    qa.debug_log_dir = "./debug_logs"
                    os.makedirs(qa.debug_log_dir, exist_ok=True)
                print(f"🐛 Debug logging enabled - logs will be saved to: {qa.debug_log_dir}")
                continue
            
            if question.lower() == 'debug off':
                qa.debug_log_dir = None
                print("🔇 Debug logging disabled")
                continue
            
            if not question:
                print("Please enter a question.")
                continue
            
            qa.ask_and_display(question)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()


if __name__ == "__main__":
    main() 