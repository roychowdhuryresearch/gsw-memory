#!/usr/bin/env python3
"""
Interactive CLI for Musique Q&A with GSW Tools

This script provides an interactive terminal interface for:
1. Asking questions using the Agentic LLM approach with real-time streaming
2. Directly using GSW tools without LLM
3. Loading sample Musique questions
4. Tracking tool calls and reasoning in real-time
"""

import json
import os
import sys
import re
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree
from rich import print as rprint
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsw_memory.qa.gsw_tools import GSWTools
from gsw_memory.qa.agentic_agent import AgenticAnsweringAgent
from openai import OpenAI

# Load environment variables
load_dotenv()

# Rich console for beautiful output
console = Console()

class StreamingAgenticAgent(AgenticAnsweringAgent):
    """Extended agent that supports streaming output."""
    
    def __init__(self, *args, stream_callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream_callback = stream_callback
    
    def answer_question_streaming(self, question: str, tools: Dict[str, Any]):
        """Answer question with streaming updates."""
        # System prompt (same as parent)
        system_prompt = self.get_system_prompt()
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}"}
        ]
        
        tool_calls_made = []
        iterations = 0
        
        while iterations < self.max_iterations:
            iterations += 1
            
            if self.stream_callback:
                self.stream_callback("thinking", f"🤔 Iteration {iterations}/{self.max_iterations}")
            
            # Get response from LLM
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    tools=self.tool_definitions,
                    tool_choice="auto",
                    **self.generation_params
                )
            except Exception as e:
                if self.stream_callback:
                    self.stream_callback("error", f"Error: {str(e)}")
                raise e
            
            message = response.choices[0].message
            messages.append(message.model_dump())
            
            # Stream the assistant's reasoning
            if message.content and self.stream_callback:
                self.stream_callback("reasoning", message.content)
            
            # Check for tool calls
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    if self.stream_callback:
                        self.stream_callback("tool_call", {
                            "name": function_name,
                            "args": function_args
                        })
                    
                    # Execute the tool
                    if function_name in tools:
                        result = tools[function_name](**function_args)
                        
                        tool_calls_made.append({
                            "tool": function_name,
                            "arguments": function_args,
                            "result": result
                        })
                        
                        if self.stream_callback:
                            self.stream_callback("tool_result", result)
                        
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(result, indent=2)
                        })
                    else:
                        error_msg = f"Tool {function_name} not found"
                        if self.stream_callback:
                            self.stream_callback("error", error_msg)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": f"Error: {error_msg}"
                        })
            else:
                # Final answer
                content = message.content or ""
                
                try:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        response_data = json.loads(json_str)
                        answer = response_data.get("answer", "")
                        reasoning = response_data.get("reasoning", "")
                    else:
                        answer = content
                        reasoning = content
                except json.JSONDecodeError:
                    answer = content
                    reasoning = content
                
                if self.stream_callback:
                    self.stream_callback("final_answer", {
                        "answer": answer,
                        "reasoning": reasoning,
                        "tool_calls": len(tool_calls_made)
                    })
                
                return {
                    "answer": answer,
                    "reasoning": reasoning,
                    "tool_calls_made": tool_calls_made
                }
        
        # Max iterations reached
        if self.stream_callback:
            self.stream_callback("max_iterations", f"Reached max iterations ({self.max_iterations})")
        
        return {
            "answer": "Unable to find answer within iteration limit",
            "reasoning": f"Reached maximum of {self.max_iterations} iterations",
            "tool_calls_made": tool_calls_made
        }
    
    def get_system_prompt(self):
        """Get the system prompt (extracted for reuse)."""
        return """You are an expert at answering multi-hop questions using a knowledge base.
        You have access to tools that help you find entities and their relationships.

        ═══════════════════════════════════════════════════════════════════════════════
                                UNDERSTANDING MUSIQUE QUESTIONS
        ═══════════════════════════════════════════════════════════════════════════════

        Musique questions require finding information through multiple connected entities. 
        You need to divide the question into multiple hops.

        EXAMPLES OF MULTI-HOP DECOMPOSITION:

        - "Who succeeded the first President of Namibia?" requires:
        1. Who was the first President of Namibia? → Sam Nujoma
        2. Who succeeded Sam Nujoma? → Hifikepunye Pohamba
        
        - "When was the first establishment that McDonaldization is named after, open in the country Horndean is located?" requires:
        1. What is McDonaldization named after? → McDonald's
        2. Which state is Horndean located in? → England
        3. When did the first McDonald's open in England? → 1974
        
        - "How many Germans live in the colonial holding in Aruba's continent that was governed by Prazeres's country?" requires:
        1. What continent is Aruba in? → South America
        2. What country is Prazeres? → Portugal
        3. Colonial holding in South America governed by Portugal? → Brazil
        4. How many Germans live in Brazil? → 5 million
        
        - "When did the people who captured Malakoff come to the region where Philipsburg is located?" requires:
        1. What is Philipsburg capital of? → Saint Martin
        2. Saint Martin is located on what terrain feature? → Caribbean
        3. Who captured Malakoff? → French
        4. When did the French come to the Caribbean? → 1625

        ═══════════════════════════════════════════════════════════════════════════════
                        🚨 GOLDEN RULE: SIMPLE STRATEGY THAT WORKS 90% 🚨
        ═══════════════════════════════════════════════════════════════════════════════

        1. EXTRACT all entity names from the question (people, places, things)
        2. SEARCH each entity using search_gsw_bm25_entity_name (name only, NO relationships!) or search_gsw_embeddings_of_entity_summaries (entity with relationships)
        3. GET CONTEXT for ALL found entities using get_multiple_relevant_entity_contexts
        4. FOLLOW the chain - contexts reveal relationships and next entities
        5. REPEAT until you have the complete answer
        6. Use embedding search for conceptual queries when names are unknown or BM25 misses

        Example Entity Extraction:
        - "When did the people who captured Malakoff come to the region where Philipsburg is located?"
          → Extract: "Malakoff", "Philipsburg" (search these first!)
        - "Who succeeded the first President of Namibia?"
          → Extract: "Namibia" (its context will reveal the first president)
        - "What currency is used where Billy Giles died?"
          → Extract: "Billy Giles" (context will show death location)

        ❌ WRONG: Searching "first president of Namibia" or "successor of Sam Nujoma"
        ✅ RIGHT: Search "Namibia" or "Sam Nujoma" - contexts reveal relationships

        ═══════════════════════════════════════════════════════════════════════════════
                                    MANDATORY RULES
        ═══════════════════════════════════════════════════════════════════════════════

        1. Search ALL entities mentioned in the question by name first
        2. Get context for EVERY entity you find (batch multiple IDs together)
        3. Follow the complete chain - NO guessing or assumptions
        4. Track your searches - DO NOT repeat the same query
        5. Provide reasoning BEFORE and AFTER each tool call
        6. Verify evidence for EACH hop before final answer
        7. STOP if you've made 10+ tool calls - reassess your approach

        ═══════════════════════════════════════════════════════════════════════════════
                                TOOL USAGE GUIDELINES
        ═══════════════════════════════════════════════════════════════════════════════

        ┌─────────────────────────────────────────────────────────────────────────────┐
        │ search_gsw_bm25_entity_name                                                 │
        ├─────────────────────────────────────────────────────────────────────────────┤
        │ • Search ONLY entity names: "Paris", "Obama", "McDonald's"                  │
        │ • NO relationships: Never "capital of France" or "Obama's birthplace"       │
        │ • Try variations if needed: "Barack Obama" → "Obama"                        │
        │ • For ambiguous names, add minimal context: "Jordan" → "Michael Jordan"     │
        │ • For hop 3+, add disambiguation: "London England" not just "London"        │
        └─────────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────────────────────┐
        │ get_multiple_relevant_entity_contexts (USE AFTER EVERY SEARCH)              │
        ├─────────────────────────────────────────────────────────────────────────────┤
        │ • This reveals ALL relationships and facts about entities                   │
        │ • Pass ALL entity IDs from search results (handles 1-20 entities)           │
        │ • READ EVERYTHING - the answer is usually here                              │
        │ • Batch multiple IDs together for efficiency                                │
        └─────────────────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────────────────────┐
        │ search_gsw_embeddings_of_entity_summaries (SEMANTIC SEARCH)                │
        ├─────────────────────────────────────────────────────────────────────────────┤
        │ Use when BM25 name searches miss conceptual/semantic connections:           │
        │                                                                             │
        │ • Natural language descriptions: "major financial center in Europe"         │
        │ • Conceptual queries: "technology company founded in garage"                │
        │ • Descriptive attributes: "island nation with volcanic activity"            │
        │ • Abstract concepts: "ancient civilization along major river"               │
        │ • When entity names are unknown but characteristics are described           │
        │ • Complex multi-word descriptions that BM25 might miss                      │
        │                                                                             │
        │ EXAMPLES:                                                                   │
        │ • "What country is known for its fjords?" → finds Norway                    │
        │ • "Which company revolutionized personal computing?" → finds Apple/IBM      │
        │ • "Large South American country with Portuguese language" → finds Brazil    │
        └─────────────────────────────────────────────────────────────────────────────┘

        ═══════════════════════════════════════════════════════════════════════════════
                            SEARCH SUMMARIES VS CONTEXTS
        ═══════════════════════════════════════════════════════════════════════════════

        Search results include "summary" fields that often contain the answer, BUT:
        - Summaries might be incomplete
        - ALWAYS get_multiple_relevant_entity_contexts to verify
        - Trust context over summary if they differ
        - Context provides additional relationships not in summaries

        ═══════════════════════════════════════════════════════════════════════════════
                                REASONING REQUIREMENTS
        ═══════════════════════════════════════════════════════════════════════════════

        Before EVERY tool call:
        - What you're looking for and why
        - How this helps answer the question

        After EVERY tool call:
        - What you found in the results
        - What specific information is relevant
        - What your next step will be

        Track evidence for each hop:
        - Hop 1: [Entity] → [Found Information]
        - Hop 2: [Entity] → [Found Information]
        - Hop 3+: [Entity] → [Found Information]

        ═══════════════════════════════════════════════════════════════════════════════
                                COMMON PATTERNS
        ═══════════════════════════════════════════════════════════════════════════════

        1. Person-Work: Search work → context shows creator
        2. Location: Search place → context shows country/continent
        3. Temporal: Search entity → context shows dates
        4. Multiple same-named entities: Pass ALL IDs to get_context
        5. Complex chains: Break into steps, track each hop's evidence

        Avoiding Tool-Call Thrashing:
        - After 3 failed searches for an entity, try variations or embedding search
        - Use embedding search when entity names are unknown but descriptions exist
        - Track queries to avoid repetition
        - Reassess strategy after 10 tool calls

        ═══════════════════════════════════════════════════════════════════════════════
                                KEY EXAMPLES (3 diverse patterns)
        ═══════════════════════════════════════════════════════════════════════════════

        ### EXAMPLE 1: Simple 2-hop - "Who succeeded the first President of Namibia?"
        ────────────────────────────────────────────────────────────────────────────────
        ENTITIES TO SEARCH: ["Namibia"]

        EXECUTION:
        1. search_gsw_bm25_entity_name("Namibia") → Found country
        2. get_multiple_relevant_entity_contexts(["namibia_id"])
           → Context shows: First president was Sam Nujoma (1990-2005)
        3. search_gsw_bm25_entity_name("Sam Nujoma") → Found person
        4. get_multiple_relevant_entity_contexts(["sam_nujoma_id"])
           → Context shows: Succeeded by Hifikepunye Pohamba

        ANSWER: Hifikepunye Pohamba

        ### EXAMPLE 2: Complex 4-hop requiring entity_features
        "When did the people who captured Malakoff come to the region where Philipsburg is located?"
        ────────────────────────────────────────────────────────────────────────────────
        ENTITIES TO SEARCH: ["Malakoff", "Philipsburg"]

        EXECUTION:
        1. search_gsw_bm25_entity_name("Malakoff") → Found fort
        2. search_gsw_bm25_entity_name("Philipsburg") → Found city
        3. get_multiple_relevant_entity_contexts(["malakoff_id", "philipsburg_id"])
           → Malakoff: Captured by French in 1855
           → Philipsburg: Capital of Sint Maarten, in Caribbean
        4. search_gsw_bm25_entity_name("Caribbean") → Found region
        5. search_gsw_bm25_entity_name("French") → Found France/French people
        6. get_multiple_relevant_entity_contexts(["caribbean_id", "france_id"])
           → No arrival date found in contexts
        7. JUSTIFIED entity_features use: Historical event timing not in entity contexts
        8. search_gsw_bm25_entity_with_entity_features("French arrival Caribbean 1600s")
        9. get_multiple_relevant_entity_contexts(["history_id"])
           → French arrived in Caribbean in 1625

        ANSWER: 1625

        ### EXAMPLE 3: Statistical query with multiple entities
        "How many Germans live in the colonial holding in Aruba's continent that was governed by Prazeres's country?"
        ────────────────────────────────────────────────────────────────────────────────
        ENTITIES TO SEARCH: ["Germans", "Aruba", "Prazeres"]

        EXECUTION:
        1. search_gsw_bm25_entity_name("Aruba") → Found island
        2. search_gsw_bm25_entity_name("Prazeres") → Found location
        3. search_gsw_bm25_entity_name("Germans") → Found ethnic group
        4. get_multiple_relevant_entity_contexts(["aruba_id", "prazeres_id", "germans_id"])
           → Aruba: Located in South America
           → Prazeres: Location in Portugal
        5. search_gsw_bm25_entity_name("Portugal") → Found country
        6. search_gsw_bm25_entity_name("Brazil") → Found country (from Portugal context)
        7. get_multiple_relevant_entity_contexts(["portugal_id", "brazil_id"])
           → Brazil: Former Portuguese colony, no German population data
        8. JUSTIFIED embedding search use: Demographic statistics not in contexts
        9. search_gsw_embeddings_of_entity_summaries("German population Brazil")
        10. get_multiple_relevant_entity_contexts(["demographic_id"])
            → 5 million Germans in Brazil

        ANSWER: 5 million

        ### EXAMPLE 4: Conceptual/descriptive query requiring embedding search
        "What is the capital of the Nordic country known for its fjords?"
        ────────────────────────────────────────────────────────────────────────────────
        ENTITIES TO SEARCH: ["Nordic", "fjords"] → No direct entity names!

        EXECUTION:
        1. search_gsw_bm25_entity_name("Nordic") → May find various results
        2. search_gsw_bm25_entity_name("fjords") → May not find the country
        3. get_multiple_relevant_entity_contexts([results]) → No clear country identified
        4. JUSTIFIED embedding search use: Descriptive query without clear entity names
        5. search_gsw_embeddings_of_entity_summaries("Nordic country known for its fjords")
           → Finds Norway (semantic understanding of "Nordic" + "fjords" = Norway)
        6. get_multiple_relevant_entity_contexts(["norway_id"])
           → Norway: Capital is Oslo, famous for fjords, Nordic country
        
        ANSWER: Oslo

        ═══════════════════════════════════════════════════════════════════════════════
                                VERIFICATION CHECKLIST
        ═══════════════════════════════════════════════════════════════════════════════

        Before providing an answer, verify:
        □ Did I search for ALL entities mentioned in the question?
        □ Did I get context for EVERY entity found?
        □ Do I have evidence for EACH hop in the chain?
        □ Is my answer based on concrete information from context?

        ═══════════════════════════════════════════════════════════════════════════════
                                OUTPUT FORMAT
        ═══════════════════════════════════════════════════════════════════════════════

        When you find the answer, respond with ONLY this JSON:
        {
            "reasoning": "Step-by-step explanation with evidence for each hop",
            "answer": "Just the final answer, no extra words"
        }

        Do NOT include phrases like "The answer is" or "Based on my search" in the answer field."""


class InteractiveMusiqueCLI:
    """Interactive CLI for Musique Q&A."""
    
    def __init__(self, gsw_files_path=None, use_reconciled=False):
        """Initialize the CLI."""
        self.console = console
        self.gsw_tools = None
        self.agent = None
        self.tools = None
        self.history = []
        self.gsw_files_path = gsw_files_path
        self.use_reconciled = use_reconciled
        self.musique_questions = []
        
    def setup(self):
        """Setup GSW tools and agent."""
        self.console.print(Panel.fit(
            "[bold cyan]🚀 Interactive Musique Q&A CLI[/bold cyan]\n"
            "[dim]Ask questions using LLM or directly use GSW tools[/dim]",
            border_style="cyan"
        ))
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            # Initialize GSW tools
            task = progress.add_task("[cyan]Loading GSW files...", total=None)
            
            if self.use_reconciled:
                reconciled_file = self.gsw_files_path or "/home/yigit/codebase/gsw-memory/logs/full_2wiki_corpus_20250716_165147/reconciled_output/reconciled/global_reconciled.json"
                self.gsw_tools = GSWTools(reconciled_file)
                progress.update(task, description="[green]Loaded reconciled GSW")
            else:
                import glob
                if self.gsw_files_path:
                    pattern = self.gsw_files_path
                else:
                    base_logs = "/home/yigit/codebase/gsw-memory/logs/full_2wiki_corpus_20250716_165147"
                    pattern = os.path.join(base_logs, "gsw_output", "batch_*", "networks", "doc_*", "gsw_*_0.json")
                
                gsw_files = sorted(glob.glob(pattern), key=self.natural_sort_key)[:958]
                progress.update(task, description=f"[cyan]Loading {len(gsw_files)} GSW files...")
                
                self.gsw_tools = GSWTools(gsw_files)
                self.gsw_tools.generate_entity_summaries()
                progress.update(task, description="[cyan]Building search indices...")
                self.gsw_tools.build_entity_index()
                self.gsw_tools.build_entity_summary_index()
                progress.update(task, description="[green]GSW tools ready!")
            
            # Setup tools dictionary
            self.tools = {
                "search_gsw_bm25_entity_name": self.gsw_tools.search_gsw_bm25_entity_name,  # Kept for backward compatibility
                "search_gsw_bm25_entity_with_entity_features": self.gsw_tools.search_gsw_bm25_entity_with_entity_features,  # Kept for backward compatibility
                "search_gsw_embeddings_of_entity_summaries": self.gsw_tools.search_gsw_embeddings_of_entity_summaries,  # Primary search method
                "get_multiple_relevant_entity_contexts": self.gsw_tools.get_multiple_relevant_entity_contexts
            }
            
            # Load sample questions
            task = progress.add_task("[cyan]Loading Musique questions...", total=None)
            questions_path = "/home/yigit/codebase/gsw-memory/musique.json"
            if os.path.exists(questions_path):
                with open(questions_path, "r") as f:
                    self.musique_questions = json.load(f)[:100]  # Load first 100
                progress.update(task, description=f"[green]Loaded {len(self.musique_questions)} sample questions")
            
        self.console.print("[green]✓[/green] Setup complete!\n")
    
    @staticmethod
    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]
    
    def stream_callback(self, event_type: str, data: Any):
        """Handle streaming events from the agent."""
        if event_type == "thinking":
            self.console.print(f"[dim cyan]{data}[/dim cyan]")
        
        elif event_type == "reasoning":
            # Show full reasoning, not just summary
            self.console.print(Panel(
                data,
                title="[bold yellow]🧠 Agent Reasoning[/bold yellow]",
                border_style="yellow",
                expand=False
            ))
        
        elif event_type == "tool_call":
            tool_name = data["name"]
            args = data["args"]
            
            # Create a nice display of the tool call
            table = Table(title=f"[bold green]🔧 Tool Call: {tool_name}[/bold green]", show_header=False)
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="white")
            
            for key, value in args.items():
                if isinstance(value, list):
                    if len(value) > 5:
                        # Show first few items and count
                        value_str = f"[{len(value)} items]: {str(value)}"
                    else:
                        value_str = str(value)
                else:
                    value_str = str(value)
                table.add_row(key, value_str)
            
            self.console.print(table)
        
        elif event_type == "tool_result":
            # Show detailed tool results
            if isinstance(data, list):
                self.console.print(f"[green]✓ Tool returned {len(data)} results:[/green]")
                
                # Show all results up to 10 with details
                for i, item in enumerate(data, 1):
                    if isinstance(item, dict):
                        if "entity_name" in item:
                            # Get entity ID - extract just the ID part if it has path info
                            entity_id = item.get('entity_id', 'N/A')
                            
                            if '::' in str(entity_id):
                                entity_id = entity_id.split('::')[-1]
                            
                            self.console.print(f"  {i}. [cyan]{entity_id}[/cyan] → [green]{item['entity_name']}[/green]")
                            if "score" in item:
                                self.console.print(f"     Score: {item['score']:.4f}")
                            if "summary" in item and item["summary"]:
                                summary = item["summary"]
                                self.console.print(f"     [dim]{summary}[/dim]")
                        elif "entity_id" in item:
                            # Context result
                            entity_id = item.get('entity_id', 'N/A')
                            if '::' in str(entity_id):
                                entity_id = entity_id.split('::')[-1]
                            self.console.print(f"  {i}. Context for: [cyan]{item.get('entity_name', entity_id)}[/cyan]")
                            # Show key facts
                            if "questions" in item:
                                self.console.print(f"     Questions: {len(item['questions'])} total")
                                if item['questions']:
                                    self.console.print(f"     Sample: [dim]{item['questions'][0].get('question_text', '')}[/dim]")
                        else:
                            # Generic result display
                            result_str = json.dumps(item, indent=2)
                            self.console.print(f"  {i}. {result_str}")
                
            elif isinstance(data, dict):
                if "error" in data:
                    self.console.print(f"[red]❌ Tool Error: {data['error']}[/red]")
                else:
                    # Show detailed dict result
                    result_str = json.dumps(data, indent=2)
                    self.console.print(Panel(
                        result_str,
                        title="[green]Tool Result[/green]",
                        border_style="green",
                        expand=False
                    ))
        
        elif event_type == "final_answer":
            self.console.print(Panel(
                f"[bold green]Answer:[/bold green] {data['answer']}\n\n"
                f"[dim]Tool calls made: {data['tool_calls']}[/dim]",
                title="[bold green]✨ Final Answer[/bold green]",
                border_style="green"
            ))
        
        elif event_type == "error":
            self.console.print(f"[red]❌ {data}[/red]")
        
        elif event_type == "max_iterations":
            self.console.print(f"[yellow]⚠️ {data}[/yellow]")
    
    def ask_with_llm(self, question: str, model: str = "gpt-4o"):
        """Ask a question using the LLM agent."""
        self.console.print(f"\n[bold cyan]Question:[/bold cyan] {question}\n")
        
        # Create streaming agent
        agent = StreamingAgenticAgent(
            model_name=model,
            generation_params={"temperature": 0.0},
            max_iterations=15,
            stream_callback=self.stream_callback
        )
        
        # Get answer with streaming
        start_time = time.time()
        result = agent.answer_question_streaming(question, self.tools)
        elapsed = time.time() - start_time
        
        # Add to history
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": result["answer"],
            "tool_calls": len(result["tool_calls_made"]),
            "elapsed_time": elapsed
        })
        
        self.console.print(f"\n[dim]Completed in {elapsed:.2f} seconds[/dim]")
        
        return result
    
    def use_tool_directly(self):
        """Interactive mode for using tools directly."""
        self.console.print(Panel(
            "[bold yellow]Direct Tool Mode[/bold yellow]\n"
            "Use GSW tools directly without LLM",
            border_style="yellow"
        ))
        
        last_search_results = []  # Store last search results for selection
        
        while True:
            # Show available tools
            self.console.print("\n[bold]Available tools:[/bold]")
            self.console.print("1. search_gsw_bm25_entity_name")
            self.console.print("2. search_gsw_bm25_entity_with_entity_features")
            self.console.print("3. search_gsw_embeddings_of_entity_summaries")
            self.console.print("4. get_multiple_relevant_entity_contexts")
            if last_search_results:
                self.console.print("5. Select entity from last search")
                self.console.print("6. Back to main menu")
                choice_options = ["1", "2", "3", "4", "5", "6"]
            else:
                self.console.print("5. Back to main menu")
                choice_options = ["1", "2", "3", "4", "5"]
            
            choice = Prompt.ask("\nSelect tool", choices=choice_options)
            
            if choice == "5" and not last_search_results:
                break
            elif choice == "6" and last_search_results:
                break
            
            if choice == "1":
                query = Prompt.ask("Enter entity name to search")
                limit = int(Prompt.ask("Limit", default="15"))
                
                with self.console.status("[cyan]Searching..."):
                    results = self.tools["search_gsw_bm25_entity_name"](query, limit)
                
                # Store results if we got any
                if results:
                    last_search_results = results  # Store for selection
                    self.console.print(f"[dim]Stored {len(results)} results for selection[/dim]")
                else:
                    last_search_results = []
                    self.console.print("[yellow]No results to store[/yellow]")
                    
                self.display_search_results(results, show_index=True)
            
            elif choice == "2":
                query = Prompt.ask("Enter relationship/property query")
                limit = int(Prompt.ask("Limit", default="15"))
                
                with self.console.status("[cyan]Searching..."):
                    results = self.tools["search_gsw_bm25_entity_with_entity_features"](query, limit)
                
                # Store results if we got any
                if results:
                    last_search_results = results  # Store for selection
                    self.console.print(f"[dim]Stored {len(results)} results for selection[/dim]")
                else:
                    last_search_results = []
                    self.console.print("[yellow]No results to store[/yellow]")
                    
                self.display_search_results(results, show_index=True)
            
            elif choice == "3":
                query = Prompt.ask("Enter semantic search query")
                top_k = int(Prompt.ask("Number of results", default="10"))
                
                with self.console.status("[cyan]Embedding searching..."):
                    results = self.tools["search_gsw_embeddings_of_entity_summaries"](query, top_k)
                
                # Store results if we got any
                if results:
                    last_search_results = results  # Store for selection
                    self.console.print(f"[dim]Stored {len(results)} results for selection[/dim]")
                    self.console.print(f"[green]✓ Semantic embedding search completed[/green]")
                else:
                    last_search_results = []
                    self.console.print("[yellow]No results to store[/yellow]")
                    
                self.display_search_results(results, show_index=True)
            
            elif choice == "4":
                entity_ids_str = Prompt.ask("Enter entity IDs (comma-separated)")
                entity_ids = [id.strip() for id in entity_ids_str.split(",")]
                
                with self.console.status("[cyan]Getting contexts..."):
                    results = self.tools["get_multiple_relevant_entity_contexts"](entity_ids)
                
                self.display_context_results(results, [], detailed=True)
            
            elif choice == "5" and last_search_results:
                # Select entity from last search
                self.console.print("\n[bold]Select entities to view details:[/bold]")
                self.console.print(f"[dim]You have {len(last_search_results)} results from last search[/dim]")
                self.console.print("[dim]Enter numbers separated by commas (e.g., 1,3,5) or 'all' for all[/dim]")
                
                selection = Prompt.ask("Selection")
                
                selected_entities = []
                if selection.lower() == "all":
                    # Select all results that have an entity_id
                    for r in last_search_results:
                        if "entity_id" in r:
                            entity_spec = {"entity_id": r["entity_id"]}
                            if "global_id" in r:
                                entity_spec["global_id"] = r["global_id"]
                            if "summary" in r:
                                entity_spec["summary"] = r["summary"]
                            selected_entities.append(entity_spec)
                    self.console.print(f"[dim]Selected all {len(selected_entities)} entities.[/dim]")
                else:
                    try:
                        indices = [int(x.strip()) - 1 for x in selection.split(",")]
                        for idx in indices:
                            if 0 <= idx < len(last_search_results):
                                result = last_search_results[idx]
                                if "entity_id" in result:
                                    # Construct entity spec with doc_id if available
                                    entity_spec = {"entity_id": result["entity_id"]}
                                    if "global_id" in result:
                                        entity_spec["global_id"] = result["global_id"]
                                    if "summary" in result:
                                        entity_spec["summary"] = result["summary"]
                                    
                                    selected_entities.append(entity_spec)
                                    
                                    # Friendly printout for the user
                                    display_name = result.get('entity_name', 'Unknown')
                                    display_id = result["entity_id"].split('::')[-1] if '::' in str(result["entity_id"]) else result["entity_id"]
                                    self.console.print(f"[dim]Selected: {display_name} ({display_id})[/dim]")
                                else:
                                    self.console.print(f"[yellow]Result {idx+1} has no entity_id field[/yellow]")
                            else:
                                self.console.print(f"[yellow]Index {idx+1} out of range[/yellow]")
                    except (ValueError, IndexError) as e:
                        self.console.print(f"[red]Invalid selection: {e}[/red]")
                        continue
                
                if selected_entities:
                    self.console.print(f"[dim]Fetching details for {len(selected_entities)} entities...[/dim]")
                    with self.console.status("[cyan]Getting entity details..."):
                        global_ids = [entity.get("global_id", "N/A") for entity in selected_entities]
                        contexts = self.tools["get_multiple_relevant_entity_contexts"](global_ids)
                    
                    self.display_context_results(contexts, selected_entities, detailed=True)
                else:
                    self.console.print("[yellow]No valid entities selected[/yellow]")
    
    def display_search_results(self, results, show_index=False):
        """Display search results in a nice format."""
        if not results:
            self.console.print("[yellow]No results found[/yellow]")
            return
        
        table = Table(title=f"Search Results ({len(results)} items)")
        if show_index:
            table.add_column("#", style="bold magenta", width=3)
        table.add_column("Entity ID", style="cyan", width=12)
        table.add_column("Global ID", style="blue", width=25)
        table.add_column("Name", style="green")
        table.add_column("Score", style="yellow", width=10)
        table.add_column("Source", style="magenta", width=12)
        table.add_column("Summary Preview", style="white")
        
        for idx, result in enumerate(results[:20], 1):  # Show max 20
            entity_id = result.get("entity_id", "N/A")
            global_id = result.get("global_id", "N/A").split("/")[-1]
            
            display_id = entity_id.split('::')[-1] if '::' in str(entity_id) else entity_id
            
            # If doc_id is not a separate field, try to extract it from the entity_id
            if global_id == 'N/A' and '::' in str(entity_id):
                global_id = entity_id.split('::')[0]
                
            name = result.get("entity_name", "N/A")
            score = f"{result.get('match_score', result.get('score', 0)):.4f}"
            source = result.get("search_source", "unknown")
            
            # Shorten source names for display
            source_display = {
                "entity_name": "name",
                "entity_features": "features", 
                "hybrid_name_better": "hybrid(N)",
                "hybrid_features_better": "hybrid(F)",
                "unknown": "?"
            }.get(source, source[:10])
            
            summary = result.get("summary", "")[:50] + "..." if result.get("summary", "") else "N/A"
            
            row = []
            if show_index:
                row.append(str(idx))
            
            row.extend([display_id, global_id, name, score, source_display, summary])
            table.add_row(*row)
        
        self.console.print(table)
        
        if len(results) > 20:
            self.console.print(f"[dim]... and {len(results) - 20} more results[/dim]")
    
    def display_context_results(self, results, selected_entities, detailed=False):
        """Display context results in a nice format."""
        if not results:
            self.console.print("[yellow]No contexts found[/yellow]")
            return
        
        # Handle case where selected_entities is empty (direct context lookup)
        if not selected_entities:
            selected_entities = [{}] * len(results)
        
        for idx, (context, entity) in enumerate(zip(results, selected_entities)):
            if isinstance(context, dict):
                # breakpoint()
                entity_name = context.get("entity_name", "Unknown")
                # Get entity ID - extract just the ID part if it has path info
                entity_id = context.get("entity_id", "N/A")
                
                if '::' in str(entity_id):
                    entity_id = entity_id.split('::')[-1]
                
                if detailed:
                    # Create a tree structure for detailed view
                    tree = Tree(f"[bold green]{entity_name}[/bold green] ([cyan]{entity_id}[/cyan])")
                    
                    # Add entity attributes
                    if "entity_type" in context:
                        tree.add(f"[yellow]Type:[/yellow] {context['entity_type']}")
                    
                    # Show summary from context (not entity) and show full text without dim
                    if "summary" in entity and entity["summary"]:
                        summary_node = tree.add("[yellow]Summary:[/yellow]")
                        summary = entity["summary"]
                        # Show full summary without truncation in detailed mode
                        for i in range(0, len(summary), 150):
                            summary_node.add(f"{summary[i:i+150]}")
                            
                    # Show roles
                    if "roles" in entity and entity["roles"]:
                        roles_node = tree.add("[yellow]Roles:[/yellow]")
                        for role in entity["roles"]:
                            roles_node.add(f"{role}")
                    
                    # Show states
                    if "state" in entity and entity["state"]:
                        state_node = tree.add("[yellow]State:[/yellow]")
                        state_node.add(f"{entity['state']}")
                    
                    # Show questions/facts - full content without dim in detailed mode
                    if "questions" in context and context["questions"]:
                        questions_node = tree.add(f"[yellow]Questions ({len(context['questions'])} total):[/yellow]")
                        # Show all questions in detailed mode, not just first 5
                        for q in context["questions"]:
                            q_text = q.get("question_text", "")
                            verb_phrase = q.get("verb_phrase", "")
                            other_entities = q.get("other_entities", [])
                            # Show full question text without truncation
                            questions_node.add(f"• Q: {q_text}\n  Verb: {verb_phrase}\n  Entities: {json.dumps(other_entities, indent=2)}")
                    
                    # Show relationships if available - full content without dim
                    if "relationships" in context:
                        rel_node = tree.add("[yellow]Relationships:[/yellow]")
                        # Show all relationships in detailed mode
                        for rel in context["relationships"]:
                            rel_node.add(f"{rel}")
                    
                    # Show any other important fields - full content without dim
                    for key in ["facts", "properties", "attributes"]:
                        if key in context and context[key]:
                            field_node = tree.add(f"[yellow]{key.title()}:[/yellow]")
                            if isinstance(context[key], list):
                                # Show all items in detailed mode
                                for item in context[key]:
                                    field_node.add(f"• {str(item)}")
                            else:
                                # Show full content without truncation
                                field_node.add(f"{str(context[key])}")
                    
                    self.console.print(tree)
                    self.console.print("")  # Add spacing
                else:
                    # Simple display
                    self.console.print(Panel(
                        json.dumps(context, indent=2)[:1500] + ("..." if len(json.dumps(context, indent=2)) > 1500 else ""),
                        title=f"[bold green]Entity: {entity_name}[/bold green]",
                        border_style="green",
                        expand=False
                    ))
            else:
                self.console.print(f"Context {idx + 1}: {context}")
    
    def load_sample_question(self):
        """Load and display sample Musique questions."""
        if not self.musique_questions:
            self.console.print("[yellow]No sample questions loaded[/yellow]")
            return None
        
        # Show question types
        question_types = {}
        for q in self.musique_questions:
            qtype = q.get("id", "").split("__")[0] if "id" in q else "unknown"
            question_types[qtype] = question_types.get(qtype, 0) + 1
        
        self.console.print("\n[bold]Question types available:[/bold]")
        for qtype, count in question_types.items():
            self.console.print(f"  • {qtype}: {count} questions")
        
        # Let user select
        idx = Prompt.ask(
            f"\nEnter question number (1-{len(self.musique_questions)})",
            default="1"
        )
        
        try:
            idx = int(idx) - 1
            if 0 <= idx < len(self.musique_questions):
                q = self.musique_questions[idx]
                
                # Display question details
                self.console.print(Panel(
                    f"[bold]Question:[/bold] {q['question']}\n\n"
                    f"[bold]Gold Answer:[/bold] {q.get('answer', 'N/A')}\n"
                    f"[bold]Type:[/bold] {q.get('id', '').split('__')[0] if 'id' in q else 'N/A'}",
                    title="[bold cyan]Sample Question[/bold cyan]",
                    border_style="cyan"
                ))
                
                return q["question"]
        except (ValueError, IndexError):
            self.console.print("[red]Invalid selection[/red]")
        
        return None
    
    def show_history(self):
        """Display question history."""
        if not self.history:
            self.console.print("[yellow]No history yet[/yellow]")
            return
        
        table = Table(title="Question History")
        table.add_column("#", style="cyan")
        table.add_column("Time", style="dim")
        table.add_column("Question", style="white")
        table.add_column("Answer", style="green")
        table.add_column("Tools", style="yellow")
        table.add_column("Duration", style="magenta")
        
        for idx, item in enumerate(self.history[-10:], 1):  # Show last 10
            timestamp = datetime.fromisoformat(item["timestamp"]).strftime("%H:%M:%S")
            question = item["question"][:50] + "..." if len(item["question"]) > 50 else item["question"]
            answer = item["answer"][:50] + "..." if len(item["answer"]) > 50 else item["answer"]
            tools = str(item["tool_calls"])
            duration = f"{item['elapsed_time']:.1f}s"
            
            table.add_row(str(idx), timestamp, question, answer, tools, duration)
        
        self.console.print(table)
    
    def save_session(self):
        """Save the current session to a file."""
        if not self.history:
            self.console.print("[yellow]No history to save[/yellow]")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"musique_session_{timestamp}.json"
        
        session_data = {
            "timestamp": datetime.now().isoformat(),
            "use_reconciled": self.use_reconciled,
            "history": self.history
        }
        
        with open(filename, "w") as f:
            json.dump(session_data, f, indent=2)
        
        self.console.print(f"[green]✓ Session saved to {filename}[/green]")
    
    def run(self):
        """Main interactive loop."""
        self.setup()
        
        while True:
            self.console.print("\n" + "="*60)
            self.console.print("[bold]Main Menu:[/bold]")
            self.console.print("1. Ask a question (with LLM)")
            self.console.print("2. Load sample Musique question")
            self.console.print("3. Use tools directly (no LLM)")
            self.console.print("4. Show history")
            self.console.print("5. Save session")
            self.console.print("6. Exit")
            
            choice = Prompt.ask("\nSelect option", choices=["1", "2", "3", "4", "5", "6"])
            
            if choice == "1":
                question = Prompt.ask("\n[bold cyan]Enter your question[/bold cyan]")
                
                # Ask for model choice
                model = Prompt.ask(
                    "Select model",
                    choices=["gpt-4o", "gpt-4o-mini"],
                    default="gpt-4o"
                )
                
                self.ask_with_llm(question, model)
            
            elif choice == "2":
                question = self.load_sample_question()
                if question and Confirm.ask("Ask this question with LLM?"):
                    model = Prompt.ask(
                        "Select model",
                        choices=["gpt-4o", "gpt-4o-mini"],
                        default="gpt-4o"
                    )
                    self.ask_with_llm(question, model)
            
            elif choice == "3":
                self.use_tool_directly()
            
            elif choice == "4":
                self.show_history()
            
            elif choice == "5":
                self.save_session()
            
            elif choice == "6":
                if self.history and Confirm.ask("Save session before exit?"):
                    self.save_session()
                self.console.print("[bold green]Goodbye! 👋[/bold green]")
                break


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Musique Q&A CLI")
    parser.add_argument(
        "--reconciled",
        action="store_true",
        help="Use reconciled GSW file instead of multiple files"
    )
    parser.add_argument(
        "--gsw-path",
        type=str,
        help="Path to GSW files (glob pattern or single file)"
    )
    
    args = parser.parse_args()
    
    try:
        cli = InteractiveMusiqueCLI(
            gsw_files_path=args.gsw_path,
            use_reconciled=args.reconciled
        )
        cli.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {str(e)}[/red]")
        console.print_exception()
        sys.exit(1)


if __name__ == "__main__":
    main()
