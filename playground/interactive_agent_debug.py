#!/usr/bin/env python
"""
Interactive debugging tool for GSW agentic agent.

This tool allows you to:
- Test individual questions interactively
- Inspect tool calls and responses
- Debug GSW search results
- Test different configurations
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import sys
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich.json import JSON
from rich.syntax import Syntax
from rich import print as rprint

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.gsw_memory.qa.agentic_agent import AgenticAnsweringAgent
from src.gsw_memory.qa.gsw_tools import GSWTools


console = Console()


class InteractiveAgentDebugger:
    """Interactive debugger for the agentic agent."""
    
    def __init__(self, gsw_dir: str, model_name: str = "gpt-4o", use_multi_file: bool = False, search_method: str = "bm25"):
        """Initialize the debugger with GSW directory."""
        self.gsw_dir = Path(gsw_dir)
        self.model_name = model_name
        self.search_method = search_method
        
        # Initialize GSW tools
        console.print("[bold blue]Loading GSW tools...[/bold blue]")
        
        if use_multi_file:
            # Load individual GSW files from subdirectories
            import glob
            gsw_files_pattern = str(self.gsw_dir / "gsw_output_global_ids" / "networks" / "doc_*" / "gsw_*_0.json")
            gsw_files = glob.glob(gsw_files_pattern)
            if not gsw_files:
                raise ValueError(f"No GSW JSON files found with pattern: {gsw_files_pattern}")
            console.print(f"[cyan]Loading {len(gsw_files)} individual GSW files...[/cyan]")
            self.gsw_tools = GSWTools(gsw_files)
        else:
            # Use reconciled file
            reconciled_file = self.gsw_dir / "reconciled_output" / "reconciled" / "global_reconciled.json"
            if not reconciled_file.exists():
                raise ValueError(f"Reconciled file not found: {reconciled_file}")
            console.print(f"[cyan]Loading reconciled GSW from {reconciled_file.name}[/cyan]")
            self.gsw_tools = GSWTools(str(reconciled_file))
        
        # Initialize agent
        self.agent = AgenticAnsweringAgent(
            model_name=model_name,
            generation_params={"temperature": 1.0},
            max_iterations=10
        )
        
        # Create tools mapping
        # Build search index
        console.print("[bold blue]Building search index...[/bold blue]")
        self.gsw_tools.build_index()
        
        # Configure tools based on search method
        if search_method == "embeddings":
            self.tools = {
                "search_gsw_entity_embeddings": self.gsw_tools.search_gsw_entity_embeddings,
                "get_multiple_entity_contexts": self.gsw_tools.get_multiple_entity_contexts
            }
        else:  # bm25
            self.tools = {
                "search_gsw_bm25": self.gsw_tools.search_gsw_bm25,
                "get_multiple_entity_contexts": self.gsw_tools.get_multiple_entity_contexts
            }
        
        console.print(f"[bold green]✓ Loaded GSW from {self.gsw_dir}[/bold green]")
        console.print(f"[bold green]✓ Using model: {model_name}[/bold green]")
        console.print(f"[bold green]✓ Search method: {search_method}[/bold green]")
    
    def display_tool_call(self, tool_call: Dict[str, Any], index: int):
        """Display a single tool call in a formatted way."""
        tool_name = tool_call["tool"]
        args = tool_call["arguments"]
        result = tool_call["result"]
        
        # Create panel for tool call
        panel_title = f"[bold cyan]Tool Call #{index + 1}: {tool_name}[/bold cyan]"
        
        # Format arguments
        args_table = Table(show_header=True, header_style="bold magenta")
        args_table.add_column("Parameter", style="cyan")
        args_table.add_column("Value", style="green")
        
        for key, value in args.items():
            if isinstance(value, list):
                value_str = ", ".join(str(v) for v in value)
            else:
                value_str = str(value)
            args_table.add_row(key, value_str)
        
        console.print(Panel(args_table, title=panel_title, expand=False))
        
        # Display results based on tool type
        if tool_name in ["search_gsw_bm25", "search_gsw_embeddings"]:
            self._display_search_results(result)
        elif tool_name == "get_multiple_entity_contexts":
            self._display_entity_contexts(result)
    
    def _display_search_results(self, results: list):
        """Display search results in a formatted table."""
        if not results:
            console.print("[yellow]No search results found.[/yellow]")
            return
        
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Entity ID", style="green", width=15)
        table.add_column("Name", style="white", width=30)
        table.add_column("Source File", style="dim", width=50)
        
        for result in results[:10]:  # Show top 10
            score = f"{result.get('match_score', 0):.2f}"
            entity_id = result.get('entity_id', 'N/A')
            name = result.get('entity_name', 'N/A')
            
            # Get source file name only (not full path)
            source_file = result.get('source_file', '')
            if source_file:
                source_file = Path(source_file).name
            
            table.add_row(score, entity_id, name, source_file)
        
        console.print(table)
        
        if len(results) > 10:
            console.print(f"[dim]... and {len(results) - 10} more results[/dim]")
    
    def _display_entity_contexts(self, contexts: Union[Dict[str, Any], List[Dict[str, Any]]]):
        """Display entity contexts in a formatted way."""
        # Handle both single dict and list of dicts
        if isinstance(contexts, dict):
            contexts = [contexts]
        
        for context in contexts:
            entity_name = context.get('entity_name', 'Unknown')
            entity_id = context.get('entity_id', 'N/A')
            global_id = context.get('global_id', 'N/A')
            
            console.print(f"\n[bold yellow]Entity: {entity_name} (ID: {entity_id})[/bold yellow]")
            console.print(f"[dim]Global ID: {global_id}[/dim]")
            
            questions = context.get('questions', [])
            if questions:
                for i, q in enumerate(questions[:3]):  # Show first 3
                    console.print(f"\n[cyan]Question {i+1}:[/cyan]")
                    console.print(f"  Text: {q.get('question_text', 'N/A')}")
                    console.print(f"  Verb: {q.get('verb_phrase', 'N/A')}")
                    
                    other_entities = q.get('other_entities', [])
                    if other_entities:
                        console.print(f"  Related entities:")
                        for oe in other_entities[:3]:
                            console.print(f"    - {oe.get('entity_name', 'Unknown')} (ID: {oe.get('entity_id', 'N/A')})")
                        if len(other_entities) > 3:
                            console.print(f"[dim]    ... and {len(other_entities) - 3} more[/dim]")
                
                if len(questions) > 3:
                    console.print(f"[dim]... and {len(questions) - 3} more questions[/dim]")
            else:
                console.print("[yellow]No questions found for this entity.[/yellow]")
    
    def process_question(self, question: str, verbose: bool = True):
        """Process a single question and display results."""
        console.print(f"\n[bold blue]Processing question:[/bold blue] {question}")
        
        try:
            # Get response from agent
            response = self.agent.answer_question(question, self.tools)
            
            # Display tool calls if verbose
            if verbose and response.tool_calls_made:
                console.print(f"\n[bold magenta]Tool Calls Made: {len(response.tool_calls_made)}[/bold magenta]")
                for i, tool_call in enumerate(response.tool_calls_made):
                    self.display_tool_call(tool_call, i)
            
            # Display reasoning
            console.print("\n[bold green]Reasoning:[/bold green]")
            console.print(Panel(response.reasoning, expand=False))
            
            # Display answer
            console.print("\n[bold green]Final Answer:[/bold green]")
            console.print(Panel(response.answer, style="bold green", expand=False))
            
            return response
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            import traceback
            if verbose:
                console.print(traceback.format_exc())
            return None
    
    def run_interactive_mode(self):
        """Run the interactive debugging mode."""
        console.print("\n[bold cyan]GSW Agent Interactive Debugger[/bold cyan]")
        console.print("Commands:")
        console.print("  - Type a question to test the agent")
        console.print("  - 'verbose on/off' - Toggle detailed output")
        console.print("  - 'test <type>' - Run predefined test questions")
        console.print("  - 'search <query>' - Test search directly")
        console.print("  - 'entity <id>' - Get entity context directly")
        console.print("  - 'help' - Show this help")
        console.print("  - 'quit' or 'exit' - Exit the debugger\n")
        console.print(f"[cyan]Current search method: {self.search_method}[/cyan]\n")
        
        verbose = True
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold yellow]Query[/bold yellow]")
                
                if user_input.lower() in ['quit', 'exit']:
                    console.print("[bold blue]Goodbye![/bold blue]")
                    break
                
                elif user_input.lower() == 'help':
                    self.run_interactive_mode()  # Show help again
                
                elif user_input.lower().startswith('verbose'):
                    parts = user_input.split()
                    if len(parts) > 1:
                        verbose = parts[1].lower() == 'on'
                        console.print(f"[green]Verbose mode: {'ON' if verbose else 'OFF'}[/green]")
                
                elif user_input.lower().startswith('test'):
                    self._run_test_questions(user_input, verbose)
                
                elif user_input.lower().startswith('search'):
                    query = user_input[6:].strip()
                    if query:
                        if self.search_method == "embeddings":
                            results = self.gsw_tools.search_gsw_embeddings(query, limit=10)
                        else:
                            results = self.gsw_tools.search_gsw_bm25(query, limit=10)
                        self._display_search_results(results)
                
                elif user_input.lower().startswith('entity'):
                    entity_id = user_input[6:].strip()
                    if entity_id:
                        contexts = self.gsw_tools.get_multiple_entity_contexts([entity_id])
                        self._display_entity_contexts(contexts)
                
                else:
                    # Process as a question
                    self.process_question(user_input, verbose)
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    def _run_test_questions(self, command: str, verbose: bool):
        """Run predefined test questions."""
        parts = command.split(maxsplit=1)
        test_type = parts[1] if len(parts) > 1 else "all"
        
        test_questions = {
            "simple": [
                "Who directed Forrest Gump?",
                "What year was The Shawshank Redemption released?"
            ],
            "multihop": [
                "What is the birth year of the director of Forrest Gump?",
                "Which film was released first, Forrest Gump or The Shawshank Redemption?",
                "Who is older, the director of Forrest Gump or the director of The Shawshank Redemption?"
            ],
            "complex": [
                "What university did the director of Forrest Gump attend?",
                "How many Academy Awards did the director of Forrest Gump win?"
            ]
        }
        
        if test_type == "all":
            questions = sum(test_questions.values(), [])
        elif test_type in test_questions:
            questions = test_questions[test_type]
        else:
            console.print(f"[red]Unknown test type: {test_type}[/red]")
            console.print(f"Available: {', '.join(test_questions.keys())}, all")
            return
        
        console.print(f"\n[bold cyan]Running {len(questions)} test questions...[/bold cyan]")
        
        for i, question in enumerate(questions, 1):
            console.print(f"\n[bold]Test {i}/{len(questions)}[/bold]")
            self.process_question(question, verbose)


def main():
    parser = argparse.ArgumentParser(description="Interactive GSW Agent Debugger")
    parser.add_argument(
        "--gsw-dir",
        type=str,
        default="../logs/full_2wiki_corpus_20250710_202211",
        help="Path to GSW output directory (default: ../logs/full_2wiki_corpus_20250710_202211)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        help="Model to use (default: gpt-4o)"
    )
    parser.add_argument(
        "--multi-file",
        action="store_true",
        help="Use individual GSW files instead of reconciled (for testing multi-file mode)"
    )
    parser.add_argument(
        "--search-method",
        type=str,
        choices=["bm25", "embeddings"],
        default="embeddings",
        help="Search method to use: bm25 or embeddings (default: embeddings)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to test (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    # Create debugger
    debugger = InteractiveAgentDebugger(
        args.gsw_dir, 
        args.model, 
        use_multi_file=True,
        search_method=args.search_method
    )
    
    if args.question:
        # Non-interactive mode - just process one question
        debugger.process_question(args.question, verbose=True)
    else:
        # Interactive mode
        debugger.run_interactive_mode()


if __name__ == "__main__":
    main()