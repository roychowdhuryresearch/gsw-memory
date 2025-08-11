#!/usr/bin/env python3
"""
Interactive debugging tool for hypernode summary queries.

This tool allows you to:
- Test questions against hypernode summaries interactively
- Inspect search results and similarity scores
- Debug query embedding and retrieval
- Test different configurations
"""

import json
import argparse
import glob
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple
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

import numpy as np

# Similarity computation imports
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SIMILARITY_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn")
    SIMILARITY_AVAILABLE = False

# VLLM for embeddings
try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: VLLM not available. Install with: pip install vllm>=0.8.5")
    VLLM_AVAILABLE = False

# OpenAI for answering
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False


console = Console()


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Create instruction for Qwen embedding model."""
    return f'Instruct: {task_description}\nQuery: {query}'


class InteractiveHypernodeQuery:
    """Interactive debugger for hypernode summary queries."""
    
    def __init__(self, summaries_dir: str, model_name: str = "gpt-4o"):
        """Initialize the debugger with hypernode summaries."""
        self.summaries_dir = Path(summaries_dir)
        self.model_name = model_name
        self.summaries = []
        self.embeddings = None
        self.embedding_model = None
        
        # Load summaries and embeddings
        console.print("[bold blue]Loading hypernode summaries...[/bold blue]")
        self._load_summaries()
        self._load_embeddings()
        self._initialize_embedding_model()
        
        console.print(f"[bold green]✓ Loaded {len(self.summaries)} hypernode summaries[/bold green]")
        console.print(f"[bold green]✓ Using model: {model_name}[/bold green]")
        console.print(f"[bold green]✓ Embeddings: {'Available' if self.embeddings is not None else 'Not available'}[/bold green]")
    
    def _load_summaries(self) -> None:
        """Load hypernode summaries from JSON file."""
        summaries_file = self.summaries_dir / "hypernode_summaries.json"
        if not summaries_file.exists():
            raise FileNotFoundError(f"Summaries file not found: {summaries_file}")
        
        with open(summaries_file, 'r') as f:
            self.summaries = json.load(f)
    
    def _load_embeddings(self) -> None:
        """Load precomputed embeddings for summaries."""
        embeddings_file = self.summaries_dir / "hypernode_embeddings.json"
        if not embeddings_file.exists():
            console.print("[yellow]Warning: No precomputed embeddings found[/yellow]")
            return
        
        with open(embeddings_file, 'r') as f:
            embeddings_data = json.load(f)
        
        self.embeddings = np.array(embeddings_data['embeddings'])
    
    def _initialize_embedding_model(self) -> None:
        """Initialize the Qwen embedding model."""
        if not VLLM_AVAILABLE:
            console.print("[yellow]Warning: VLLM not available. Query embedding disabled.[/yellow]")
            return
        
        try:
            console.print("[cyan]Initializing Qwen embedding model...[/cyan]")
            self.embedding_model = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")
            console.print("[green]✓ Qwen embedding model initialized[/green]")
        except Exception as e:
            console.print(f"[red]Error initializing embedding model: {e}[/red]")
            self.embedding_model = None
    
    def _embed_query(self, query: str) -> Optional[np.ndarray]:
        """Embed a query using the Qwen model."""
        if not self.embedding_model:
            return None
        
        # Use the same task description as for summary embeddings
        task = 'Given a comprehensive entity summary, create an embedding optimized for semantic search and question answering retrieval'
        instructed_query = get_detailed_instruct(task, query)
        
        try:
            outputs = self.embedding_model.embed([instructed_query])
            embedding = np.array(outputs[0].outputs.embedding)
            return embedding
        except Exception as e:
            console.print(f"[red]Error embedding query: {e}[/red]")
            return None
    
    def search_summaries(self, query: str, top_k: int = 5, verbose: bool = True) -> List[Tuple[Dict[str, Any], float]]:
        """Search for relevant hypernode summaries."""
        if verbose:
            console.print(f"[cyan]Searching for: '{query}'[/cyan]")
        
        if not SIMILARITY_AVAILABLE:
            console.print("[red]Error: Similarity computation not available[/red]")
            return []
        
        # Embed the query
        query_embedding = self._embed_query(query)
        if query_embedding is None:
            console.print("[yellow]Warning: Could not embed query, using text search[/yellow]")
            return self._text_search_fallback(query, top_k, verbose)
        
        # If no precomputed embeddings, fall back to text search
        if self.embeddings is None:
            console.print("[yellow]Warning: No embeddings available, using text search[/yellow]")
            return self._text_search_fallback(query, top_k, verbose)
        
        # Compute similarities
        query_embedding = query_embedding.reshape(1, -1)
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar summaries
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            summary = self.summaries[idx]
            similarity = similarities[idx]
            results.append((summary, similarity))
        
        if verbose:
            self._display_search_results(results)
        
        return results
    
    def _text_search_fallback(self, query: str, top_k: int = 5, verbose: bool = True) -> List[Tuple[Dict[str, Any], float]]:
        """Fallback text-based search when embeddings are not available."""
        query_words = set(query.lower().split())
        
        scored_summaries = []
        for summary in self.summaries:
            # Count word matches in summary text and name
            summary_text = summary['summary'].lower()
            name_text = summary['name'].lower()
            variations_text = ' '.join(summary['entity_variations']).lower()
            
            summary_words = set(summary_text.split()) | set(name_text.split()) | set(variations_text.split())
            
            # Simple overlap score
            overlap = len(query_words & summary_words)
            score = overlap / len(query_words) if query_words else 0.0
            
            scored_summaries.append((summary, score))
        
        # Sort by score and return top-k
        scored_summaries.sort(key=lambda x: x[1], reverse=True)
        results = scored_summaries[:top_k]
        
        if verbose:
            console.print("[yellow]Using text-based search[/yellow]")
            self._display_search_results(results)
        
        return results
    
    def _display_search_results(self, results: List[Tuple[Dict[str, Any], float]]):
        """Display search results in a formatted table."""
        if not results:
            console.print("[yellow]No search results found.[/yellow]")
            return
        
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Score", style="cyan", width=8)
        table.add_column("Hypernode", style="green", width=25)
        table.add_column("Variations", style="white", width=40)
        table.add_column("Documents", style="dim", width=20)
        
        for summary, similarity in results:
            score = f"{similarity:.3f}"
            name = summary['name']
            variations = ', '.join(summary['entity_variations'][:3])  # Show first 3
            if len(summary['entity_variations']) > 3:
                variations += f" (+{len(summary['entity_variations']) - 3} more)"
            
            docs = f"{len(summary['source_documents'])} docs"
            
            table.add_row(score, name, variations, docs)
        
        console.print(table)
    
    def _display_summary_details(self, summary: Dict[str, Any]):
        """Display detailed information about a hypernode summary."""
        console.print(f"\n[bold yellow]Hypernode: {summary['name']}[/bold yellow]")
        console.print(f"[dim]ID: {summary['hypernode_id']}[/dim]")
        console.print(f"[dim]Confidence: {summary.get('confidence_score', 'N/A')}[/dim]")
        
        console.print(f"\n[cyan]Entity Variations:[/cyan]")
        for variation in summary['entity_variations']:
            console.print(f"  - {variation}")
        
        console.print(f"\n[cyan]Source Documents:[/cyan]")
        for doc in summary['source_documents']:
            console.print(f"  - {doc}")
        
        console.print(f"\n[cyan]Summary:[/cyan]")
        console.print(Panel(summary['summary'], expand=False))
        
        console.print(f"\n[cyan]Entities Included:[/cyan]")
        for entity_id in summary['entities_included']:
            console.print(f"  - {entity_id}")
    
    def answer_question(self, query: str, relevant_summaries: List[Tuple[Dict[str, Any], float]], verbose: bool = True) -> str:
        """Answer a question using retrieved hypernode summaries."""
        if not OPENAI_AVAILABLE:
            return "[red]Error: OpenAI not available for question answering[/red]"
        
        if not relevant_summaries:
            return "[yellow]No relevant summaries found to answer your question.[/yellow]"
        
        # Display the summaries being passed to the LLM
        if verbose:
            console.print(f"\n[bold magenta]Context Summaries Passed to LLM ({len(relevant_summaries)} summaries):[/bold magenta]")
            for i, (summary, similarity) in enumerate(relevant_summaries, 1):
                console.print(f"\n[cyan]Summary {i} - {summary['name']} (Similarity: {similarity:.3f})[/cyan]")
                console.print(f"[dim]Variations: {', '.join(summary['entity_variations'])}[/dim]")
                console.print(f"[dim]Documents: {', '.join(summary['source_documents'])}[/dim]")
                console.print(Panel(summary['summary'], title=f"Summary Content", expand=False))
        
        # Prepare context from retrieved summaries
        context_parts = []
        for i, (summary, similarity) in enumerate(relevant_summaries, 1):
            context_parts.append(f"""
{i}. **{summary['name']}** (Similarity: {similarity:.3f})
   - Variations: {', '.join(summary['entity_variations'])}
   - Summary: {summary['summary']}
""")
        
        context = "\n".join(context_parts)
        
        # Create answering prompt
        prompt = f"""You are an AI assistant that answers questions based on entity information from multiple documents. You have been provided with relevant entity summaries below.

RELEVANT ENTITY INFORMATION:
{context}

QUESTION: {query}

INSTRUCTIONS:
- Answer the question based ONLY on the information provided in the entity summaries above
- If the information is not sufficient to answer the question, say so clearly
- Cite which entities/documents your answer is based on
- Be concise but comprehensive
- If multiple entities are relevant, explain their relationships
- Reason about your answer using the docs first and put those in <reasoning> tags.
- For your final answer, but it within <answer> tags. Note to ONLY include the answer in the <answer> tags.

Example: 
1. Who is the top goal scorer in the 2024 World Cup?
<reasoning>
From the docs, we can see that the top goal scorer in the 2024 World Cup was Lionel Messi.
</reasoning>
<answer>
Lionel Messi
</answer>

ANSWER:"""
        
        try:
            if verbose:
                console.print("[cyan]Generating answer with GPT-4o...[/cyan]")
            
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that answers questions based on provided entity information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"[red]Error generating answer: {e}[/red]"
    
    def process_question(self, question: str, top_k: int = 5, verbose: bool = True):
        """Process a single question and display results."""
        console.print(f"\n[bold blue]Processing question:[/bold blue] {question}")
        
        try:
            # Search for relevant summaries
            relevant_summaries = self.search_summaries(question, top_k, verbose)
            
            if not relevant_summaries:
                console.print("[yellow]No relevant summaries found.[/yellow]")
                return None
            
            # Generate answer
            answer = self.answer_question(question, relevant_summaries, verbose)
            
            # Display answer
            console.print("\n[bold green]Answer:[/bold green]")
            console.print(Panel(answer, style="bold green", expand=False))
            
            return answer
            
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            import traceback
            if verbose:
                console.print(traceback.format_exc())
            return None
    
    def run_interactive_mode(self):
        """Run the interactive debugging mode."""
        console.print("\n[bold cyan]Hypernode Summary Interactive Query Tool[/bold cyan]")
        console.print("Commands:")
        console.print("  - Type a question to query hypernode summaries")
        console.print("  - 'search <query>' - Search summaries without answering")
        console.print("  - 'show <hypernode_name>' - Show detailed info about a hypernode")
        console.print("  - 'list' - List all available hypernodes")
        console.print("  - 'stats' - Show summary statistics")
        console.print("  - 'verbose on/off' - Toggle detailed output")
        console.print("  - 'topk <number>' - Set number of results to retrieve")
        console.print("  - 'test' - Run predefined test questions")
        console.print("  - 'help' - Show this help")
        console.print("  - 'quit' or 'exit' - Exit the tool\\n")
        
        verbose = True
        top_k = 10
        
        while True:
            try:
                user_input = Prompt.ask("\\n[bold yellow]Query[/bold yellow]")
                
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
                
                elif user_input.lower().startswith('topk'):
                    parts = user_input.split()
                    if len(parts) > 1:
                        try:
                            top_k = int(parts[1])
                            console.print(f"[green]Top-k set to: {top_k}[/green]")
                        except ValueError:
                            console.print("[red]Invalid number for top-k[/red]")
                
                elif user_input.lower().startswith('search'):
                    query = user_input[6:].strip()
                    if query:
                        self.search_summaries(query, top_k, verbose)
                
                elif user_input.lower().startswith('show'):
                    name = user_input[4:].strip()
                    if name:
                        # Find hypernode by name
                        found = False
                        for summary in self.summaries:
                            if name.lower() in summary['name'].lower():
                                self._display_summary_details(summary)
                                found = True
                                break
                        if not found:
                            console.print(f"[yellow]No hypernode found matching: {name}[/yellow]")
                
                elif user_input.lower() == 'list':
                    self._list_hypernodes()
                
                elif user_input.lower() == 'stats':
                    self._show_stats()
                
                elif user_input.lower() == 'test':
                    self._run_test_questions(verbose, top_k)
                
                else:
                    # Process as a question
                    self.process_question(user_input, top_k, verbose)
                    
            except KeyboardInterrupt:
                console.print("\\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    def _list_hypernodes(self):
        """List all available hypernodes."""
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Name", style="green", width=30)
        table.add_column("Variations", style="white", width=40)
        table.add_column("Docs", style="dim", width=10)
        table.add_column("Entities", style="dim", width=10)
        
        for summary in self.summaries[:20]:  # Show first 20
            name = summary['name']
            variations = ', '.join(summary['entity_variations'][:2])
            if len(summary['entity_variations']) > 2:
                variations += f" (+{len(summary['entity_variations']) - 2})"
            docs = str(len(summary['source_documents']))
            entities = str(len(summary['entities_included']))
            
            table.add_row(name, variations, docs, entities)
        
        console.print(table)
        if len(self.summaries) > 20:
            console.print(f"[dim]... and {len(self.summaries) - 20} more hypernodes[/dim]")
    
    def _show_stats(self):
        """Show statistics about the hypernode summaries."""
        if not self.summaries:
            console.print("[yellow]No summaries loaded.[/yellow]")
            return
        
        total_summaries = len(self.summaries)
        total_docs = len(set().union(*[s['source_documents'] for s in self.summaries]))
        total_entities = sum(len(s['entities_included']) for s in self.summaries)
        
        # Variations statistics
        variation_counts = [len(s['entity_variations']) for s in self.summaries]
        avg_variations = np.mean(variation_counts)
        
        # Document spread statistics  
        doc_counts = [len(s['source_documents']) for s in self.summaries]
        avg_docs = np.mean(doc_counts)
        
        console.print(f"[bold cyan]Hypernode Summary Statistics:[/bold cyan]")
        console.print(f"  Total hypernodes: {total_summaries}")
        console.print(f"  Total unique documents: {total_docs}")
        console.print(f"  Total entities consolidated: {total_entities}")
        console.print(f"  Average variations per hypernode: {avg_variations:.1f}")
        console.print(f"  Average documents per hypernode: {avg_docs:.1f}")
        console.print(f"  Embeddings available: {'Yes' if self.embeddings is not None else 'No'}")
    
    def _run_test_questions(self, verbose: bool, top_k: int):
        """Run predefined test questions."""
        test_questions = [
            "Who is John Smith?",
            "What happened in New York?",
            "Tell me about the investigation",
            "Who are the main characters?",
            "What events took place in 2023?"
        ]
        
        console.print(f"\\n[bold cyan]Running {len(test_questions)} test questions...[/bold cyan]")
        
        for i, question in enumerate(test_questions, 1):
            console.print(f"\\n[bold]Test {i}/{len(test_questions)}[/bold]")
            self.process_question(question, top_k, verbose)


def find_latest_summaries_dir(base_pattern: str) -> Optional[str]:
    """Find the most recent hypernode summaries directory."""
    dirs = glob.glob(base_pattern)
    if not dirs:
        return None
    
    # Sort by creation time and return the latest
    latest_dir = max(dirs, key=os.path.getctime)
    return latest_dir


def main():
    """Main entry point for the interactive tool."""
    parser = argparse.ArgumentParser(description="Interactive Hypernode Summary Query Tool")
    parser.add_argument(
        "--summaries-dir",
        type=str,
        help="Directory containing hypernode summaries (auto-detects latest if not provided)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="Model to use for answering (default: gpt-4o)"
    )
    parser.add_argument(
        "--question",
        type=str,
        help="Single question to test (non-interactive mode)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top results to retrieve (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Find summaries directory
    if args.summaries_dir:
        summaries_dir = args.summaries_dir
    else:
        # Auto-detect latest summaries directory
        base_pattern = "/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/hypernode_summaries_*"
        summaries_dir = find_latest_summaries_dir(base_pattern)
        
        if not summaries_dir:
            console.print(f"[red]Error: No hypernode summaries found. Please run generate_hypernode_summaries.py first.[/red]")
            return
        
        console.print(f"[green]Using summaries from: {summaries_dir}[/green]")
    
    # Create interactive tool
    try:
        tool = InteractiveHypernodeQuery(summaries_dir, args.model)
    except Exception as e:
        console.print(f"[red]Error initializing tool: {e}[/red]")
        return
    
    if args.question:
        # Non-interactive mode - just process one question
        tool.process_question(args.question, args.top_k, verbose=True)
    else:
        # Interactive mode
        tool.run_interactive_mode()


if __name__ == "__main__":
    main()