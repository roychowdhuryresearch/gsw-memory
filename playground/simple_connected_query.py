#!/usr/bin/env python3
"""
Simple connected hypernode query: 
1. Get top-5 initial matches via embedding similarity
2. Retrieve all summaries connected to those matches
3. Return combined set
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import sys
import numpy as np

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Rich console imports
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt
    RICH_AVAILABLE = True
except ImportError:
    print("Warning: Rich not available")
    RICH_AVAILABLE = False

# VLLM for embeddings
try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    print("Warning: VLLM not available")
    VLLM_AVAILABLE = False

# Similarity computation
try:
    from sklearn.metrics.pairwise import cosine_similarity
    SIMILARITY_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available")
    SIMILARITY_AVAILABLE = False

# OpenAI for question answering
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

console = Console() if RICH_AVAILABLE else None


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Create instruction for Qwen embedding model."""
    return f'Instruct: {task_description}\nQuery: {query}'


class SimpleConnectedQuery:
    """Simple connected hypernode query system."""
    
    def __init__(self, summaries_dir: str, connectivity_dir: str):
        self.summaries_dir = Path(summaries_dir)
        self.connectivity_dir = Path(connectivity_dir)
        
        # Load data
        self.summaries = self._load_summaries()
        self.embeddings = self._load_embeddings()
        self.super_hypernodes = {}  # Will be populated by _load_connectivity
        self.hypernode_to_super = self._load_connectivity()  # hypernode_id -> super_hypernode_id
        self.embedding_model = self._init_embedding_model()
        
        console.print(f"[green]✓ Loaded {len(self.summaries)} summaries[/green]")
        console.print(f"[green]✓ Super-hypernodes: {len(self.super_hypernodes)} clusters[/green]")
        console.print(f"[green]✓ Cluster membership: {len(self.hypernode_to_super)} hypernodes mapped[/green]")
    
    def _load_summaries(self) -> List[Dict[str, Any]]:
        """Load hypernode summaries."""
        summaries_file = self.summaries_dir / "hypernode_summaries.json"
        with open(summaries_file, 'r') as f:
            return json.load(f)
    
    def _load_embeddings(self) -> np.ndarray:
        """Load precomputed embeddings."""
        embeddings_file = self.summaries_dir / "hypernode_embeddings.json"
        with open(embeddings_file, 'r') as f:
            data = json.load(f)
        return np.array(data['embeddings'])
    
    def _load_connectivity(self) -> Dict[str, str]:
        """Load super-hypernode cluster membership mapping."""
        # Load super_hypernodes.json which contains connected components
        super_hypernodes_file = self.connectivity_dir / "super_hypernodes.json"
        if not super_hypernodes_file.exists():
            console.print(f"[yellow]Warning: {super_hypernodes_file} not found[/yellow]")
            return {}
        
        with open(super_hypernodes_file, 'r') as f:
            super_hypernodes = json.load(f)
        
        # Build mapping: hypernode_id -> super_hypernode_id
        hypernode_to_super = {}
        for super_id, super_data in super_hypernodes.items():
            constituent_hypernodes = super_data.get('constituent_hypernodes', [])
            for hypernode_id in constituent_hypernodes:
                hypernode_to_super[hypernode_id] = super_id
        
        # Store the full super_hypernodes data for later use
        self.super_hypernodes = super_hypernodes
        
        return hypernode_to_super
    
    def _init_embedding_model(self):
        """Initialize embedding model."""
        if not VLLM_AVAILABLE:
            return None
        try:
            return LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")
        except Exception as e:
            console.print(f"[red]Error initializing embedding model: {e}[/red]")
            return None
    
    def search_with_connections(self, query: str, top_k: int = 5, rerank_top_k: int = 10) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Search and return (initial_matches, connected_summaries, reranked_top_summaries).
        """
        console.print(f"[cyan]Searching for: '{query}'[/cyan]")
        
        # Step 1: Get top-k initial matches
        initial_matches = self._get_initial_matches(query, top_k)
        console.print(f"[green]Found {len(initial_matches)} initial matches[/green]")
        
        # Step 2: Get connected summaries
        connected_summaries = self._get_connected_summaries(initial_matches)
        console.print(f"[green]Found {len(connected_summaries)} connected summaries[/green]")
        
        # Step 3: Rerank all summaries and get top-k
        all_summaries = initial_matches + connected_summaries
        reranked_summaries = self._rerank_summaries(query, all_summaries, rerank_top_k)
        console.print(f"[green]Reranked to top {len(reranked_summaries)} most relevant summaries[/green]")
        
        return initial_matches, connected_summaries, reranked_summaries
    
    def _get_initial_matches(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Get top-k initial matches using embedding similarity."""
        if not self.embedding_model or not SIMILARITY_AVAILABLE:
            console.print("[yellow]Falling back to text search[/yellow]")
            return self._text_search(query, top_k)
        
        try:
            # Embed query
            task = 'Given a fact based query, retrieve relevant summaries that answer the query.'
            instructed_query = get_detailed_instruct(task, query)
            
            query_output = self.embedding_model.embed([instructed_query])
            query_embedding = np.array([query_output[0].outputs.embedding])
            
            # Compute similarities
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
            
            # Get top-k
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                summary = self.summaries[idx]
                score = float(similarities[idx])
                summary_with_score = summary.copy()
                summary_with_score['similarity_score'] = score
                results.append(summary_with_score)
            
            return results
            
        except Exception as e:
            console.print(f"[red]Error in semantic search: {e}[/red]")
            return self._text_search(query, top_k)
    
    def _text_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Fallback text search."""
        query_lower = query.lower()
        scored_summaries = []
        
        for summary in self.summaries:
            score = 0.0
            if query_lower in summary['name'].lower():
                score += 1.0
            for var in summary['entity_variations']:
                if query_lower in var.lower():
                    score += 0.5
            if query_lower in summary['summary'].lower():
                score += 0.3
            
            if score > 0:
                summary_with_score = summary.copy()
                summary_with_score['similarity_score'] = score
                scored_summaries.append(summary_with_score)
        
        scored_summaries.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_summaries[:top_k]
    
    def _get_connected_summaries(self, initial_matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get all summaries from the same super-hypernode clusters as initial matches."""
        # Get IDs of initial matches
        initial_ids = {summary['hypernode_id'] for summary in initial_matches}
        
        # Find which super-hypernodes contain the initial matches
        super_hypernode_ids = set()
        cluster_info = {}  # super_id -> cluster data
        
        for summary in initial_matches:
            hid = summary['hypernode_id']
            if hid in self.hypernode_to_super:
                super_id = self.hypernode_to_super[hid]
                super_hypernode_ids.add(super_id)
                if super_id in self.super_hypernodes:
                    cluster_info[super_id] = self.super_hypernodes[super_id]
        
        # Get ALL constituent hypernodes from those super-hypernodes
        all_cluster_hypernode_ids = set()
        for super_id in super_hypernode_ids:
            if super_id in cluster_info:
                constituent_hypernodes = cluster_info[super_id].get('constituent_hypernodes', [])
                all_cluster_hypernode_ids.update(constituent_hypernodes)
        
        # Remove initial IDs to avoid duplicates
        connected_ids = all_cluster_hypernode_ids - initial_ids
        
        # Get summaries for connected IDs
        id_to_summary = {s['hypernode_id']: s for s in self.summaries}
        connected_summaries = []
        
        for conn_id in connected_ids:
            if conn_id in id_to_summary:
                summary = id_to_summary[conn_id].copy()
                # Add info about which cluster this came from
                super_id = self.hypernode_to_super.get(conn_id)
                if super_id in cluster_info:
                    summary['connected_via'] = f'cluster_{super_id}_{cluster_info[super_id]["super_hypernode_name"]}'
                    summary['cluster_size'] = len(cluster_info[super_id].get('constituent_hypernodes', []))
                else:
                    summary['connected_via'] = 'cluster_connection'
                connected_summaries.append(summary)
        
        return connected_summaries
    
    def _rerank_summaries(self, query: str, summaries: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank summaries by relevance to query and return top-k."""
        if not summaries:
            return []
        
        # Try LLM reranking first
        llm_result = self._llm_rerank_summaries(query, summaries, top_k)
        if llm_result:
            return llm_result
        
        # Fallback to embedding reranking
        if not self.embedding_model or not SIMILARITY_AVAILABLE:
            console.print("[yellow]Reranking unavailable without embeddings, using original order[/yellow]")
            return summaries[:top_k]
        
        try:
            # Generate query embedding
            task = 'Given a fact based query, retrieve relevant summaries that answer the query.'
            instructed_query = get_detailed_instruct(task, query)
            query_output = self.embedding_model.embed([instructed_query])
            query_embedding = np.array([query_output[0].outputs.embedding])
            
            # Get embeddings for all summaries
            # Use precomputed embeddings where available
            id_to_idx = {s['hypernode_id']: i for i, s in enumerate(self.summaries)}
            
            summary_embeddings = []
            for summary in summaries:
                hid = summary['hypernode_id']
                if hid in id_to_idx:
                    idx = id_to_idx[hid]
                    embedding = self.embeddings[idx]
                    summary_embeddings.append(embedding)
                else:
                    # Generate embedding on the fly if needed
                    text = f"{summary['name']}: {summary['summary']}"
                    instructed = get_detailed_instruct(task, text)
                    output = self.embedding_model.embed([instructed])
                    embedding = np.array(output[0].outputs.embedding)
                    summary_embeddings.append(embedding)
            
            summary_embeddings = np.array(summary_embeddings)
            
            # Compute similarities
            similarities = cosine_similarity(query_embedding, summary_embeddings)[0]
            
            # Add similarity scores to summaries
            for i, summary in enumerate(summaries):
                summary['rerank_score'] = float(similarities[i])
            
            # Sort by similarity and return top-k
            summaries.sort(key=lambda x: x['rerank_score'], reverse=True)
            return summaries[:top_k]
            
        except Exception as e:
            console.print(f"[red]Error during reranking: {e}[/red]")
            return summaries[:top_k]
    
    def _llm_rerank_summaries(self, query: str, summaries: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Rerank summaries using LLM reasoning for multi-hop queries."""
        if not OPENAI_AVAILABLE or not summaries:
            return None
        
        try:
            client = OpenAI()
            
            # Format summaries for the prompt
            formatted_summaries = []
            for i, summary in enumerate(summaries, 1):
                formatted_summaries.append(
                    f"{i}. {summary['name']}\n"
                    f"   Summary: {summary['summary'][:200]}...\n"
                    f"   Variations: {', '.join(summary['entity_variations'][:3])}\n"
                )
            
            summaries_text = "\n".join(formatted_summaries)
            
            prompt = f"""Query: {query}

Summaries to rank:
{summaries_text}

Task: Rank these summaries by how useful they are for answering the query. Consider:
1. Direct relevance to the question
2. Multi-hop connections (bridging entities that connect relevant information)
3. Supporting context that helps build a complete answer

Return only the top {top_k} most useful summaries as a JSON list of numbers (e.g., [3, 1, 7, 2, 5]).
Think about multi-hop reasoning - sometimes a summary that doesn't directly mention the query terms might be crucial for connecting the dots."""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at ranking information for multi-hop question answering. Focus on identifying both direct answers and crucial bridging information."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            # Parse the response to get ranking
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON array from response
            import re
            json_match = re.search(r'\[[\d,\s]+\]', response_text)
            if not json_match:
                console.print("[yellow]Could not parse LLM ranking response, falling back to embedding reranking[/yellow]")
                return None
            
            import json as json_module
            ranking_indices = json_module.loads(json_match.group())
            
            # Reorder summaries based on LLM ranking
            reranked_summaries = []
            for rank_pos, summary_idx in enumerate(ranking_indices):
                if 1 <= summary_idx <= len(summaries):
                    summary = summaries[summary_idx - 1].copy()  # Convert to 0-based index
                    summary['rerank_score'] = 1.0 - (rank_pos * 0.1)  # Assign decreasing scores
                    summary['rerank_method'] = 'llm'
                    reranked_summaries.append(summary)
            
            console.print(f"[green]LLM reranking successful: {len(reranked_summaries)} summaries ranked[/green]")
            return reranked_summaries[:top_k]
            
        except Exception as e:
            console.print(f"[yellow]LLM reranking failed ({e}), falling back to embedding reranking[/yellow]")
            return None
    
    def display_results(self, initial_matches: List[Dict[str, Any]], 
                       connected_summaries: List[Dict[str, Any]],
                       reranked_summaries: List[Dict[str, Any]] = None):
        """Display search results."""
        
        # Display initial matches
        console.print("\n[bold yellow]🎯 INITIAL MATCHES[/bold yellow]")
        if initial_matches:
            table = Table(show_header=True, header_style="bold green")
            table.add_column("Score", width=8)
            table.add_column("Name", width=25)
            table.add_column("Variations", width=40)
            table.add_column("Docs", width=8)
            
            for summary in initial_matches:
                score = f"{summary.get('similarity_score', 0):.3f}"
                name = summary['name']
                variations = ', '.join(summary['entity_variations'][:3])
                if len(summary['entity_variations']) > 3:
                    variations += f" (+{len(summary['entity_variations'])-3})"
                docs = str(len(summary['source_documents']))
                
                table.add_row(score, name, variations, docs)
            
            console.print(table)
        else:
            console.print("[dim]No initial matches found[/dim]")
        
        # Display connected summaries
        console.print(f"\n[bold blue]🔗 CONNECTED SUMMARIES ({len(connected_summaries)})[/bold blue]")
        console.print("[dim]From same super-hypernode clusters as initial matches[/dim]")
        
        if connected_summaries:
            table = Table(show_header=True, header_style="bold blue")
            table.add_column("Name", width=20)
            table.add_column("Cluster", width=15)
            table.add_column("Variations", width=30)
            table.add_column("Docs", width=8)
            table.add_column("Summary Preview", width=40)
            
            for summary in connected_summaries[:10]:  # Show first 10
                name = summary['name']
                
                # Extract cluster info
                cluster_info = summary.get('connected_via', 'unknown')
                if cluster_info.startswith('cluster_'):
                    parts = cluster_info.split('_', 2)
                    if len(parts) >= 3:
                        cluster_display = f"{parts[1]}:{parts[2]}"
                    else:
                        cluster_display = parts[1] if len(parts) > 1 else "unknown"
                else:
                    cluster_display = "unknown"
                
                variations = ', '.join(summary['entity_variations'][:2])
                if len(summary['entity_variations']) > 2:
                    variations += f" (+{len(summary['entity_variations'])-2})"
                docs = str(len(summary['source_documents']))
                preview = summary['summary'][:60] + "..." if len(summary['summary']) > 60 else summary['summary']
                
                table.add_row(name, cluster_display, variations, docs, preview)
            
            console.print(table)
            
            if len(connected_summaries) > 10:
                console.print(f"[dim]... and {len(connected_summaries) - 10} more connected summaries[/dim]")
        else:
            console.print("[dim]No connected summaries found[/dim]")
        
        # Display reranked summaries
        if reranked_summaries is not None:
            # Check reranking method
            rerank_method = "embedding"
            if reranked_summaries and reranked_summaries[0].get('rerank_method') == 'llm':
                rerank_method = "LLM (GPT-4o-mini)"
            
            console.print(f"\n[bold magenta]🏆 TOP RERANKED SUMMARIES ({len(reranked_summaries)})[/bold magenta]")
            console.print(f"[dim]Best matches reranked by {rerank_method} for multi-hop query relevance[/dim]")
            
            if reranked_summaries:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Rank", width=6)
                table.add_column("Score", width=8)
                table.add_column("Name", width=20)
                table.add_column("Source", width=12)
                table.add_column("Summary Preview", width=45)
                
                for i, summary in enumerate(reranked_summaries, 1):
                    rank = str(i)
                    score = f"{summary.get('rerank_score', 0):.3f}"
                    name = summary['name']
                    
                    # Determine source
                    if summary in initial_matches:
                        source = "[green]Initial[/green]"
                    else:
                        source = "[blue]Connected[/blue]"
                    
                    preview = summary['summary'][:65] + "..." if len(summary['summary']) > 65 else summary['summary']
                    
                    table.add_row(rank, score, name, source, preview)
                
                console.print(table)
            else:
                console.print("[dim]No reranked summaries available[/dim]")
        
        # Summary stats
        total_summaries = len(initial_matches) + len(connected_summaries)
        reranked_count = len(reranked_summaries) if reranked_summaries else 0
        console.print(f"\n[bold]📊 SUMMARY: {total_summaries} total summaries ({len(initial_matches)} initial + {len(connected_summaries)} connected) → {reranked_count} reranked[/bold]")
    
    def answer_question(self, query: str, reranked_summaries: List[Dict[str, Any]]) -> str:
        """Generate an answer using GPT-4o based on reranked summaries."""
        if not OPENAI_AVAILABLE:
            return "[yellow]OpenAI not available for answer generation[/yellow]"
        
        if not reranked_summaries:
            return "No relevant information found to answer your question."
        
        # Prepare context using reranked summaries
        context_parts = []
        for i, summary in enumerate(reranked_summaries, 1):
            relevance_score = summary.get('rerank_score', 0)
            context_parts.append(
                f"{i}. {summary['name']} (Relevance: {relevance_score:.3f})\n"
                f"   Summary: {summary['summary']}\n"
                f"   Variations: {', '.join(summary['entity_variations'][:5])}\n"
            )
        
        context = "\n".join(context_parts)
        
        try:
            client = OpenAI()
            
            prompt = f"""Question: {query}

Relevant Entity Summaries (ranked by relevance):
{context}

Please answer the question based on the provided summaries. Focus on the most relevant entities (higher relevance scores). Be specific and cite which entities you're drawing information from."""
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided entity summaries. Be concise but comprehensive."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"[red]Error generating answer: {e}[/red]"
    
    def interactive_mode(self):
        """Run interactive query session."""
        console.print("\n[bold blue]🔍 Cluster-Aware Hypernode Query[/bold blue]")
        console.print("[dim]Search gets top-5 matches + all summaries from same clusters[/dim]")
        console.print("[dim]Type 'help' for commands, 'quit' to exit[/dim]\n")
        
        while True:
            try:
                query = Prompt.ask("[bold cyan]Query")
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'help':
                    self._show_help()
                elif query.lower().startswith('test'):
                    self._run_test_queries()
                else:
                    # Search and display results
                    initial, connected, reranked = self.search_with_connections(query, top_k=5, rerank_top_k=10)
                    self.display_results(initial, connected, reranked)
                    
                    # Generate answer using reranked summaries
                    console.print("\n[yellow]Generating answer with GPT-4o using reranked summaries...[/yellow]")
                    answer = self.answer_question(query, reranked)
                    console.print(Panel(answer, title="[bold green]Answer[/bold green]", border_style="green"))
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
        
        console.print("\n[bold blue]👋 Goodbye![/bold blue]")
    
    def _show_help(self):
        """Show help information."""
        help_text = """[bold]Available Commands:[/bold]
        
• [cyan]<search query>[/cyan] - Search and generate answer with GPT-4o
• [cyan]test[/cyan] - Run predefined test queries
• [cyan]help[/cyan] - Show this help
• [cyan]quit[/cyan] - Exit

[bold]How it works:[/bold]
1. Finds top-5 hypernode summaries using embedding similarity
2. Gets ALL summaries from the same super-hypernode clusters
3. Reranks all retrieved summaries by query relevance
4. Shows initial matches, connected summaries, and top reranked results
5. Generates answer using the most relevant reranked summaries
6. This ensures complete entity coverage with optimal relevance
        """
        console.print(Panel(help_text, title="[bold yellow]Help[/bold yellow]"))
    
    def _run_test_queries(self):
        """Run predefined test queries."""
        test_queries = [
            "Nicki Minaj",
            "Lothair II", 
            "British films",
            "medieval history"
        ]
        
        for query in test_queries:
            console.print(f"\n{'='*60}")
            console.print(f"[bold magenta]TEST: {query}[/bold magenta]")
            console.print(f"{'='*60}")
            
            initial, connected, reranked = self.search_with_connections(query, top_k=5, rerank_top_k=8)
            self.display_results(initial, connected, reranked)
            
            if query != test_queries[-1]:  # Not the last query
                input("\nPress Enter for next test query...")


def main():
    """Main function - run interactive connected query system."""
    summaries_dir = "/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/hypernode_summaries_1755036912"
    
    # Auto-detect latest connectivity directory
    playground_dir = Path(summaries_dir).parent
    connectivity_dirs = list(playground_dir.glob("super_hypernodes_*"))
    if not connectivity_dirs:
        console.print("[red]No connectivity directories found[/red]")
        console.print("[yellow]Run hypernode_connected_components.py first to generate connectivity data[/yellow]")
        return
    
    connectivity_dir = max(connectivity_dirs, key=lambda p: p.stat().st_mtime)
    console.print(f"[green]Using connectivity from: {connectivity_dir.name}[/green]")
    
    # Initialize and run interactive system
    query_system = SimpleConnectedQuery(summaries_dir, str(connectivity_dir))
    query_system.interactive_mode()


if __name__ == "__main__":
    main()