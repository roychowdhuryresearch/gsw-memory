#!/usr/bin/env python3
"""
Interactive tool to test embedding similarity between text strings.
Uses the same Qwen3-Embedding-8B model and approach as hypernode clustering.
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

sys.path.append(str(Path(__file__).parent.parent))

try:
    from vllm import LLM
    VLLM_AVAILABLE = True
except ImportError:
    print("Error: VLLM not available. Install with: pip install vllm>=0.8.5")
    sys.exit(1)

console = Console()


def get_detailed_instruct(task_description: str, query: str) -> str:
    """Create instruction for Qwen embedding model."""
    return f'Instruct: {task_description}\nQuery: {query}'


class EmbeddingSimilarityTester:
    """Interactive embedding similarity tester."""
    
    def __init__(self):
        """Initialize the tester with Qwen model."""
        self.model = None
        self.threshold = 0.85
        self.task = 'Given a summary of an entity, retrieve other similar entity summaries.'
        self.history = []
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the Qwen embedding model."""
        console.print("[cyan]Initializing Qwen3-Embedding-8B model...[/cyan]")
        try:
            self.model = LLM(model="Qwen/Qwen3-Embedding-8B", task="embed")
            console.print("[green]✓ Model initialized successfully[/green]")
        except Exception as e:
            console.print(f"[red]Error initializing model: {e}[/red]")
            sys.exit(1)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two text strings."""
        # Prepare inputs
        input1 = get_detailed_instruct(self.task, text1)
        input2 = get_detailed_instruct(self.task, text2)
        
        # Generate embeddings
        outputs = self.model.embed([input1, input2])
        
        # Extract embeddings
        embedding1 = np.array([outputs[0].outputs.embedding])
        embedding2 = np.array([outputs[1].outputs.embedding])
        
        # Compute similarity
        similarity = cosine_similarity(embedding1, embedding2)[0][0]
        
        # Add to history
        self.history.append({
            'text1': text1,
            'text2': text2,
            'similarity': similarity,
            'above_threshold': similarity >= self.threshold
        })
        
        return similarity
    
    def display_result(self, text1: str, text2: str, similarity: float):
        """Display similarity result with rich formatting."""
        above_threshold = similarity >= self.threshold
        
        # Create result table
        table = Table(show_header=False, title="[bold cyan]Similarity Result[/bold cyan]")
        table.add_column("Field", style="cyan", width=20)
        table.add_column("Value", style="white")
        
        table.add_row("Text 1", text1[:100] + "..." if len(text1) > 100 else text1)
        table.add_row("Text 2", text2[:100] + "..." if len(text2) > 100 else text2)
        table.add_row("Similarity Score", f"[bold yellow]{similarity:.4f}[/bold yellow]")
        table.add_row("Threshold", f"{self.threshold:.4f}")
        table.add_row("Would Cluster?", 
                     f"[bold green]✓ YES[/bold green]" if above_threshold else f"[bold red]✗ NO[/bold red]")
        
        console.print(table)
        
        # Interpretation
        interpretation = self._get_interpretation(similarity)
        console.print(Panel(interpretation, title="[bold]Interpretation[/bold]", expand=False))
    
    def _get_interpretation(self, similarity: float) -> str:
        """Get interpretation of similarity score."""
        if similarity >= 0.95:
            return "[bold green]Very high similarity[/bold green] - Almost certainly the same entity"
        elif similarity >= 0.85:
            return "[bold green]High similarity[/bold green] - Would be clustered together as same entity"
        elif similarity >= 0.70:
            return "[yellow]Moderate similarity[/yellow] - Related but not clustered (below threshold)"
        elif similarity >= 0.50:
            return "[yellow]Low similarity[/yellow] - Somewhat related entities"
        else:
            return "[red]Very low similarity[/red] - Likely unrelated entities"
    
    def show_history(self):
        """Display history of comparisons."""
        if not self.history:
            console.print("[yellow]No comparison history yet.[/yellow]")
            return
        
        table = Table(show_header=True, header_style="bold cyan", title="[bold]Comparison History[/bold]")
        table.add_column("#", style="dim", width=4)
        table.add_column("Text 1", width=30)
        table.add_column("Text 2", width=30)
        table.add_column("Similarity", style="yellow", width=10)
        table.add_column("Clustered?", width=10)
        
        for i, item in enumerate(self.history[-10:], 1):  # Show last 10
            text1 = item['text1'][:27] + "..." if len(item['text1']) > 30 else item['text1']
            text2 = item['text2'][:27] + "..." if len(item['text2']) > 30 else item['text2']
            sim = f"{item['similarity']:.4f}"
            clustered = "[green]✓[/green]" if item['above_threshold'] else "[red]✗[/red]"
            
            table.add_row(str(i), text1, text2, sim, clustered)
        
        console.print(table)
    
    def run_examples(self):
        """Run example comparisons."""
        examples = [
            ("Barack Obama", "Obama"),
            ("Barack Obama", "President Obama"),
            ("United States", "USA"),
            ("New York", "New York City"),
            ("John Smith", "J. Smith"),
            ("Apple Inc.", "Apple"),
            ("John Smith", "Jane Smith"),
            ("New York", "Los Angeles"),
        ]
        
        console.print("[bold cyan]Running example comparisons...[/bold cyan]\n")
        
        for text1, text2 in examples:
            similarity = self.compute_similarity(text1, text2)
            self.display_result(text1, text2, similarity)
            console.print("")
    
    def run_interactive(self):
        """Run interactive mode."""
        console.print("\n[bold cyan]Embedding Similarity Interactive Tester[/bold cyan]")
        console.print(f"[dim]Using Qwen3-Embedding-8B model[/dim]")
        console.print(f"[dim]Current threshold: {self.threshold}[/dim]\n")
        
        console.print("[bold]Commands:[/bold]")
        console.print("  - Enter two texts to compare (press Enter after each)")
        console.print("  - 'examples' - Run predefined examples")
        console.print("  - 'history' - Show comparison history")
        console.print("  - 'threshold <value>' - Change clustering threshold")
        console.print("  - 'clear' - Clear history")
        console.print("  - 'help' - Show this help")
        console.print("  - 'quit' or 'exit' - Exit the tool\n")
        
        while True:
            try:
                text1 = Prompt.ask("\n[bold yellow]Text 1[/bold yellow]")
                
                if text1.lower() in ['quit', 'exit']:
                    console.print("[bold blue]Goodbye![/bold blue]")
                    break
                
                elif text1.lower() == 'help':
                    self.run_interactive()  # Show help again
                    return
                
                elif text1.lower() == 'examples':
                    self.run_examples()
                    continue
                
                elif text1.lower() == 'history':
                    self.show_history()
                    continue
                
                elif text1.lower() == 'clear':
                    self.history = []
                    console.print("[green]History cleared[/green]")
                    continue
                
                elif text1.lower().startswith('threshold'):
                    parts = text1.split()
                    if len(parts) > 1:
                        try:
                            self.threshold = float(parts[1])
                            console.print(f"[green]Threshold set to: {self.threshold}[/green]")
                        except ValueError:
                            console.print("[red]Invalid threshold value[/red]")
                    continue
                
                text2 = Prompt.ask("[bold yellow]Text 2[/bold yellow]")
                
                if not text1 or not text2:
                    console.print("[red]Both texts must be non-empty![/red]")
                    continue
                
                # Compute and display similarity
                console.print("\n[cyan]Computing similarity...[/cyan]")
                similarity = self.compute_similarity(text1, text2)
                self.display_result(text1, text2, similarity)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
            except Exception as e:
                console.print(f"[bold red]Error:[/bold red] {str(e)}")


if __name__ == "__main__":
    tester = EmbeddingSimilarityTester()
    tester.run_interactive()