#!/usr/bin/env python3
"""
Interactive Question Decomposition Testing Tool

This tool allows you to:
- Test question decomposition with different prompts
- Modify decomposition prompts interactively
- Save and load prompt versions
- Compare decomposition results
- Debug and optimize prompts in real-time
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import sys
import tempfile
import subprocess
import os

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.columns import Columns
from rich.text import Text
from rich.markdown import Markdown

# OpenAI for question decomposition
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

console = Console()

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Default decomposition prompt
DEFAULT_PROMPT = """Your task is to break down a complex multi-hop question into the most efficient sequence of single-hop, **atomic** questions.

## Your Main Goal: Build Smart Bridges, Don't Just Collect Nouns
The most critical skill is to convert complex logical clauses (like "despite," "the country where," "the year before") into a single, powerful **bridging question**. This question should use a known entity as context to find the next one. Avoid finding all the entities separately and then trying to figure out how they connect.

---
## A Simple Analogy for Efficiency

**Question:** "What is the phone number of the mother of the tallest player on the Lakers?"

** Inefficient Path:**
1.  Who are the players on the Lakers?
2.  What are all their heights?
3.  Who is the mother of the tallest player? *(This step is a logical leap)*

** Efficient Path:**
1.  Who is the tallest player on the Lakers?
2.  Who is the mother of `<ENTITY_Q1>`?
3.  What is the phone number of `<ENTITY_Q2>`?

---
## How to Decompose a Question
This process follows a logical flow from high-level analysis to the fine-tuning of your question chain.

### 1. Analyze the Query's Components
First, break down the original question into its fundamental building blocks. Identify the core **entities** (people, places, organizations), their **properties** (attributes like rank, location, date), and the **relationships** that connect them.

### 2. Construct an Atomic Chain
Next, formulate a sequence of questions where each question retrieves a single fact.
* **Isolate Comparisons:** Don't ask "who is faster?" Ask for the specific rank or time of each person involved.
* **Link with Placeholders:** Use `<ENTITY_Qn>` to pass the answer from a previous question (`Qn`) into the next one.

### 3. Optimize for Efficiency and Precision
Your final goal is the **shortest and most direct path** to the answer.
* **Embed Constraints to Build Bridges:** If a piece of information is only a filter (like a date or location), embed it as a constraint in the next question instead of asking for it directly.


## Formatting
Format each decomposed question as follows:

<decomposition>
Question: [the question text]
Requires retrieval: [true/false]
</decomposition>

- Any question that requires factual information from a knowledge base **MUST** have `Requires retrieval: true`.
- A question only has `Requires retrieval: false` if it involves a simple logical step or comparison based *only* on the previously retrieved answers (this is rare).

---

## Gold Standard Example (Atomic Decomposition)

Question: "When was the town where the headquarters of the only music label larger than the label that produced Take Me to the Ball Game explored?"

**Correct Decomposition (Atomic):**
<decomposition>
1. Question: Which label produced Take Me to the Ball Game?
   Requires retrieval: true
2. Question: What is the ranking of <ENTITY_Q1> among music labels?
   Requires retrieval: true
3. Question: Which music label is the larger than <ENTITY_Q2> in the country?
   Requires retrieval: true
4. Question: Where are the headquarters of <ENTITY_Q3> located?
   Requires retrieval: true
5. Question: When was <ENTITY_Q4> explored?
   Requires retrieval: true
</decomposition>

*Reasoning (handled by the system later): The logic correctly separates the lookup for the first label (StarTone), its rank (second), the label with the higher rank (Harmonia), its location (Clearwater), and the final fact about that location (1823). No single question attempts to bridge these facts.*

---

## Efficiency Example: Good vs. Bad Decomposition

Question: "What was the political party of the U.S. President who signed the Civil Rights Act of 1964, despite having previously led the party whose southern bloc largely opposed it?"

** Inefficient Decomposition (Avoid This):**
<decomposition>
1.  Question: Which political party's southern bloc opposed the Civil Rights Act of 1964?
    Requires retrieval: true
2.  Question: Who signed the Civil Rights Act of 1964?
    Requires retrieval: true
3.  Question: What was the political party of <ENTITY_Q2>?
    Requires retrieval: true
</decomposition>
*Reasoning for avoidance: This chain is broken. Step 1 finds a political party, but that information is never used. Step 2 makes a logical leap to find the president, completely ignoring the complex clause. This fails to follow the logic of the original question.*

** Efficient Decomposition (Correct):**
<decomposition>
1.  Question: Which political party's southern bloc largely opposed the Civil Rights Act of 1964?
    Requires retrieval: true
2.  Question: Which U.S. President, who was previously a Senate Majority Leader for the `<ENTITY_Q1>`, signed the Civil Rights Act of 1964?
    Requires retrieval: true
3.  Question: What was the political party of `<ENTITY_Q2>`?
    Requires retrieval: true
</decomposition>
*Reasoning for correctness: This chain is efficient and logically sound. Step 2 is a perfect "contextual bridge." It uses the party from Step 1 as a constraint to resolve the "despite" clause and identify the correct person (Lyndon B. Johnson), ensuring the full logic of the question is followed.*

---

## Further Examples

Question: "When was the first establishment that Mc-Donaldization is named after, open in the country Horndean is located?"
Decomposition:
<decomposition>
1. Question: What is McDonaldization named after?
   Requires retrieval: true
2. Question: Which state is Horndean located in?
   Requires retrieval: true
3. Question: When did the first <ENTITY_Q1> open in <ENTITY_Q2>?
   Requires retrieval: true
</decomposition>
Question: "How many Germans live in the colonial holding in Aruba's continent that was governed by Prazeres's country?
Decomposition:
<decomposition>
1. Question: In what continent is Aruba located?
   Requires retrieval: true
2. Question: What country is Prazeres?
   Requires retrieval: true
3. Question: Colonial holding in <ENTITY_Q1> governed by <ENTITY_Q2>?
   Requires retrieval: true
4. How many Germans live in <ENTITY_Q3>?
   Requires retrieval: true
</decomposition>

Question: "When did the people who captured Malakoff come to the region where Philipsburg is located?
Decomposition:
<decomposition>
1. Question: What is Philipsburg capital of?
   Requires retrieval: true
2. Question: What terrain feature is <ENTITY_Q1> located in?
   Requires retrieval: true
3. Who captured Malakoff?
   Requires retrieval: true
4. When did <ENTITY_Q3> come to <ENTITY_Q4>?
   Requires retrieval: true
</decomposition>

## Important Constraints
-   **AVOID YES/NO QUESTIONS.**
-   **AVOID OVER-DECOMPOSITION.** Each question should seek a meaningful entity or property.
-   DON'T break "When was John Doe born?" into "Who is John Doe?" -> "English", then "When was English born?".
-   DO ask directly: "When was John Doe born?".

Now decompose this question with provided format:
Question: "{question}"
Decomposition:"""

# Old prompt from multi_hop_qa_chains_n_hops.py for comparison
OLD_PROMPT = """Break down this multi-hop question into a sequence of single-hop questions.
The FIRST question should keep the original specific entity/information from the question.
SUBSEQUENT questions should use <ENTITY> as a placeholder if it requires answers from the previous step to form the question.

IMPORTANT: Avoid over-decomposition. Avoid yes/no questions. Each question should extract meaningful entities (proper nouns like names, places), not single-word descriptors. Keep questions at an appropriate granularity level.

For each question, indicate whether it requires retrieval from the knowledge base, any question that requires factual information MUST require retrieval.
The only case where retreival is not required is if the question just requires comparison of responses from previous questions.

Format each decomposed question as:
Question: [the question text]
Requires retrieval: [true/false]

Examples:

Question: "What is the birth year of the spouse of the director of Casablanca?"
Decomposition:
1. Question: Who directed Casablanca?
   Requires retrieval: true
2. Question: Who was <ENTITY_Q1>'s spouse?
   Requires retrieval: true
3. Question: What is <ENTITY_Q2>'s birth year?
   Requires retrieval: true

Question: "Which film has the director who is older, Dune or The Dark Knight?"
Decomposition:
1. Question: Who directed Dune?
   Requires retrieval: true
2. Question: Who directed The Dark Knight?
   Requires retrieval: true
3. Question: When was <ENTITY_Q1> born?
   Requires retrieval: true
4. Question: When was <ENTITY_Q2> born?
   Requires retrieval: true
5. Question: Who is older, <ENTITY_Q3> or <ENTITY_Q4>?
   Requires retrieval: false


IMPORTANT:
    AVOID over-decomposition like this:
    DON'T break "Who is John Doe?" into:
    1. Who is John Doe? → "English"
    2. When was <ENTITY_Q1> born? → "When was English born?"

    DO ask directly: "When was John Doe born?"

Now decompose this question:
Question: "{question}"
Decomposition:"""


class QuestionDecomposer:
    """Interactive question decomposition testing tool."""
    
    def __init__(self, model: str = "gpt-4o", verbose: bool = True):
        """Initialize the decomposer.
        
        Args:
            model: OpenAI model to use for decomposition
            verbose: Whether to show detailed output
        """
        self.model = model
        self.verbose = verbose
        self.current_prompt = DEFAULT_PROMPT
        self.prompt_history = []
        self.saved_prompts = {}
        self.decomposition_cache = {}
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI()
                if verbose:
                    console.print(f"[green]✓ OpenAI client initialized (model: {model})[/green]")
            except Exception as e:
                console.print(f"[red]Error initializing OpenAI: {e}[/red]")
        else:
            console.print("[red]OpenAI not available. Please install with: pip install openai[/red]")
    
    def decompose_question(self, question: str, prompt_template: Optional[str] = None) -> List[Dict[str, Any]]:
        """Decompose a multi-hop question using the current or specified prompt.
        
        Args:
            question: The question to decompose
            prompt_template: Optional custom prompt template (uses current_prompt if None)
            
        Returns:
            List of decomposed questions with metadata
        """
        if not self.openai_client:
            console.print("[red]OpenAI client not available[/red]")
            return []
        
        template = prompt_template or self.current_prompt
        prompt = template.format(question=question)
        
        try:
            response = self.openai_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that breaks down complex questions into simple steps."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                seed=42,
                max_tokens=300
            )
            
            decomposition_text = response.choices[0].message.content
            console.print(f"[green]✓ Decomposition successful[/green]")
            console.print(f"[cyan]{decomposition_text}[/cyan]")
            
            # Parse the response
            questions = []
            lines = decomposition_text.strip().split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Handle various formats: "1. Question:", "Question:", "- Question:", etc.
                if re.match(r'^[\d]+[\.)\s]*Question:', line) or line.startswith('Question:') or re.match(r'^[-*]\s*Question:', line):
                    question_match = re.search(r'Question:\s*(.+)', line)
                    if question_match:
                        question_text = question_match.group(1).strip()
                        
                        # Check for requires_retrieval flag
                        requires_retrieval = True  # Default
                        
                        if 'Requires retrieval:' in line:
                            retrieval_match = re.search(r'Requires retrieval:\s*(true|false)', line, re.IGNORECASE)
                            if retrieval_match:
                                requires_retrieval = retrieval_match.group(1).lower() == 'true'
                        elif i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            # Check if next line has requires retrieval (handle indented format too)
                            if 'Requires retrieval:' in next_line or re.match(r'^[-*]\s*Requires retrieval:', next_line):
                                retrieval_match = re.search(r'Requires retrieval:\s*(true|false)', next_line, re.IGNORECASE)
                                if retrieval_match:
                                    requires_retrieval = retrieval_match.group(1).lower() == 'true'
                                i += 1
                        
                        questions.append({
                            "question": question_text,
                            "requires_retrieval": requires_retrieval
                        })
                # Handle markdown formatted output when both markers are on the same line
                elif "**Question:**" in line and "**Requires retrieval:**" in line:
                    # Extract question text between the two markers
                    question_match = re.search(r'\*\*Question:\*\*\s*(.+?)\s*\*\*Requires retrieval:\*\*', line)
                    if question_match:
                        question_text = question_match.group(1).strip()
                        
                        # Extract requires_retrieval value
                        requires_retrieval = True  # Default
                        retrieval_match = re.search(r'\*\*Requires retrieval:\*\*\s*(true|false)', line, re.IGNORECASE)
                        if retrieval_match:
                            requires_retrieval = retrieval_match.group(1).lower() == 'true'
                        
                        questions.append({
                            "question": question_text,
                            "requires_retrieval": requires_retrieval
                        })
                # Handle markdown formatted output with **Question:** (retrieval might be on next line)
                elif "**Question:**" in line:
                    # Extract question text - it ends at end of line since no **Requires retrieval:** on this line
                    question_match = re.search(r'\*\*Question:\*\*\s*(.+)', line)
                    if question_match:
                        question_text = question_match.group(1).strip()
                        
                        # Check for requires_retrieval flag on next line
                        requires_retrieval = True  # Default
                        
                        if i + 1 < len(lines):
                            next_line = lines[i + 1].strip()
                            if "**Requires retrieval:**" in next_line:
                                retrieval_match = re.search(r'\*\*Requires retrieval:\*\*\s*(true|false)', next_line, re.IGNORECASE)
                                if retrieval_match:
                                    requires_retrieval = retrieval_match.group(1).lower() == 'true'
                                i += 1  # Skip the next line since we've processed it
                        
                        questions.append({
                            "question": question_text,
                            "requires_retrieval": requires_retrieval
                        })
                
                i += 1
            
            # Store in cache
            cache_key = f"{question}|||{template[:100]}"
            self.decomposition_cache[cache_key] = {
                "result": questions,
                "raw_response": decomposition_text
            }
            
            return questions
            
        except Exception as e:
            console.print(f"[red]Error during decomposition: {e}[/red]")
            return []
    
    def display_decomposition(self, question: str, decomposed: List[Dict[str, Any]], title: str = "Decomposition"):
        """Display decomposition results in a nice format.
        
        Args:
            question: Original question
            decomposed: List of decomposed questions
            title: Panel title
        """
        content = f"[bold]Original:[/bold] {question}\n\n"
        content += "[bold]Decomposed Questions:[/bold]\n"
        
        for i, q in enumerate(decomposed, 1):
            retrieval = "✓" if q["requires_retrieval"] else "✗"
            content += f"{i}. {q['question']} [retrieval: {retrieval}]\n"
        
        panel = Panel(content, title=title, expand=False, border_style="cyan")
        console.print(panel)
    
    def compare_decompositions(self, question: str, prompt1: str, prompt2: str):
        """Compare decompositions from two different prompts side by side.
        
        Args:
            question: The question to decompose
            prompt1: First prompt template
            prompt2: Second prompt template
        """
        console.print(f"\n[bold cyan]Comparing Decompositions[/bold cyan]")
        console.print(f"[dim]Question: {question}[/dim]\n")
        
        # Get decompositions
        result1 = self.decompose_question(question, prompt1)
        result2 = self.decompose_question(question, prompt2)
        
        # Create side-by-side display
        panel1_content = ""
        for i, q in enumerate(result1, 1):
            retrieval = "✓" if q["requires_retrieval"] else "✗"
            panel1_content += f"{i}. {q['question'][:50]}... [{retrieval}]\n"
        
        panel2_content = ""
        for i, q in enumerate(result2, 1):
            retrieval = "✓" if q["requires_retrieval"] else "✗"
            panel2_content += f"{i}. {q['question'][:50]}... [{retrieval}]\n"
        
        panel1 = Panel(panel1_content or "No decomposition", title="Prompt 1", border_style="green")
        panel2 = Panel(panel2_content or "No decomposition", title="Prompt 2", border_style="blue")
        
        columns = Columns([panel1, panel2], equal=True)
        console.print(columns)
    
    def edit_prompt_interactive(self):
        """Open the current prompt in an editor for modification."""
        # Create temporary file with current prompt
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(self.current_prompt)
            temp_path = f.name
        
        # Try to open with system editor
        editor = None
        for candidate in ['nano', 'vim', 'vi', 'emacs', 'notepad']:
            try:
                subprocess.run(['which', candidate], capture_output=True, check=True)
                editor = candidate
                break
            except:
                continue
        
        if editor:
            console.print(f"[cyan]Opening editor ({editor})...[/cyan]")
            subprocess.run([editor, temp_path])
            
            # Read modified content
            with open(temp_path, 'r') as f:
                new_prompt = f.read()
            
            # Clean up
            Path(temp_path).unlink()
            
            if new_prompt != self.current_prompt:
                self.prompt_history.append(self.current_prompt)
                self.current_prompt = new_prompt
                console.print("[green]✓ Prompt updated[/green]")
            else:
                console.print("[yellow]No changes made[/yellow]")
        else:
            console.print("[red]No suitable editor found. Please edit manually.[/red]")
            console.print("\n[cyan]Current prompt:[/cyan]")
            console.print(Panel(self.current_prompt, expand=False))
            
            console.print("\n[yellow]Paste your new prompt (end with Ctrl+D on Unix or Ctrl+Z on Windows):[/yellow]")
            lines = []
            try:
                while True:
                    lines.append(input())
            except EOFError:
                pass
            
            new_prompt = '\n'.join(lines)
            if new_prompt and new_prompt != self.current_prompt:
                self.prompt_history.append(self.current_prompt)
                self.current_prompt = new_prompt
                console.print("[green]✓ Prompt updated[/green]")
    
    def save_prompt(self, name: str, filepath: Optional[Path] = None):
        """Save current prompt to memory or file.
        
        Args:
            name: Name for the saved prompt
            filepath: Optional file path to save to
        """
        self.saved_prompts[name] = self.current_prompt
        
        if filepath:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                "name": name,
                "prompt": self.current_prompt,
                "timestamp": datetime.now().isoformat(),
                "model": self.model
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            
            console.print(f"[green]✓ Prompt saved to {filepath}[/green]")
        else:
            console.print(f"[green]✓ Prompt saved as '{name}'[/green]")
    
    def load_prompt(self, name_or_path: str):
        """Load a saved prompt by name or from file.
        
        Args:
            name_or_path: Name of saved prompt or file path
        """
        if name_or_path in self.saved_prompts:
            self.prompt_history.append(self.current_prompt)
            self.current_prompt = self.saved_prompts[name_or_path]
            console.print(f"[green]✓ Loaded prompt '{name_or_path}'[/green]")
        else:
            filepath = Path(name_or_path)
            if filepath.exists():
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                self.prompt_history.append(self.current_prompt)
                self.current_prompt = data["prompt"]
                self.saved_prompts[data["name"]] = data["prompt"]
                console.print(f"[green]✓ Loaded prompt from {filepath}[/green]")
            else:
                console.print(f"[red]Prompt '{name_or_path}' not found[/red]")
    
    def batch_test(self, questions: List[str]):
        """Test decomposition on multiple questions.
        
        Args:
            questions: List of questions to test
        """
        results = []
        
        console.print(f"\n[bold cyan]Batch Testing {len(questions)} Questions[/bold cyan]\n")
        
        for i, question in enumerate(questions, 1):
            console.print(f"[cyan]Question {i}/{len(questions)}:[/cyan] {question}")
            
            decomposed = self.decompose_question(question)
            results.append({
                "original": question,
                "decomposed": decomposed,
                "count": len(decomposed),
                "retrieval_count": sum(1 for q in decomposed if q["requires_retrieval"])
            })
            
            # Brief summary
            console.print(f"  → {len(decomposed)} sub-questions, {results[-1]['retrieval_count']} require retrieval\n")
        
        # Summary table
        table = Table(title="Batch Test Summary")
        table.add_column("Question", style="cyan", no_wrap=False)
        table.add_column("Sub-Q", justify="center")
        table.add_column("Retrieval", justify="center")
        
        for result in results:
            table.add_row(
                result["original"][:50] + "..." if len(result["original"]) > 50 else result["original"],
                str(result["count"]),
                str(result["retrieval_count"])
            )
        
        console.print(table)
        return results
    
    def run_interactive(self):
        """Run the interactive testing interface."""
        console.print("\n[bold cyan]🔍 Interactive Question Decomposition Tester[/bold cyan]")
        console.print("Commands:")
        console.print("  - Type a question to decompose it")
        console.print("  - 'edit' - Edit the current prompt")
        console.print("  - 'show' - Show current prompt")
        console.print("  - 'save <name>' - Save current prompt")
        console.print("  - 'load <name>' - Load saved prompt")
        console.print("  - 'compare' - Compare two prompts")
        console.print("  - 'oldnew' - Compare old vs new prompts")
        console.print("  - 'batch' - Test batch of questions")
        console.print("  - 'history' - Show prompt history")
        console.print("  - 'export' - Export results to file")
        console.print("  - 'model <name>' - Change model")
        console.print("  - 'help' - Show this help")
        console.print("  - 'quit' - Exit\n")
        
        sample_questions = [
            "What is the birth year of the spouse of the director of Casablanca?",
            "When did Lothair II's mother die?",
            "Which film has the director who is older, Dune or The Dark Knight?",
            "What nationality is the director of the film in which Hugh Jackman played Wolverine?",
            "In which city was the lead actor of Inception born?"
        ]
        
        while True:
            try:
                user_input = Prompt.ask("\n[bold yellow]Enter command or question[/bold yellow]")
                
                if user_input.lower() == 'quit':
                    console.print("[bold blue]Goodbye![/bold blue]")
                    break
                
                elif user_input.lower() == 'help':
                    console.print("\nSample questions to try:")
                    for q in sample_questions[:3]:
                        console.print(f"  - {q}")
                
                elif user_input.lower() == 'edit':
                    self.edit_prompt_interactive()
                
                elif user_input.lower() == 'show':
                    console.print("\n[bold]Current Prompt Template:[/bold]")
                    syntax = Syntax(self.current_prompt, "text", theme="monokai", line_numbers=True)
                    console.print(Panel(syntax, expand=False, border_style="cyan"))
                
                elif user_input.lower().startswith('save '):
                    name = user_input[5:].strip()
                    if name:
                        save_to_file = Confirm.ask("Save to file?", default=False)
                        if save_to_file:
                            filepath = Prompt.ask("File path", default=f"prompts/{name}.json")
                            self.save_prompt(name, Path(filepath))
                        else:
                            self.save_prompt(name)
                
                elif user_input.lower().startswith('load '):
                    name_or_path = user_input[5:].strip()
                    if name_or_path:
                        self.load_prompt(name_or_path)
                
                elif user_input.lower() == 'oldnew':
                    console.print("\n[bold cyan]Comparing Old Prompt vs New Prompt[/bold cyan]")
                    question = Prompt.ask("Question to test")
                    
                    console.print("\n[yellow]Running decompositions...[/yellow]")
                    
                    # Get decompositions
                    console.print("\n[dim]Using OLD prompt (from multi_hop_qa_chains_n_hops.py)...[/dim]")
                    old_result = self.decompose_question(question, OLD_PROMPT)
                    
                    console.print("\n[dim]Using NEW prompt (current DEFAULT_PROMPT with enhanced instructions)...[/dim]")
                    new_result = self.decompose_question(question, DEFAULT_PROMPT)
                    
                    # Create detailed comparison display
                    console.print(f"\n[bold cyan]Comparison Results for:[/bold cyan] {question}\n")
                    
                    # Old prompt panel
                    old_content = "[bold]Old Prompt (Simple):[/bold]\n"
                    for i, q in enumerate(old_result, 1):
                        retrieval = "✓" if q["requires_retrieval"] else "✗"
                        old_content += f"{i}. {q['question']} [retrieval: {retrieval}]\n"
                    
                    # New prompt panel  
                    new_content = "[bold]New Prompt (Enhanced):[/bold]\n"
                    for i, q in enumerate(new_result, 1):
                        retrieval = "✓" if q["requires_retrieval"] else "✗"
                        new_content += f"{i}. {q['question']} [retrieval: {retrieval}]\n"
                    
                    panel_old = Panel(old_content or "No decomposition", title="Old Prompt", border_style="red")
                    panel_new = Panel(new_content or "No decomposition", title="New Prompt", border_style="green")
                    
                    columns = Columns([panel_old, panel_new], equal=True)
                    console.print(columns)
                    
                    # Summary comparison
                    console.print(f"\n[bold]Summary:[/bold]")
                    console.print(f"  Old: {len(old_result)} questions, {sum(1 for q in old_result if q['requires_retrieval'])} requiring retrieval")
                    console.print(f"  New: {len(new_result)} questions, {sum(1 for q in new_result if q['requires_retrieval'])} requiring retrieval")
                
                elif user_input.lower() == 'compare':
                    console.print("\n[cyan]Select prompts to compare:[/cyan]")
                    
                    # List available prompts
                    if self.saved_prompts:
                        console.print("Saved prompts:")
                        for name in self.saved_prompts:
                            console.print(f"  - {name}")
                    
                    use_current = Confirm.ask("Use current prompt as first?", default=True)
                    if use_current:
                        prompt1 = self.current_prompt
                    else:
                        name1 = Prompt.ask("First prompt name")
                        prompt1 = self.saved_prompts.get(name1, self.current_prompt)
                    
                    name2 = Prompt.ask("Second prompt name (or 'default' for original)")
                    if name2.lower() == 'default':
                        prompt2 = DEFAULT_PROMPT
                    else:
                        prompt2 = self.saved_prompts.get(name2, self.current_prompt)
                    
                    question = Prompt.ask("Question to test")
                    self.compare_decompositions(question, prompt1, prompt2)
                
                elif user_input.lower() == 'batch':
                    console.print("\n[cyan]Batch testing mode[/cyan]")
                    use_samples = Confirm.ask("Use sample questions?", default=True)
                    
                    if use_samples:
                        test_questions = sample_questions
                    else:
                        console.print("Enter questions (one per line, empty line to finish):")
                        test_questions = []
                        while True:
                            q = input()
                            if not q:
                                break
                            test_questions.append(q)
                    
                    if test_questions:
                        self.batch_test(test_questions)
                
                elif user_input.lower() == 'history':
                    if self.prompt_history:
                        console.print("\n[bold]Prompt History:[/bold]")
                        for i, prompt in enumerate(self.prompt_history[-5:], 1):
                            console.print(f"\n[cyan]Version -{len(self.prompt_history) - i + 1}:[/cyan]")
                            console.print(Panel(prompt[:200] + "..." if len(prompt) > 200 else prompt, expand=False))
                    else:
                        console.print("[yellow]No prompt history available[/yellow]")
                
                elif user_input.lower() == 'export':
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"decomposition_results_{timestamp}.json"
                    
                    export_data = {
                        "timestamp": datetime.now().isoformat(),
                        "model": self.model,
                        "current_prompt": self.current_prompt,
                        "saved_prompts": self.saved_prompts,
                        "cache": self.decomposition_cache
                    }
                    
                    with open(filename, 'w') as f:
                        json.dump(export_data, f, indent=2)
                    
                    console.print(f"[green]✓ Results exported to {filename}[/green]")
                
                elif user_input.lower().startswith('model '):
                    new_model = user_input[6:].strip()
                    if new_model:
                        self.model = new_model
                        console.print(f"[green]✓ Model changed to {new_model}[/green]")
                
                else:
                    # Treat as a question to decompose
                    decomposed = self.decompose_question(user_input)
                    
                    if decomposed:
                        self.display_decomposition(user_input, decomposed)
                        
                        # Ask if user wants to see raw response
                        show_raw = Confirm.ask("Show raw LLM response?", default=False)
                        if show_raw:
                            cache_key = f"{user_input}|||{self.current_prompt[:100]}"
                            if cache_key in self.decomposition_cache:
                                raw = self.decomposition_cache[cache_key]["raw_response"]
                                console.print("\n[bold]Raw Response:[/bold]")
                                console.print(Panel(raw, expand=False, border_style="dim"))
                    else:
                        console.print("[red]Failed to decompose question[/red]")
                        console.print(f"[red]{decomposed}[/red]")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Use 'quit' to exit[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Interactive Question Decomposition Tester")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use")
    parser.add_argument("--quiet", action="store_true", help="Less verbose output")
    
    args = parser.parse_args()
    
    try:
        decomposer = QuestionDecomposer(model=args.model, verbose=not args.quiet)
        decomposer.run_interactive()
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())