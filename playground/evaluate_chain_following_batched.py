#!/usr/bin/env python3
"""
Batched Chain-Following Multi-Hop Question Answering Evaluation Script

Uses curator for parallel LLM calls to significantly speed up evaluation.
Processes decomposition and answer generation in batches while maintaining
compatibility with the ChainFollowingMultiHopQA retrieval logic.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from dataclasses import dataclass
from datetime import datetime
import time

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import our chain-following multi-hop QA system for retrieval
from playground.multi_hop_qa_chains_n_hops import ChainFollowingMultiHopQA

# Import evaluation utilities
from src.gsw_memory.evaluation.hipporag_eval import evaluate_qa_batch, format_evaluation_report

# Import curator for batched LLM processing
try:
    from bespokelabs import curator
    CURATOR_AVAILABLE = True
except ImportError:
    print("Warning: Curator not available. Install with: pip install bespokelabs-curator")
    CURATOR_AVAILABLE = False

console = Console()

# Setup logging
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"chain_following_eval_batched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


class QuestionDecomposer(curator.LLM):
    """Curator class for decomposing multi-hop questions in parallel."""
    
    return_completions_object = True
    
    def __init__(self, **kwargs):
        """Initialize the question decomposer."""
        super().__init__(**kwargs)
    
    def prompt(self, input):
        """Create a decomposition prompt for each question."""
        decomposition_prompt = f"""Your task is to break down a complex multi-hop question into the most efficient sequence of single-hop, **atomic** questions.

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
Format each decomposed question as:
Question: [the question text]
Requires retrieval: [true/false]

- Any question that requires factual information from a knowledge base **MUST** have `Requires retrieval: true`.
- A question only has `Requires retrieval: false` if it involves a simple logical step or comparison based *only* on the previously retrieved answers (this is rare).

---

## Gold Standard Example (Atomic Decomposition)

Question: "When was the town where the headquarters of the only music label larger than the label that produced Take Me to the Ball Game explored?"

**Correct Decomposition (Atomic):**
1. Question: Which label produced Take Me to the Ball Game?
   Requires retrieval: true
2. Question: What is the ranking of StarTone Records among music labels?
   Requires retrieval: true
3. Question: Which music label is the larger than <ENTITY_Q2> in the country?
   Requires retrieval: true
4. Question: Where are the headquarters of <ENTITY_Q3> located?
   Requires retrieval: true
5. Question: When was <ENTITY_Q4> explored?
   Requires retrieval: true

*Reasoning (handled by the system later): The logic correctly separates the lookup for the first label (StarTone), its rank (second), the label with the higher rank (Harmonia), its location (Clearwater), and the final fact about that location (1823). No single question attempts to bridge these facts.*

---

## Efficiency Example: Good vs. Bad Decomposition

Question: "What was the political party of the U.S. President who signed the Civil Rights Act of 1964, despite having previously led the party whose southern bloc largely opposed it?"

** Inefficient Decomposition (Avoid This):**
1.  Question: Which political party's southern bloc opposed the Civil Rights Act of 1964?
    Requires retrieval: true
2.  Question: Who signed the Civil Rights Act of 1964?
    Requires retrieval: true
3.  Question: What was the political party of <ENTITY_Q2>?
    Requires retrieval: true
*Reasoning for avoidance: This chain is broken. Step 1 finds a political party, but that information is never used. Step 2 makes a logical leap to find the president, completely ignoring the complex clause. This fails to follow the logic of the original question.*

** Efficient Decomposition (Correct):**
1.  Question: Which political party's southern bloc largely opposed the Civil Rights Act of 1964?
    Requires retrieval: true
2.  Question: Which U.S. President, who was previously a Senate Majority Leader for the `<ENTITY_Q1>`, signed the Civil Rights Act of 1964?
    Requires retrieval: true
3.  Question: What was the political party of `<ENTITY_Q2>`?
    Requires retrieval: true
*Reasoning for correctness: This chain is efficient and logically sound. Step 2 is a perfect "contextual bridge." It uses the party from Step 1 as a constraint to resolve the "despite" clause and identify the correct person (Lyndon B. Johnson), ensuring the full logic of the question is followed.*

---

## Further Examples

Question: "When was the first establishment that Mc-Donaldization is named after, open in the country Horndean is located?"
Decomposition:
1. Question: What is McDonaldization named after?
   Requires retrieval: true
2. Question: Which state is Horndean located in?
   Requires retrieval: true
3. Question: When did the first <ENTITY_Q1> open in <ENTITY_Q2>?
   Requires retrieval: true
   
Question: "How many Germans live in the colonial holding in Aruba's continent that was governed by Prazeres's country?
Decomposition:
1. Question: In what continent is Aruba located?
   Requires retrieval: true
2. Question: What country is Prazeres?
   Requires retrieval: true
3. Question: Colonial holding in <ENTITY_Q1> governed by <ENTITY_Q2>?
   Requires retrieval: true
4. How many Germans live in <ENTITY_Q3>?
   Requires retrieval: true

Question: "When did the people who captured Malakoff come to the region where Philipsburg is located?
Decomposition:
1. Question: What is Philipsburg capital of?
   Requires retrieval: true
2. Question: What terrain feature is <ENTITY_Q1> located in?
   Requires retrieval: true
3. Who captured Malakoff?
   Requires retrieval: true
4. When did <ENTITY_Q3> come to <ENTITY_Q4>?
   Requires retrieval: true

## Important Constraints
-   **AVOID YES/NO QUESTIONS.**
-   **AVOID OVER-DECOMPOSITION.** Each question should seek a meaningful entity or property.
    -   DON'T break "When was John Doe born?" into "Who is John Doe?" -> "English", then "When was English born?".
    -   DO ask directly: "When was John Doe born?".

Now decompose this question:
Question: "{input['question']}"
Decomposition:"""
        
        return [
            {"role": "system", "content": "You are a helpful assistant that breaks down complex questions into simple steps."},
            {"role": "user", "content": decomposition_prompt}
        ]
    
    def parse(self, input, response):
        """Parse the decomposition response."""
        decomposition_text = response["choices"][0]["message"]["content"]
        
        # Parse the response into structured format
        questions = []
        lines = decomposition_text.strip().split('\n')
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            if re.match(r'^[\d]+[\.)\s]*Question:', line) or line.startswith('Question:') or re.match(r'^-\s*Question:', line):
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
                        if 'Requires retrieval:' in next_line:
                            retrieval_match = re.search(r'Requires retrieval:\s*(true|false)', next_line, re.IGNORECASE)
                            if retrieval_match:
                                requires_retrieval = retrieval_match.group(1).lower() == 'true'
                            i += 1
                    
                    questions.append({
                        "question": question_text,
                        "requires_retrieval": requires_retrieval
                    })
            
            i += 1
        
        # If parsing fails, return the original question
        if not questions:
            questions = [{"question": input['question'], "requires_retrieval": True}]
        
        return [{
            "question_id": input['question_id'],
            "original_question": input['question'],
            "decomposed_questions": questions,
            "raw_response": decomposition_text
        }]


class AnswerGenerator(curator.LLM):
    """Curator class for generating answers in parallel."""
    
    return_completions_object = True
    
    def __init__(self, **kwargs):
        """Initialize the answer generator."""
        super().__init__(**kwargs)
    
    def prompt(self, input):
        """Create an answer generation prompt."""
        evidence_text = '\n'.join(input['evidence'])
        
        # Include decomposition if available
        decomposition_text = ""
        if 'decomposed_questions' in input and input['decomposed_questions']:
            decomposition_text = f"""
In order to answer the question, the multi-hop question is broken down into the following single-hop questions:
Decomposition: {input['decomposed_questions']}
"""
        
        prompt_text = f"""Answer the following multi-hop question using ONLY the provided evidence.

Question: {input['question']}
{decomposition_text}

Available Evidence (Q&A pairs from knowledge base):
{evidence_text}

Instructions:
1. Use ONLY the Q&A pairs provided above
2. Be sure to check all the Q&A pairs for the answer
3. Do NOT use any external knowledge
4. If the evidence doesn't contain the answer, say "Cannot determine from available evidence"
5. Be concise and direct

Please respond in the following format, you MUST use the <reasoning> and <answer> tags and close them appropriately, since the evaluator will be looking for them:

<reasoning>
Reasoning about the question and the evidence.
</reasoning>
<answer>
Only the final answer, respond with a single word or phrase only.
</answer>
"""
        
        # Store the full prompt for later reference
        self._last_prompt = prompt_text
        
        return [
            {"role": "system", "content": "You are a helpful assistant that answers questions using only provided evidence."},
            {"role": "user", "content": prompt_text}
        ]
    
    def parse(self, input, response):
        """Parse the answer from the response."""
        answer_text = response["choices"][0]["message"]["content"]
        
        # Extract answer from tags with robust error handling
        final_answer = None
        
        if '<answer>' in answer_text:
            if '</answer>' in answer_text:
                # Both tags present - use regex
                answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', answer_text, re.DOTALL | re.IGNORECASE)
                if answer_match:
                    final_answer = answer_match.group(1).strip()
            else:
                # Only opening tag present - extract everything after it
                parts = answer_text.split('<answer>')
                if len(parts) > 1:
                    final_answer = parts[1].strip()
        
        # Fallback to full response if no answer was extracted
        if final_answer is None:
            final_answer = answer_text.strip()
        
        # Calculate token count for the user prompt only (excluding system prompt)
        token_count = 0
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
            # Count tokens in user prompt only (the variable part)
            token_count = len(encoding.encode(self._last_prompt))
        except ImportError:
            # tiktoken not available
            pass
        except Exception:
            # Any other error in token counting
            pass
        
        return [{
            "question_id": input['question_id'],
            "question": input['question'],
            "predicted_answer": final_answer,
            "full_response": answer_text,
            "evidence_count": len(input['evidence']),
            "final_prompt": self._last_prompt,
            "token_count": token_count
        }]


@dataclass
class BatchedChainEvaluationResult:
    """Container for batched chain-following evaluation results."""
    question_id: str
    question: str
    predicted_answer: str
    gold_answers: List[str]
    processing_time: float
    decomposed_questions: List[Dict[str, Any]]
    evidence_count: int
    token_count: int = 0
    chains_info: Dict[str, Any] = None  # Chain-specific information
    final_prompt: Optional[str] = None
    error: Optional[str] = None


class BatchedChainFollowingEvaluator:
    """Batched evaluator for chain-following multi-hop QA using curator for parallel LLM calls."""
    
    def __init__(self, num_documents: int = 200, num_questions: int = 20, verbose: bool = False,
                 chain_top_k: int = 15, max_entities_per_hop: int = 5):
        """Initialize batched chain-following evaluator.
        
        Args:
            num_documents: Number of documents to load
            num_questions: Number of questions to evaluate
            verbose: Whether to show detailed output
            chain_top_k: Number of top chains to select after reranking
            max_entities_per_hop: Maximum entities to consider at each hop
        """
        self.num_documents = num_documents
        self.num_questions = num_questions
        self.data_dir = Path("/home/yigit/codebase/gsw-memory/playground_data")
        self.verbose = verbose
        self.chain_top_k = chain_top_k
        self.max_entities_per_hop = max_entities_per_hop
        
        console.print("[bold blue]Initializing Batched Chain-Following QA Evaluator...[/bold blue]")
        console.print(f"  Chain selection: Top {chain_top_k} chains")
        console.print(f"  Max entities per hop: {max_entities_per_hop}")
        
        # Initialize the retrieval system (single instance for all questions)
        self.qa_system = ChainFollowingMultiHopQA(
            num_documents=num_documents, 
            verbose=False, 
            show_prompt=False,
            chain_top_k=chain_top_k,
            max_entities_per_hop=max_entities_per_hop
        )
        
        # Initialize curator classes if available
        if CURATOR_AVAILABLE:
            self.decomposer = QuestionDecomposer(model_name="gpt-4o", generation_params={"temperature": 0.0, "max_tokens": 600})
            self.answer_generator = AnswerGenerator(model_name="gpt-4o-mini", generation_params={"temperature": 0.0})
            console.print("[green]✓ Curator initialized for parallel processing[/green]")
        else:
            console.print("[yellow]⚠ Curator not available - will fall back to sequential processing[/yellow]")
        
        console.print(f"[cyan]Ready to evaluate {num_questions} questions on {num_documents} documents[/cyan]")
    
    def load_questions_and_answers(self) -> List[Tuple[str, str, List[str]]]:
        """Load questions and gold answers from Musique dataset.
        
        Returns:
            List of tuples: (question_id, question, gold_answers)
        """
        console.print("[cyan]Loading Musique questions...[/cyan]")
        
        questions_file = self.data_dir / "musique.json"
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        with open(questions_file, 'r') as f:
            data = json.load(f)
        
        # Extract first N questions
        questions_data = []
        for i, item in enumerate(data[:self.num_questions]):
            question_id = item.get("id", f"q_{i}")
            question = item["question"]
            gold_answers = item.get("answer", [])
            
            # Ensure gold_answers is a list
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
            
            # Optional: filter by hop count if needed
            # if "3hop1" not in question_id:
            #     continue
            
            if "3hop1" not in question_id:
                continue
            questions_data.append((question_id, question, gold_answers))
        
        console.print(f"[green]✓ Loaded {len(questions_data)} questions[/green]")
        return questions_data
    
    def run_batched_decomposition(self, questions_data: List[Tuple[str, str, List[str]]]) -> Dict[str, Dict]:
        """Run batched question decomposition using curator.
        
        Args:
            questions_data: List of (question_id, question, gold_answers)
            
        Returns:
            Dictionary mapping question_id to decomposition results
        """
        console.print("\n[bold cyan]Stage 1: Batched Question Decomposition[/bold cyan]")
        
        if not CURATOR_AVAILABLE:
            # Fallback to sequential processing
            console.print("[yellow]Using sequential decomposition (curator not available)[/yellow]")
            decomposition_results = {}
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
                task = progress.add_task("Decomposing questions...", total=len(questions_data))
                
                for question_id, question, _ in questions_data:
                    decomposed = self.qa_system.decompose_question(question)
                    decomposition_results[question_id] = {
                        "original_question": question,
                        "decomposed_questions": decomposed
                    }
                    progress.update(task, advance=1)
            
            return decomposition_results
        
        # Prepare inputs for batched decomposition
        decompose_inputs = [
            {"question_id": qid, "question": question}
            for qid, question, _ in questions_data
        ]
        
        console.print(f"[cyan]Decomposing {len(decompose_inputs)} questions in parallel...[/cyan]")
        start_time = time.time()
        
        # Run batched decomposition
        decomposition_dataset = self.decomposer(decompose_inputs)
        
        elapsed = time.time() - start_time
        console.print(f"[green]✓ Decomposition complete in {elapsed:.1f}s ({elapsed/len(decompose_inputs):.2f}s per question)[/green]")
        
        # Convert to dictionary for easy lookup
        decomposition_results = {
            item["question_id"]: item
            for item in decomposition_dataset
        }
        
        return decomposition_results
    
    def run_chain_retrieval_stage(self, questions_data: List[Tuple[str, str, List[str]]], 
                                  decomposition_results: Dict[str, Dict]) -> Tuple[Dict[str, List[str]], Dict[str, Dict]]:
        """Run chain-following information retrieval for all questions.
        
        Args:
            questions_data: Original questions data
            decomposition_results: Decomposition results from Stage 1
            
        Returns:
            Tuple of (evidence_by_question, chains_info_by_question)
        """
        console.print("\n[bold cyan]Stage 2: Chain-Following Information Retrieval[/bold cyan]")
        
        evidence_by_question = {}
        chains_info_by_question = {}
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
            task = progress.add_task("Building and selecting reasoning chains...", total=len(questions_data))
            
            for question_id, question, _ in questions_data:
                if self.verbose:
                    console.print(f"\n[dim]Processing: {question}[/dim]")
                
                decomposed = decomposition_results[question_id]["decomposed_questions"]
                
                # Use the chain-following retrieval logic
                all_evidence = []
                chains_info = {}
                entities_by_question = {}
                
                # Get retrieval questions
                retrieval_questions = [q for q in decomposed if q["requires_retrieval"]]
                
                # Check if we have dependent chains
                has_dependent_chains = False
                for i, q_info in enumerate(decomposed):
                    if q_info.get("requires_retrieval", True):
                        if self.qa_system.is_question_referenced_in_future(i, decomposed):
                            has_dependent_chains = True
                            break
                
                if has_dependent_chains and len(retrieval_questions) >= 2:
                    # Use chain-following approach
                    current_chains = []
                    q1_qa_pairs = None
                    
                    for q_idx, q_info in enumerate(retrieval_questions):
                        q_num = f"Q{q_idx + 1}"
                        
                        if q_idx == 0:
                            # Q1 - Get Q&A pairs, extract entities
                            q1_qa_pairs = self.qa_system.search_and_collect_evidence(
                                q_info['question'], 
                                top_k_entities=20, 
                                top_k_qa=15
                            )
                            q1_entities, _ = self.qa_system.extract_entities_from_qa_pairs(
                                q1_qa_pairs, 
                                max_entities=self.max_entities_per_hop
                            )
                            entities_by_question[q_num] = q1_entities
                        
                        elif q_idx == 1:
                            # Q1→Q2 - Form complete chains, rerank, select top K
                            question_template = q_info['question']
                            actual_questions, has_substitution, _ = self.qa_system.substitute_entities(
                                question_template, 
                                entities_by_question
                            )
                            
                            if has_substitution:
                                # Collect Q2 Q&A pairs for each Q1 entity
                                q2_qa_pairs_by_entity = {}
                                for actual_q in actual_questions:
                                    # Find which Q1 entity this is for
                                    entity_used = None
                                    for ent in entities_by_question["Q1"]:
                                        if ent in actual_q:
                                            entity_used = ent
                                            break
                                    
                                    if entity_used:
                                        qa_pairs = self.qa_system.search_and_collect_evidence(
                                            actual_q, 
                                            top_k_entities=20, 
                                            top_k_qa=15
                                        )
                                        q2_qa_pairs_by_entity[entity_used] = qa_pairs
                                
                                # Form complete Q1→Q2 chains
                                current_chains_2hop = self.qa_system.form_reasoning_chains(
                                    q1_qa_pairs, 
                                    q2_qa_pairs_by_entity
                                )
                                current_chains = [
                                    self.qa_system.convert_2hop_to_nhop_format(chain) 
                                    for chain in current_chains_2hop
                                ]
                                
                                # Rerank and filter to top K chains
                                if current_chains:
                                    sorted_chains = self.qa_system.rerank_chains_against_original(
                                        current_chains, 
                                        question
                                    )
                                    current_chains = sorted_chains[:self.chain_top_k]
                        
                        else:
                            # Q1→Q2→Q3... - Extend existing chains
                            if current_chains:
                                question_template = q_info['question']
                                
                                # Extract entities from current chains
                                current_entities = []
                                for chain in current_chains:
                                    if 'qa_chain' in chain and chain['qa_chain']:
                                        last_qa = chain['qa_chain'][-1]
                                        answer_names = last_qa.get('answer_names', last_qa.get('answers', []))
                                        if isinstance(answer_names, str):
                                            answer_names = [answer_names]
                                        current_entities.extend([name for name in answer_names if name])
                                
                                current_entities = list(set(current_entities))
                                entities_by_question[f"Q{q_idx}"] = current_entities
                                
                                actual_questions, has_substitution, current_entities = self.qa_system.substitute_entities(
                                    question_template, 
                                    entities_by_question
                                )
                                
                                if has_substitution:
                                    # Collect Q&A pairs for current entities
                                    qa_pairs_by_entity = {}
                                    for actual_q in actual_questions:
                                        entity_used = None
                                        for entity in current_entities:
                                            if entity in actual_q:
                                                entity_used = entity
                                                break
                                        
                                        if entity_used:
                                            qa_pairs = self.qa_system.search_and_collect_evidence(
                                                actual_q, 
                                                top_k_entities=20, 
                                                top_k_qa=15
                                            )
                                            if entity_used not in qa_pairs_by_entity:
                                                qa_pairs_by_entity[entity_used] = []
                                            qa_pairs_by_entity[entity_used].extend(qa_pairs)
                                    
                                    # Extend current chains
                                    extended_chains = self.qa_system.extend_chains_to_next_hop(
                                        current_chains, 
                                        qa_pairs_by_entity, 
                                        q_idx
                                    )
                                    
                                    # Rerank and filter extended chains
                                    if extended_chains:
                                        sorted_chains = self.qa_system.rerank_n_hop_chains(
                                            extended_chains, 
                                            question
                                        )
                                        current_chains = sorted_chains[:self.chain_top_k]
                    
                    # Extract evidence from final chains
                    if current_chains:
                        # Final reranking
                        sorted_chains = self.qa_system.rerank_n_hop_chains(
                            current_chains, 
                            question
                        ) if len(current_chains) > 1 else current_chains
                        
                        selected_chains = sorted_chains[:self.chain_top_k]
                        
                        chains_info = {
                            'total_chains': len(current_chains),
                            'selected_chains': len(selected_chains),
                            'top_score': selected_chains[0]['chain_score'] if selected_chains else 0.0,
                            'score_range': f"{selected_chains[-1]['chain_score']:.3f} - {selected_chains[0]['chain_score']:.3f}" if selected_chains else "N/A"
                        }
                        
                        # Extract unique Q&A pairs from selected chains
                        all_evidence = self.qa_system.extract_unique_qa_pairs_from_n_hop_chains(selected_chains)
                    else:
                        chains_info = {'fallback': True, 'reason': 'No chains formed'}
                        # Fallback to simple approach
                        all_evidence = self._simple_retrieval_fallback(decomposed, question_id)
                
                else:
                    # Use simple approach for non-chain questions
                    chains_info = {'fallback': True, 'reason': 'No dependent chains'}
                    all_evidence = self._simple_retrieval_fallback(decomposed, question_id)
                
                evidence_by_question[question_id] = all_evidence
                chains_info_by_question[question_id] = chains_info
                
                if self.verbose:
                    console.print(f"  [green]Collected {len(all_evidence)} pieces of evidence[/green]")
                    if chains_info and not chains_info.get('fallback'):
                        console.print(f"  [cyan]Chains: {chains_info['selected_chains']}/{chains_info['total_chains']}[/cyan]")
                
                progress.update(task, advance=1)
        
        console.print(f"[green]✓ Chain retrieval complete for {len(evidence_by_question)} questions[/green]")
        return evidence_by_question, chains_info_by_question
    
    def _simple_retrieval_fallback(self, decomposed: List[Dict[str, Any]], question_id: str) -> List[str]:
        """Fallback to simple retrieval for non-chain questions."""
        all_evidence = []
        entities_by_question = {}
        
        for i, q_info in enumerate(decomposed):
            if not q_info["requires_retrieval"]:
                continue
            
            question_template = q_info["question"]
            is_referenced = self.qa_system.is_question_referenced_in_future(i, decomposed)
            
            actual_questions, _, _ = self.qa_system.substitute_entities(question_template, entities_by_question)
            
            for actual_q in actual_questions:
                qa_pairs = self.qa_system.search_and_collect_evidence(actual_q)
                
                if is_referenced:
                    entities, qa_pair_used = self.qa_system.extract_entities_from_qa_pairs(qa_pairs)
                    entities_by_question[f"Q{i+1}"] = list(set(entities))
                    all_evidence.extend(qa_pair_used)
                else:
                    # Format as evidence strings
                    for qa in qa_pairs[:10]:
                        q_text = qa.get('question', '')
                        answer_names = qa.get('answer_names', qa.get('answers', []))
                        if isinstance(answer_names, str):
                            answer_names = [answer_names]
                        answer_text = ', '.join(str(name) for name in answer_names if name)
                        answer_rolestates = ', '.join(qa.get('answer_rolestates', []))
                        if q_text and answer_text:
                            all_evidence.append(f"Q: {q_text} A: {answer_text} {answer_rolestates}")
        
        return list(set(all_evidence))
    
    def run_batched_answer_generation(self, questions_data: List[Tuple[str, str, List[str]]], 
                                     evidence_by_question: Dict[str, List[str]],
                                     decomposition_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """Run batched answer generation using curator.
        
        Args:
            questions_data: Original questions data
            evidence_by_question: Evidence collected in Stage 2
            decomposition_results: Decomposition results from Stage 1
            
        Returns:
            Dictionary mapping question_id to answer results
        """
        console.print("\n[bold cyan]Stage 3: Batched Answer Generation[/bold cyan]")
        
        if not CURATOR_AVAILABLE:
            # Fallback to sequential processing
            console.print("[yellow]Using sequential answer generation (curator not available)[/yellow]")
            answer_results = {}
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
                task = progress.add_task("Generating answers...", total=len(questions_data))
                
                for question_id, question, _ in questions_data:
                    evidence = evidence_by_question.get(question_id, [])
                    decomposed = decomposition_results[question_id]["decomposed_questions"]
                    answer = self.qa_system.generate_answer(question, evidence, decomposed)
                    
                    # Extract final answer
                    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', answer, re.DOTALL | re.IGNORECASE)
                    if answer_match:
                        final_answer = answer_match.group(1).strip()
                    else:
                        final_answer = answer.strip()
                    
                    # Get the final prompt and token count
                    final_prompt = getattr(self.qa_system, '_last_prompt', None)
                    
                    # Calculate token count
                    token_count = 0
                    if final_prompt:
                        try:
                            import tiktoken
                            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
                            token_count = len(encoding.encode(final_prompt))
                        except:
                            pass
                    
                    answer_results[question_id] = {
                        "predicted_answer": final_answer,
                        "full_response": answer,
                        "evidence_count": len(evidence),
                        "final_prompt": final_prompt,
                        "token_count": token_count
                    }
                    progress.update(task, advance=1)
            
            return answer_results
        
        # Prepare inputs for batched answer generation
        answer_inputs = [
            {
                "question_id": qid,
                "question": question,
                "evidence": evidence_by_question.get(qid, []),
                "decomposed_questions": decomposition_results[qid]["decomposed_questions"]
            }
            for qid, question, _ in questions_data
        ]
        
        console.print(f"[cyan]Generating answers for {len(answer_inputs)} questions in parallel...[/cyan]")
        start_time = time.time()
        
        # Run batched answer generation
        answer_dataset = self.answer_generator(answer_inputs)
        
        elapsed = time.time() - start_time
        console.print(f"[green]✓ Answer generation complete in {elapsed:.1f}s ({elapsed/len(answer_inputs):.2f}s per question)[/green]")
        
        # Convert to dictionary for easy lookup
        answer_results = {
            item["question_id"]: item
            for item in answer_dataset
        }
        
        return answer_results
    
    def run_evaluation(self) -> List[BatchedChainEvaluationResult]:
        """Run complete batched evaluation pipeline.
        
        Returns:
            List of evaluation results
        """
        console.print(f"\n[bold magenta]Running Batched Chain-Following Evaluation on {self.num_questions} Questions[/bold magenta]")
        total_start = time.time()
        
        # Load questions
        questions_data = self.load_questions_and_answers()
        
        # Stage 1: Batched Decomposition
        decomposition_results = self.run_batched_decomposition(questions_data)
        
        # Stage 2: Chain-Following Information Retrieval
        evidence_by_question, chains_info_by_question = self.run_chain_retrieval_stage(questions_data, decomposition_results)
        
        # Stage 3: Batched Answer Generation
        answer_results = self.run_batched_answer_generation(questions_data, evidence_by_question, decomposition_results)
        
        # Compile results
        results = []
        for question_id, question, gold_answers in questions_data:
            decomposed = decomposition_results[question_id]["decomposed_questions"]
            answer_data = answer_results[question_id]
            chains_info = chains_info_by_question.get(question_id, {})
            
            result = BatchedChainEvaluationResult(
                question_id=question_id,
                question=question,
                predicted_answer=answer_data["predicted_answer"],
                gold_answers=gold_answers,
                processing_time=0.0,  # Will be updated with total time
                decomposed_questions=decomposed,
                evidence_count=answer_data["evidence_count"],
                token_count=answer_data.get("token_count", 0),
                chains_info=chains_info,
                final_prompt=answer_data.get("final_prompt", None),
                error=None
            )
            results.append(result)
        
        total_elapsed = time.time() - total_start
        console.print(f"\n[bold green]✓ Evaluation complete in {total_elapsed:.1f}s ({total_elapsed/len(results):.2f}s per question)[/bold green]")
        
        # Update processing time
        for result in results:
            result.processing_time = total_elapsed / len(results)
        
        return results
    
    def compute_metrics(self, results: List[BatchedChainEvaluationResult]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        """Compute evaluation metrics.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Tuple of (overall_metrics, per_example_metrics)
        """
        console.print("\n[cyan]Computing evaluation metrics...[/cyan]")
        
        # Filter out error cases
        valid_results = [r for r in results if r.error is None]
        
        if not valid_results:
            console.print("[red]No valid results to evaluate![/red]")
            return {}, []
        
        # Prepare data for evaluation
        gold_answers_list = [r.gold_answers for r in valid_results]
        predicted_answers = [r.predicted_answer for r in valid_results]
        
        # Compute metrics
        overall_metrics, per_example_metrics = evaluate_qa_batch(gold_answers_list, predicted_answers)
        
        return overall_metrics, per_example_metrics
    
    def save_results(self, results: List[BatchedChainEvaluationResult], overall_metrics: Dict[str, float], 
                     per_example_metrics: List[Dict[str, Any]]) -> None:
        """Save evaluation results to JSON file.
        
        Args:
            results: List of evaluation results
            overall_metrics: Overall performance metrics
            per_example_metrics: Per-question metrics
        """
        output_file = LOG_DIR / f"chain_following_batched_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_data = {
            "evaluation_info": {
                "num_documents": self.num_documents,
                "num_questions": self.num_questions,
                "chain_top_k": self.chain_top_k,
                "max_entities_per_hop": self.max_entities_per_hop,
                "timestamp": datetime.now().isoformat(),
                "batched": True,
                "curator_available": CURATOR_AVAILABLE
            },
            "overall_metrics": overall_metrics,
            "chain_statistics": self._compute_chain_statistics(results),
            "per_question_results": []
        }
        
        # Add per-question details
        for result, metrics in zip(results, per_example_metrics):
            question_data = {
                "question_id": result.question_id,
                "question": result.question,
                "predicted_answer": result.predicted_answer,
                "gold_answers": result.gold_answers,
                "metrics": metrics,
                "processing_time": result.processing_time,
                "decomposed_questions": result.decomposed_questions,
                "evidence_count": result.evidence_count,
                "token_count": result.token_count,
                "chains_info": result.chains_info,
                "final_prompt": result.final_prompt,
                "error": result.error
            }
            output_data["per_question_results"].append(question_data)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"[green]✓ Results saved to: {output_file}[/green]")
    
    def _compute_chain_statistics(self, results: List[BatchedChainEvaluationResult]) -> Dict[str, Any]:
        """Compute statistics about chain formation and selection."""
        chain_stats = {
            "questions_with_chains": 0,
            "questions_with_fallback": 0,
            "total_chains_formed": 0,
            "total_chains_selected": 0,
            "avg_chains_per_question": 0,
            "avg_chain_score": 0
        }
        
        chain_counts = []
        chain_scores = []
        
        for result in results:
            if result.chains_info:
                if result.chains_info.get('fallback'):
                    chain_stats["questions_with_fallback"] += 1
                else:
                    chain_stats["questions_with_chains"] += 1
                    chain_stats["total_chains_formed"] += result.chains_info.get('total_chains', 0)
                    chain_stats["total_chains_selected"] += result.chains_info.get('selected_chains', 0)
                    chain_counts.append(result.chains_info.get('total_chains', 0))
                    if result.chains_info.get('top_score'):
                        chain_scores.append(result.chains_info['top_score'])
        
        if chain_counts:
            chain_stats["avg_chains_per_question"] = sum(chain_counts) / len(chain_counts)
        
        if chain_scores:
            chain_stats["avg_chain_score"] = sum(chain_scores) / len(chain_scores)
        
        return chain_stats


def main(verbose: bool = False):
    """Main evaluation function.
    
    Args:
        verbose: Whether to show detailed output
    """
    console.print("\n[bold cyan]🚀 Batched Chain-Following Multi-Hop QA Evaluation[/bold cyan]")
    console.print("Using parallel processing for decomposition and answer generation")
    console.print("Using chain-following approach for information retrieval")
    
    try:
        # Initialize evaluator
        evaluator = BatchedChainFollowingEvaluator(
            num_documents=-1,  # Load all documents
            num_questions=-1,  # Evaluate all questions
            verbose=verbose,
            chain_top_k=15,     # Select top 15 chains
            max_entities_per_hop=5  # Max 5 entities per hop
        )
        
        # Run batched evaluation
        results = evaluator.run_evaluation()
        
        # Compute metrics
        overall_metrics, per_example_metrics = evaluator.compute_metrics(results)
        
        # Display results
        console.print("\n" + "="*60)
        console.print("[bold green]Evaluation Results:[/bold green]")
        console.print(format_evaluation_report(overall_metrics, per_example_metrics, show_examples=5))
        
        # Display chain statistics
        chain_stats = evaluator._compute_chain_statistics(results)
        console.print("\n[bold cyan]Chain Formation Statistics:[/bold cyan]")
        console.print(f"Questions with chains: {chain_stats['questions_with_chains']}")
        console.print(f"Questions with fallback: {chain_stats['questions_with_fallback']}")
        console.print(f"Average chains per question: {chain_stats['avg_chains_per_question']:.1f}")
        console.print(f"Average chain score: {chain_stats['avg_chain_score']:.3f}")
        console.print(f"Total chains formed: {chain_stats['total_chains_formed']:,}")
        console.print(f"Total chains selected: {chain_stats['total_chains_selected']:,}")
        
        # Display token usage summary
        console.print("\n[bold cyan]Token Usage Summary:[/bold cyan]")
        total_tokens = sum(r.token_count for r in results)
        avg_tokens = total_tokens / len(results) if results else 0
        min_tokens = min((r.token_count for r in results), default=0)
        max_tokens = max((r.token_count for r in results), default=0)
        
        console.print(f"Total tokens: {total_tokens:,}")
        console.print(f"Average tokens per question: {avg_tokens:.0f}")
        console.print(f"Min tokens: {min_tokens:,}")
        console.print(f"Max tokens: {max_tokens:,}")
        
        # Save results
        evaluator.save_results(results, overall_metrics, per_example_metrics)
        
        console.print("\n[bold green]✓ Batched chain-following evaluation completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    # Set verbose=True for detailed output during development
    main(verbose=False)