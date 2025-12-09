#!/usr/bin/env python3
"""
Batched Chain-Following Multi-Hop Question Answering Evaluation Script

Uses curator for parallel LLM calls with the chain-following approach.
Implements smart chain following:
1. Decomposes questions into sub-questions (batched)
2. Processes retrieval with entity substitution and chain formation
3. Reranks chains against original questions
4. Generates answers using oracle-style prompting (batched)
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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from pydantic import BaseModel
from typing import List

import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3,4'
os.environ["OPENAI_API_KEY"] = "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"
# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import our chain-following multi-hop QA system for retrieval
from playground.multi_hop_qa_chains import ChainFollowingMultiHopQA

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
LOG_FILE = LOG_DIR / f"multihop_qa_chains_batched_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"


class DecomposedQuestion(BaseModel):
    question: str
    requires_retrieval: bool

class DecomposedQuestionList(BaseModel):
    questions: List[DecomposedQuestion]

class ChainQuestionDecomposer(curator.LLM):
    """Curator class for decomposing multi-hop questions in parallel."""
    
    # return_completions_object = True
    
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
  **Important note for bridges:** There can be no `<ENTITY_Qn>` in the first question if the nth question DOES NOT require retrieval.

## Formatting
Format each decomposed question as follows:

<decomposition>
Question: [the question text]
Requires retrieval: [true/false]

And provide the response in the following json format:
{{
  "questions": [
    {{
      "question": "the decomposed question text",
      "requires_retrieval": "true/false"
    }}
  ]
}}

Examples:

Input: "What is the birth year of the spouse of the director of Casablanca?"
Output:
{{
    "questions": [
        {{
            "question": "Who directed Casablanca?",
            "requires_retrieval": "true"
        }},
        {{
            "question": "Who was <ENTITY_Q1>'s spouse?",
            "requires_retrieval": "true"
        }},
        {{
            "question": "What is <ENTITY_Q2>'s birth year?",
            "requires_retrieval": "true"
        }}
    ]
}}

Input: "Which film has the director who is older, Dune or The Dark Knight?"
Output:
{{
    "questions": [
        {{
            "question": "Who directed Dune?",
            "requires_retrieval": "true"
        }},
        {{
            "question": "Who directed The Dark Knight?",
            "requires_retrieval": "true"
        }},
        {{
            "question": "Who is older, <ENTITY_Q1> or <ENTITY_Q2>?",
            "requires_retrieval": "true"
        }},
        {{
            "question": "Who is older, <ENTITY_Q1> or <ENTITY_Q2>?",
            "requires_retrieval": "false"
        }}
    ]
}}


IMPORTANT:
    AVOID over-decomposition like this:
    DON'T break "Who is John Doe?" into:
    1. Who is John Doe? → "English"
    2. When was <ENTITY_Q1> born? → "When was English born?"

    DO ask directly: "When was John Doe born?"

Now decompose this question:
Input: "{input['question']}"
Output:
"""
        
        return [
            {"role": "system", "content": "You are a helpful assistant that breaks down complex questions into simple steps."},
            {"role": "user", "content": decomposition_prompt}
        ]
    
    def parse(self, input, response: DecomposedQuestionList):
        """Parse the decomposition response."""

        # convert string to json
        response = json.loads(response)
        questions = [{"question" : q["question"], "requires_retrieval" : q["requires_retrieval"]} for q in response["questions"]]
        # questions = [{"question" : q.question, "requires_retrieval" : q.requires_retrieval} for q in response.questions]
        
        return [{
            "question_id": input['question_id'],
            "original_question": input['question'],
            "decomposed_questions": questions,
            # "raw_response": decomposition_text
        }]




class ChainAnswerGenerator(curator.LLM):
    """Curator class for generating answers with oracle-style prompting in parallel."""
    
    return_completions_object = True
    
    def __init__(self, **kwargs):
        """Initialize the answer generator."""
        super().__init__(**kwargs)
    
    def prompt(self, input):
        """Create an oracle-style answer generation prompt."""
        evidence_text = '\n'.join(input['evidence'])
        
        # Build oracle-style prompt with one-shot example
        prompt_text = f"""
{evidence_text}
\n\nQuestion: {input['question']}
\n\nThought: 

"""
        
        # Store the full prompt for later reference
        self._last_prompt = prompt_text
        
        # One-shot example with Q&A pairs format (concise entity-focused answers)
        one_shot_docs = (
            """ Q: Who directed The Last Horse? A: Edgar Neville
                Q: When was The Last Horse released? A: 1950
                Q: When was the University of Southampton founded? A: 1862
                Q: Where is the University of Southampton located? A: Southampton
                Q: What is the population of Stanton Township? A: 505
                Q: Where is Stanton Township? A: Champaign County, Illinois
                Q: Who is Neville A. Stanton? A: British Professor of Human Factors and Ergonomics
                Q: Where does Neville A. Stanton work? A: University of Southampton
                Q: What is Neville A. Stanton's profession? A: Professor
                Q: Who directed Finding Nemo? A: Andrew Stanton
                Q: When was Finding Nemo released? A: 2003
                Q: What company produced Finding Nemo? A: Pixar Animation Studios"""
            # """Wikipedia Title: Milk and Honey (album)\nMilk and Honey is an album by John Lennon and Yoko Ono released in 1984. Following the compilation "The John Lennon Collection", it is Lennon's eighth and final studio album, and the first posthumous release of new Lennon music, having been recorded in the last months of his life during and following the sessions for their 1980 album "Double Fantasy". It was assembled by Yoko Ono in association with the Geffen label.\n\n"""
            # """Wikipedia Title: John Lennon Museum\nJohn Lennon Museum (ジョン・レノン・ミュージアム , Jon Renon Myūjiamu ) was a museum located inside the Saitama Super Arena in Chūō-ku, Saitama, Saitama Prefecture, Japan. It was established to preserve knowledge of John Lennon's life and musical career. It displayed Lennon's widow Yoko Ono's collection of his memorabilia as well as other displays. The museum opened on October 9, 2000, the 60th anniversary of Lennon’s birth, and closed on September 30, 2010, when its exhibit contract with Yoko Ono expired. A tour of the museum began with a welcoming message and short film narrated by Yoko Ono (in Japanese with English headphones available), and ended at an avant-garde styled "reflection room" full of chairs facing a slide show of moving words and images. After this room there was a gift shop with John Lennon memorabilia available.\n\n"""
            # """Wikipedia Title: Walls and Bridges\nWalls and Bridges is the fifth studio album by English musician John Lennon. It was issued by Apple Records on 26 September 1974 in the United States and on 4 October in the United Kingdom. Written, recorded and released during his 18-month separation from Yoko Ono, the album captured Lennon in the midst of his "Lost Weekend". "Walls and Bridges" was an American "Billboard" number-one album and featured two hit singles, "Whatever Gets You thru the Night" and "#9 Dream". The first of these was Lennon's first number-one hit in the United States as a solo artist, and his only chart-topping single in either the US or Britain during his lifetime.\n\n"""
            # """Wikipedia Title: Nobody Loves You (When You're Down and Out)\n"Nobody Loves You (When You're Down and Out)" is a song written by John Lennon released on his 1974 album "Walls and Bridges". The song is included on the 1986 compilation "Menlove Ave.", the 1990 boxset "Lennon", the 1998 boxset "John Lennon Anthology", the 2005 two-disc compilation "", and the 2010 boxset "Gimme Some Truth".\n\n"""
            # """Wikipedia Title: Give Peace a Chance\n"Give Peace a Chance" is an anti-war song written by John Lennon (credited to Lennon–McCartney), and performed with Yoko Ono in Montreal, Quebec, Canada. Released as a single in 1969 by the Plastic Ono Band on Apple Records (catalogue Apple 13 in the United Kingdom, Apple 1809 in the United States), it is the first solo single issued by Lennon, released when he was still a member of the Beatles, and became an anthem of the American anti-war movement during the 1970s. It peaked at number 14 on the "Billboard" Hot 100 and number 2 on the British singles chart.\n"""
        )
        
        # System message for advanced reading comprehension
        rag_qa_system = (
            'As an advanced reading comprehension assistant, your task is to analyze precise QA pairs extracted from the documents and corresponding questions meticulously. '
            'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
            'Conclude with "Answer: " to present only a concise, definitive response, devoid of additional elaborations.'
            # 'If you don\'t know the answer, say "No Answer".'
            # 'You serve as an intelligent assistant, adept at facilitating users through complex, multi-hop reasoning across multiple documents. This task is illustrated through demonstrations, each consisting of a document set paired with a relevant question and its multi-hop reasoning thoughts. Your task is to generate one thought for current step, DON\'T generate the whole thoughts at once! If you reach what you believe to be the final step, conclude with "Answer: " to present only a concise, definitive response, devoid of additional elaborations..'
            # '\n\n'
        )
        
        # One-shot example input
        one_shot_input = (
            f"{one_shot_docs}"
            "\n\nQuestion: "
            "When was Neville A. Stanton's employer founded?"
            '\nThought: '
            # '\n\nQuestion: '
            # f"Nobody Loves You was written by John Lennon and released on what album that was issued by Apple Records, and was written, recorded, and released during his 18 month separation from Yoko Ono?"
            # '\nThought: '
        )
        
        # One-shot example output
        one_shot_output = (
            "From the QA pairs, the employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
            "\nAnswer: 1862."
            # f"The album issued by Apple Records, and written, recorded, and released during John Lennon's 18 month separation from Yoko Ono is Walls and Bridges. Nobody Loves You was written by John Lennon on Walls and Bridges album."
            # '\nAnswer: Walls and Bridges.'
            # '\n\n'
        )
        
        # Build the prompt template
        prompt_messages = [
            {"role": "system", "content": rag_qa_system},
            {"role": "user", "content": one_shot_input},
            {"role": "assistant", "content": one_shot_output},
            {"role": "user", "content": prompt_text}
        ]    

        return prompt_messages

    def parse(self, input, response):
        """Parse the answer from the response with oracle-style format."""
        answer_text = response["choices"][0]["message"]["content"]
        
        # Parse answer with new format (Answer: format)
        if 'Answer: ' in answer_text:
            final_answer = answer_text.split('Answer: ')[-1].strip()
            # Remove any trailing period if it's just a number/date
            if final_answer.endswith('.') and final_answer[:-1].replace(',', '').replace(' ', '').isdigit():
                final_answer = final_answer[:-1]
        else:
            # Fallback to full response if no "Answer:" found
            final_answer = answer_text.strip()
        
        # Focus on evidence only for reporting token usage
        evidence_text = '\n'.join(input.get('evidence', []))
        token_count = 0
        try:
            import tiktoken
            encoding = tiktoken.encoding_for_model("gpt-4o-mini")
            token_count = len(encoding.encode(evidence_text))
        except Exception:
            # Fallback rough estimate if tokenizer is unavailable
            token_count = max(1, len(evidence_text) // 4)
        
        return [{
            "question_id": input['question_id'],
            "question": input['question'],
            "predicted_answer": final_answer,
            "full_response": answer_text,
            "evidence_count": len(input['evidence']),
            "evidence": input['evidence'],
            "final_prompt": None,
            "token_count": token_count
        }]


@dataclass
class ChainBatchedEvaluationResult:
    """Container for chain-based batched evaluation results."""
    question_id: str
    question: str
    predicted_answer: str
    full_response: str
    gold_answers: List[str]
    processing_time: float
    decomposed_questions: List[Dict[str, Any]]
    evidence_count: int
    evidence: List[str]
    chains_info: Dict[str, Any]
    token_count: int = 0
    final_prompt: Optional[str] = None
    error: Optional[str] = None


class ChainBatchedMultiHopQAEvaluator:
    """Batched evaluator for chain-following multi-hop QA using curator for parallel LLM calls."""
    
    def __init__(self, num_documents: int = 200, num_questions: int = 20, verbose: bool = False,
                 chain_top_k: int = 15, use_bm25: bool = False):
        """Initialize batched chain evaluator.
        
        Args:
            num_documents: Number of documents to load
            num_questions: Number of questions to evaluate
            verbose: Whether to show detailed output
            chain_top_k: Number of top chains to select after reranking
        """
        self.num_documents = num_documents
        self.num_questions = num_questions
        self.data_dir = Path("/mnt/SSD6/yigit/gsw-memory/playground_data/")
        self.verbose = verbose
        self.chain_top_k = chain_top_k
        self.use_bm25 = use_bm25
        
        console.print("[bold blue]Initializing Chain-Based Batched Multi-Hop QA Evaluator...[/bold blue]")
        console.print(f"  Chain selection: Top {chain_top_k} chains")
        
        # Initialize the chain-following retrieval system (single instance for all questions)
        self.qa_system = ChainFollowingMultiHopQA(
            num_documents=num_documents, 
            verbose=False, 
            show_prompt=False,
            chain_top_k=chain_top_k,
            chain_following_mode="cumulative", 
            beam_width=5, 
            reranker_model_name="voyage",
            use_chain_reranker=True,
            use_bm25=use_bm25

        )
        # Initialize curator classes if available
        if CURATOR_AVAILABLE:
            os.environ["HOSTED_VLLM_API_KEY"] = "token-abc123"
            self.decomposer = ChainQuestionDecomposer(
                # model_name="gpt-4o",
                model_name="hosted_vllm/yigitturali/qwen3-8b-qa-decomp-gsw-rank-256-gpt5-golden-large",
                backend = "litellm",
                backend_params = {
                    "base_url": "http://127.0.0.1:8000/v1",
                    "request_timeout": 600.0,  
                    "max_concurrent_requests": 32,
                    "max_requests_per_minute": 120,
                    "max_tokens_per_minute": 200000,
                    "seconds_to_pause_on_rate_limit": 5,
                    "require_all_responses": False,
                },
                generation_params={"temperature": 0}, 
                # response_format=DecomposedQuestionList
            )
            self.answer_generator = ChainAnswerGenerator(
                # model_name="gpt-4o-mini", 
                # generation_params={"temperature": 0},
                model_name="hosted_vllm/Qwen/Qwen3-8B",
                backend = "litellm",
                backend_params = {
                    "base_url": "http://127.0.0.1:6379/v1",
                    "request_timeout": 600.0,  
                    "max_concurrent_requests": 32,
                    "max_requests_per_minute": 120,
                    "max_tokens_per_minute": 200000,
                    "seconds_to_pause_on_rate_limit": 5,
                    "require_all_responses": False,
                },
                generation_params={"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0, "max_tokens": 1000}
            )
            console.print("[green]✓ Curator initialized for parallel processing[/green]")
        else:
            console.print("[yellow]⚠ Curator not available - will fall back to sequential processing[/yellow]")
        
        console.print(f"[cyan]Ready to evaluate {num_questions} questions on {num_documents} documents[/cyan]")
    
    def load_questions_and_answers(self) -> List[Tuple[str, str, List[str]]]:
        """Load questions and gold answers from 2WikiMultihopQA dataset.
        
        Returns:
            List of tuples: (question_id, question, gold_answers)
        """
        console.print("[cyan]Loading Musique questions...[/cyan]")
        
        questions_file = self.data_dir / "/mnt/SSD6/yigit/gsw-memory/playground_data/musique_unanswerable.json"
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        
        with open(questions_file, 'r') as f:
            data = json.load(f)
        
        # Extract first N questions
        questions_data = []
        for i, item in enumerate(data[:self.num_questions]):
            # if "2hop" in item["id"] or "3hop1" in item["id"]:
            question_id = item.get("_id", f"q_{i}")
            question = item["question"]
            gold_answers = item.get("answer", [])
            # gold_answers = item.get("gold_ans", [])
            # gold_answers = item.get("reference", [])
            # gold_answers = item.get("possible_answers", [])
            # In the dataset, possible_answers is a JSON-encoded string for lists; e.g., '["cartoonist", "graphic artist", "animator", "illustrator"]'
            # Fix to always decode it if it is a string:
            # if isinstance(gold_answers, str):
            #     try:
            #         gold_answers = json.loads(gold_answers)
            #     except Exception:
            #         gold_answers = [gold_answers]
                
            # else:
            #     continue
            
            # Ensure gold_answers is a list and add aliases
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]
                gold_answers.extend(item.get("answer_aliases", []))
            else:
                # answer is already a list, still need to add aliases
                answer_aliases = item.get("answer_aliases", [])
                gold_answers = gold_answers + answer_aliases
            
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
            for item in decomposition_dataset.dataset
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
            task = progress.add_task("Retrieving evidence with chain following...", total=len(questions_data))
            
            for question_id, question, _ in questions_data:
                if self.verbose:
                    console.print(f"\n[dim]Processing: {question}[/dim]")
                try:
                    decomposed = decomposition_results[question_id]["decomposed_questions"]
                except:
                    print(f"Error: {question_id} not found in decomposition results") #TODO: Handle this error
                    print("Skipping question...") #TODO: Handle this error
                    continue

                # Reuse the exact retrieval logic from ChainFollowingMultiHopQA
                all_evidence, chains_info, _ = self.qa_system.collect_evidence(
                    question, decomposed=decomposed
                )
                
                evidence_by_question[question_id] = all_evidence
                chains_info_by_question[question_id] = chains_info
                
                if self.verbose:
                    console.print(f"  [green]Collected {len(all_evidence)} pieces of evidence[/green]")
                    if chains_info and not chains_info.get('fallback', False):
                        console.print(f"  [dim]Chains: {chains_info.get('total_chains', 0)} formed, {chains_info.get('selected_chains', 0)} selected[/dim]")
                    else:
                        console.print(f"  [dim]Used fallback retrieval[/dim]")
                
                # Debug: Print first few evidence items if verbose
                if self.verbose and all_evidence:
                    console.print(f"  [dim]Sample evidence: {all_evidence[0][:80]}...[/dim]")
                elif self.verbose and not all_evidence:
                    console.print(f"  [red]WARNING: No evidence collected for question![/red]")
                
                progress.update(task, advance=1)
        
        console.print(f"[green]✓ Chain retrieval complete for {len(evidence_by_question)} questions[/green]")
        return evidence_by_question, chains_info_by_question
    
    # Note: retrieval is now delegated entirely to ChainFollowingMultiHopQA.collect_evidence
    
    def run_batched_answer_generation(self, questions_data: List[Tuple[str, str, List[str]]], 
                                     evidence_by_question: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Run batched answer generation using curator with oracle-style prompting.
        
        Args:
            questions_data: Original questions data
            evidence_by_question: Evidence collected in Stage 2
            
        Returns:
            Dictionary mapping question_id to answer results
        """
        console.print("\n[bold cyan]Stage 3: Batched Answer Generation (Oracle-Style)[/bold cyan]")
        
        if not CURATOR_AVAILABLE:
            # Fallback to sequential processing
            console.print("[yellow]Using sequential answer generation (curator not available)[/yellow]")
            answer_results = {}
            
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), console=console) as progress:
                task = progress.add_task("Generating answers...", total=len(questions_data))
                
                for question_id, question, _ in questions_data:
                    evidence = evidence_by_question.get(question_id, [])
                    answer = self.qa_system.generate_answer(question, evidence)
                    
                    # Parse answer with new format
                    if 'Answer: ' in answer:
                        final_answer = answer.split('Answer: ')[-1].strip()
                        if final_answer.endswith('.') and final_answer[:-1].replace(',', '').replace(' ', '').isdigit():
                            final_answer = final_answer[:-1]
                    else:
                        final_answer = answer.strip()
                    
                    # Focus on evidence only for outputs
                    final_prompt = None
                    # Compute token count over evidence text
                    ev_text = '\n'.join(evidence)
                    try:
                        import tiktoken
                        encoding = tiktoken.encoding_for_model("gpt-4o-mini")
                        token_count = len(encoding.encode(ev_text))
                    except Exception:
                        token_count = max(1, len(ev_text) // 4)
                    
                    answer_results[question_id] = {
                        "predicted_answer": final_answer,
                        "full_response": answer,
                        "evidence_count": len(evidence),
                        "evidence": evidence,
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
                "evidence": evidence_by_question.get(qid, [])
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
        answer_results = {item["question_id"]: item for item in answer_dataset.dataset}
        
        return answer_results
    
    def run_evaluation(self) -> List[ChainBatchedEvaluationResult]:
        """Run complete batched evaluation pipeline with chain following.
        
        Returns:
            List of evaluation results
        """
        console.print(f"\n[bold magenta]Running Chain-Based Batched Evaluation on {self.num_questions} Questions[/bold magenta]")
        total_start = time.time()
        
        # Load questions
        questions_data = self.load_questions_and_answers()
        
        # Stage 1: Batched Decomposition
        decomposition_results = self.run_batched_decomposition(questions_data)
        
        # Stage 2: Chain-Following Information Retrieval
        evidence_by_question, chains_info_by_question = self.run_chain_retrieval_stage(
            questions_data, decomposition_results
        )
        
        # Stage 3: Batched Answer Generation with Oracle-Style Prompting
        answer_results = self.run_batched_answer_generation(questions_data, evidence_by_question)
        
        # Compile results
        results = []
        for question_id, question, gold_answers in questions_data:
            try:
                decomposed = decomposition_results[question_id]["decomposed_questions"]
                answer_data = answer_results[question_id]
                chains_info = chains_info_by_question.get(question_id, {})
                
                result = ChainBatchedEvaluationResult(
                    question_id=question_id,
                    question=question,
                    predicted_answer=answer_data["predicted_answer"],
                    full_response=answer_data.get("full_response", ""),
                    gold_answers=gold_answers,
                    processing_time=0.0,  # Will be updated with total time
                    decomposed_questions=decomposed,
                    evidence_count=answer_data["evidence_count"],
                    evidence=answer_data.get("evidence", []),
                    chains_info=chains_info,
                    token_count=answer_data.get("token_count", 0),
                    final_prompt=answer_data.get("final_prompt", None),
                    error=None
                )
                results.append(result)
            except Exception as e:
                print(f"Error: {question_id} not found in answer results") #TODO: Handle this error
                print("Skipping question...") #TODO: Handle this error
                continue
        
        total_elapsed = time.time() - total_start
        console.print(f"\n[bold green]✓ Evaluation complete in {total_elapsed:.1f}s ({total_elapsed/len(results):.2f}s per question)[/bold green]")
        
        # Update processing time
        for result in results:
            result.processing_time = total_elapsed / len(results)
        
        return results
    
    def compute_metrics(self, results: List[ChainBatchedEvaluationResult]) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
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

        # Compute metrics with substring matching enabled
        overall_metrics, per_example_metrics = evaluate_qa_batch(
            gold_answers_list,
            predicted_answers,
        )

        return overall_metrics, per_example_metrics
    
    def save_results(self, results: List[ChainBatchedEvaluationResult], overall_metrics: Dict[str, float], 
                     per_example_metrics: List[Dict[str, Any]]) -> None:
        """Save evaluation results to JSON file.
        
        Args:
            results: List of evaluation results
            overall_metrics: Overall performance metrics
            per_example_metrics: Per-question metrics
        """
        output_file = LOG_DIR / f"multihop_qa_chains_batched_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_data = {
            "evaluation_info": {
                "num_documents": self.num_documents,
                "num_questions": self.num_questions,
                "chain_top_k": self.chain_top_k,
                "timestamp": datetime.now().isoformat(),
                "batched": True,
                "chain_following": True,
                "oracle_prompting": True,
                "curator_available": CURATOR_AVAILABLE
            },
            "overall_metrics": overall_metrics,
            "per_question_results": []
        }
        
        # Add per-question details
        for result, metrics in zip(results, per_example_metrics):
            question_data = {
                "question_id": result.question_id,
                "question": result.question,
                "predicted_answer": result.predicted_answer,
                "full_response": result.full_response,
                "gold_answers": result.gold_answers,
                "metrics": metrics,
                "processing_time": result.processing_time,
                "decomposed_questions": result.decomposed_questions,
                "evidence_count": result.evidence_count,
                "evidence": result.evidence,
                "chains_info": result.chains_info,
                "token_count": result.token_count,
                "final_prompt": result.final_prompt,
                "error": result.error
            }
            output_data["per_question_results"].append(question_data)
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        console.print(f"[green]✓ Results saved to: {output_file}[/green]")
    
    def display_chain_statistics(self, results: List[ChainBatchedEvaluationResult]) -> None:
        """Display statistics about chain formation and selection.
        
        Args:
            results: List of evaluation results
        """
        console.print("\n[bold cyan]Chain Statistics:[/bold cyan]")
        
        chain_results = [r for r in results if r.chains_info and not r.chains_info.get('fallback', False)]
        fallback_results = [r for r in results if r.chains_info and r.chains_info.get('fallback', False)]
        
        if chain_results:
            total_chains = sum(r.chains_info.get('total_chains', 0) for r in chain_results)
            selected_chains = sum(r.chains_info.get('selected_chains', 0) for r in chain_results)
            avg_chains = total_chains / len(chain_results) if chain_results else 0
            avg_selected = selected_chains / len(chain_results) if chain_results else 0
            
            console.print(f"Questions using chain following: {len(chain_results)}")
            console.print(f"Questions using fallback: {len(fallback_results)}")
            console.print(f"Total chains formed: {total_chains}")
            console.print(f"Total chains selected: {selected_chains}")
            console.print(f"Average chains per question: {avg_chains:.1f}")
            console.print(f"Average selected chains: {avg_selected:.1f}")
        else:
            console.print("No chain-following results found (all used fallback)")


def main(verbose: bool = False):
    """Main evaluation function.
    
    Args:
        verbose: Whether to show detailed output
    """
    console.print("\n[bold cyan]🚀 Chain-Based Batched Multi-Hop QA Evaluation on 2WikiMultihopQA[/bold cyan]")
    console.print("Using chain-following approach with oracle-style prompting")
    console.print("Parallel processing for decomposition and answer generation")
    
    try:
        # Initialize evaluator
        evaluator = ChainBatchedMultiHopQAEvaluator(
            num_documents=-1, 
            num_questions=1000, 
            verbose=verbose,
            use_bm25=True
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
        evaluator.display_chain_statistics(results)
        
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
        
        console.print("\n[bold green]✓ Chain-based batched evaluation completed successfully![/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        raise


if __name__ == "__main__":
    # Set verbose=True for detailed output during development
    main(verbose=False)
