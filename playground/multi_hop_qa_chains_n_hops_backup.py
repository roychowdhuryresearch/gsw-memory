#!/usr/bin/env python3
"""
Chain-Following Multi-Hop Question Answering System

An enhanced version that implements smart chain following:
1. Decomposes questions into sub-questions  
2. Processes each question sequentially with entity substitution
3. For terminal questions: Forms complete reasoning chains
4. Reranks chains against the original query
5. Selects top-k most coherent chains
6. Extracts unique Q&A pairs from selected chains

This addresses the exponential explosion problem by focusing on 
semantically coherent reasoning paths.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from datetime import datetime
from collections import defaultdict
import numpy as np
import json
import time

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel

# Add the parent directory to the path
sys.path.append(str(Path(__file__).parent.parent))

# Import our enhanced entity searcher
from playground.simple_entity_search import EntitySearcher

# OpenAI for question decomposition and final reasoning
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI not available. Install with: pip install openai")
    OPENAI_AVAILABLE = False

console = Console()


class ChainDebugger:
    """Debug tracker for chain formation and entity-evidence mapping."""
    
    def __init__(self, debug_dir: str = "chain_debug", enabled: bool = True):
        self.enabled = enabled
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        
        self.current_state = {
            "question_id": None,
            "original_question": None,
            "decomposed_questions": [],
            "chain_evolution": [],
            "final_chains": [],
            "entity_evidence_map": defaultdict(dict),
            "final_evidence_usage": {},
            "fallback_reason": None,
            "total_time": 0,
            "start_time": None
        }
    
    def start_tracking(self, question_id: str, question: str, decomposed: List[Dict[str, Any]]):
        """Initialize tracking for a new question."""
        if not self.enabled:
            return
        
        self.current_state = {
            "question_id": question_id,
            "original_question": question,
            "decomposed_questions": decomposed,
            "chain_evolution": [],
            "final_chains": [],
            "entity_evidence_map": defaultdict(dict),
            "final_evidence_usage": {},
            "fallback_reason": None,
            "total_time": 0,
            "start_time": time.time()
        }
    
    def track_step(self, step: int, step_name: str, question_template: str, 
                  substituted_questions: List[str] = None, entities_used: List[str] = None):
        """Track the start of a new step."""
        if not self.enabled:
            return
        
        step_data = {
            "step": step,
            "step_name": step_name,
            "question_template": question_template,
            "substituted_questions": substituted_questions or [],
            "entities_used": entities_used or [],
            "entity_evidence_map": {},
            "chains_before": 0,
            "chains_formed": 0,
            "chains_after_filtering": 0,
            "chains": [],
            "entities_extracted": {},
            "entities_filtered": [],
            "searches_performed": [],
            "step_duration": 0,
            "timestamp": datetime.now().isoformat(),
            "step_start_time": time.time()
        }
        
        self.current_state["chain_evolution"].append(step_data)
    
    def track_entity_search(self, entity: str, substituted_question: str, 
                           search_results: List[Dict[str, Any]], step_idx: int = -1):
        """Track what evidence was collected for a specific entity."""
        if not self.enabled:
            return
        
        if step_idx == -1:
            step_idx = len(self.current_state["chain_evolution"]) - 1
        
        if step_idx < 0 or step_idx >= len(self.current_state["chain_evolution"]):
            return
        
        # Process evidence collected
        evidence_list = []
        entities_discovered = []
        
        for i, qa_pair in enumerate(search_results):
            evidence_item = {
                "qa_pair": {
                    "question": qa_pair.get("question", ""),
                    "answer_names": qa_pair.get("answer_names", []),
                    "answer_rolestates": qa_pair.get("answer_rolestates", []),
                    "source_entity": qa_pair.get("source_entity", ""),
                    "doc_id": qa_pair.get("doc_id", ""),
                    "entity_score": qa_pair.get("entity_score", 0.0)
                },
                "rank": i + 1,
                "used_in_chain": False  # Will be updated later
            }
            evidence_list.append(evidence_item)
            
            # Extract new entities discovered
            answer_names = qa_pair.get("answer_names", [])
            if isinstance(answer_names, str):
                answer_names = [answer_names]
            entities_discovered.extend([name for name in answer_names if name and name != entity])
        
        entity_evidence_info = {
            "substituted_question": substituted_question,
            "evidence_collected": evidence_list,
            "total_evidence_found": len(evidence_list),
            "evidence_after_filtering": len(evidence_list),  # Will be updated if filtering occurs
            "entities_discovered": list(set(entities_discovered))
        }
        
        # Store in current step and global entity map
        step_data = self.current_state["chain_evolution"][step_idx]
        step_data["entity_evidence_map"][entity] = entity_evidence_info
        self.current_state["entity_evidence_map"][entity] = entity_evidence_info
        
        # Track the search performed
        step_data["searches_performed"].append({
            "query": substituted_question,
            "for_entity": entity,
            "results_count": len(search_results),
            "top_entities": entities_discovered[:5],
            "qa_pairs_retrieved": len(evidence_list)
        })
    
    def track_chain_formation(self, chains_before: int, chains_formed: List[Dict[str, Any]], 
                             chains_after_filtering: List[Dict[str, Any]], step_idx: int = -1):
        """Track chain formation and filtering at a step."""
        if not self.enabled:
            return
        
        if step_idx == -1:
            step_idx = len(self.current_state["chain_evolution"]) - 1
        
        if step_idx < 0 or step_idx >= len(self.current_state["chain_evolution"]):
            return
        
        step_data = self.current_state["chain_evolution"][step_idx]
        step_data["chains_before"] = chains_before
        step_data["chains_formed"] = len(chains_formed)
        step_data["chains_after_filtering"] = len(chains_after_filtering)
        
        # Store detailed chain information
        for i, chain in enumerate(chains_after_filtering):
            chain_info = {
                "chain_id": f"chain_{step_idx}_{i}",
                "qa_chain": chain.get("qa_chain", []),
                "entity_bridges": chain.get("entity_bridges", []),
                "evidence_path": self._extract_evidence_path(chain),
                "chain_text": chain.get("chain_text", ""),
                "score": chain.get("chain_score", 0.0),
                "selected": True,  # These are the filtered chains
                "rank": i + 1
            }
            step_data["chains"].append(chain_info)
        
        # Mark evidence as used in chains
        self._mark_evidence_usage(chains_after_filtering, step_idx)
    
    def _extract_evidence_path(self, chain: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract the evidence path showing which entity led to which evidence."""
        evidence_path = []
        qa_chain = chain.get("qa_chain", [])
        entity_bridges = chain.get("entity_bridges", [])
        
        for i, qa_pair in enumerate(qa_chain):
            path_item = {
                "hop": i + 1,
                "entity_queried": entity_bridges[i] if i < len(entity_bridges) else "unknown",
                "evidence_used": {
                    "question": qa_pair.get("question", ""),
                    "answer": qa_pair.get("answer_names", []),
                    "source_entity": qa_pair.get("source_entity", "")
                }
            }
            evidence_path.append(path_item)
        
        return evidence_path
    
    def _mark_evidence_usage(self, chains: List[Dict[str, Any]], step_idx: int):
        """Mark which evidence pieces were used in the given chains."""
        if step_idx < 0 or step_idx >= len(self.current_state["chain_evolution"]):
            return
        
        step_data = self.current_state["chain_evolution"][step_idx]
        
        # Get all Q&A pairs used in chains
        used_qa_pairs = set()
        for chain in chains:
            for qa_pair in chain.get("qa_chain", []):
                qa_key = (qa_pair.get("question", ""), tuple(qa_pair.get("answer_names", [])))
                used_qa_pairs.add(qa_key)
        
        # Mark evidence as used
        for entity, evidence_info in step_data["entity_evidence_map"].items():
            for evidence_item in evidence_info["evidence_collected"]:
                qa_pair = evidence_item["qa_pair"]
                qa_key = (qa_pair["question"], tuple(qa_pair["answer_names"]))
                if qa_key in used_qa_pairs:
                    evidence_item["used_in_chain"] = True
    
    def finish_step(self, step_idx: int = -1):
        """Finish tracking for the current step."""
        if not self.enabled:
            return
        
        if step_idx == -1:
            step_idx = len(self.current_state["chain_evolution"]) - 1
        
        if step_idx < 0 or step_idx >= len(self.current_state["chain_evolution"]):
            return
        
        step_data = self.current_state["chain_evolution"][step_idx]
        step_data["step_duration"] = time.time() - step_data["step_start_time"]
        del step_data["step_start_time"]  # Remove temporary field
    
    def set_final_chains(self, final_chains: List[Dict[str, Any]]):
        """Set the final selected chains."""
        if not self.enabled:
            return
        
        self.current_state["final_chains"] = final_chains
        
        # Calculate final evidence usage statistics
        total_searches = sum(len(step["searches_performed"]) for step in self.current_state["chain_evolution"])
        total_qa_pairs = sum(len(evidence_info["evidence_collected"]) 
                           for evidence_info in self.current_state["entity_evidence_map"].values())
        
        qa_pairs_used = 0
        entities_that_contributed = set()
        entity_contribution_count = defaultdict(int)
        
        for chain in final_chains:
            for qa_pair in chain.get("qa_chain", []):
                qa_pairs_used += 1
                source_entity = qa_pair.get("source_entity", "")
                if source_entity:
                    entities_that_contributed.add(source_entity)
                    entity_contribution_count[source_entity] += 1
        
        self.current_state["final_evidence_usage"] = {
            "total_searches": total_searches,
            "total_qa_pairs_retrieved": total_qa_pairs,
            "qa_pairs_used_in_chains": qa_pairs_used,
            "entities_that_contributed": list(entities_that_contributed),
            "entity_contribution_count": dict(entity_contribution_count)
        }
    
    def set_fallback_reason(self, reason: str):
        """Set the reason for fallback if used."""
        if not self.enabled:
            return
        
        self.current_state["fallback_reason"] = reason
    
    def save_debug_info(self) -> str:
        """Save all debug information to files."""
        if not self.enabled:
            return ""
        
        if self.current_state["start_time"]:
            self.current_state["total_time"] = time.time() - self.current_state["start_time"]
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        question_id = self.current_state.get("question_id", "unknown")
        save_dir = self.debug_dir / timestamp / question_id
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save complete debug state
        debug_file = save_dir / "full_debug.json"
        with open(debug_file, 'w') as f:
            json.dump(self.current_state, f, indent=2, default=str)
        
        # Save entity-evidence mapping separately
        entity_evidence_file = save_dir / "entity_evidence_map.json"
        with open(entity_evidence_file, 'w') as f:
            json.dump(dict(self.current_state["entity_evidence_map"]), f, indent=2, default=str)
        
        # Save chain evolution
        chain_evolution_file = save_dir / "chain_evolution.json"
        with open(chain_evolution_file, 'w') as f:
            json.dump(self.current_state["chain_evolution"], f, indent=2, default=str)
        
        # Save human-readable summary
        summary_file = save_dir / "summary.txt"
        with open(summary_file, 'w') as f:
            self._write_summary(f)
        
        return str(save_dir)
    
    def _write_summary(self, f):
        """Write a human-readable summary of the debug information."""
        f.write(f"Chain Debug Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Question: {self.current_state['original_question']}\n")
        f.write(f"Question ID: {self.current_state['question_id']}\n")
        f.write(f"Total Time: {self.current_state['total_time']:.2f}s\n\n")
        
        f.write(f"Decomposed Questions:\n")
        for i, q in enumerate(self.current_state['decomposed_questions'], 1):
            f.write(f"  {i}. {q.get('question', '')}\n")
        f.write(f"\n")
        
        if self.current_state['fallback_reason']:
            f.write(f"Fallback Used: {self.current_state['fallback_reason']}\n\n")
        
        f.write(f"Chain Evolution:\n")
        f.write(f"===============\n")
        for step in self.current_state['chain_evolution']:
            f.write(f"\n{step['step_name']}:\n")
            f.write(f"  Template: {step['question_template']}\n")
            f.write(f"  Duration: {step['step_duration']:.2f}s\n")
            f.write(f"  Searches: {len(step['searches_performed'])}\n")
            f.write(f"  Chains before: {step['chains_before']}\n")
            f.write(f"  Chains formed: {step['chains_formed']}\n")
            f.write(f"  Chains after filtering: {step['chains_after_filtering']}\n")
            
            if step['entity_evidence_map']:
                f.write(f"  Entity Evidence:\n")
                for entity, evidence_info in step['entity_evidence_map'].items():
                    f.write(f"    {entity}: {evidence_info['total_evidence_found']} evidence pieces\n")
        
        f.write(f"\nFinal Evidence Usage:\n")
        f.write(f"====================\n")
        usage = self.current_state['final_evidence_usage']
        f.write(f"Total searches: {usage.get('total_searches', 0)}\n")
        f.write(f"Total Q&A pairs retrieved: {usage.get('total_qa_pairs_retrieved', 0)}\n")
        f.write(f"Q&A pairs used in chains: {usage.get('qa_pairs_used_in_chains', 0)}\n")
        f.write(f"Entities that contributed: {len(usage.get('entities_that_contributed', []))}\n")
        
        if usage.get('entity_contribution_count'):
            f.write(f"\nEntity Contributions:\n")
            for entity, count in usage['entity_contribution_count'].items():
                f.write(f"  {entity}: {count} Q&A pairs\n")


class ChainFollowingMultiHopQA:
    """Chain-following multi-hop QA system with intelligent chain reranking."""
    
    def __init__(self, num_documents: int = 200, verbose: bool = True, show_prompt: bool = False, 
                 chain_top_k: int = 15, max_entities_per_hop: int = 5, debug_mode: bool = False, 
                 debug_dir: str = "chain_debug", use_elbow_entity: bool = True,
                 use_elbow_qa: bool = True, elbow_min_keep: int = 5):
        """Initialize the chain-following multi-hop QA system.
        
        Args:
            num_documents: Number of documents to load
            verbose: Whether to show detailed output
            show_prompt: Whether to show the full LLM prompt
            chain_top_k: Number of top chains to select after reranking
            max_entities_per_hop: Maximum entities to consider at each hop to prevent explosion
            debug_mode: Whether to enable detailed debug tracking and file saving
            debug_dir: Directory to save debug files (only used if debug_mode=True)
            use_elbow_entity: Whether to use elbow method for entity filtering
            use_elbow_qa: Whether to use elbow method for Q&A pair filtering
            elbow_min_keep: Minimum number of items to keep regardless of elbow
        """
        self.verbose = verbose
        self.show_prompt = show_prompt
        self.chain_top_k = chain_top_k
        self.max_entities_per_hop = max_entities_per_hop
        self.debug_mode = debug_mode
        self.debug_dir = debug_dir
        self.use_elbow_entity = use_elbow_entity
        self.use_elbow_qa = use_elbow_qa
        self.elbow_min_keep = elbow_min_keep
        
        if verbose:
            console.print("[bold blue]Initializing Chain-Following Multi-Hop QA System...[/bold blue]")
            console.print(f"  Chain selection: Top {chain_top_k} chains")
            if debug_mode:
                console.print(f"  Debug mode: ENABLED (saving to {debug_dir})")
        
        # Initialize debugger
        self.debugger = ChainDebugger(debug_dir=debug_dir, enabled=debug_mode)
        
        # Initialize entity searcher
        self.entity_searcher = EntitySearcher(
            num_documents, 
            cache_dir="/home/yigit/codebase/gsw-memory/.gsw_cache",
            verbose=False  # Keep entity searcher quiet
        )
        
        # Initialize OpenAI client
        self.openai_client = None
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI()
                if verbose:
                    console.print("[green]✓ OpenAI client initialized[/green]")
            except Exception as e:
                if verbose:
                    console.print(f"[yellow]Warning: Could not initialize OpenAI: {e}[/yellow]")
        
        if verbose:
            console.print("[bold green]✓ System ready[/bold green]")
    
    def decompose_question(self, question: str) -> List[Dict[str, Any]]:
        """Decompose a multi-hop question into single-hop questions.
        
        Reuses the decomposition logic from the original implementation.
        """
        if not self.openai_client:
            return [{"question": question, "requires_retrieval": True}]
        
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
-   ** THERE CANNOT BE <ENTITY_Qn> IF NTH QUESTION DOES NOT REQUIRE RETRIEVAL.**
-   **AVOID OVER-DECOMPOSITION.** Each question should seek a meaningful entity or property.
-   DON'T break "When was John Doe born?" into "Who is John Doe?" -> "English", then "When was English born?".
-   DO ask directly: "When was John Doe born?".

Now decompose this question with provided format:
Question: "{question}"
Decomposition:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that breaks down complex questions into simple steps."},
                    {"role": "user", "content": decomposition_prompt}
                ],
                temperature=0,
                max_tokens=600
            )
            
            decomposition_text = response.choices[0].message.content
            
            # Parse the response
            questions = []
            # First try to extract content within <decomposition> tags
            decomposition_match = re.search(r'<decomposition>(.*?)</decomposition>', decomposition_text, re.DOTALL)
            if decomposition_match:
                # Parse content within decomposition tags
                content_to_parse = decomposition_match.group(1).strip()
            else:
                # Fallback to parsing the entire response
                content_to_parse = decomposition_text.strip()
            
            lines = content_to_parse.split('\n')
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
            
            if self.verbose and questions:
                console.print(f"[cyan]Decomposed into {len(questions)} questions:[/cyan]")
                for i, q in enumerate(questions, 1):
                    retrieval_str = "✓" if q["requires_retrieval"] else "✗"
                    console.print(f"  {i}. {q['question']} [{retrieval_str}]")
            
            return questions if questions else [{"question": question, "requires_retrieval": True}]
            
        except Exception as e:
            if self.verbose:
                console.print(f"[red]Error in decomposition: {e}[/red]")
            return [{"question": question, "requires_retrieval": True}]
    
    def substitute_entities(self, question_template: str, entities_by_question: Dict[str, List[str]]) -> List[str]:
        """Substitute entity placeholders with actual entity names.
        
        Args:
            question_template: Question with placeholders like <ENTITY> or <ENTITY_Q1>
            entities_by_question: Dict mapping Q1, Q2, etc. to entity names
            
        Returns:
            List of substituted questions (one per entity combination)
        """
        if "<ENTITY" not in question_template:
            return ([question_template], False, [])
        
        substituted_questions = []
        q_key = None
        entities = []
        
        
        # Handle simple <ENTITY> placeholder (use most recent question's entities)
        if "<ENTITY>" in question_template:
            # Find the most recent question number
            if entities_by_question:
                last_q = max(entities_by_question.keys(), key=lambda x: int(x[1:]))
                entities = entities_by_question.get(last_q, [])
                
                for entity_name in entities:
                    q = question_template.replace("<ENTITY>", entity_name)
                    substituted_questions.append(q)
        
        # Handle indexed placeholders like <ENTITY_Q1>, <ENTITY_Q2>
        else:
            # Find all entity references
            refs = re.findall(r'<ENTITY_Q(\d+)>', question_template)
            if refs:
                # For now, just substitute with the first entity from each referenced question
                # (In a more complex version, we could do cartesian product)
                q = question_template
                for ref in refs:
                    q_key = f"Q{ref}"
                    if q_key in entities_by_question and entities_by_question[q_key]:
                        # Use the first entity from this question
                        for entity in entities_by_question[q_key]:
                            q = question_template.replace(f"<ENTITY_Q{ref}>", entity)
                            substituted_questions.append(q)
                    try:        
                        entities.extend(entities_by_question[q_key])
                    except:
                        print(f"Error extending entities for {q_key}")
                        entities_by_question[q_key] = [f"<ENTITY_Q{ref}>"]
                        entities.extend([f"<ENTITY_Q{ref}>"])
                
        return (substituted_questions, True, entities_by_question[q_key]) if substituted_questions else ([question_template], False, [])
    
    def search_and_collect_evidence(self, question: str, top_k_entities: int = 10, top_k_qa: int = 15) -> List[Dict[str, Any]]:
        """Search for a question and collect relevant Q&A pairs.
        
        Args:
            question: The question to search for
            top_k_entities: Number of top entities to retrieve (before elbow filtering)
            top_k_qa: Number of top Q&A pairs to keep after reranking (before elbow filtering)
            
        Returns:
            List of relevant Q&A pairs with metadata
        """
        # Search for relevant entities
        search_results = self.entity_searcher.search(
            query=question,
            top_k=top_k_entities,
            verbose=False
        )
        
        # Apply elbow method to entity results if enabled
        # if self.use_elbow_entity and search_results:
        #     elbow_cutoff = self._find_elbow_cutoff(search_results)
        #     if elbow_cutoff < len(search_results):
        #         if self.verbose:
        #             console.print(f"[cyan]Entity elbow filtering: {len(search_results)} → {elbow_cutoff} entities[/cyan]")
        #         search_results = search_results[:elbow_cutoff]
        
        # Extract all Q&A pairs from search results
        all_qa_pairs = []
        for entity, score in search_results:
            qa_pairs = entity.get('qa_pairs', [])
            for qa in qa_pairs:
                qa_with_context = qa.copy()
                qa_with_context['source_entity'] = entity['name']
                qa_with_context['source_entity_id'] = entity['id']
                qa_with_context['doc_id'] = entity['doc_id']
                qa_with_context['entity_score'] = score
                qa_with_context['search_question'] = question  # Track what question led to this
                all_qa_pairs.append(qa_with_context)
        
        # Rerank Q&A pairs if we have embedding capability
        if hasattr(self.entity_searcher, '_rerank_qa_pairs') and all_qa_pairs:
            reranked = self.entity_searcher._rerank_qa_pairs(question, all_qa_pairs, top_k=top_k_qa)
            
            # Apply elbow method to Q&A pairs if enabled
            if self.use_elbow_qa and reranked:
                # Extract similarity scores from reranked Q&A pairs
                qa_with_scores = [(qa, qa.get('similarity_score', qa.get('entity_score', 0))) 
                                  for qa in reranked]
                elbow_cutoff = self._find_elbow_cutoff(qa_with_scores)
                if elbow_cutoff < len(reranked):
                    if self.verbose:
                        console.print(f"[cyan]Q&A elbow filtering: {len(reranked)} → {elbow_cutoff} pairs[/cyan]")
                    reranked = reranked[:elbow_cutoff]
            
            return reranked
        
        # Otherwise just return top k by entity score
        return all_qa_pairs[:top_k_qa]
    
    def _find_elbow_cutoff(self, items_with_scores: List[Tuple[Any, float]], 
                          min_keep: int = None, relative_drop_threshold: float = 0.0) -> int:
        """Find the elbow point in a sorted list of items with scores using unsupervised detection.
        
        The method finds the point where the relative drop in similarity is maximum,
        indicating a natural boundary between relevant and less relevant items.
        
        Args:
            items_with_scores: List of (item, score) tuples sorted by score descending
            min_keep: Minimum number of items to keep (default: self.elbow_min_keep)
            relative_drop_threshold: Minimum relative drop (as fraction of current score) to consider
            
        Returns:
            Index to cut off at (items[:cutoff] will be kept)
        """
        if min_keep is None:
            min_keep = self.elbow_min_keep
            
        if len(items_with_scores) <= min_keep:
            return len(items_with_scores)
        
        # Extract scores
        scores = [score for _, score in items_with_scores]
        
        # Calculate relative drops (normalized by current score to make it scale-invariant)
        max_relative_drop = 0
        elbow_idx = len(scores)  # Default to keeping all
        
        for i in range(min_keep - 1, len(scores) - 1):
            if scores[i] > 0:  # Avoid division by zero
                absolute_drop = scores[i] - scores[i + 1]
                relative_drop = absolute_drop / scores[i]
                
                # Find the maximum relative drop
                if relative_drop > max_relative_drop and relative_drop > relative_drop_threshold:
                    max_relative_drop = relative_drop
                    elbow_idx = i + 1  # Keep items up to and including index i
        
        # Alternative: use second derivative approach for smoother curves
        if elbow_idx == len(scores) and len(scores) > min_keep + 2:
            # Calculate second derivatives
            second_derivatives = []
            for i in range(1, len(scores) - 1):
                second_deriv = scores[i-1] - 2*scores[i] + scores[i+1]
                second_derivatives.append((i, second_deriv))
            
            # Find the point with maximum curvature (highest second derivative)
            if second_derivatives:
                max_curvature_idx = max(second_derivatives, key=lambda x: x[1])[0]
                if max_curvature_idx >= min_keep:
                    elbow_idx = max_curvature_idx
        
        if self.verbose and elbow_idx < len(items_with_scores):
            console.print(f"[yellow]Elbow detected: Keeping top {elbow_idx} items (relative drop: {max_relative_drop:.2%})[/yellow]")
        
        return elbow_idx
    
    def is_question_referenced_in_future(self, current_index: int, decomposed: List[Dict[str, Any]]) -> bool:
        """Check if the current question index is referenced in any future questions.
        
        Args:
            current_index: Current question index (0-based)
            decomposed: List of all decomposed questions
            
        Returns:
            True if any future question references this one with <ENTITY_Q{}>
        """
        current_q_ref = f"<ENTITY_Q{current_index + 1}>"  # Q1 is index 0, so add 1
        
        # Check all subsequent questions (only those that require retrieval)
        for future_index in range(current_index + 1, len(decomposed)):
            future_q_info = decomposed[future_index]
            # Skip non-retrieval questions when checking for references
            if not future_q_info.get("requires_retrieval", True):
                continue
            
            future_question = future_q_info.get("question", "")
            if current_q_ref in future_question:
                return True
        
        return False
    
    def extract_entities_from_qa_pairs(self, qa_pairs: List[Dict[str, Any]], max_entities: int = 5) -> Tuple[List[str], List[str]]:
        """Extract unique entity names from Q&A pairs.
        
        Args:
            qa_pairs: List of Q&A pairs with answer information
            max_entities: Maximum number of unique entities to extract
            
        Returns:
            Tuple of (unique entity names, evidence strings used)
        """
        unique_entities = []
        qa_pair_used = []
        seen = set()
        
        for qa in qa_pairs:
            # Get answer names (could be a list or string)
            answer_names = qa.get('answer_names', qa.get('answers', []))
            if isinstance(answer_names, str):
                answer_names = [answer_names]
            
            for name in answer_names:
                if name and name not in seen:
                    unique_entities.append(name)
                    answer_text = ', '.join(qa.get('answer_names', qa.get('answer_ids', [])))
                    answer_rolestates = ', '.join(qa.get('answer_rolestates', []))
                    qa_pair_used.append(f"Q: {qa['question']} A: {answer_text} {answer_rolestates}")
                    seen.add(name)
                    if len(unique_entities) >= max_entities:
                        return unique_entities, qa_pair_used
        
        return unique_entities, qa_pair_used

    def form_reasoning_chains(self, q1_qa_pairs: List[Dict[str, Any]], q2_qa_pairs_by_entity: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Form complete reasoning chains by combining Q1 and Q2 Q&A pairs.
        
        Args:
            q1_qa_pairs: Q&A pairs from the first question
            q2_qa_pairs_by_entity: Q2 Q&A pairs grouped by entity from Q1
            
        Returns:
            List of complete reasoning chains
        """
        chains = []
        
        # For each Q1 Q&A pair
        for q1_qa in q1_qa_pairs:
            # Get the answer entities from Q1
            q1_answer_names = q1_qa.get('answer_names', q1_qa.get('answers', []))
            if isinstance(q1_answer_names, str):
                q1_answer_names = [q1_answer_names]
            
            # For each entity found in Q1
            for entity in q1_answer_names:
                if entity and entity in q2_qa_pairs_by_entity:
                    # Get Q2 Q&A pairs for this entity
                    q2_qa_pairs = q2_qa_pairs_by_entity[entity]
                    
                    # Create a chain for each Q2 Q&A pair
                    for q2_qa in q2_qa_pairs:
                        # Format the complete chain as a single text
                        chain_text = self._format_chain(q1_qa, q2_qa)
                        
                        chain = {
                            'chain_text': chain_text,
                            'q1_qa': q1_qa,
                            'q2_qa': q2_qa,
                            'entity_bridge': entity
                        }
                        chains.append(chain)
        
        return chains
    
    def convert_2hop_to_nhop_format(self, chain: Dict[str, Any]) -> Dict[str, Any]:
        """Convert 2-hop chain format to N-hop chain format for compatibility.
        
        Args:
            chain: Chain in 2-hop format {q1_qa, q2_qa, entity_bridge, chain_text}
            
        Returns:
            Chain in N-hop format {qa_chain, entity_bridges, chain_text}
        """
        if 'qa_chain' in chain:
            # Already in N-hop format
            return chain
        
        # Convert 2-hop format to N-hop format
        if 'q1_qa' in chain and 'q2_qa' in chain:
            return {
                'qa_chain': [chain['q1_qa'], chain['q2_qa']],
                'entity_bridges': [chain['entity_bridge']] if 'entity_bridge' in chain else [],
                'chain_text': chain.get('chain_text', ''),
                'chain_score': chain.get('chain_score', 0.0)
            }
        
        return chain
    
    def extend_chains_to_next_hop(self, current_chains: List[Dict[str, Any]], 
                                  next_level_qa: Any, level_idx: int) -> List[Dict[str, Any]]:
        """Extend existing chains by one hop using next level Q&A pairs.
        
        Args:
            current_chains: Existing partial chains to extend
            next_level_qa: Q&A pairs for next level (list or dict)
            level_idx: Index of the level being added (0-based)
            
        Returns:
            Extended chains with one additional hop
        """
        extended_chains = []
        
        if not current_chains:
            return []
            
        for chain in current_chains:
            # Convert 2-hop chain to N-hop format if needed
            normalized_chain = self.convert_2hop_to_nhop_format(chain)
            qa_chain = normalized_chain['qa_chain']
            entity_bridges = normalized_chain['entity_bridges']
            
            if isinstance(next_level_qa, list):
                # Non-substituted question - extend with filtered entities
                entities_with_qa = []
                for qa_pair in next_level_qa:
                    answer_names = qa_pair.get('answer_names', qa_pair.get('answers', []))
                    if isinstance(answer_names, str):
                        answer_names = [answer_names]
                    
                    for entity in answer_names:
                        if entity:
                            score = qa_pair.get('entity_score', 1.0 / (next_level_qa.index(qa_pair) + 1))
                            entities_with_qa.append((entity, qa_pair, score))
                
                # Filter to top entities
                entities_with_qa.sort(key=lambda x: x[2], reverse=True)
                seen_entities = set()
                filtered_entities = []
                for entity, qa_pair, score in entities_with_qa:
                    if entity not in seen_entities:
                        filtered_entities.append((entity, qa_pair))
                        seen_entities.add(entity)
                        if len(filtered_entities) >= self.max_entities_per_hop:
                            break
                
                # Extend current chain with each filtered entity
                for entity, qa_pair in filtered_entities:
                    extended_chain = {
                        'qa_chain': qa_chain + [qa_pair],
                        'entity_bridges': entity_bridges + [entity],
                        'chain_text': self._format_n_hop_chain(qa_chain + [qa_pair])
                    }
                    extended_chains.append(extended_chain)
                    
            elif isinstance(next_level_qa, dict):
                # Substituted question - use entities from last Q&A pair to find Q&A pairs
                last_qa_entities = []
                if qa_chain:
                    last_qa = qa_chain[-1]
                    answer_names = last_qa.get('answer_names', last_qa.get('answers', []))
                    if isinstance(answer_names, str):
                        answer_names = [answer_names]
                    last_qa_entities = [name for name in answer_names if name]
                
                # Try each entity from the last Q&A pair
                for last_entity in last_qa_entities:
                    
                    if last_entity in next_level_qa:
                        qa_pairs_for_entity = next_level_qa[last_entity]
                        
                        # Filter entities for this specific chain
                        entities_with_qa = []
                        for qa_pair in qa_pairs_for_entity:
                            answer_names = qa_pair.get('answer_names', qa_pair.get('answers', []))
                            if isinstance(answer_names, str):
                                answer_names = [answer_names]
                            
                            for entity in answer_names:
                                if entity:
                                    score = qa_pair.get('entity_score', 1.0 / (qa_pairs_for_entity.index(qa_pair) + 1))
                                    entities_with_qa.append((entity, qa_pair, score))
                        
                        # Filter to top entities
                        entities_with_qa.sort(key=lambda x: x[2], reverse=True)
                        seen_entities = set()
                        filtered_entities = []
                        for entity, qa_pair, score in entities_with_qa:
                            if entity not in seen_entities:
                                filtered_entities.append((entity, qa_pair))
                                seen_entities.add(entity)
                                if len(filtered_entities) >= self.max_entities_per_hop:
                                    break
                        
                        # Extend current chain with each filtered entity
                        for entity, qa_pair in filtered_entities:
                            extended_chain = {
                                'qa_chain': qa_chain + [qa_pair],
                                'entity_bridges': entity_bridges + [entity],
                                'chain_text': self._format_n_hop_chain(qa_chain + [qa_pair])
                            }
                            extended_chains.append(extended_chain)
        
        return extended_chains
    
    def form_n_hop_chains(self, qa_pairs_by_level: List[Any]) -> List[Dict[str, Any]]:
        """Form complete reasoning chains for N-hop questions.
        
        Args:
            qa_pairs_by_level: List where each element is either:
                - List of Q&A pairs (for first question or non-substituted questions)
                - Dict mapping entities to Q&A pairs (for substituted questions)
                
        Returns:
            List of complete N-hop reasoning chains
        """
        if not qa_pairs_by_level:
            return []
        
        # Build chains recursively
        def build_chains_recursive(level: int, partial_chain: List[Dict[str, Any]], 
                                 entity_bridges: List[str]) -> List[Dict[str, Any]]:
            """Recursively build chains across all levels."""
            
            if level >= len(qa_pairs_by_level):
                # We've reached the end, create a complete chain
                return [{
                    'qa_chain': partial_chain.copy(),
                    'entity_bridges': entity_bridges.copy(),
                    'chain_text': self._format_n_hop_chain(partial_chain)
                }]
            
            chains = []
            current_level_data = qa_pairs_by_level[level]
            
            if isinstance(current_level_data, list):
                # First level or non-substituted question - just Q&A pairs
                if level == 0:
                    # First question - start new chains
                    # Extract unique entities from all Q&A pairs and filter to top K
                    entities_with_qa = []
                    for qa_pair in current_level_data:
                        # Get entities from this Q&A pair for bridging
                        answer_names = qa_pair.get('answer_names', qa_pair.get('answers', []))
                        if isinstance(answer_names, str):
                            answer_names = [answer_names]
                        
                        # Track each entity with its source Q&A pair and score
                        for entity in answer_names:
                            if entity:
                                # Use entity_score if available, otherwise use position in list as proxy
                                score = qa_pair.get('entity_score', 1.0 / (current_level_data.index(qa_pair) + 1))
                                entities_with_qa.append((entity, qa_pair, score))
                    
                    # Sort by score and take top K unique entities
                    entities_with_qa.sort(key=lambda x: x[2], reverse=True)
                    seen_entities = set()
                    filtered_entities = []
                    for entity, qa_pair, score in entities_with_qa:
                        if entity not in seen_entities:
                            filtered_entities.append((entity, qa_pair))
                            seen_entities.add(entity)
                            if len(filtered_entities) >= self.max_entities_per_hop:
                                break
                    
                    # Now build chains only with filtered entities
                    for entity, qa_pair in filtered_entities:
                        sub_chains = build_chains_recursive(
                            level + 1, 
                            [qa_pair], 
                            [entity]
                        )
                        chains.extend(sub_chains)
                else:
                    # Non-substituted question at a later level
                    # Extract and filter entities same as level 0
                    entities_with_qa = []
                    for qa_pair in current_level_data:
                        answer_names = qa_pair.get('answer_names', qa_pair.get('answers', []))
                        if isinstance(answer_names, str):
                            answer_names = [answer_names]
                        
                        for entity in answer_names:
                            if entity:
                                score = qa_pair.get('entity_score', 1.0 / (current_level_data.index(qa_pair) + 1))
                                entities_with_qa.append((entity, qa_pair, score))
                    
                    # Sort by score and take top K unique entities
                    entities_with_qa.sort(key=lambda x: x[2], reverse=True)
                    seen_entities = set()
                    filtered_entities = []
                    for entity, qa_pair, score in entities_with_qa:
                        if entity not in seen_entities:
                            filtered_entities.append((entity, qa_pair))
                            seen_entities.add(entity)
                            if len(filtered_entities) >= self.max_entities_per_hop:
                                break
                    
                    # Build chains with filtered entities
                    for entity, qa_pair in filtered_entities:
                        new_chain = partial_chain + [qa_pair]
                        new_bridges = entity_bridges + [entity]
                        sub_chains = build_chains_recursive(
                            level + 1,
                            new_chain,
                            new_bridges
                        )
                        chains.extend(sub_chains)
            
            elif isinstance(current_level_data, dict):
                # Substituted question - dict mapping entities to Q&A pairs
                if not partial_chain:
                    # This shouldn't happen - substituted questions should not be first
                    return []
                
                # Get the last entity from the previous level
                if entity_bridges:
                    last_entity = entity_bridges[-1]
                    
                    if last_entity in current_level_data:
                        # Get Q&A pairs for this entity
                        qa_pairs_for_entity = current_level_data[last_entity]
                        
                        if level == len(qa_pairs_by_level) - 1:
                            # Last level - don't need to filter entities, just add all chains
                            for qa_pair in qa_pairs_for_entity:
                                new_chain = partial_chain + [qa_pair]
                                chains.append({
                                    'qa_chain': new_chain,
                                    'entity_bridges': entity_bridges.copy(),
                                    'chain_text': self._format_n_hop_chain(new_chain)
                                })
                        else:
                            # Not last level - need to filter entities for next hop
                            entities_with_qa = []
                            for qa_pair in qa_pairs_for_entity:
                                # Get entities from this Q&A pair for next bridging
                                answer_names = qa_pair.get('answer_names', qa_pair.get('answers', []))
                                if isinstance(answer_names, str):
                                    answer_names = [answer_names]
                                
                                for entity in answer_names:
                                    if entity:
                                        score = qa_pair.get('entity_score', 1.0 / (qa_pairs_for_entity.index(qa_pair) + 1))
                                        entities_with_qa.append((entity, qa_pair, score))
                            
                            # Sort by score and take top K unique entities
                            entities_with_qa.sort(key=lambda x: x[2], reverse=True)
                            seen_entities = set()
                            filtered_entities = []
                            for entity, qa_pair, score in entities_with_qa:
                                if entity not in seen_entities:
                                    filtered_entities.append((entity, qa_pair))
                                    seen_entities.add(entity)
                                    if len(filtered_entities) >= self.max_entities_per_hop:
                                        break
                            
                            # Build chains with filtered entities
                            for entity, qa_pair in filtered_entities:
                                new_chain = partial_chain + [qa_pair]
                                new_bridges = entity_bridges + [entity]
                                sub_chains = build_chains_recursive(
                                    level + 1,
                                    new_chain,
                                    new_bridges
                                )
                                chains.extend(sub_chains)
            
            return chains
        
        # Start building chains from level 0
        return build_chains_recursive(0, [], [])
    
    def _format_chain(self, q1_qa: Dict[str, Any], q2_qa: Dict[str, Any]) -> str:
        """Format a complete reasoning chain as text for embedding.
        
        Args:
            q1_qa: First question Q&A pair
            q2_qa: Second question Q&A pair
            
        Returns:
            Formatted chain text
        """
        # Format Q1
        q1_question = q1_qa.get('question', '')
        q1_answer_names = q1_qa.get('answer_names', q1_qa.get('answers', []))
        if isinstance(q1_answer_names, str):
            q1_answer_names = [q1_answer_names]
        q1_answer = ', '.join(str(name) for name in q1_answer_names if name)
        
        # Format Q2
        q2_question = q2_qa.get('question', '')
        q2_answer_names = q2_qa.get('answer_names', q2_qa.get('answers', []))
        if isinstance(q2_answer_names, str):
            q2_answer_names = [q2_answer_names]
        q2_answer = ', '.join(str(name) for name in q2_answer_names if name)
        
        # Create complete chain
        chain_text = f"Q: {q1_question} A: {q1_answer}. Q: {q2_question} A: {q2_answer}"
        return chain_text
    
    def _format_n_hop_chain(self, qa_chain: List[Dict[str, Any]]) -> str:
        """Format an N-hop reasoning chain as text for embedding.
        
        Args:
            qa_chain: List of Q&A pairs forming the complete chain
            
        Returns:
            Formatted chain text
        """
        chain_parts = []
        
        for qa in qa_chain:
            question = qa.get('question', '')
            answer_names = qa.get('answer_names', qa.get('answers', []))
            if isinstance(answer_names, str):
                answer_names = [answer_names]
            answer = ', '.join(str(name) for name in answer_names if name)
            
            if question and answer:
                chain_parts.append(f"Q: {question} A: {answer}")
        
        return ". ".join(chain_parts)
    
    def rerank_chains_against_original(self, chains: List[Dict[str, Any]], original_question: str) -> List[Dict[str, Any]]:
        """Rerank complete reasoning chains against the original query.
        
        Args:
            chains: List of complete reasoning chains
            original_question: The original multi-hop question
            
        Returns:
            Chains sorted by relevance to original question
        """
        if not chains:
            return []
        
        # Get embedding for original question
        original_embedding = self.entity_searcher._embed_query(original_question)
        if original_embedding is None:
            if self.verbose:
                console.print("[yellow]Could not embed original question for chain reranking[/yellow]")
            return chains  # Return unsorted if embedding fails
        
        # Calculate similarity for each chain
        for chain in chains:
            chain_embedding = self.entity_searcher._embed_query(chain['chain_text'])
            if chain_embedding is not None:
                # Calculate cosine similarity
                similarity = np.dot(original_embedding, chain_embedding) / (
                    np.linalg.norm(original_embedding) * np.linalg.norm(chain_embedding)
                )
                chain['chain_score'] = float(similarity)
            else:
                chain['chain_score'] = 0.0
        
        # Sort by chain score (highest first)
        sorted_chains = sorted(chains, key=lambda x: x['chain_score'], reverse=True)
        
        if self.verbose:
            console.print(f"[dim]Reranked {len(chains)} chains by similarity to original question[/dim]")
            if sorted_chains:
                console.print(f"[dim]Top chain score: {sorted_chains[0]['chain_score']:.3f}, Bottom: {sorted_chains[-1]['chain_score']:.3f}[/dim]")
        
        return sorted_chains
    
    def rerank_n_hop_chains(self, chains: List[Dict[str, Any]], original_question: str) -> List[Dict[str, Any]]:
        """Rerank N-hop reasoning chains against the original query.
        
        Args:
            chains: List of N-hop reasoning chains
            original_question: The original multi-hop question
            
        Returns:
            Chains sorted by relevance to original question
        """
        if not chains:
            return []
        
        # Get embedding for original question
        original_embedding = self.entity_searcher._embed_query(original_question)
        if original_embedding is None:
            if self.verbose:
                console.print("[yellow]Could not embed original question for chain reranking[/yellow]")
            return chains  # Return unsorted if embedding fails
        
        # Calculate similarity for each chain
        for chain in chains:
            chain_embedding = self.entity_searcher._embed_query(chain['chain_text'])
            if chain_embedding is not None:
                # Calculate cosine similarity
                similarity = np.dot(original_embedding, chain_embedding) / (
                    np.linalg.norm(original_embedding) * np.linalg.norm(chain_embedding)
                )
                chain['chain_score'] = float(similarity)
            else:
                chain['chain_score'] = 0.0
        
        # Sort by chain score (highest first)
        sorted_chains = sorted(chains, key=lambda x: x['chain_score'], reverse=True)
        
        if self.verbose:
            console.print(f"[dim]Reranked {len(chains)} N-hop chains by similarity to original question[/dim]")
            if sorted_chains:
                console.print(f"[dim]Top chain score: {sorted_chains[0]['chain_score']:.3f}, Bottom: {sorted_chains[-1]['chain_score']:.3f}[/dim]")
        
        return sorted_chains
    
    def extract_unique_qa_pairs_from_chains(self, selected_chains: List[Dict[str, Any]]) -> List[str]:
        """Extract unique Q&A pairs from selected chains for final evidence.
        
        Maintains the ranking order of chains while deduplicating Q&A pairs.
        Q&A pairs from higher-ranked chains appear first in the output.
        
        Args:
            selected_chains: Top-k selected reasoning chains (already sorted by relevance)
            
        Returns:
            List of unique Q&A pair strings in ranked order
        """
        seen = set()  # Track what we've already added
        unique_qa_pairs = []  # Maintains insertion order
        
        for chain in selected_chains:  # Iterate in ranked order (best chains first)
            q1_qa = chain['q1_qa']
            q2_qa = chain['q2_qa']
            
            # Format Q1 Q&A pair
            q1_question = q1_qa.get('question', '')
            q1_answer_names = q1_qa.get('answer_names', q1_qa.get('answers', []))
            if isinstance(q1_answer_names, str):
                q1_answer_names = [q1_answer_names]
            q1_answer = ', '.join(str(name) for name in q1_answer_names if name)
            q1_rolestates = ', '.join(q1_qa.get('answer_rolestates', []))
            
            if q1_question and q1_answer:
                q1_formatted = f"Q: {q1_question} A: {q1_answer}"
                if q1_rolestates:
                    q1_formatted += f" {q1_rolestates}"
                if q1_formatted not in seen:
                    seen.add(q1_formatted)
                    unique_qa_pairs.append(q1_formatted)
            
            # Format Q2 Q&A pair
            q2_question = q2_qa.get('question', '')
            q2_answer_names = q2_qa.get('answer_names', q2_qa.get('answers', []))
            if isinstance(q2_answer_names, str):
                q2_answer_names = [q2_answer_names]
            q2_answer = ', '.join(str(name) for name in q2_answer_names if name)
            q2_rolestates = ', '.join(q2_qa.get('answer_rolestates', []))
            
            if q2_question and q2_answer:
                q2_formatted = f"Q: {q2_question} A: {q2_answer}"
                if q2_rolestates:
                    q2_formatted += f" {q2_rolestates}"
                if q2_formatted not in seen:
                    seen.add(q2_formatted)
                    unique_qa_pairs.append(q2_formatted)
        
        return unique_qa_pairs  # Returns list in ranked order
    
    def extract_unique_qa_pairs_from_n_hop_chains(self, selected_chains: List[Dict[str, Any]]) -> List[str]:
        """Extract unique Q&A pairs from N-hop chains for final evidence.
        
        Maintains the ranking order of chains while deduplicating Q&A pairs.
        Q&A pairs from higher-ranked chains appear first in the output.
        
        Args:
            selected_chains: Top-k selected N-hop reasoning chains (already sorted by relevance)
            
        Returns:
            List of unique Q&A pair strings in ranked order
        """
        seen = set()  # Track what we've already added
        unique_qa_pairs = []  # Maintains insertion order
        
        for chain in selected_chains:  # Iterate in ranked order (best chains first)
            qa_chain = chain.get('qa_chain', [])
            if not qa_chain:
                chain = self.convert_2hop_to_nhop_format(chain)
                qa_chain = chain.get('qa_chain', [])
            
            for qa in qa_chain:
                # Format Q&A pair
                question = qa.get('question', '')
                answer_names = qa.get('answer_names', qa.get('answers', []))
                if isinstance(answer_names, str):
                    answer_names = [answer_names]
                answer = ', '.join(str(name) for name in answer_names if name)
                rolestates = ', '.join(qa.get('answer_rolestates', []))
                
                if question and answer:
                    formatted = f"Q: {question} A: {answer}"
                    if rolestates:
                        formatted += f" {rolestates}"
                    if formatted not in seen:
                        seen.add(formatted)
                        unique_qa_pairs.append(formatted)
        
        return unique_qa_pairs  # Returns list in ranked order

    def process_multihop_question(self, question: str, final_topk: int = 10) -> Dict[str, Any]:
        """Process a multi-hop question with chain-following approach.
        
        Args:
            question: The multi-hop question to answer
            final_topk: Maximum evidence items for final questions (unused in chain version)
            
        Returns:
            Dictionary containing the answer and collected evidence
        """
        import time
        start_time = time.time()
        
        # Generate question ID for debugging
        question_id = f"q_{hash(question) % 10000:04d}"
        
        if self.verbose:
            console.print(f"\n[bold blue]Processing with Chain Following: {question}[/bold blue]")
            if self.debug_mode:
                console.print(f"[dim]Debug ID: {question_id}[/dim]")
        
        # Step 1: Decompose the question
        decomposed = self.decompose_question(question)
        
        # Initialize debug tracking
        self.debugger.start_tracking(question_id, question, decomposed)
        
        # Step 2: Initialize storage
        all_evidence = []  # Final evidence after chain selection
        entities_by_question = {}
        chains_info = {}  # For debugging/analysis
        
        # Step 3: Process questions - now handles N-hop
        retrieval_questions = [q for q in decomposed if q["requires_retrieval"]]
        
        # Check if any retrieval question creates dependencies for future retrieval questions
        has_dependent_chains = False
        for i, q_info in enumerate(decomposed):
            if q_info.get("requires_retrieval", True):
                if self.is_question_referenced_in_future(i, decomposed):
                    has_dependent_chains = True
                    break
        
        if not has_dependent_chains:
            # No dependent chains found, use simple approach
            if self.verbose:
                console.print("[yellow]No dependent chains detected, using simple approach[/yellow]")
            self.debugger.set_fallback_reason("No dependent chains detected")
            return self._fallback_to_simple(question, decomposed)
        
        # Step 3: Process questions with incremental chain building and filtering
        current_chains = []  # Active chains that get extended and filtered at each step
        q1_qa_pairs = None   # Store Q1 for initial chain formation
        
        for q_idx, q_info in enumerate(retrieval_questions):
            q_num = f"Q{q_idx + 1}"
            
            if self.verbose:
                console.print(f"\n[cyan]{q_num}: {q_info['question']}[/cyan]")
            
            if q_idx == 0:
                # Step 3.1: Q1 - Get Q&A pairs, extract entities (max 5)
                self.debugger.track_step(q_idx, q_num, q_info['question'])
                
                q1_qa_pairs = self.search_and_collect_evidence(q_info['question'], top_k_entities=20, top_k_qa=15)
                q1_entities, _ = self.extract_entities_from_qa_pairs(q1_qa_pairs, max_entities=5)
                entities_by_question[q_num] = q1_entities
                
                # Track entity search for Q1 (no specific entity, just the question)
                self.debugger.track_entity_search("Q1_DIRECT", q_info['question'], q1_qa_pairs)
                self.debugger.finish_step()
                
                if self.verbose:
                    console.print(f"  [green]Q1 entities (max 5): {', '.join(q1_entities[:3])}{'...' if len(q1_entities) > 3 else ''}[/green]")
            
            elif q_idx == 1:
                # Step 3.2: Q1→Q2 - Form complete chains, rerank, select top K
                question_template = q_info['question']
                
                if self.verbose:
                    console.print(f"  [cyan]Q1→Q2: Forming complete chains[/cyan]")
                
                # Substitute Q1 entities into Q2
                actual_questions, has_substitution, _ = self.substitute_entities(question_template, entities_by_question)
                
                # Track this step
                self.debugger.track_step(q_idx, f"{q_num}→Q2", question_template, actual_questions, entities_by_question.get("Q1", []))
                
                if has_substitution:
                    # Collect Q2 Q&A pairs for each Q1 entity
                    q2_qa_pairs_by_entity = {}
                    for actual_q in actual_questions:
                        if self.verbose:
                            console.print(f"    → {actual_q}")
                        
                        # Find which Q1 entity this is for
                        entity_used = None
                        for ent in entities_by_question["Q1"]:
                            if ent in actual_q:
                                entity_used = ent
                                break
                        
                        if entity_used:
                            qa_pairs = self.search_and_collect_evidence(actual_q, top_k_entities=20, top_k_qa=15)
                            q2_qa_pairs_by_entity[entity_used] = qa_pairs
                            
                            # Track entity search
                            self.debugger.track_entity_search(entity_used, actual_q, qa_pairs)
                    
                    # Form complete Q1→Q2 chains
                    current_chains_2hop = self.form_reasoning_chains(q1_qa_pairs, q2_qa_pairs_by_entity)
                    current_chains = [self.convert_2hop_to_nhop_format(chain) for chain in current_chains_2hop]
                    
                    if self.verbose:
                        console.print(f"    [yellow]Formed {len(current_chains)} Q1→Q2 chains[/yellow]")
                    
                    # Rerank and filter to top K chains
                    if current_chains:
                        sorted_chains = self.rerank_chains_against_original(current_chains, question)
                        old_chains = current_chains.copy()
                        current_chains = sorted_chains[:self.chain_top_k]
                        
                        # Track chain formation and filtering
                        self.debugger.track_chain_formation(0, old_chains, current_chains)
                        
                        if self.verbose:
                            console.print(f"    [green]Filtered to top {len(current_chains)} chains[/green]")
                    
                    self.debugger.finish_step()
                else:
                    if self.verbose:
                        console.print(f"    [yellow]No substitution needed, using fallback[/yellow]")
                    self.debugger.set_fallback_reason("No entity substitution in Q2")
                    self.debugger.finish_step()
                    return self._fallback_to_simple(question, decomposed)
            
            else:
                # Step 3.3: Q1→Q2→Q3... - Extend existing chains, rerank, select top K
                question_template = q_info['question']
                
                if not current_chains:
                    if self.verbose:
                        console.print(f"  [red]No chains to extend for Q{q_idx + 1}[/red]")
                    break
                
                if self.verbose:
                    console.print(f"  [cyan]Q1→...→Q{q_idx + 1}: Extending {len(current_chains)} chains[/cyan]")
                
                # Extract UNIQUE entities from the last hop/step Q&A pairs (not just bridge entities)
                current_entities = []
                for chain in current_chains:
                    # For N-hop chains, get entities from the last Q&A pair (representing last hop)
                    if 'qa_chain' in chain and chain['qa_chain']:
                        last_qa = chain['qa_chain'][-1]
                        answer_names = last_qa.get('answer_names', last_qa.get('answers', []))
                        if isinstance(answer_names, str):
                            answer_names = [answer_names]
                        current_entities.extend([name for name in answer_names if name])
                
                # Keep only unique entities
                current_entities = list(set(current_entities))
                
                # Use current entities for substitution
                entities_by_question[f"Q{q_idx}"] = list(set(current_entities))
                actual_questions, has_substitution, current_entities = self.substitute_entities(question_template, entities_by_question)
                
                # Track this step
                self.debugger.track_step(q_idx, f"Q1→...→Q{q_idx + 1}", question_template, actual_questions, current_entities)
                
                if has_substitution:
                    # Collect Q&A pairs for current entities
                    qa_pairs_by_entity = {}
                    for actual_q in actual_questions:
                        if self.verbose:
                            console.print(f"    → {actual_q}")
                        
                        # Find which entity this is for
                        entity_used = None
                        for entity in current_entities:
                            if entity in actual_q:
                                entity_used = entity
                                break
                        
                        if entity_used:
                            qa_pairs = self.search_and_collect_evidence(actual_q, top_k_entities=20, top_k_qa=15)
                            if entity_used not in qa_pairs_by_entity:
                                qa_pairs_by_entity[entity_used] = []
                            qa_pairs_by_entity[entity_used].extend(qa_pairs)
                            
                            # Track entity search
                            self.debugger.track_entity_search(entity_used, actual_q, qa_pairs)
                    
                    # Extend current chains
                    old_chains_count = len(current_chains)
                    extended_chains = self.extend_chains_to_next_hop(current_chains, qa_pairs_by_entity, q_idx)
                    
                    if self.verbose:
                        console.print(f"    [yellow]Extended to {len(extended_chains)} chains[/yellow]")
                    
                    # Rerank and filter extended chains
                    if extended_chains:
                        sorted_chains = self.rerank_n_hop_chains(extended_chains, question)
                        old_extended = extended_chains.copy()
                        current_chains = sorted_chains[:self.chain_top_k]
                        
                        # Track chain extension and filtering
                        self.debugger.track_chain_formation(old_chains_count, old_extended, current_chains)
                        
                        if self.verbose:
                            console.print(f"    [green]Filtered to top {len(current_chains)} chains[/green]")
                    else:
                        current_chains = []
                        if self.verbose:
                            console.print(f"    [red]No chains could be extended[/red]")
                    
                    self.debugger.finish_step()
                else:
                    if self.verbose:
                        console.print(f"    [yellow]No substitution needed for Q{q_idx + 1}[/yellow]")
                    self.debugger.finish_step()
        
        # Final chains are in current_chains
        chains = current_chains
        
        if self.verbose:
            console.print(f"\n[yellow]Final: {len(chains)} complete reasoning chains[/yellow]")
        
        # Step 4: Final processing of selected chains
        if chains:
            # Chains are already filtered at each step, but do one final ranking to be sure
            sorted_chains = self.rerank_n_hop_chains(chains, question) if len(chains) > 1 else chains
            
            # Select top-k chains (may already be at the right count, but ensure consistency)
            selected_chains = sorted_chains[:self.chain_top_k]
            chains_info = {
                'total_chains': len(chains),
                'selected_chains': len(selected_chains),
                'top_score': selected_chains[0]['chain_score'] if selected_chains else 0.0,
                'score_range': f"{selected_chains[-1]['chain_score']:.3f} - {selected_chains[0]['chain_score']:.3f}" if selected_chains else "N/A"
            }
            
            if self.verbose:
                console.print(f"[green]Selected top {len(selected_chains)} chains (score range: {chains_info['score_range']})[/green]")
            
            # Set final chains for debugging
            self.debugger.set_final_chains(selected_chains)
            
            # Step 7: Extract unique Q&A pairs from selected chains
            all_evidence = self.extract_unique_qa_pairs_from_n_hop_chains(selected_chains)
            
            if self.verbose:
                console.print(f"[green]Extracted {len(all_evidence)} unique Q&A pairs from selected chains[/green]")
        else:
            if self.verbose:
                console.print("[yellow]No chains formed, falling back to simple approach[/yellow]")
            self.debugger.set_fallback_reason("No chains formed")
            return self._fallback_to_simple(question, decomposed)
        
        # Step 8: Generate final answer
        answer = self.generate_answer(question, all_evidence, decomposed)
        
        elapsed_time = time.time() - start_time
        
        # Save debug information if enabled
        debug_path = ""
        if self.debug_mode:
            debug_path = self.debugger.save_debug_info()
            if self.verbose:
                console.print(f"[dim]Debug files saved to: {debug_path}[/dim]")
        
        result = {
            "question": question,
            "answer": answer,
            "evidence_count": len(all_evidence),
            "time_taken": elapsed_time,
            "decomposed_questions": decomposed,
            "entities_found": entities_by_question,
            "chains_info": chains_info,
            "final_prompt": getattr(self, '_last_prompt', None)
        }
        
        if debug_path:
            result["debug_path"] = debug_path
        
        return result
    
    def _fallback_to_simple(self, question: str, decomposed: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback to simple approach for non-2-hop questions."""
        if self.verbose:
            console.print("[yellow]Using fallback simple approach[/yellow]")
        
        # Simple implementation for fallback
        all_evidence = []
        entities_by_question = {}
        
        for i, q_info in enumerate(decomposed):
            if not q_info["requires_retrieval"]:
                continue
                
            question_template = q_info["question"]
            is_referenced = self.is_question_referenced_in_future(i, decomposed)
            
            actual_questions, _, _ = self.substitute_entities(question_template, entities_by_question)
            
            for actual_q in actual_questions:
                qa_pairs = self.search_and_collect_evidence(actual_q, top_k_entities=20)
                
                if is_referenced:
                    entities, qa_pair_used = self.extract_entities_from_qa_pairs(qa_pairs)
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
        
        all_evidence = list(set(all_evidence))
        answer = self.generate_answer(question, all_evidence, decomposed)
        
        # Save debug information if enabled
        debug_path = ""
        if self.debug_mode:
            debug_path = self.debugger.save_debug_info()
            if self.verbose:
                console.print(f"[dim]Debug files saved to: {debug_path}[/dim]")
        
        result = {
            "question": question,
            "answer": answer,
            "evidence_count": len(all_evidence),
            "decomposed_questions": decomposed,
            "entities_found": entities_by_question,
            "chains_info": {"fallback": True}
        }
        
        if debug_path:
            result["debug_path"] = debug_path
        
        return result
    
    def generate_answer(self, original_question: str, all_evidence: List[str], 
                       decomposed_questions: List[Dict[str, Any]] = None) -> str:
        """Generate final answer using collected evidence.
        
        Args:
            original_question: The original multi-hop question
            all_evidence: All collected Q&A pairs
            decomposed_questions: The decomposed questions (for context)
            
        Returns:
            Final answer string
        """
        if not self.openai_client:
            return "OpenAI client not available for answer generation"
        
        if not all_evidence:
            return "No evidence found to answer the question"
        
        # Format evidence
        evidence_text = '\n'.join(all_evidence)
        
        # Format decomposed questions
        decomposed_questions_text = '\n'.join([f"Q{i+1}: {q['question']}" for i, q in enumerate(decomposed_questions)])
        
        # Create a simple, clear prompt
        prompt = f"""Answer the following multi-hop question using ONLY the provided evidence.

Question: {original_question}
In order to answer the question, the multi-hop question is broken down into the following single-hop questions and evidence are gathered from the knowledge base:
Decomposition: 
{decomposed_questions_text}

Available Evidence (Q&A pairs from knowledge base):
{evidence_text}

Instructions:
1. Use ONLY the Q&A pairs provided above
2. Be sure to check all the Q&A pairs for the answer
3. Do NOT use any external knowledge
4. If the evidence doesn't contain the answer, say "Cannot determine from available evidence"
5. Be concise and direct

Please respond in the following format:

<reasoning>
Reasoning about the question and the evidence.
</reasoning>
<answer>
Only the final answer, respond with a single word or phrase only.
</answer>

"""

        # Store the prompt for later access
        self._last_prompt = prompt

        # Show the prompt if requested
        if self.show_prompt:
            console.print("\n[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
            console.print("[bold cyan]FULL LLM PROMPT:[/bold cyan]")
            console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")
            console.print(Panel(prompt, expand=False, border_style="blue"))
            
            # Count tokens if tiktoken is available
            try:
                import tiktoken
                encoding = tiktoken.encoding_for_model("gpt-4o-mini")
                token_count = len(encoding.encode(prompt))
                console.print(f"\n[bold yellow]📊 Prompt tokens: {token_count}[/bold yellow]")
            except ImportError:
                console.print("[dim]Install tiktoken for token counting: pip install tiktoken[/dim]")
            except Exception:
                pass
            
            console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")

        try:
            if self.verbose:
                console.print(f"\n[cyan]Generating answer from {len(all_evidence)} Q&A pairs...[/cyan]")
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions using only provided evidence."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def run_interactive_mode(self):
        """Run interactive question-answering mode."""
        console.print("\n[bold cyan]🔗 Chain-Following Multi-Hop QA System[/bold cyan]")
        console.print("Commands:")
        console.print("  - Type a multi-hop question to get an answer")
        console.print(f"  - 'chains <number>' - Set number of chains to select (current: {self.chain_top_k})")
        console.print("  - 'prompt on/off' - Toggle showing the full LLM prompt")
        console.print("  - 'verbose on/off' - Toggle detailed output")
        console.print("  - 'help' - Show this help")
        console.print("  - 'quit' or 'exit' - Exit\n")
        
        console.print(f"[dim]Current settings: verbose={self.verbose}, show_prompt={self.show_prompt}, chains={self.chain_top_k}[/dim]")
        
        while True:
            try:
                user_input = Prompt.ask("[bold yellow]Question[/bold yellow]")
                
                if user_input.lower() in ['quit', 'exit']:
                    console.print("[bold blue]Goodbye![/bold blue]")
                    break
                
                elif user_input.lower() == 'help':
                    console.print("Ask multi-hop questions like:")
                    console.print("  - 'What is the birth year of the spouse of the director of Casablanca?'")
                    console.print("  - 'When did Lothair II's mother die?'")
                    console.print("  - 'Which film was released first, Dune or The Dark Knight?'")
                
                elif user_input.lower().startswith('chains '):
                    try:
                        self.chain_top_k = int(user_input.split()[1])
                        console.print(f"[green]✓ Chain selection set to top {self.chain_top_k}[/green]")
                    except:
                        console.print("[yellow]Invalid number. Use 'chains <integer>'[/yellow]")
                
                elif user_input.lower().startswith('prompt '):
                    setting = user_input.lower().split()[1]
                    if setting == 'on':
                        self.show_prompt = True
                        console.print("[green]✓ Prompt display enabled[/green]")
                    elif setting == 'off':
                        self.show_prompt = False
                        console.print("[green]✓ Prompt display disabled[/green]")
                    else:
                        console.print("[yellow]Use 'prompt on' or 'prompt off'[/yellow]")
                
                elif user_input.lower().startswith('verbose '):
                    setting = user_input.lower().split()[1]
                    if setting == 'on':
                        self.verbose = True
                        console.print("[green]✓ Verbose mode enabled[/green]")
                    elif setting == 'off':
                        self.verbose = False
                        console.print("[green]✓ Verbose mode disabled[/green]")
                    else:
                        console.print("[yellow]Use 'verbose on' or 'verbose off'[/yellow]")
                
                else:
                    # Process the question
                    result = self.process_multihop_question(user_input)
                    
                    # Display results
                    console.print(f"\n[bold green]Answer:[/bold green]")
                    console.print(Panel(result['answer'], expand=False))
                    
                    # Show statistics
                    console.print(f"\n[dim]Statistics:[/dim]")
                    console.print(f"  Evidence used: {result['evidence_count']} Q&A pairs")
                    console.print(f"  Time taken: {result['time_taken']:.2f} seconds")
                    
                    # Show chain info if available
                    chains_info = result.get('chains_info', {})
                    if chains_info and not chains_info.get('fallback', False):
                        console.print(f"  Chains formed: {chains_info['total_chains']}")
                        console.print(f"  Chains selected: {chains_info['selected_chains']}")
                        console.print(f"  Score range: {chains_info['score_range']}")
                    
                    # Optionally show entities found
                    if result.get('entities_found') and self.verbose:
                        console.print("\n[dim]Entities discovered:[/dim]")
                        for q_num, entities in result['entities_found'].items():
                            if entities:
                                console.print(f"  {q_num}: {', '.join(entities[:3])}{'...' if len(entities) > 3 else ''}")
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrupted. Type 'quit' to exit.[/yellow]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")


def main():
    """Main entry point."""
    console.print("\n[bold cyan]🔗 Chain-Following Multi-Hop QA System[/bold cyan]")
    console.print("Initializing...")
    
    try:
        qa_system = ChainFollowingMultiHopQA(num_documents=-1, verbose=True, debug_mode=True)
        qa_system.run_interactive_mode()
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())