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


class ChainFollowingMultiHopQA:
    """Chain-following multi-hop QA system with intelligent chain reranking."""
    
    def __init__(self, num_documents: int = 200, verbose: bool = True, show_prompt: bool = False, 
                 chain_top_k: int = 15, max_entities_per_hop: int = 5):
        """Initialize the chain-following multi-hop QA system.
        
        Args:
            num_documents: Number of documents to load
            verbose: Whether to show detailed output
            show_prompt: Whether to show the full LLM prompt
            chain_top_k: Number of top chains to select after reranking
            max_entities_per_hop: Maximum entities to consider at each hop to prevent explosion
        """
        self.verbose = verbose
        self.show_prompt = show_prompt
        self.chain_top_k = chain_top_k
        self.max_entities_per_hop = max_entities_per_hop
        
        if verbose:
            console.print("[bold blue]Initializing Chain-Following Multi-Hop QA System...[/bold blue]")
            console.print(f"  Chain selection: Top {chain_top_k} chains")
        
        # Initialize entity searcher
        self.entity_searcher = EntitySearcher(
            num_documents, 
            # cache_dir="/home/yigit/codebase/gsw-memory/.gsw_cache",
            cache_dir="/mnt/SSD1/shreyas/SM_GSW/musique/.gsw_cache",
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
        
        decomposition_prompt = f"""Break down this multi-hop question into a sequence of single-hop questions.
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

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that breaks down complex questions into simple steps."},
                    {"role": "user", "content": decomposition_prompt}
                ],
                temperature=0.0,
                max_tokens=300
            )
            
            decomposition_text = response.choices[0].message.content
            
            # Parse the response
            questions = []
            lines = decomposition_text.strip().split('\n')
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                if re.match(r'^[\d]+[\.)\s]*Question:', line) or line.startswith('Question:'):
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
                            
                    entities.extend(entities_by_question[q_key])
                
        return (substituted_questions, True, entities_by_question[q_key]) if substituted_questions else ([question_template], False, [])
    
    def search_and_collect_evidence(self, question: str, top_k_entities: int = 10, top_k_qa: int = 15) -> List[Dict[str, Any]]:
        """Search for a question and collect relevant Q&A pairs.
        
        Args:
            question: The question to search for
            top_k_entities: Number of top entities to retrieve
            top_k_qa: Number of top Q&A pairs to keep after reranking
            
        Returns:
            List of relevant Q&A pairs with metadata
        """
        # Search for relevant entities
        search_results = self.entity_searcher.search(
            query=question,
            top_k=top_k_entities,
            verbose=False
        )
        
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
            return reranked
        
        # Otherwise just return top k by entity score
        return all_qa_pairs[:top_k_qa]
    
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
        
        if self.verbose:
            console.print(f"\n[bold blue]Processing with Chain Following: {question}[/bold blue]")
        
        # Step 1: Decompose the question
        decomposed = self.decompose_question(question)
        
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
                q1_qa_pairs = self.search_and_collect_evidence(q_info['question'], top_k_entities=20, top_k_qa=15)
                q1_entities, _ = self.extract_entities_from_qa_pairs(q1_qa_pairs, max_entities=10)
                entities_by_question[q_num] = q1_entities
                
                if self.verbose:
                    console.print(f"  [green]Q1 entities (max 5): {', '.join(q1_entities[:3])}{'...' if len(q1_entities) > 3 else ''}[/green]")
            
            elif q_idx == 1:
                # Step 3.2: Q1→Q2 - Form complete chains, rerank, select top K
                question_template = q_info['question']
                
                if self.verbose:
                    console.print(f"  [cyan]Q1→Q2: Forming complete chains[/cyan]")
                
                # Substitute Q1 entities into Q2
                actual_questions, has_substitution, _ = self.substitute_entities(question_template, entities_by_question)
                
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
                    
                    # Form complete Q1→Q2 chains
                    current_chains = self.form_reasoning_chains(q1_qa_pairs, q2_qa_pairs_by_entity)
                    
                    if self.verbose:
                        console.print(f"    [yellow]Formed {len(current_chains)} Q1→Q2 chains[/yellow]")
                    
                    # Rerank and filter to top K chains
                    if current_chains:
                        sorted_chains = self.rerank_chains_against_original(current_chains, question)
                        current_chains = sorted_chains[:self.chain_top_k]
                        
                        if self.verbose:
                            console.print(f"    [green]Filtered to top {len(current_chains)} chains[/green]")
                else:
                    if self.verbose:
                        console.print(f"    [yellow]No substitution needed, using fallback[/yellow]")
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
                    # For 2-hop chains, get entities from the Q2 answer (last hop)
                    elif 'q2_qa' in chain:
                        q2_qa = chain['q2_qa']
                        answer_names = q2_qa.get('answer_names', q2_qa.get('answers', []))
                        if isinstance(answer_names, str):
                            answer_names = [answer_names]
                        current_entities.extend([name for name in answer_names if name])
                    # Last resort fallback to bridge entities
                    elif 'entity_bridge' in chain:
                        current_entities.append(chain['entity_bridge'])
                    elif 'entity_bridges' in chain and chain['entity_bridges']:
                        current_entities.append(chain['entity_bridges'][-1])
                
                # Keep only unique entities
                current_entities = list(set(current_entities))
                
                # Use current entities for substitution
                entities_by_question[f"Q{q_idx}"] = list(set(current_entities))
                actual_questions, has_substitution, current_entities = self.substitute_entities(question_template, entities_by_question)
                
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
                    
                    # Extend current chains
                    extended_chains = self.extend_chains_to_next_hop(current_chains, qa_pairs_by_entity, q_idx)
                    
                    if self.verbose:
                        console.print(f"    [yellow]Extended to {len(extended_chains)} chains[/yellow]")
                    
                    # Rerank and filter extended chains
                    if extended_chains:
                        sorted_chains = self.rerank_n_hop_chains(extended_chains, question)
                        current_chains = sorted_chains[:self.chain_top_k]
                        
                        if self.verbose:
                            console.print(f"    [green]Filtered to top {len(current_chains)} chains[/green]")
                    else:
                        current_chains = []
                        if self.verbose:
                            console.print(f"    [red]No chains could be extended[/red]")
                else:
                    if self.verbose:
                        console.print(f"    [yellow]No substitution needed for Q{q_idx + 1}[/yellow]")
        
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
            
            # Step 7: Extract unique Q&A pairs from selected chains
            all_evidence = self.extract_unique_qa_pairs_from_n_hop_chains(selected_chains)
            
            if self.verbose:
                console.print(f"[green]Extracted {len(all_evidence)} unique Q&A pairs from selected chains[/green]")
        else:
            if self.verbose:
                console.print("[yellow]No chains formed, falling back to simple approach[/yellow]")
            return self._fallback_to_simple(question, decomposed)
        
        # Step 8: Generate final answer
        answer = self.generate_answer(question, all_evidence, decomposed)
        
        elapsed_time = time.time() - start_time
        
        return {
            "question": question,
            "answer": answer,
            "evidence_count": len(all_evidence),
            "time_taken": elapsed_time,
            "decomposed_questions": decomposed,
            "entities_found": entities_by_question,
            "chains_info": chains_info,
            "final_prompt": getattr(self, '_last_prompt', None)
        }
    
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
        
        return {
            "question": question,
            "answer": answer,
            "evidence_count": len(all_evidence),
            "decomposed_questions": decomposed,
            "entities_found": entities_by_question,
            "chains_info": {"fallback": True}
        }
    
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
        
        # Create a simple, clear prompt
        prompt = f"""Answer the following multi-hop question using ONLY the provided evidence.

Question: {original_question}

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
        qa_system = ChainFollowingMultiHopQA(num_documents=-1, verbose=True)
        qa_system.run_interactive_mode()
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())