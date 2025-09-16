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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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
                 chain_top_k: int = 15, beam_width: int = 5, entity_top_k: int = 20, qa_rerank_top_k: int = 15,
                 final_only_evidence: bool = False):
        """Initialize the chain-following multi-hop QA system.
        
        Args:
            num_documents: Number of documents to load
            verbose: Whether to show detailed output
            show_prompt: Whether to show the full LLM prompt
            chain_top_k: Number of top chains to select after reranking
        """
        self.verbose = verbose
        self.show_prompt = show_prompt
        self.chain_top_k = chain_top_k
        self.beam_width = beam_width
        self.entity_top_k = entity_top_k
        self.qa_rerank_top_k = qa_rerank_top_k
        self.final_only_evidence = final_only_evidence
        
        if verbose:
            console.print("[bold blue]Initializing Chain-Following Multi-Hop QA System...[/bold blue]")
            console.print(f"  Chain selection: Top {chain_top_k} chains")
            console.print(f"  Beam width per hop: {beam_width}; Entity@{entity_top_k}, QA@{qa_rerank_top_k}")
        
        # Initialize entity searcher
        self.entity_searcher = EntitySearcher(
            num_documents, 
            cache_dir="/home/shreyas/NLP/SM/gensemworkspaces/gsw-memory/playground/.gsw_cache",
            path_to_gsw_files="/mnt/SSD1/shreyas/SM_GSW/2wiki/networks",
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

    def _extract_top_entities_from_qa(self, qa_pairs: List[Dict[str, Any]], max_entities: int) -> List[Tuple[str, Dict[str, Any]]]:
        """From reranked QA pairs, pick top unique answer entities with their source QA pair.

        Returns list of (entity_name, qa_pair_used) up to max_entities.
        """
        unique: List[Tuple[str, Dict[str, Any]]] = []
        seen = set()
        for qa in qa_pairs:
            answer_names = qa.get('answer_names', qa.get('answers', []))
            if isinstance(answer_names, str):
                answer_names = [answer_names]
            for name in answer_names:
                if name and name not in seen:
                    unique.append((name, qa))
                    seen.add(name)
                    if len(unique) >= max_entities:
                        return unique
        return unique
    
    def decompose_question(self, question: str) -> List[Dict[str, Any]]:
        """Decompose a multi-hop question into single-hop questions.
        
        Reuses the decomposition logic from the original implementation.
        """
        if not self.openai_client:
            return [{"question": question, "requires_retrieval": True}]
        
        decomposition_prompt = f"""Break down this multi-hop question into a sequence of single-hop questions.
The FIRST question should keep the original specific entity/information from the question.
SUBSEQUENT questions should use <ENTITY> as a placeholder if it requires answers from the previous step to form the question.

IMPORTANT: Avoid over-decomposition. Each question should extract meaningful entities (proper nouns like names, places), not single-word descriptors. Keep questions at an appropriate granularity level.

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
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that breaks down complex questions into simple steps."},
                    {"role": "user", "content": decomposition_prompt}
                ]
                # temperature=0.1,
                # max_tokens=300
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
            return ([question_template], False)
        
        substituted_questions = []
        
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
        
        return (substituted_questions, True) if substituted_questions else ([question_template], False)
    
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
        
        # Deduplicate Q&A pairs before reranking using (question, answer_ids|answer_names) as key
        if all_qa_pairs:
            seen_keys: Dict[Any, int] = {}
            deduped_qa_pairs: List[Dict[str, Any]] = []
            for qa in all_qa_pairs:
                q_text = (qa.get('question', '') or '').strip()
                ans_ids = qa.get('answer_ids', [])
                if isinstance(ans_ids, list) and len(ans_ids) > 0:
                    ans_key = tuple(ans_ids)
                else:
                    ans_names = qa.get('answer_names', qa.get('answers', []))
                    if isinstance(ans_names, str):
                        ans_names = [ans_names]
                    ans_key = tuple(str(n).strip().lower() for n in ans_names if n)
                key = (q_text, ans_key)

                existing_idx = seen_keys.get(key)
                if existing_idx is None:
                    seen_keys[key] = len(deduped_qa_pairs)
                    deduped_qa_pairs.append(qa)
                else:
                    # Prefer the entry with higher entity_score (keep richer context)
                    prev = deduped_qa_pairs[existing_idx]
                    prev_score = float(prev.get('entity_score', 0.0) or 0.0)
                    new_score = float(qa.get('entity_score', 0.0) or 0.0)
                    if new_score > prev_score:
                        deduped_qa_pairs[existing_idx] = qa

            all_qa_pairs = deduped_qa_pairs

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
    
    def identify_reasoning_chains(self, decomposed: List[Dict[str, Any]]) -> List[List[int]]:
        """Identify which questions belong to which reasoning chains.
        
        Args:
            decomposed: List of decomposed questions with metadata
            
        Returns:
            List of chains, where each chain is a list of question indices (0-based)
        """
        # Only consider questions that require retrieval
        retrieval_questions = [(i, q) for i, q in enumerate(decomposed) if q.get("requires_retrieval", True)]
        
        if not retrieval_questions:
            return []
        
        # Build dependency graph
        dependencies = {}  # question_idx -> [list of dependent question indices]
        dependents = {}    # question_idx -> [list of questions that depend on this one]
        
        for i, (q_idx, q_info) in enumerate(retrieval_questions):
            question_text = q_info.get("question", "")
            dependencies[q_idx] = []
            
            # Find all <ENTITY_Q*> references in this question
            import re
            refs = re.findall(r'<ENTITY_Q(\d+)>', question_text)
            for ref in refs:
                dep_q_idx = int(ref) - 1  # Q1 is index 0
                if dep_q_idx < len(decomposed):
                    dependencies[q_idx].append(dep_q_idx)
                    
                    # Build reverse dependency
                    if dep_q_idx not in dependents:
                        dependents[dep_q_idx] = []
                    dependents[dep_q_idx].append(q_idx)
        
        # Find root questions (no dependencies) and build chains from them
        chains = []
        visited = set()
        
        def build_chain_from_root(root_idx: int) -> List[int]:
            """Build a reasoning chain starting from a root question."""
            chain = [root_idx]
            current = root_idx
            
            # Follow the dependency chain
            while current in dependents:
                # For now, follow the first dependent (could be extended for branching)
                next_questions = [q for q in dependents[current] if q not in visited]
                if next_questions:
                    next_q = next_questions[0]
                    chain.append(next_q)
                    visited.add(next_q)
                    current = next_q
                else:
                    break
            
            return chain
        
        # Find all root questions and build chains
        for q_idx, _ in retrieval_questions:
            if q_idx not in visited and not dependencies[q_idx]:  # Root question
                chain = build_chain_from_root(q_idx)
                if len(chain) > 1:  # Only include multi-step chains
                    chains.append(chain)
                    for idx in chain:
                        visited.add(idx)
        
        return chains
    
    def process_reasoning_chain(self, chain_indices: List[int], decomposed: List[Dict[str, Any]], 
                               original_question: str) -> List[str]:
        """Process a single reasoning chain using beam search over answer entities.

        Implements the flow:
        1) For each question in the chain, search top-K entities (20), collect and rerank QA pairs.
        2) Select top-5 unique answer entities from the reranked QA pairs to continue the chain.
        3) Maintain beams with accumulated evidence and scores across hops.
        4) At the end, send final QA pairs (configurable) for answer generation.
        """
        if self.verbose:
            chain_questions = [decomposed[i]['question'] for i in chain_indices]
            console.print(f"  [cyan]Processing chain: {' → '.join(chain_questions)}[/cyan]")

        # Pre-compute embedding for the original question to use in chain-level reranking
        orig_emb = None
        try:
            orig_emb = self.entity_searcher._embed_query(original_question)
        except Exception:
            orig_emb = None

        # Beam state per hop
        beams: List[Dict[str, Any]] = []  # each: {'entities_by_qidx': {q_idx: name}, 'evidence_pairs': [qa,...], 'score': float}

        def substitute_from_state(q: str, state: Dict[int, str]) -> str:
            out = q
            # Indexed placeholders
            for ref in re.findall(r'<ENTITY_Q(\d+)>', q):
                ref_idx = int(ref) - 1
                if ref_idx in state:
                    out = out.replace(f'<ENTITY_Q{ref}>', state[ref_idx])
            # Generic placeholder
            if '<ENTITY>' in out and state:
                last_idx = sorted(state.keys())[-1]
                out = out.replace('<ENTITY>', state[last_idx])
            return out

        for step, q_idx in enumerate(chain_indices):
            q_info = decomposed[q_idx]
            question_template = q_info['question']
            if self.verbose:
                console.print(f"    Step {step + 1}: {question_template}")

            candidate_expansions: List[Dict[str, Any]] = []

            prior_states = beams if step > 0 else [{'entities_by_qidx': {}, 'evidence_pairs': [], 'score': 0.0}]

            is_final_hop = (step == len(chain_indices) - 1)

            for state in prior_states:
                concrete_q = question_template if step == 0 else substitute_from_state(question_template, state['entities_by_qidx'])
                if self.verbose:
                    console.print(f"      → {concrete_q}")

                qa_pairs = self.search_and_collect_evidence(concrete_q, top_k_entities=self.entity_top_k, top_k_qa=self.qa_rerank_top_k)

                if is_final_hop:
                    # Final hop: select by top QA pairs directly (no entity grouping)
                    for qa_used in qa_pairs:
                        # Build new candidate including this QA
                        new_state = {
                            'entities_by_qidx': dict(state['entities_by_qidx']),
                            'evidence_pairs': list(state['evidence_pairs']),
                            'score': 0.0,
                        }
                        # Keep entity map consistent if there is an answer entity
                        answer_names = qa_used.get('answer_names', qa_used.get('answers', []))
                        if isinstance(answer_names, str):
                            answer_names = [answer_names]
                        if answer_names:
                            new_state['entities_by_qidx'][q_idx] = answer_names[0]
                        new_state['evidence_pairs'].append(qa_used)
                        new_state['last_hop_score'] = float(qa_used.get('similarity_score', 0.0))

                        # Score chain vs original
                        chain_text_parts = []
                        for qa in new_state['evidence_pairs']:
                            q_text = qa.get('question', '')
                            ans = qa.get('answer_names', qa.get('answers', []))
                            if isinstance(ans, str):
                                ans = [ans]
                            a_text = ', '.join(str(x) for x in ans if x)
                            if q_text and a_text:
                                chain_text_parts.append(f"Q: {q_text} A: {a_text}")
                        chain_text = " | ".join(chain_text_parts) if chain_text_parts else ""
                        try:
                            if orig_emb is not None and chain_text:
                                emb = self.entity_searcher._embed_chain(chain_text)
                                if emb is not None:
                                    sim = float(np.dot(orig_emb, emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(emb)))
                                else:
                                    sim = -1.0
                            else:
                                sim = -1.0
                        except Exception:
                            sim = -1.0
                        new_state['chain_score'] = sim
                        candidate_expansions.append(new_state)
                else:
                    # Non-final hop: entity-by-best-QA selection
                    per_entity_best: Dict[str, Dict[str, Any]] = {}
                    for qa_used in qa_pairs:
                        answer_names = qa_used.get('answer_names', qa_used.get('answers', []))
                        if isinstance(answer_names, str):
                            answer_names = [answer_names]
                        answer_ids = qa_used.get('answer_ids', [])
                        for idx, ent_name in enumerate(answer_names):
                            if not ent_name:
                                continue
                            # Prefer stable ID when available, else fall back to name
                            ent_key = None
                            try:
                                if isinstance(answer_ids, list) and idx < len(answer_ids) and answer_ids[idx]:
                                    ent_key = str(answer_ids[idx])
                            except Exception:
                                ent_key = None
                            if not ent_key:
                                ent_key = str(ent_name)

                            new_state = {
                                'entities_by_qidx': dict(state['entities_by_qidx']),
                                'evidence_pairs': list(state['evidence_pairs']),
                                'score': 0.0,
                            }
                            new_state['entities_by_qidx'][q_idx] = ent_name
                            new_state['evidence_pairs'].append(qa_used)
                            new_state['last_hop_score'] = float(qa_used.get('similarity_score', 0.0))

                            # Score chain vs original
                            chain_text_parts = []
                            for qa in new_state['evidence_pairs']:
                                q_text = qa.get('question', '')
                                ans = qa.get('answer_names', qa.get('answers', []))
                                if isinstance(ans, str):
                                    ans = [ans]
                                a_text = ', '.join(str(x) for x in ans if x)
                                if q_text and a_text:
                                    chain_text_parts.append(f"Q: {q_text} A: {a_text}")
                            chain_text = " | ".join(chain_text_parts) if chain_text_parts else ""
                            try:
                                if orig_emb is not None and chain_text:
                                    emb = self.entity_searcher._embed_chain(chain_text)
                                    if emb is not None:
                                        sim = float(np.dot(orig_emb, emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(emb)))
                                    else:
                                        sim = -1.0
                                else:
                                    sim = -1.0
                            except Exception:
                                sim = -1.0
                            new_state['chain_score'] = sim

                            prev = per_entity_best.get(ent_key)
                            if prev is None or (new_state['chain_score'], new_state['last_hop_score']) > (
                                prev.get('chain_score', -1.0), prev.get('last_hop_score', 0.0)
                            ):
                                per_entity_best[ent_key] = new_state

                    candidate_expansions.extend(per_entity_best.values())

            if not candidate_expansions:
                beams = []
                break

            # Prune to beam width globally using chain-level reranking against the original question.
            # Fallback to current-hop score if embeddings are unavailable.
            if any('chain_score' in s and s['chain_score'] is not None for s in candidate_expansions):
                candidate_expansions.sort(
                    key=lambda s: (s.get('chain_score', -1.0), s.get('last_hop_score', 0.0)),
                    reverse=True,
                )
            else:
                candidate_expansions.sort(key=lambda s: s.get('last_hop_score', 0.0), reverse=True)
            beams = candidate_expansions[: self.beam_width]
            if self.verbose:
                console.print(f"      Expanded to {len(candidate_expansions)} beams → kept top {len(beams)}")

        # Evidence from final beams
        if not beams:
            return []

        evidence: List[str] = []
        seen = set()
        for state in beams:
            qa_list = state['evidence_pairs'] if not self.final_only_evidence else ([state['evidence_pairs'][-1]] if state['evidence_pairs'] else [])
            for qa in qa_list:
                q_text = qa.get('question', '')
                answer_names = qa.get('answer_names', qa.get('answers', []))
                if isinstance(answer_names, str):
                    answer_names = [answer_names]
                answer_text = ', '.join(str(name) for name in answer_names if name)
                answer_rolestates = ', '.join(qa.get('answer_rolestates', []))
                if q_text and answer_text:
                    formatted = f"Q: {q_text} A: {answer_text}"
                    if answer_rolestates:
                        formatted += f" {answer_rolestates}"
                    if formatted not in seen:
                        seen.add(formatted)
                        evidence.append(formatted)

        if self.verbose:
            console.print(f"      Final evidence: {len(evidence)} unique Q&A pairs")
        return evidence
    
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
        
        # Step 3: Identify reasoning chains
        chains = self.identify_reasoning_chains(decomposed)
        
        if not chains:
            # No chains found, use simple approach
            if self.verbose:
                console.print("[yellow]No reasoning chains detected, using simple approach[/yellow]")
            return self._fallback_to_simple(question, decomposed)
        
        if self.verbose:
            console.print(f"\n[cyan]Identified {len(chains)} reasoning chains:[/cyan]")
            for i, chain in enumerate(chains, 1):
                chain_questions = [decomposed[idx]['question'] for idx in chain]
                console.print(f"  Chain {i}: {' → '.join(chain_questions)}")
        
        # Step 4: Process each chain independently
        all_evidence = []
        chain_entities = {}
        
        for chain_idx, chain_indices in enumerate(chains):
            if self.verbose:
                console.print(f"\n[yellow]Processing Chain {chain_idx + 1}:[/yellow]")
            
            chain_evidence = self.process_reasoning_chain(chain_indices, decomposed, question)
            all_evidence.extend(chain_evidence)
            
            # Store entities for this chain (for debugging)
            chain_entities[f"Chain_{chain_idx + 1}"] = f"{len(chain_evidence)} evidence items"
        
        # Update entities_by_question for compatibility
        entities_by_question = chain_entities
        
        # Update chains_info for compatibility
        chains_info = {
            'total_chains': len(chains),
            'selected_chains': len(chains),
            'multi_chain_approach': True
        }
        
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

    def collect_evidence(self, question: str, decomposed: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """Collect evidence only (no answer generation) using the chain-following approach.

        Args:
            question: The multi-hop question to retrieve evidence for.
            decomposed: Optional precomputed decomposition to avoid an LLM call. If provided,
                        it will be used directly; otherwise this method will call decompose_question().

        Returns:
            Tuple of (evidence strings list, chains_info dict, decomposed questions list)
        """
        if self.verbose:
            console.print(f"\n[bold blue]Collecting evidence (chain-following): {question}[/bold blue]")

        # Decompose the question if not provided (allows external batched decomposition)
        if decomposed is None:
            decomposed = self.decompose_question(question)

        # Identify reasoning chains
        chains = self.identify_reasoning_chains(decomposed)

        # If no chains, use a simple fallback retrieval that preserves order
        if not chains:
            if self.verbose:
                console.print("[yellow]No reasoning chains detected, using simple fallback retrieval[/yellow]")

            all_evidence: List[str] = []
            entities_by_question: Dict[str, List[str]] = {}

            for i, q_info in enumerate(decomposed):
                if not q_info.get("requires_retrieval", True):
                    continue

                question_template = q_info.get("question", "")
                is_referenced = self.is_question_referenced_in_future(i, decomposed)

                actual_questions, _ = self.substitute_entities(question_template, entities_by_question)
                for actual_q in actual_questions:
                    qa_pairs = self.search_and_collect_evidence(actual_q, top_k_entities=self.entity_top_k, top_k_qa=self.qa_rerank_top_k)

                    if is_referenced:
                        entities, qa_pair_used = self.extract_entities_from_qa_pairs(qa_pairs)
                        entities_by_question[f"Q{i+1}"] = list(dict.fromkeys(entities))
                        all_evidence.extend(qa_pair_used)
                    else:
                        # Format as evidence strings
                        for qa in qa_pairs[:5]: # Limit to number of QA pairs per non chain question.
                            q_text = qa.get('question', '')
                            answer_names = qa.get('answer_names', qa.get('answers', []))
                            if isinstance(answer_names, str):
                                answer_names = [answer_names]
                            answer_text = ', '.join(str(name) for name in answer_names if name)
                            answer_rolestates = ', '.join(qa.get('answer_rolestates', []))
                            if q_text and answer_text:
                                all_evidence.append(f"Q: {q_text} A: {answer_text} {answer_rolestates}")

            # Deduplicate while preserving order
            seen = set()
            unique_evidence: List[str] = []
            for ev in all_evidence:
                if ev not in seen:
                    seen.add(ev)
                    unique_evidence.append(ev)

            chains_info = {"fallback": True}
            return unique_evidence, chains_info, decomposed

        # Process each identified chain independently using beam search
        all_evidence: List[str] = []
        for chain_indices in chains:
            chain_evidence = self.process_reasoning_chain(chain_indices, decomposed, question)
            all_evidence.extend(chain_evidence)

        # Deduplicate across chains while preserving order
        seen = set()
        unique_evidence: List[str] = []
        for ev in all_evidence:
            if ev not in seen:
                seen.add(ev)
                unique_evidence.append(ev)

        chains_info = {
            'total_chains': len(chains),
            'selected_chains': len(chains),
            'multi_chain_approach': True
        }

        return unique_evidence, chains_info, decomposed
    
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
            
            actual_questions, _ = self.substitute_entities(question_template, entities_by_question)
            
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
        
        # Remove duplicates while preserving order
        seen = set()
        unique_evidence = []
        for evidence in all_evidence:
            if evidence not in seen:
                seen.add(evidence)
                unique_evidence.append(evidence)
        all_evidence = unique_evidence
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
        """Generate final answer using collected evidence with oracle-style prompting.
        
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
        
        # Build oracle-style prompt with one-shot example
        prompt_text = f"""
{evidence_text}
\n\nQuestion: {original_question}
\n\nThought: 

"""
        
        # Store the prompt for later access
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
        )
        
        # System message for advanced reading comprehension
        rag_qa_system = (
            'As an advanced reading comprehension assistant, your task is to analyze precise QA pairs extracted from the documents and corresponding questions meticulously. '
            'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
            'Conclude with "Answer: " to present only a concise, definitive response, devoid of additional elaborations.'
        )
        
        # One-shot example input
        one_shot_input = (
            f"{one_shot_docs}"
            "\n\nQuestion: "
            "When was Neville A. Stanton's employer founded?"
            '\nThought: '
        )
        
        # One-shot example output
        one_shot_output = (
            "From the QA pairs, the employer of Neville A. Stanton is University of Southampton. The University of Southampton was founded in 1862. "
            "\nAnswer: 1862."
        )
        
        # Build the prompt template
        prompt_messages = [
            {"role": "system", "content": rag_qa_system},
            {"role": "user", "content": one_shot_input},
            {"role": "assistant", "content": one_shot_output},
            {"role": "user", "content": prompt_text}
        ]

        # Show the prompt if requested
        if self.show_prompt:
            console.print("\n[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
            console.print("[bold cyan]FULL LLM PROMPT:[/bold cyan]")
            console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]\n")
            console.print(Panel(f"System: {rag_qa_system}\n\nUser: {one_shot_input}\n\nAssistant: {one_shot_output}\n\nUser: {prompt_text}", expand=False, border_style="blue"))
            
            # Count tokens if tiktoken is available
            try:
                import tiktoken
                encoding = tiktoken.encoding_for_model("gpt-4o-mini")
                token_count = len(encoding.encode(str(prompt_messages)))
                console.print(f"\n[bold yellow]📊 Prompt tokens: ~{token_count}[/bold yellow]")
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
                messages=prompt_messages,
                temperature=0.0,
                max_tokens=1000
            )
            
            answer_text = response.choices[0].message.content
            
            # Parse answer with new format
            if 'Answer: ' in answer_text:
                final_answer = answer_text.split('Answer: ')[-1].strip()
                # Remove any trailing period if it's just a number/date
                if final_answer.endswith('.') and final_answer[:-1].replace(',', '').replace(' ', '').isdigit():
                    final_answer = final_answer[:-1]
                return answer_text  # Return full response including reasoning
            else:
                # Fallback if no "Answer:" found
                if self.verbose:
                    console.print("[yellow]Warning: No 'Answer:' found in response, using full text[/yellow]")
                return answer_text
            
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
