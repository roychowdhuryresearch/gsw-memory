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
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import sys
from datetime import datetime
from collections import defaultdict
import numpy as np
import voyageai
from pydantic import BaseModel

import os 
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table
from rich.panel import Panel
from itertools import product

# Optional HTTP client for external reranker
try:
    import requests as _requests
except Exception:
    _requests = None
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


class DecomposedQuestion(BaseModel):
    question: str
    requires_retrieval: bool

class DecomposedQuestionList(BaseModel):
    questions: List[DecomposedQuestion]


class ChainFollowingMultiHopQA:
    """Chain-following multi-hop QA system with intelligent chain reranking."""
    
    def __init__(self, num_documents: int = 200, verbose: bool = True, show_prompt: bool = False,
                 chain_top_k: int = 15, beam_width: int = 5, entity_top_k: int = 20, qa_rerank_top_k: int = 15,
                 final_only_evidence: bool = False, chain_following_mode: str = "cumulative", use_bm25: bool = False, alpha: float = 0.5,
                 use_chain_reranker: bool = False, reranker_model_name: str = None,
                 reranker_instruction: str = "Given a question, score the chain of QA pairs based on how likely it is to lead to the answer",
                 reranker_endpoint_url: Optional[str] = None, reranker_http_timeout: float = 15.0,
                 multi_dep_quality_threshold: float = 0.3, use_gpu_for_qa_index: bool = True):
        """Initialize the chain-following multi-hop QA system.

        Args:
            num_documents: Number of documents to load
            verbose: Whether to show detailed output
            show_prompt: Whether to show the full LLM prompt
            chain_top_k: Number of top chains to select after reranking
            chain_following_mode: How to score chains - "similarity" (cosine sim to original) or "cumulative" (product of QA scores)
            use_gpu_for_qa_index: Whether to use GPU for Q&A FAISS index (default: False, uses CPU to save GPU memory)
        """
        self.verbose = verbose
        self.show_prompt = show_prompt
        self.chain_top_k = chain_top_k
        self.beam_width = beam_width
        self.entity_top_k = entity_top_k
        self.qa_rerank_top_k = qa_rerank_top_k
        self.final_only_evidence = final_only_evidence
        self.chain_following_mode = chain_following_mode
        self.use_bm25 = use_bm25
        self.alpha = alpha # If weighting cumulative and chain scores
        self.use_chain_reranker = use_chain_reranker
        self.reranker_instruction = reranker_instruction
        self.reranker_model_name = reranker_model_name
        self.reranker_endpoint_url = reranker_endpoint_url
        self.reranker_http_timeout = reranker_http_timeout
        self.multi_dep_quality_threshold = multi_dep_quality_threshold
        self._reranker = None
        self.voyage_client = None
        self.rerank_instruction = "Given a question, you need to score the chain of decomposed questions based on how likely it is to lead to the answer. Make sure to focues on both questions"
        self.use_bm25 = use_bm25
        if self.reranker_model_name == "voyage":
            self.voyage_client = voyageai.Client()

        if verbose:
            console.print("[bold blue]Initializing Chain-Following Multi-Hop QA System...[/bold blue]")
            console.print(f"  Chain selection: Top {chain_top_k} chains")
            console.print(f"  Beam width per hop: {beam_width}; Entity@{entity_top_k}, QA@{qa_rerank_top_k}")
            console.print(f"  Chain following mode: {chain_following_mode}")
            if self.use_chain_reranker:
                if self.reranker_endpoint_url:
                    console.print(f"  Chain reranker: HTTP endpoint {self.reranker_endpoint_url}")
                else:
                    console.print("  Chain reranker: in-process (Qwen score API)")
        
        # Initialize entity searcher
        self.entity_searcher = EntitySearcher(
            num_documents,
            cache_dir="/mnt/SSD1/shreyas/SM_GSW/musique/.gsw_cache_4_1_mini",
            path_to_gsw_files="/mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini",
            verbose=False,  # Keep entity searcher quiet
            use_bm25=self.use_bm25,
            rebuild_cache=False,
            use_gpu_for_qa_index=use_gpu_for_qa_index
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

    def _format_chain_text_from_state(self, state: Dict[str, Any]) -> str:
        """Format state's evidence pairs into a single chain text.

        Returns "Q: ... A: ... | Q: ... A: ..." style string.
        """
        parts = []
        for qa in state.get('evidence_pairs', []):
            q_text = qa.get('question', '')
            ans = qa.get('answer_names', qa.get('answers', []))
            rolestate = qa.get('answer_rolestates', [])
            if isinstance(ans, str):
                ans = [ans]
            a_text = ', '.join(str(x) for x in ans if x)
            r_text = ', '.join(str(x) for x in rolestate if x)
            if q_text and a_text:
                parts.append(f"Q: {q_text} A: {a_text} {r_text}")
        return " | ".join(parts) if parts else ""

    def _rerank_states(self, original_question: str, states: List[Dict[str, Any]]) -> None:
        """Use cross-encoder reranker to set state['chain_score'] in [0,1].

        No fusion: reranker score replaces existing chain_score for pruning.
        Safe no-op if reranker is unavailable or states empty.
        """
        if not self.use_chain_reranker or not states:
            return

        # Build chain texts
        chain_texts = [self._format_chain_text_from_state(s) for s in states]
        valid = [i for i, t in enumerate(chain_texts) if t]
        if not valid:
            return

        # Prefer HTTP reranker when endpoint is configured
        if self.reranker_model_name == "voyage":
            docs_payload = [chain_texts[i] for i in valid]
            reranking = self.voyage_client.rerank(original_question, docs_payload, model="rerank-2.5", top_k=len(valid))
            for r in reranking.results:
                states[r.index]["chain_score"] = r.relevance_score

        elif self.reranker_endpoint_url:
            docs_payload = [chain_texts[i] for i in valid]
            payload = {
                "model": self.reranker_model_name or "tomaarsen/Qwen3-Reranker-8B-seq-cls",
                "query": original_question,
                "documents": docs_payload,
            }
            try:
                if _requests is not None:
                    resp = _requests.post(
                        self.reranker_endpoint_url,
                        headers={"accept": "application/json", "Content-Type": "application/json"},
                        json=payload,
                        timeout=self.reranker_http_timeout,
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                    else:
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                else:
                    from urllib import request as _urlreq
                    req = _urlreq.Request(
                        self.reranker_endpoint_url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"accept": "application/json", "Content-Type": "application/json"},
                        method="POST",
                    )
                    with _urlreq.urlopen(req, timeout=self.reranker_http_timeout) as r:
                        data = json.loads(r.read().decode("utf-8"))

                results = data.get("results", [])
                for item in results:
                    try:
                        idx_in_batch = int(item.get("index", -1))
                        score = float(item.get("relevance_score", 0.0))
                        if 0 <= idx_in_batch < len(valid):
                            states[valid[idx_in_batch]]["chain_score"] = score
                    except Exception:
                        continue
            except Exception as e:
                if self.verbose:
                    console.print(f"[yellow]HTTP rerank failed; keeping existing scores: {e}[/yellow]")
        elif self._reranker is not None:
            raise ValueError("Reranker is currently not served, serve as http endpoint")

    def _rerank_evidence(self, question: str, all_evidence: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Rerank the evidence using the chain reranker."""
        if not self.use_chain_reranker or not all_evidence:
            return all_evidence
        
        # Build chain texts

        reranking = self.voyage_client.rerank(question, all_evidence, model="rerank-2.5", top_k=top_k) # For simple fallback hardcode to 1.
        reranked_evidence = []
        for r in reranking.results:
            reranked_evidence.append(r.document)

        return reranked_evidence

    def _compute_similarity_score(self, state: Dict[str, Any], orig_emb: np.ndarray) -> float:
        """Compute similarity-based score between chain text and original question.

        Args:
            state: State dict with 'evidence_pairs' containing QA pairs
            orig_emb: Original question embedding

        Returns:
            Similarity score (cosine similarity) or -1.0 on error
        """
        eps = 1e-6
        # Build chain text from evidence pairs
        chain_text_parts = []
        for qa in state.get('evidence_pairs', []):
            q_text = qa.get('question', '')
            ans = qa.get('answer_names', qa.get('answers', []))
            if isinstance(ans, str):
                ans = [ans]
            a_text = ', '.join(str(x) for x in ans if x)
            rolestate = ', '.join(qa.get('answer_rolestates', []))
            if q_text and a_text:
                chain_text_parts.append(f"Q: {q_text} A: {a_text}, {rolestate}")

        chain_text = " | ".join(chain_text_parts) if chain_text_parts else ""

        # Compute similarity score
        try:
            if orig_emb is not None and chain_text:
                emb = self.entity_searcher._embed_chain(chain_text)
                if emb is not None:
                    similarity = float(np.dot(orig_emb, emb) / (np.linalg.norm(orig_emb) * np.linalg.norm(emb)))
                    normalized_score = 0.5 * (similarity + 1)
                    normalized_score = max(eps, min(normalized_score, 1))
                    return float(normalized_score)
            return eps
        except Exception:
            return eps

    def _compute_cumulative_score(self, state: Dict[str, Any]) -> float:
        """Compute cumulative score as product of individual QA pair scores.

        Args:
            state: State dict with 'evidence_pairs' containing QA pairs

        Returns:
            Cumulative score (product of QA scores)
        """
        evidence_pairs = state.get('evidence_pairs', [])
        eps = 1e-6
        if not evidence_pairs:
            return eps

        
        # Product of all QA pair scores
        normalized_scores = []
        for qa in evidence_pairs:
            # Get the best available score for this QA pair
            score = qa.get('similarity_score', qa.get('entity_score', 0.0))
            if score is None: 
                score = -1 
            try: 
                score = float(score)
            except Exception as e:
                print(f"Error converting score to float: {e}, defaulting to -1")
                score = -1

            normalized_score = 0.5 * (score + 1)
            normalized_score = max(eps, min(normalized_score, 1))
            normalized_scores.append(normalized_score)

        #Geometric mean
        geometric_mean = float(np.exp(np.mean(np.log(normalized_scores))))

        return float(geometric_mean)


    def _compute_combined_score(self, state: Dict[str, Any], orig_emb: np.ndarray, alpha: float = 0.5) -> float:
        """
        Combine both cumulative and chain similarity scores.
        """

        cumulative_score = self._compute_cumulative_score(state)
        chain_score = self._compute_similarity_score(state, orig_emb)
        return alpha * cumulative_score + (1 - alpha) * chain_score


    def _harmonic_mean(self, scores: List[float]) -> float:
        """Compute harmonic mean of scores.

        Harmonic mean heavily penalizes low outliers, making it ideal for
        multi-dependency scoring where one weak parent breaks the chain.

        Args:
            scores: List of scores (all should be > 0)

        Returns:
            Harmonic mean: n / (1/s1 + 1/s2 + ... + 1/sn)
        """
        if not scores:
            return 1e-6

        # Filter out zero/negative scores
        valid_scores = [s for s in scores if s > 1e-6]
        if not valid_scores:
            return 1e-6

        n = len(valid_scores)
        harmonic = n / sum(1.0 / s for s in valid_scores)
        return float(harmonic)

    def _score_chain_state(self, state: Dict[str, Any], orig_emb: np.ndarray = None) -> Dict[str, Any]:
        """Score a chain state using the configured scoring method.

        Args:
            state: State dict with 'evidence_pairs' containing QA pairs
            orig_emb: Original question embedding (required for similarity mode)

        Returns:
            Updated state with 'chain_score' added
        """
        if self.chain_following_mode == "cumulative":
            state['chain_score'] = self._compute_cumulative_score(state)
        # elif self.chain_following_mode == "similarity":
        #     state['chain_score'] = self._compute_similarity_score(state, orig_emb)
        # elif self.chain_following_mode == "combined":
        #     state['chain_score'] = self._compute_combined_score(state, orig_emb, self.alpha)
        else:
            raise ValueError(f"Invalid chain following mode: {self.chain_following_mode}")

        return state

    def _create_expansion_state(self, base_state: Dict[str, Any], qa_used: Dict[str, Any], q_idx: int, is_last_hop: bool = False) -> Dict[str, Any]:
        """Create a new expanded state by adding a QA pair to the base state.

        Args:
            base_state: Base state to expand from
            qa_used: QA pair to add
            q_idx: Question index for entity mapping

        Returns:
            New state with QA added and scores updated
        """
        new_state = {
            'entities_by_qidx': dict(base_state.get('entities_by_qidx', {})),
            'evidence_pairs': list(base_state.get('evidence_pairs', [])),
            'score': 0.0,
        }

        # Update entity mapping if there's an answer entity
        answer_names = qa_used.get('answer_names', qa_used.get('answers', []))
        if isinstance(answer_names, str):
            answer_names = [answer_names]
        if answer_names:
            new_state['entities_by_qidx'][q_idx] = answer_names[0] if not is_last_hop else answer_names

        # Add QA pair to evidence
        new_state['evidence_pairs'].append(qa_used)
        new_state['last_hop_score'] = float(qa_used.get('similarity_score', 0.0))

        return new_state

    def _prune_to_beam_width(self, candidates: List[Dict[str, Any]], beam_width: int) -> List[Dict[str, Any]]:
        """Prune candidate states to beam width using chain scores.

        Args:
            candidates: List of candidate states with scores
            beam_width: Maximum number of states to keep

        Returns:
            Top beam_width states sorted by score
        """
        if not candidates:
            return []

        # Sort by chain score (primary) and last hop score (secondary)
        if any('chain_score' in s and s['chain_score'] is not None for s in candidates):
            candidates.sort(
                key=lambda s: (s.get('chain_score', -1.0), s.get('last_hop_score', 0.0)),
                reverse=True
            )
        else:
            # Fallback to last hop score if chain scores unavailable
            candidates.sort(key=lambda s: s.get('last_hop_score', 0.0), reverse=True)

        return candidates[:beam_width]

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


Question: [the question text]
Requires retrieval: [True/False]

And provide the response in the following JSON format:
{{
  "questions": [
    {{
      "question": "the decomposed question text",
      "requires_retrieval": "True/False"
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
Input: "{question}"
Output:
"""

        try:
            response = self.openai_client.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that breaks down complex questions into simple steps."},
                    {"role": "user", "content": decomposition_prompt}
                ],
                response_format=DecomposedQuestionList
            )

            # Parse the structured response using Pydantic
            parsed_response = response.choices[0].message.parsed
            questions = [
                {
                    "question": q.question,
                    "requires_retrieval": q.requires_retrieval
                }
                for q in parsed_response.questions
            ]
            
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
    
    def search_and_collect_evidence(self, question: str, question_embedding: np.ndarray = None, top_k_entities: int = 10, top_k_qa: int = 15) -> List[Dict[str, Any]]:
        """ABLATION: Search using ONLY entity-based search (no dual search).

        Uses only entity-based search, NOT direct Q&A search.

        Args:
            question: The question to search for
            question_embedding: The embedding of the question
            top_k_entities: Number of top entities to retrieve
            top_k_qa: Number of top Q&A pairs to keep after reranking

        Returns:
            List of relevant Q&A pairs with metadata
        """
        if self.verbose:
            console.print(f"[yellow]ABLATION: Using entity-based search only (no dual search)[/yellow]")

        # ABLATION: Only entity-based search (no direct Q&A search)
        search_results = self.entity_searcher.search(
            query=question,
            query_embedding=question_embedding,
            top_k=top_k_entities,
            verbose=False
        )

        # Extract Q&A pairs from entity search results
        all_qa_pairs = []
        for entity, score in search_results:
            qa_pairs = entity.get('qa_pairs', [])
            for qa in qa_pairs:
                qa_with_context = qa.copy()
                qa_with_context['source_entity'] = entity['name']
                qa_with_context['source_entity_id'] = entity['id']
                qa_with_context['doc_id'] = entity['doc_id']
                qa_with_context['entity_score'] = score
                qa_with_context['search_question'] = question
                qa_with_context['source_method'] = 'entity_search'
                all_qa_pairs.append(qa_with_context)

        if self.verbose:
            console.print(f"[cyan]Entity-based search found {len(all_qa_pairs)} Q&A pairs[/cyan]")

        # Rerank Q&A pairs if we have embedding capability
        if hasattr(self.entity_searcher, '_rerank_qa_pairs') and all_qa_pairs:
            # Usage voyage reranker to rerank the qa pairs

            qa_texts = []
            for qa in all_qa_pairs:
                # Combine question and answer for embedding
                answer_names = qa.get('answer_names', qa.get('answers', []))
                answer_rolestates = qa.get('answer_rolestates', [])
                qa_text = f"{qa['question']} {', '.join(answer_names)} {', '.join(answer_rolestates)}"
                qa_texts.append(qa_text)

            reranking = self.voyage_client.rerank(question, qa_texts, model="rerank-2.5", top_k=len(qa_texts))
            for r in reranking.results:
                all_qa_pairs[r.index]['similarity_score'] = r.relevance_score

            # Sort by similarity score from list of dicts
            all_qa_pairs = sorted(all_qa_pairs, key=lambda x: x['similarity_score'], reverse=True)
            reranked = all_qa_pairs[:top_k_qa]
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

        This enhanced version handles both linear chains and DAG structures where
        a question can depend on multiple previous questions.

        Args:
            decomposed: List of decomposed questions with metadata

        Returns:
            List of chains, where each chain is a list of question indices (0-based)
            For DAG structures, returns the connected component in topological order
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

        # Find connected components (handles both linear chains and DAGs)
        chains = []
        visited = set()

        def topological_sort_component(component_nodes: set) -> List[int]:
            """Perform topological sort on a component of the dependency graph."""
            # Create in-degree map for the component
            in_degree = {}
            for node in component_nodes:
                in_degree[node] = 0
                for dep in dependencies.get(node, []):
                    if dep in component_nodes:
                        in_degree[node] += 1

            # Find nodes with no dependencies within component
            queue = [node for node in component_nodes if in_degree[node] == 0]
            result = []

            while queue:
                # Sort queue to ensure deterministic ordering
                queue.sort()
                node = queue.pop(0)
                result.append(node)

                # Process dependents
                for dependent in dependents.get(node, []):
                    if dependent in component_nodes:
                        in_degree[dependent] -= 1
                        if in_degree[dependent] == 0:
                            queue.append(dependent)

            return result

        def find_connected_component(start_idx: int) -> set:
            """Find all questions in the connected component containing start_idx."""
            component = set()
            to_explore = [start_idx]

            while to_explore:
                current = to_explore.pop()
                if current in component:
                    continue
                component.add(current)

                # Add dependencies and dependents
                for dep in dependencies.get(current, []):
                    if dep not in component:
                        to_explore.append(dep)
                for dependent in dependents.get(current, []):
                    if dependent not in component:
                        to_explore.append(dependent)

            return component

        # Process each unvisited question
        for q_idx, _ in retrieval_questions:
            if q_idx not in visited:
                # Find the connected component
                component = find_connected_component(q_idx)

                # Only process multi-question components
                if len(component) > 1:
                    # Perform topological sort to get proper ordering
                    sorted_component = topological_sort_component(component)
                    chains.append(sorted_component)

                    # Mark all nodes as visited
                    visited.update(component)
                elif len(component) == 1:
                    # Single question - mark as visited but don't create a chain
                    visited.add(q_idx)

        return chains
    
    def process_reasoning_chain(self, chain_indices: List[int], decomposed: List[Dict[str, Any]],
                               original_question: str) -> List[str]:
        """Process a single reasoning chain using beam search over answer entities.

        Enhanced to handle DAG structures where questions can have multiple dependencies.

        Implements the flow:
        1) For each question in the chain, search top-K entities (20), collect and rerank QA pairs.
        2) Select top-5 unique answer entities from the reranked QA pairs to continue the chain.
        3) Maintain beams with accumulated evidence and scores across hops.
        4) Handle multi-dependency questions by combining entity states from multiple parents.
        5) At the end, send final QA pairs (configurable) for answer generation.
        """
        if self.verbose:
            chain_questions = [decomposed[i]['question'] for i in chain_indices]
            console.print(f"  [cyan]Processing chain: {' → '.join(chain_questions)}[/cyan]")

        # Pre-compute embedding for the original question to use in chain-level reranking
        # orig_emb = None
        # try:
        #     orig_emb = self.entity_searcher._embed_query(original_question)
        # except Exception:
        #     orig_emb = None

        # Build dependency map for this chain
        dependencies = {}
        for q_idx in chain_indices:
            question_text = decomposed[q_idx].get("question", "")
            deps = []
            refs = re.findall(r'<ENTITY_Q(\d+)>', question_text)
            for ref in refs:
                dep_idx = int(ref) - 1
                if dep_idx in chain_indices:
                    deps.append(dep_idx)
            dependencies[q_idx] = deps

        # Process questions in topological order (chain_indices should already be sorted)
        completed_questions = {}  # q_idx -> List of beam states

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

            # Get dependencies for this question
            deps = dependencies.get(q_idx, [])

            if not deps:
                # No dependencies - this is a root question
                prior_states = [{'entities_by_qidx': {}, 'evidence_pairs': [], 'score': 0.0}]
            elif len(deps) == 1:
                # Single dependency - use existing logic
                prior_states = completed_questions.get(deps[0], [])
            else:
                # Multiple dependencies - need to combine states with harmonic mean pruning
                dep_states = [completed_questions.get(dep, [])[: self.beam_width] for dep in deps]

                # Score all possible combinations with harmonic mean BEFORE creating merged states
                combinations = []
                for state_combination in product(*dep_states):
                    if not state_combination:
                        continue

                    # Compute harmonic mean of parent chain scores
                    parent_scores = [s.get('chain_score', 0.5) for s in state_combination]
                    h_mean = self._harmonic_mean(parent_scores)

                    combinations.append({
                        'harmonic_score': h_mean,
                        'states': state_combination
                    })

                # Sort by harmonic mean (best combinations first)
                combinations.sort(key=lambda x: x['harmonic_score'], reverse=True)

                # Take top beam_width combinations that meet quality threshold
                top_combinations = [
                    c for c in combinations[:self.beam_width] #TODO: Try beam_width*2 or different approach
                    if c['harmonic_score'] >= self.multi_dep_quality_threshold
                ]

                # Fallback: if all combinations are below threshold, take best 1
                if not top_combinations and combinations:
                    top_combinations = combinations[:1]

                if self.verbose and len(combinations) > len(top_combinations):
                    console.print(f"      [dim]Pruned {len(combinations)} combinations → {len(top_combinations)} via harmonic mean (threshold={self.multi_dep_quality_threshold:.2f})[/dim]")

                # NOW create merged states only for selected high-quality combinations
                prior_states = []
                for combo in top_combinations:
                    combined_state = {
                        'entities_by_qidx': {},
                        'evidence_pairs': [],
                        'pre_combination_score': combo['harmonic_score']  # Track for debugging
                    }
                    for parent_state in combo['states']:
                        combined_state['entities_by_qidx'].update(parent_state['entities_by_qidx'])
                        combined_state['evidence_pairs'].extend(parent_state['evidence_pairs'])
                    prior_states.append(combined_state)

            is_final_hop = (step == len(chain_indices) - 1)
            substituted_questions = []
            substituted_entities = {}

            concrete_questions = []
            for state in prior_states:
                concrete_q = substitute_from_state(question_template, state['entities_by_qidx'])
                concrete_questions.append(concrete_q)
                console.print(f"Processing Question: {concrete_q}")

            if concrete_questions:
                concrete_q_embeddings = self.entity_searcher._embed_query(concrete_questions)
            else:
                concrete_q_embeddings = None
            for state, concrete_q, concrete_q_embedding in zip(prior_states, concrete_questions, concrete_q_embeddings):
                if concrete_q not in substituted_questions:
                    qa_pairs = self.search_and_collect_evidence(question=concrete_q, question_embedding=concrete_q_embedding, top_k_entities=self.entity_top_k, top_k_qa=self.qa_rerank_top_k)
                    substituted_entities[concrete_q] = qa_pairs

                else:
                    qa_pairs = substituted_entities[concrete_q]

                if is_final_hop:
                    # Final hop: select by top QA pairs directly (no entity grouping)
                    for qa_used in qa_pairs:
                        # Create new state with QA pair
                        new_state = self._create_expansion_state(state, qa_used, q_idx, is_last_hop=True)
                        # Score the chain
                        # new_state = self._score_chain_state(new_state, orig_emb)
                        new_state = self._score_chain_state(new_state)
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

                            # Create and score new state
                            new_state = self._create_expansion_state(state, qa_used, q_idx)
                            new_state['entities_by_qidx'][q_idx] = ent_name  # Override with specific entity
                            # new_state = self._score_chain_state(new_state, orig_emb)
                            new_state = self._score_chain_state(new_state)

                            # Keep best scoring chain for each entity
                            prev = per_entity_best.get(ent_key)
                            if prev is None or (new_state['chain_score'], new_state['last_hop_score']) > (
                                prev.get('chain_score', -1.0), prev.get('last_hop_score', 0.0)
                            ):
                                per_entity_best[ent_key] = new_state

                    candidate_expansions.extend(per_entity_best.values())

            if not candidate_expansions:
                completed_questions[q_idx] = []
                continue

            # Rerank candidates with cross-encoder (no fusion), then prune
            # self._rerank_states(original_question, candidate_expansions)
            if not is_final_hop:
                beams = self._prune_to_beam_width(candidate_expansions, self.beam_width)
                
            # Final rerank before evidence extraction only for final hop
            if is_final_hop:
                # self._rerank_states(original_question, candidate_expansions)
                beams = self._prune_to_beam_width(candidate_expansions, self.beam_width)
                
            if self.verbose:
                console.print(f"      Expanded to {len(candidate_expansions)} beams → kept top {len(beams)}")

            # Store completed beams for this question (for use by dependent questions)
            completed_questions[q_idx] = beams

        # Evidence from final question's beams
        # Get the last question index from the chain
        final_q_idx = chain_indices[-1] if chain_indices else None
        final_beams = completed_questions.get(final_q_idx, []) if final_q_idx is not None else []

        # Evidence from final beams

        if not final_beams:
            return []

        evidence: List[str] = []
        seen = set()
        for state in final_beams:
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

    
    def process_multihop_question(self, question: str, final_topk: int = 10) -> Dict[str, Any]:
        """Process a multi-hop question with chain-following approach.
        
        Args:
            question: The multi-hop question to answer
            final_topk: Maximum evidence items for final questions (unused in chain version)
            
        Returns:
            Dictionary containing the answer and collected evidence
        """
        import time
        
        if self.verbose:
            console.print(f"\n[bold blue]Processing with Chain Following: {question}[/bold blue]")
        
        # Step 1: Decompose the question
        decomposed = self.decompose_question(question)
        start_time = time.time()
        
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

            all_evidence = []
            # entities_by_question = {}

            query_embeddings = self.entity_searcher._embed_query([question])[0]
            qa_pairs = self.search_and_collect_evidence(question, query_embeddings, top_k_entities=20)
            for qa in qa_pairs:
                q_text = qa.get('question', '')
                answer_names = qa.get('answer_names', qa.get('answers', []))
                if isinstance(answer_names, str):
                    answer_names = [answer_names]
                answer_text = ', '.join(str(name) for name in answer_names if name)
                answer_rolestates = ', '.join(qa.get('answer_rolestates', []))
                if q_text and answer_text:
                    all_evidence.append(f"Q: {q_text} A: {answer_text} {answer_rolestates}")
            
            all_evidence = list(set(all_evidence))

            chains_info = {"fallback": True}
            return all_evidence, chains_info, decomposed


            # if self.verbose:
            #     console.print("[yellow]No reasoning chains detected, using simple fallback retrieval[/yellow]")

            # all_evidence: List[str] = []
            # entities_by_question: Dict[str, List[str]] = {}

            # for i, q_info in enumerate(decomposed):
            #     if not q_info.get("requires_retrieval", True):
            #         continue

            #     question_template = q_info.get("question", "")
            #     is_referenced = self.is_question_referenced_in_future(i, decomposed)

            #     actual_questions, _ = self.substitute_entities(question_template, entities_by_question)

            #     actual_question_embeddings = self.entity_searcher._embed_query(actual_questions)

            #     for actual_q, actual_q_embedding in zip(actual_questions, actual_question_embeddings):
            #         qa_pairs = self.search_and_collect_evidence(actual_q, actual_q_embedding, top_k_entities=self.entity_top_k, top_k_qa=self.qa_rerank_top_k)

            #         if is_referenced:
            #             entities, qa_pair_used = self.extract_entities_from_qa_pairs(qa_pairs)
            #             entities_by_question[f"Q{i+1}"] = list(dict.fromkeys(entities))
            #             all_evidence.extend(qa_pair_used)
            #         else:
            #             # Format as evidence strings
            #             for qa in qa_pairs[:5]: # Limit to number of QA pairs per non chain question.
            #                 q_text = qa.get('question', '')
            #                 answer_names = qa.get('answer_names', qa.get('answers', []))
            #                 if isinstance(answer_names, str):
            #                     answer_names = [answer_names]
            #                 answer_text = ', '.join(str(name) for name in answer_names if name)
            #                 answer_rolestates = ', '.join(qa.get('answer_rolestates', []))
            #                 if q_text and answer_text:
            #                     all_evidence.append(f"Q: {q_text} A: {answer_text} {answer_rolestates}")

            # # Deduplicate while preserving order
            # seen = set()
            # unique_evidence: List[str] = []
            # for ev in all_evidence:
            #     if ev not in seen:
            #         seen.add(ev)
            #         unique_evidence.append(ev)

            # # Rerank the evidence
            # unique_evidence = self._rerank_evidence(question=question, all_evidence=unique_evidence, top_k=5)

            # chains_info = {"fallback": True}
            # return unique_evidence, chains_info, decomposed

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

        query_embeddings = self.entity_searcher._embed_query([question])[0]
        qa_pairs = self.search_and_collect_evidence(question, query_embeddings, top_k_entities=20)
        for qa in qa_pairs:
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


        # questions_to_collect_evidence = [q["question"] for q in decomposed if q["requires_retrieval"]]
        # query_embeddings = self.entity_searcher._embed_query(questions_to_collect_evidence)

        # for question, query_embedding in zip(questions_to_collect_evidence, query_embeddings):
        #     qa_pairs = self.search_and_collect_evidence(question, query_embedding, top_k_entities=20)

        #     for qa in qa_pairs:
        #         q_text = qa.get('question', '')
        #         answer_names = qa.get('answer_names', qa.get('answers', []))
        #         if isinstance(answer_names, str):
        #             answer_names = [answer_names]
        #         answer_text = ', '.join(str(name) for name in answer_names if name)
        #         answer_rolestates = ', '.join(qa.get('answer_rolestates', []))
        #         if q_text and answer_text:
        #             all_evidence.append(f"Q: {q_text} A: {answer_text} {answer_rolestates}")

        # all_evidence = list(set(all_evidence))

        # # Rerank the evidence
        # all_evidence = self._rerank_evidence(question=question, all_evidence=all_evidence, top_k=5)

        # return {
        #     "question": question,
        #     "answer": answer,
        #     "evidence_count": len(all_evidence),
        #     "decomposed_questions": decomposed,
        #     "entities_found": entities_by_question,
        #     "chains_info": {"fallback": True}
        # }
        
        # for i, q_info in enumerate(decomposed):
        #     if not q_info["requires_retrieval"]:
        #         continue
                
        #     question_template = q_info["question"]
        #     is_referenced = self.is_question_referenced_in_future(i, decomposed)
            
        #     actual_questions, _ = self.substitute_entities(question_template, entities_by_question)

        #     actual_question_embeddings = self.entity_searcher._embed_query(actual_questions)

        #     for actual_q, actual_q_embedding in zip(actual_questions, actual_question_embeddings):
        #         qa_pairs = self.search_and_collect_evidence(actual_q, actual_q_embedding, top_k_entities=20)
                
        #         if is_referenced:
        #             entities, qa_pair_used = self.extract_entities_from_qa_pairs(qa_pairs)
        #             entities_by_question[f"Q{i+1}"] = list(set(entities))
        #             all_evidence.extend(qa_pair_used)
        #         else:
        #             # Format as evidence strings
        #             for qa in qa_pairs:
        #                 q_text = qa.get('question', '')
        #                 answer_names = qa.get('answer_names', qa.get('answers', []))
        #                 if isinstance(answer_names, str):
        #                     answer_names = [answer_names]
        #                 answer_text = ', '.join(str(name) for name in answer_names if name)
        #                 answer_rolestates = ', '.join(qa.get('answer_rolestates', []))
        #                 if q_text and answer_text:
        #                     all_evidence.append(f"Q: {q_text} A: {answer_text} {answer_rolestates}")
        
        # # Remove duplicates while preserving order
        # seen = set()
        # unique_evidence = []
        # for evidence in all_evidence:
        #     if evidence not in seen:
        #         seen.add(evidence)
        #         unique_evidence.append(evidence)
        # all_evidence = unique_evidence

        # # Rerank the evidence
        # all_evidence = self._rerank_evidence(question=question, all_evidence=all_evidence, top_k=5)

        # answer = self.generate_answer(question, all_evidence, decomposed)
        
        # return {
        #     "question": question,
        #     "answer": answer,
        #     "evidence_count": len(all_evidence),
        #     "decomposed_questions": decomposed,
        #     "entities_found": entities_by_question,
        #     "chains_info": {"fallback": True}
        # }
    
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
            )
        
        # System message for advanced reading comprehension
        rag_qa_system = (
            'As an advanced reading comprehension assistant, your task is to analyze precise QA pairs extracted from the documents and corresponding questions meticulously. '
            'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
            'Conclude with "Answer: " to present only a concise, definitive response, devoid of additional elaborations.'
            # 'If you don\'t know the answer, say "No Answer".'
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
        qa_system = ChainFollowingMultiHopQA(num_documents=-1, verbose=True, chain_following_mode="cumulative", beam_width=5, alpha=0.5, use_chain_reranker=True, reranker_model_name="voyage", multi_dep_quality_threshold=0.3, use_bm25=True)
        qa_system.run_interactive_mode()
    except Exception as e:
        console.print(f"[red]Failed to initialize: {e}[/red]")
        return 1
    
    return 0


if __name__ == "__main__":   
     exit(main())
