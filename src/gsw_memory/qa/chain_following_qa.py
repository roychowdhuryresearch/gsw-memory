"""
Chain-following multi-hop question answering.

Implements the Panini pipeline:
1. Decompose questions into atomic sub-questions
2. Identify reasoning chains (linear or DAG) via topological sort
3. Process each chain with beam search over answer entities
4. Dual retrieval (entity embeddings + QA-pair search) with reranking
5. Generate final answer from collected evidence
"""

import logging
import re
import time
from itertools import product
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .chain_answer_generator import ChainAnswerGenerator
from .chain_models import ChainFollowingResult
from .gsw_tools import GSWTools
from .question_decomposer import QuestionDecomposer

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

try:
    import voyageai

    VOYAGEAI_AVAILABLE = True
except ImportError:
    voyageai = None
    VOYAGEAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class ChainFollowingQA:
    """Chain-following multi-hop QA with beam search and dual retrieval.

    Example::

        from gsw_memory.qa import ChainFollowingQA, GSWTools

        tools = GSWTools(gsw_file_paths)
        tools.build_index()

        qa = ChainFollowingQA(tools, decomposition_model="gpt-4o")
        result = qa.ask("What is the birthplace of the performer of Changed It?")
        print(result.answer)
    """

    def __init__(
        self,
        gsw_tools: GSWTools,
        decomposition_model: str = "gpt-4o",
        answering_model: str = "gpt-4o-mini",
        beam_width: int = 5,
        chain_top_k: int = 15,
        entity_top_k: int = 20,
        qa_rerank_top_k: int = 15,
        scoring_mode: str = "cumulative",
        alpha: float = 0.5,
        allow_no_answer: bool = False,
        multi_dep_quality_threshold: float = 0.3,
        verbose: bool = False,
    ):
        """
        Args:
            gsw_tools: Initialised GSWTools instance with index already built.
            decomposition_model: Model for question decomposition.
            answering_model: Model for answer generation.
            beam_width: Beam width per hop.
            chain_top_k: Top-k chains to keep.
            entity_top_k: Entities to retrieve per hop.
            qa_rerank_top_k: QA pairs to keep after reranking per hop.
            scoring_mode: One of "cumulative", "similarity", "combined", "none".
            alpha: Weight for combined scoring (cumulative * alpha + similarity * (1-alpha)).
            allow_no_answer: Allow "No Answer" responses.
            multi_dep_quality_threshold: Minimum harmonic-mean quality for multi-dependency combinations.
            verbose: Log detailed progress.
        """
        self.gsw_tools = gsw_tools
        self.beam_width = beam_width
        self.chain_top_k = chain_top_k
        self.entity_top_k = entity_top_k
        self.qa_rerank_top_k = qa_rerank_top_k
        self.scoring_mode = scoring_mode
        self.alpha = alpha
        self.multi_dep_quality_threshold = multi_dep_quality_threshold
        self.verbose = verbose

        self.decomposer = QuestionDecomposer(model_name=decomposition_model)
        self.answer_generator = ChainAnswerGenerator(
            model_name=answering_model,
            allow_no_answer=allow_no_answer,
        )

        # Voyage reranker (initialised lazily)
        self._voyage_client = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(self, question: str) -> ChainFollowingResult:
        """Full pipeline: decompose → retrieve → answer.

        Returns a :class:`ChainFollowingResult` with the answer, evidence,
        decomposition, chain info, and timing.
        """
        start = time.time()

        evidence, chains_info, decomposed = self.collect_evidence(question)
        answer = self.generate_answer(question, evidence)

        elapsed = time.time() - start
        return ChainFollowingResult(
            question=question,
            answer=answer,
            evidence=evidence,
            evidence_count=len(evidence),
            decomposed_questions=decomposed,
            chains_info=chains_info,
            time_taken=elapsed,
        )

    def collect_evidence(
        self,
        question: str,
        decomposed: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """Retrieve evidence without generating an answer.

        Returns (evidence_strings, chains_info, decomposed_questions).
        """
        if decomposed is None:
            decomposed = self.decompose_question(question)

        chains = self.identify_reasoning_chains(decomposed)

        if not chains:
            # Fallback: simple single-hop retrieval
            all_evidence = self._fallback_retrieval(question)
            return all_evidence, {"fallback": True}, decomposed

        if self.verbose:
            logger.info("Identified %d reasoning chain(s)", len(chains))

        all_evidence: List[str] = []
        for chain_indices in chains:
            chain_evidence = self.process_reasoning_chain(
                chain_indices, decomposed, question
            )
            all_evidence.extend(chain_evidence)

        # Deduplicate preserving order
        seen: set = set()
        unique: List[str] = []
        for ev in all_evidence:
            if ev not in seen:
                seen.add(ev)
                unique.append(ev)

        chains_info = {
            "total_chains": len(chains),
            "selected_chains": len(chains),
            "multi_chain_approach": True,
        }
        return unique, chains_info, decomposed

    def decompose_question(self, question: str) -> List[Dict[str, Any]]:
        """Decompose a multi-hop question into sub-questions."""
        return self.decomposer.decompose(question)

    def generate_answer(self, question: str, evidence: List[str]) -> str:
        """Generate a final answer from evidence."""
        return self.answer_generator.generate(question, evidence)

    # ------------------------------------------------------------------
    # Chain identification (pure logic, no API)
    # ------------------------------------------------------------------

    def identify_reasoning_chains(
        self, decomposed: List[Dict[str, Any]]
    ) -> List[List[int]]:
        """Identify reasoning chains from decomposed questions.

        Handles both linear chains and DAG structures. Returns a list of
        chains where each chain is a topologically-sorted list of question
        indices (0-based).
        """
        retrieval_questions = [
            (i, q)
            for i, q in enumerate(decomposed)
            if q.get("requires_retrieval", True)
        ]
        if not retrieval_questions:
            return []

        # Build dependency graph
        dependencies: Dict[int, List[int]] = {}
        dependents: Dict[int, List[int]] = {}

        for q_idx, q_info in retrieval_questions:
            question_text = q_info.get("question", "")
            dependencies[q_idx] = []

            refs = re.findall(r"<ENTITY_Q(\d+)>", question_text)
            for ref in refs:
                dep_q_idx = int(ref) - 1  # Q1 → index 0
                if dep_q_idx < len(decomposed):
                    dependencies[q_idx].append(dep_q_idx)
                    dependents.setdefault(dep_q_idx, []).append(q_idx)

        # Find connected components
        chains: List[List[int]] = []
        visited: set = set()

        def _find_component(start: int) -> set:
            component: set = set()
            stack = [start]
            while stack:
                node = stack.pop()
                if node in component:
                    continue
                component.add(node)
                for dep in dependencies.get(node, []):
                    if dep not in component:
                        stack.append(dep)
                for dep in dependents.get(node, []):
                    if dep not in component:
                        stack.append(dep)
            return component

        def _topo_sort(nodes: set) -> List[int]:
            in_degree = {n: 0 for n in nodes}
            for n in nodes:
                for dep in dependencies.get(n, []):
                    if dep in nodes:
                        in_degree[n] += 1
            queue = sorted(n for n in nodes if in_degree[n] == 0)
            result: List[int] = []
            while queue:
                node = queue.pop(0)
                result.append(node)
                for dep in dependents.get(node, []):
                    if dep in nodes:
                        in_degree[dep] -= 1
                        if in_degree[dep] == 0:
                            queue.append(dep)
                            queue.sort()
            return result

        for q_idx, _ in retrieval_questions:
            if q_idx not in visited:
                component = _find_component(q_idx)
                if len(component) > 1:
                    chains.append(_topo_sort(component))
                visited.update(component)

        return chains

    # ------------------------------------------------------------------
    # Beam-search chain processing
    # ------------------------------------------------------------------

    def process_reasoning_chain(
        self,
        chain_indices: List[int],
        decomposed: List[Dict[str, Any]],
        original_question: str,
    ) -> List[str]:
        """Process a single reasoning chain using beam search.

        Returns a list of formatted evidence strings.
        """
        orig_emb = None
        if self.scoring_mode in ("similarity", "combined"):
            orig_emb = self.gsw_tools.embed_query(original_question)

        # Build dependency map for this chain
        dependencies: Dict[int, List[int]] = {}
        for q_idx in chain_indices:
            question_text = decomposed[q_idx].get("question", "")
            deps = []
            for ref in re.findall(r"<ENTITY_Q(\d+)>", question_text):
                dep_idx = int(ref) - 1
                if dep_idx in chain_indices:
                    deps.append(dep_idx)
            dependencies[q_idx] = deps

        completed: Dict[int, List[Dict[str, Any]]] = {}

        for step, q_idx in enumerate(chain_indices):
            question_template = decomposed[q_idx]["question"]
            deps = dependencies.get(q_idx, [])

            # Determine prior states
            if not deps:
                prior_states = [
                    {"entities_by_qidx": {}, "evidence_pairs": [], "score": 0.0}
                ]
            elif len(deps) == 1:
                prior_states = completed.get(deps[0], [])
            else:
                prior_states = self._combine_multi_dep_states(deps, completed)

            if not prior_states:
                prior_states = [
                    {"entities_by_qidx": {}, "evidence_pairs": [], "score": 0.0}
                ]

            is_final = step == len(chain_indices) - 1

            # Substitute entities and search
            candidates: List[Dict[str, Any]] = []
            substituted_cache: Dict[str, List[Dict[str, Any]]] = {}

            for state in prior_states:
                concrete_q = self._substitute_from_state(
                    question_template, state["entities_by_qidx"]
                )

                if concrete_q not in substituted_cache:
                    qa_pairs = self.search_and_collect_evidence(
                        concrete_q,
                        top_k_entities=self.entity_top_k,
                        top_k_qa=self.qa_rerank_top_k,
                    )
                    substituted_cache[concrete_q] = qa_pairs
                else:
                    qa_pairs = substituted_cache[concrete_q]

                if is_final:
                    for qa in qa_pairs:
                        new_state = self._create_expansion_state(
                            state, qa, q_idx, is_last_hop=True
                        )
                        new_state = self._score_chain_state(new_state, orig_emb)
                        candidates.append(new_state)
                else:
                    per_entity_best: Dict[str, Dict[str, Any]] = {}
                    for qa in qa_pairs:
                        answer_names = qa.get("answer_names", qa.get("answers", []))
                        if isinstance(answer_names, str):
                            answer_names = [answer_names]
                        answer_ids = qa.get("answer_ids", [])
                        for idx_e, ent_name in enumerate(answer_names):
                            if not ent_name:
                                continue
                            ent_key = None
                            if (
                                isinstance(answer_ids, list)
                                and idx_e < len(answer_ids)
                                and answer_ids[idx_e]
                            ):
                                ent_key = str(answer_ids[idx_e])
                            if not ent_key:
                                ent_key = str(ent_name)

                            new_state = self._create_expansion_state(state, qa, q_idx)
                            new_state["entities_by_qidx"][q_idx] = ent_name
                            new_state = self._score_chain_state(new_state, orig_emb)

                            prev = per_entity_best.get(ent_key)
                            if prev is None or (
                                new_state["chain_score"],
                                new_state["last_hop_score"],
                            ) > (
                                prev.get("chain_score", -1.0),
                                prev.get("last_hop_score", 0.0),
                            ):
                                per_entity_best[ent_key] = new_state
                    candidates.extend(per_entity_best.values())

            if not candidates:
                completed[q_idx] = []
                continue

            beams = self._prune_to_beam_width(candidates, self.beam_width)
            completed[q_idx] = beams

        # Extract evidence from final beams
        final_q_idx = chain_indices[-1] if chain_indices else None
        final_beams = completed.get(final_q_idx, []) if final_q_idx is not None else []
        return self._extract_evidence_from_beams(final_beams)

    # ------------------------------------------------------------------
    # Dual retrieval: entity search + QA-pair search + reranking
    # ------------------------------------------------------------------

    def search_and_collect_evidence(
        self,
        question: str,
        top_k_entities: int = 10,
        top_k_qa: int = 15,
    ) -> List[Dict[str, Any]]:
        """Dual search (entity + QA-pair), merge, and rerank."""
        # 1. Entity-based search → extract QA pairs from entity contexts
        entity_results = self.gsw_tools.search_gsw_entity_embeddings(
            question, limit=top_k_entities
        )
        entity_qa_pairs: List[Dict[str, Any]] = []
        for entity in entity_results:
            entity_id = entity.get("global_id") or entity.get("entity_id", "")
            ctx = self.gsw_tools.get_entity_context(entity_id)
            if "error" in ctx:
                continue
            for q_info in ctx.get("questions", []):
                other_names = [
                    e["entity_name"] for e in q_info.get("other_entities", [])
                ]
                other_ids = [
                    e.get("entity_id", "") for e in q_info.get("other_entities", [])
                ]
                # Include the entity itself as an answer
                all_names = [ctx["entity_name"]] + other_names
                all_ids = [ctx["entity_id"]] + other_ids
                entity_qa_pairs.append(
                    {
                        "question": q_info["question_text"],
                        "answer_names": all_names,
                        "answer_ids": all_ids,
                        "answer_rolestates": [],
                        "verb_phrase": q_info.get("verb_phrase", ""),
                        "source_file": ctx.get("source_file", ""),
                        "entity_score": entity.get("match_score", 0.0),
                        "source_method": "entity_search",
                    }
                )

        # 2. Direct QA-pair search
        direct_qa_pairs = self.gsw_tools.search_qa_pairs(question, limit=top_k_qa)

        # 3. Merge and deduplicate
        all_qa: List[Dict[str, Any]] = []
        seen_qa: set = set()

        for qa in entity_qa_pairs:
            key = (qa.get("question", ""), tuple(qa.get("answer_names", [])))
            if key not in seen_qa:
                all_qa.append(qa)
                seen_qa.add(key)

        for qa in direct_qa_pairs:
            key = (qa.get("question", ""), tuple(qa.get("answer_names", [])))
            if key not in seen_qa:
                all_qa.append(qa)
                seen_qa.add(key)

        if not all_qa:
            return []

        # 4. Rerank via VoyageAI
        if VOYAGEAI_AVAILABLE and all_qa:
            try:
                if self._voyage_client is None:
                    self._voyage_client = voyageai.Client()

                qa_texts = []
                for qa in all_qa:
                    names = qa.get("answer_names", qa.get("answers", []))
                    rolestates = qa.get("answer_rolestates", [])
                    qa_text = (
                        f"{qa['question']} {', '.join(str(n) for n in names)} "
                        f"{', '.join(str(r) for r in rolestates)}"
                    )
                    qa_texts.append(qa_text)

                reranking = self._voyage_client.rerank(
                    question, qa_texts, model="rerank-2.5", top_k=len(qa_texts)
                )
                for r in reranking.results:
                    all_qa[r.index]["similarity_score"] = r.relevance_score

                all_qa.sort(key=lambda x: x.get("similarity_score", 0.0), reverse=True)
            except Exception as e:
                logger.warning("VoyageAI reranking failed: %s", e)

        return all_qa[:top_k_qa]

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _compute_cumulative_score(self, state: Dict[str, Any]) -> float:
        eps = 1e-6
        pairs = state.get("evidence_pairs", [])
        if not pairs:
            return eps

        scores = []
        for qa in pairs:
            score = qa.get("similarity_score", qa.get("entity_score", 0.0))
            if score is None:
                score = -1
            try:
                score = float(score)
            except Exception:
                score = -1
            norm = max(eps, min(0.5 * (score + 1), 1.0))
            scores.append(norm)

        return float(np.exp(np.mean(np.log(scores))))

    def _compute_similarity_score(
        self, state: Dict[str, Any], orig_emb: Optional[np.ndarray]
    ) -> float:
        eps = 1e-6
        parts = []
        for qa in state.get("evidence_pairs", []):
            q_text = qa.get("question", "")
            ans = qa.get("answer_names", qa.get("answers", []))
            if isinstance(ans, str):
                ans = [ans]
            a_text = ", ".join(str(x) for x in ans if x)
            rolestates = ", ".join(qa.get("answer_rolestates", []))
            if q_text and a_text:
                parts.append(f"Q: {q_text} A: {a_text}, {rolestates}")

        chain_text = " | ".join(parts) if parts else ""
        if orig_emb is None or not chain_text:
            return eps

        emb = self.gsw_tools.embed_query(chain_text)
        if emb is None:
            return eps

        sim = float(
            np.dot(orig_emb, emb)
            / (np.linalg.norm(orig_emb) * np.linalg.norm(emb) + eps)
        )
        return max(eps, min(0.5 * (sim + 1), 1.0))

    def _compute_combined_score(
        self, state: Dict[str, Any], orig_emb: Optional[np.ndarray]
    ) -> float:
        cum = self._compute_cumulative_score(state)
        sim = self._compute_similarity_score(state, orig_emb)
        return self.alpha * cum + (1 - self.alpha) * sim

    def _score_chain_state(
        self, state: Dict[str, Any], orig_emb: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        if self.scoring_mode == "cumulative":
            state["chain_score"] = self._compute_cumulative_score(state)
        elif self.scoring_mode == "similarity":
            state["chain_score"] = self._compute_similarity_score(state, orig_emb)
        elif self.scoring_mode == "combined":
            state["chain_score"] = self._compute_combined_score(state, orig_emb)
        elif self.scoring_mode == "none":
            state["chain_score"] = 1.0
        else:
            raise ValueError(f"Invalid scoring mode: {self.scoring_mode}")
        return state

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_expansion_state(
        base_state: Dict[str, Any],
        qa_used: Dict[str, Any],
        q_idx: int,
        is_last_hop: bool = False,
    ) -> Dict[str, Any]:
        new_state = {
            "entities_by_qidx": dict(base_state.get("entities_by_qidx", {})),
            "evidence_pairs": list(base_state.get("evidence_pairs", [])),
            "score": 0.0,
        }
        answer_names = qa_used.get("answer_names", qa_used.get("answers", []))
        if isinstance(answer_names, str):
            answer_names = [answer_names]
        if answer_names:
            new_state["entities_by_qidx"][q_idx] = (
                answer_names if is_last_hop else answer_names[0]
            )
        new_state["evidence_pairs"].append(qa_used)
        new_state["last_hop_score"] = float(qa_used.get("similarity_score", 0.0))
        return new_state

    @staticmethod
    def _prune_to_beam_width(
        candidates: List[Dict[str, Any]], beam_width: int
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        if any("chain_score" in s and s["chain_score"] is not None for s in candidates):
            candidates.sort(
                key=lambda s: (
                    s.get("chain_score", -1.0),
                    s.get("last_hop_score", 0.0),
                ),
                reverse=True,
            )
        else:
            candidates.sort(key=lambda s: s.get("last_hop_score", 0.0), reverse=True)
        return candidates[:beam_width]

    @staticmethod
    def _substitute_from_state(template: str, entities: Dict[int, str]) -> str:
        out = template
        for ref in re.findall(r"<ENTITY_Q(\d+)>", template):
            ref_idx = int(ref) - 1
            if ref_idx in entities:
                val = entities[ref_idx]
                if isinstance(val, list):
                    val = val[0] if val else ""
                out = out.replace(f"<ENTITY_Q{ref}>", val)
        if "<ENTITY>" in out and entities:
            last_idx = sorted(entities.keys())[-1]
            val = entities[last_idx]
            if isinstance(val, list):
                val = val[0] if val else ""
            out = out.replace("<ENTITY>", val)
        return out

    @staticmethod
    def _harmonic_mean(scores: List[float]) -> float:
        valid = [s for s in scores if s > 1e-6]
        if not valid:
            return 1e-6
        return float(len(valid) / sum(1.0 / s for s in valid))

    def _combine_multi_dep_states(
        self,
        deps: List[int],
        completed: Dict[int, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        dep_states = [completed.get(dep, [])[: self.beam_width] for dep in deps]
        # Filter empty
        if any(not ds for ds in dep_states):
            return []

        combos = []
        for combo in product(*dep_states):
            parent_scores = [s.get("chain_score", 0.5) for s in combo]
            h_mean = self._harmonic_mean(parent_scores)
            combos.append({"harmonic_score": h_mean, "states": combo})

        combos.sort(key=lambda x: x["harmonic_score"], reverse=True)
        top = [
            c
            for c in combos[: self.beam_width]
            if c["harmonic_score"] >= self.multi_dep_quality_threshold
        ]
        if not top and combos:
            top = combos[:1]

        prior_states: List[Dict[str, Any]] = []
        for combo in top:
            merged = {
                "entities_by_qidx": {},
                "evidence_pairs": [],
            }
            for parent in combo["states"]:
                merged["entities_by_qidx"].update(parent["entities_by_qidx"])
                merged["evidence_pairs"].extend(parent["evidence_pairs"])
            prior_states.append(merged)

        return prior_states

    @staticmethod
    def _extract_evidence_from_beams(beams: List[Dict[str, Any]]) -> List[str]:
        evidence: List[str] = []
        seen: set = set()
        for state in beams:
            for qa in state.get("evidence_pairs", []):
                q_text = qa.get("question", "")
                answer_names = qa.get("answer_names", qa.get("answers", []))
                if isinstance(answer_names, str):
                    answer_names = [answer_names]
                answer_text = ", ".join(str(name) for name in answer_names if name)
                answer_rs = ", ".join(qa.get("answer_rolestates", []))
                if q_text and answer_text:
                    fmt = f"Q: {q_text} A: {answer_text}"
                    if answer_rs:
                        fmt += f" {answer_rs}"
                    if fmt not in seen:
                        seen.add(fmt)
                        evidence.append(fmt)
        return evidence

    def _fallback_retrieval(self, question: str) -> List[str]:
        """Simple single-hop retrieval fallback when no chains are found."""
        qa_pairs = self.search_and_collect_evidence(
            question, top_k_entities=self.entity_top_k, top_k_qa=self.qa_rerank_top_k
        )
        evidence: List[str] = []
        for qa in qa_pairs:
            q_text = qa.get("question", "")
            answer_names = qa.get("answer_names", qa.get("answers", []))
            if isinstance(answer_names, str):
                answer_names = [answer_names]
            answer_text = ", ".join(str(n) for n in answer_names if n)
            answer_rs = ", ".join(qa.get("answer_rolestates", []))
            if q_text and answer_text:
                evidence.append(f"Q: {q_text} A: {answer_text} {answer_rs}")
        return list(set(evidence))
