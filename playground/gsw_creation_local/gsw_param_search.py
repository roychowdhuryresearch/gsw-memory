#!/usr/bin/env python3
"""
GSW Parameter Search Script
Searches over generation parameters (temperature, top_p, top_k, min_p) and
evaluates each combination using LLM-as-a-judge to find optimal settings.

Usage:
    # Run with default parameter grid
    uv run python playground/gsw_creation_local/gsw_param_search.py

    # Run with custom parameters
    uv run python playground/gsw_creation_local/gsw_param_search.py \
        --temperatures 0.1 0.3 0.5 \
        --top-ps 0.9 0.95 \
        --num-samples 50
"""

from openai import OpenAI
import json
import os
import random
import sys
import glob
import re
import argparse
import itertools
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from tqdm import tqdm
from pathlib import Path
from bespokelabs import curator

# Fix sys.path and import paths
sys.path.append("/mnt/SSD3/chenda/gsw/gsw-memory")
from gsw_memory.memory.models import GSWStructure
from gsw_memory.memory.operator_utils import parse_gsw, GSWOperator
from gsw_memory.prompts.operator_prompts import PromptType, FactualExtractionPrompts
os.environ["OPENAI_API_KEY"] = "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"
os.environ["CURATOR_CACHE_DIR"] = "/mnt/SSD3/chenda/gsw/cache/curator"
# ============================================================================
# Pydantic Models for LLM-as-a-Judge
# ============================================================================

class Subscores(BaseModel):
    """Subscores for coverage, precision, and format compliance."""
    coverage: float = Field(description="Coverage score (0-1)")
    precision: float = Field(description="Precision score (0-1)")
    format_compliance: float = Field(description="Format compliance score (0-1)")


class Counts(BaseModel):
    """Counts of gold and predicted entities, facts, and violations."""
    gold_entities: int = Field(description="Number of gold entities")
    pred_entities: int = Field(description="Number of predicted entities")
    gold_facts: int = Field(description="Number of gold facts")
    pred_facts: int = Field(description="Number of predicted facts")
    covered_gold_facts: int = Field(description="Number of covered gold facts")
    hallucinated_pred_facts: int = Field(description="Number of hallucinated predicted facts")
    unscorable_pred_relations: int = Field(description="Number of unscorable predicted relations")


class CriticalViolation(BaseModel):
    """A critical violation in the GSW."""
    type: str = Field(description="Type of violation")
    message: str = Field(description="Short explanation")
    location: str = Field(description="Entity ID, verb phrase ID, or question ID")


class EntityAlignment(BaseModel):
    """Alignment between gold and predicted entities."""
    gold_entity_id: Optional[str] = Field(description="Gold entity ID", default=None)
    pred_entity_id: Optional[str] = Field(description="Predicted entity ID or null", default=None)
    match_type: str = Field(description="Match type: exact, normalized, alias, or none")
    notes: str = Field(description="Brief notes")


class MissingEntity(BaseModel):
    """An entity missing from the predicted GSW."""
    gold_entity_id: str = Field(description="Gold entity ID")
    gold_name: str = Field(description="Gold entity name")
    reason: str = Field(description="Reason: missing, role_mismatch, or bundled")


class ExtraEntity(BaseModel):
    """An extra entity in the predicted GSW."""
    pred_entity_id: str = Field(description="Predicted entity ID")
    pred_name: str = Field(description="Predicted entity name")
    reason: str = Field(description="Reason: hallucinated, unnecessary, or bundled")


class FactDetail(BaseModel):
    """Details of a fact (subject, predicate, object triple)."""
    subject: Optional[str] = Field(description="Subject entity", default=None)
    predicate: Optional[str] = Field(description="Predicate/verb phrase", default=None)
    object: Optional[str] = Field(description="Object entity", default=None)


class MissingFact(BaseModel):
    """A fact missing from the predicted GSW."""
    gold_fact: FactDetail = Field(description="Gold fact details")
    gold_verb_phrase_id: str = Field(description="Gold verb phrase ID")
    reason: str = Field(description="Reason: missing, temporal_missing, wrong_object, or wrong_subject")


class CoveredFact(BaseModel):
    """A fact covered by both gold and predicted GSW."""
    gold_fact: FactDetail = Field(description="Gold fact details")
    pred_fact: FactDetail = Field(description="Predicted fact details")
    gold_verb_phrase_id: str = Field(description="Gold verb phrase ID")
    pred_verb_phrase_id: str = Field(description="Predicted verb phrase ID")
    match_notes: str = Field(description="Brief match notes")


class HallucinatedFact(BaseModel):
    """A hallucinated fact in the predicted GSW."""
    pred_fact: FactDetail = Field(description="Predicted fact details")
    pred_verb_phrase_id: str = Field(description="Predicted verb phrase ID")
    reason: str = Field(description="Reason: not_in_gold, not_entailed_by_text, or over_specific")


class FactComparison(BaseModel):
    """Comparison of facts between gold and predicted GSW."""
    missing_facts: List[MissingFact] = Field(description="Facts missing from predicted GSW", default_factory=list)
    covered_facts: List[CoveredFact] = Field(description="Facts covered by both GSWs", default_factory=list)
    hallucinated_facts: List[HallucinatedFact] = Field(description="Hallucinated facts in predicted GSW", default_factory=list)


class FormatIssue(BaseModel):
    """A format issue in the predicted GSW."""
    type: str = Field(description="Type of format issue")
    message: str = Field(description="Short explanation")
    location: str = Field(description="Verb phrase ID, question ID, or entity ID")


class Judge_Format(BaseModel):
    """LLM-as-a-judge evaluation format for GSW comparisons."""
    overall_score: float = Field(description="Overall score between 0 and 100")
    usable_for_QA: bool = Field(description="True if the GSW is usable for QA, False otherwise")
    subscores: Subscores = Field(description="Subscores for coverage, precision, and format compliance")
    counts: Counts = Field(description="Counts of gold and predicted entities, facts, and violations")
    critical_violations: List[CriticalViolation] = Field(description="List of critical violations", default_factory=list)
    entity_alignment: List[EntityAlignment] = Field(description="List of entity alignments", default_factory=list)
    missing_entities: List[MissingEntity] = Field(description="List of missing entities", default_factory=list)
    extra_entities: List[ExtraEntity] = Field(description="List of extra entities", default_factory=list)
    fact_comparison: FactComparison = Field(description="Comparison of facts between gold and predicted GSW")
    format_issues: List[FormatIssue] = Field(description="List of format issues", default_factory=list)
    improvement_suggestions: List[str] = Field(description="List of improvement suggestions", default_factory=list)

    model_config = {"extra": "forbid"}


# ============================================================================
# LLM-as-a-Judge Prompt
# ============================================================================

LLM_AS_A_JUDGE_PROMPT = """SYSTEM:
You are a strict evaluator (LLM-as-a-judge) for GSW factual extraction graphs used in multi-hop QA.
You will compare a PREDICTED GSW against a GOLD GSW for the SAME input text.

Your job:
1) Determine how well the predicted GSW matches the gold GSW in factual coverage and structural compliance.
2) Identify missing facts, hallucinated facts, malformed entities, malformed relations, and question-format violations.
3) Output ONLY valid JSON in the schema given below. No extra commentary.

Definitions:
- A "fact" is a (subject entity, verb phrase, object entity) triple expressed by a verb_phrase_node whose questions imply that relation.
- "Coverage" means the predicted GSW contains an equivalent fact to one in the gold GSW.
- "Equivalent" means same underlying meaning, allowing:
  - minor wording differences in the verb phrase/questions,
  - re-ordering,
  - synonyms that do not change meaning (e.g., "joined" vs "enrolled" if clearly same in context),
  - but NOT allowing changed entities, times, locations, roles, or added details not in text.
- IDs may differ across graphs; match by entity name + type/role + context.

Hard constraints from the extraction spec (judge these strictly):
A) No fabrication: predicted facts must be entailed by the input text.
B) Atomic entities: do not bundle multiple separable entities into one.
C) Abbreviation/alias: if text has "Full Name (ABBR)", predicted should create:
   - entity "Full Name (ABBR)" and entity "ABBR" (alias) connected via "also known as".
   - Do NOT expand abbreviations that were not expanded in text.
D) Two questions per verb phrase node: exactly 2 questions per verb_phrase_node.
E) Each question must have exactly one unknown (answer) and no pronouns ("he", "it", "they", etc.).
F) Questions must contain complete content (no dropping "that ..." clauses when required).
G) Temporal connectivity: when gold connects a date/time via a phrase like "on/in/during", predicted should also connect that temporal info somewhere relevant.
H) Answers must be entity IDs only and must exist in entity_nodes.
I) Do not merge multiple subjects/objects into one question unless gold does (and the rule permits it).

Evaluation procedure:
1) Parse both GSWs into:
   - entity inventory: (name, role(s), states)
   - fact inventory: for each verb_phrase_node, infer intended (S, P, O) from the questions:
     * If one question is "Who <P> <O> ...?" and the answer is X => (X, P, O)
     * If one question is "What/Which <O> did <S> <P> ...?" and the answer is Y => (S, P, Y)
   If inference is ambiguous, record it as "unscorable_relation" and penalize format/clarity, not factual mismatch.
2) Align entities between gold and pred by best match on:
   - exact name match > normalized match (case/diacritics) > alias match
   - role compatibility (person vs org vs date etc.)
3) Align facts:
   - A gold fact is "covered" if an equivalent pred fact exists.
   - A pred fact is "hallucinated" if it is not entailed by input text OR has no corresponding gold fact meaningfully.
4) Score:
   - Coverage (0–1): covered_gold_facts / total_gold_facts
   - Precision (0–1): correct_pred_facts / total_pred_facts (exclude unscorable if clearly malformed)
   - Format compliance (0–1): start from 1 and subtract for each violation category (see below).
   - Overall (0–100): weighted:
       overall = 45*coverage + 35*precision + 20*format_compliance
   Provide also pass/fail for "usable_for_QA" with threshold overall>=80 and no critical violations.
Critical violations (any => usable_for_QA=false even if score high):
   - fabrication (A)
   - pronouns in questions (E)
   - answers not IDs / missing IDs (H)
   - not exactly two questions per verb phrase node (D)

Format compliance penalties (examples):
- two_questions_violation: -0.10 each verb phrase node violating
- pronoun_violation: -0.15 each occurrence
- missing_that_content: -0.10 each occurrence
- entity_bundling: -0.10 each bundled entity
- alias_rule_violation: -0.10 each missed/incorrect alias case
- temporal_disconnection: -0.05 each missing required time link
- bad_answer_ids: critical (set compliance to 0 for that node and mark critical)

Output JSON schema (MUST follow exactly):
{{
  "overall_score": 0-100 number,
  "usable_for_QA": boolean,
  "subscores": {{
    "coverage": 0-1 number,
    "precision": 0-1 number,
    "format_compliance": 0-1 number
  }},
  "counts": {{
    "gold_entities": int,
    "pred_entities": int,
    "gold_facts": int,
    "pred_facts": int,
    "covered_gold_facts": int,
    "hallucinated_pred_facts": int,
    "unscorable_pred_relations": int
  }},
  "critical_violations": [
    {{
      "type": "fabrication|pronouns|bad_answer_ids|two_questions_violation",
      "message": "short explanation",
      "location": "entity_id or verb_phrase_id or question_id"
    }}
  ],
  "entity_alignment": [
    {{
      "gold_entity_id": "e#",
      "pred_entity_id": "e# or null",
      "match_type": "exact|normalized|alias|none",
      "notes": "brief"
    }}
  ],
  "missing_entities": [
    {{ "gold_entity_id": "e#", "gold_name": "...", "reason": "missing|role_mismatch|bundled" }}
  ],
  "extra_entities": [
    {{ "pred_entity_id": "e#", "pred_name": "...", "reason": "hallucinated|unnecessary|bundled" }}
  ],
  "fact_comparison": {{
    "missing_facts": [
      {{
        "gold_fact": {{ "subject": "...", "predicate": "...", "object": "..."}},
        "gold_verb_phrase_id": "v#",
        "reason": "missing|temporal_missing|wrong_object|wrong_subject"
      }}
    ],
    "covered_facts": [
      {{
        "gold_fact": {{ "subject": "...", "predicate": "...", "object": "..."}},
        "pred_fact": {{ "subject": "...", "predicate": "...", "object": "..."}},
        "gold_verb_phrase_id": "v#",
        "pred_verb_phrase_id": "v#",
        "match_notes": "brief"
      }}
    ],
    "hallucinated_facts": [
      {{
        "pred_fact": {{ "subject": "...", "predicate": "...", "object": "..."}},
        "pred_verb_phrase_id": "v#",
        "reason": "not_in_gold|not_entailed_by_text|over_specific"
      }}
    ]
  }},
  "format_issues": [
    {{
      "type": "two_questions_violation|pronoun_violation|missing_that_content|alias_rule_violation|entity_bundling|temporal_disconnection|question_unknown_count|other",
      "message": "short explanation",
      "location": "v# or q# or e#"
    }}
  ],
  "improvement_suggestions": [
    "bullet-like string suggestions, concrete and minimal"
  ]
}}

USER (template you will receive):
<input_text>
{input_text}
</input_text>

<gold_gsw_json>
{gold_gsw_json}
</gold_gsw_json>

<pred_gsw_json>
{pred_gsw_json}
</pred_gsw_json>

Now perform the evaluation and output ONLY the JSON object.
"""


# ============================================================================
# Curator Judge Class
# ============================================================================

class GSWJudge(curator.LLM):
    """Curator-based LLM judge for evaluating GSW structures."""

    response_format = Judge_Format

    def prompt(self, input_data):
        """Create a prompt for the judge to evaluate GSWs."""
        user_content = f"""<input_text>
{input_data['text']}
</input_text>

<gold_gsw_json>
{input_data['gold_gsw_json']}
</gold_gsw_json>

<pred_gsw_json>
{input_data['pred_gsw_json']}
</pred_gsw_json>

Now perform the evaluation and output ONLY the JSON object."""

        return [
            {"role": "system", "content": LLM_AS_A_JUDGE_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def parse(self, input_data, response):
        """Parse the judge response."""
        # Response from Curator is a JSON string of the Judge_Format Pydantic object
        judgement_data = response.model_dump() if response else None
        return {
            "sample_id": input_data.get("sample_id", 0),
            "global_id": input_data.get("global_id", "unknown"),
            "text": input_data["text"],
            "judgement": judgement_data,
            "overall_score": judgement_data.get("overall_score", 0.0) if isinstance(judgement_data, dict) else 0.0
          }


# ============================================================================
# Helper Functions
# ============================================================================

def sort_natural_key(s):
    """Extract the numeric part from the path string for natural sorting."""
    match = re.search(r'doc_(\d+)', s)
    return int(match.group(1)) if match else s


def load_musique_corpus(musique_json_path: str, num_samples: int = 100):
    """Load and sample Musique corpus data."""
    print(f"Loading Musique data from {musique_json_path}...")

    # Load test data
    with open(musique_json_path) as f:
        test_musique = json.load(f)

    # Build corpus dictionaries
    test_musique_corpus = {}
    for data in test_musique:
        paragraphs = data["paragraphs"]
        for doc_idx, paragraph in enumerate(paragraphs):
            test_musique_corpus[str(data["id"]) + "_" + str(paragraph["idx"])] = {
                "global_id": f"{data['id']}_{paragraph['idx']}",
                "title": paragraph["title"],
                "text": paragraph["title"] + "\n" + paragraph["paragraph_text"],
                "id": data["id"],
                "idx": paragraph["idx"]
            }

    # Sample corpus
    test_musique_corpus_sample = list(test_musique_corpus.values())[:num_samples]

    print(f"Loaded {len(test_musique_corpus_sample)} test samples")
    return test_musique_corpus_sample


def load_golden_gsws(musique_network_dir: str, num_docs: int = 100):
    """Load golden GSW structures from pre-processed Musique data."""
    print(f"Loading golden GSWs from {musique_network_dir}...")

    musique_docs = sorted(
        glob.glob(f"{musique_network_dir}/doc_*"),
        key=sort_natural_key
    )
    musique_docs_sample = musique_docs[:num_docs]

    golden_gsws = []
    for doc_dir in tqdm(musique_docs_sample, desc="Loading golden GSWs"):
        if os.path.isdir(doc_dir):
            json_files = sorted(glob.glob(os.path.join(doc_dir, "*.json")))
            for json_file in json_files:
                try:
                    with open(json_file, 'r') as f:
                        doc_data = json.load(f)
                        doc_data = GSWStructure(**doc_data)
                        golden_gsws.append(doc_data)
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
        else:
            print(f"Warning: {doc_dir} is not a directory, skipping.")

    print(f"Loaded {len(golden_gsws)} golden GSWs")
    return golden_gsws


def generate_pred_gsws(
    corpus_sample: List[Dict],
    vllm_base_url: str = "http://127.0.0.1:6379/v1",
    generation_params: Dict = None
):
    """Generate predicted GSWs using GSWOperator with specified generation params."""
    if generation_params is None:
        generation_params = {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "max_tokens": 4096 * 3, 
            "repetition_penalty": 1.1,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3
        }

    print(f"Generating predicted GSWs with params: {generation_params}")

    os.environ["HOSTED_VLLM_API_KEY"] = "token-abc123"

    gsw_model = GSWOperator(
        model_name="hosted_vllm/Qwen/Qwen3-8B",
        backend_params={
            "base_url": vllm_base_url,
            "request_timeout": 600.0,
            "max_concurrent_requests": 32,
            "max_requests_per_minute": 120,
            "max_tokens_per_minute": 200000,
            "seconds_to_pause_on_rate_limit": 5,
            "require_all_responses": False,
        },
        generation_params=generation_params,
        prompt_type=PromptType.FACTUAL,
        backend="litellm",
        response_format=GSWStructure,
        batch=False,
    )

    gsw_response = gsw_model(corpus_sample)

    # Parse responses
    all_documents_data = {}
    for response in gsw_response.dataset:
        try:
            if response["gsw"]:
                gsw_dict = response["gsw"]
                gsw = GSWStructure(**gsw_dict)
            else:
                gsw = parse_gsw(response["graph"])
            global_id = response["global_id"]
            all_documents_data[global_id] = gsw
        except Exception as e:
            print(f"Error parsing GSW for chunk {response.get('global_id', 'unknown')}: {e}")
            continue

    print(f"Generated {len(all_documents_data)} predicted GSWs")
    return all_documents_data  # Return dict keyed by global_id for proper matching


def evaluate_gsws(corpus_sample: List[Dict], pred_gsws: List[Optional[GSWStructure]],
                  golden_gsws: List[GSWStructure], openai_api_key: str = None):
    """Evaluate predicted GSWs against golden GSWs using LLM-as-a-judge with Curator."""
    print("Evaluating GSWs with LLM-as-a-judge (Curator + GPT-4.1-mini)...")

    # Prepare input dataset for Curator
    num_samples = min(len(corpus_sample), len(pred_gsws), len(golden_gsws))
    judge_inputs = []
    skipped_samples = []

    for i in range(num_samples):
        # Skip samples where prediction failed (None)
        if pred_gsws[i] is None:
            skipped_samples.append({
                "sample_id": i,
                "global_id": corpus_sample[i].get("global_id", f"sample_{i}"),
                "error": "prediction_failed",
                "overall_score": None
            })
            continue

        judge_inputs.append({
            "sample_id": i,
            "global_id": corpus_sample[i].get("global_id", f"sample_{i}"),
            "text": corpus_sample[i]["text"],
            "gold_gsw_json": golden_gsws[i].model_dump_json(indent=2),
            "pred_gsw_json": pred_gsws[i].model_dump_json(indent=2)
        })

    if skipped_samples:
        print(f"Skipped {len(skipped_samples)} samples due to failed predictions")

    # If no valid predictions, return only skipped samples
    if not judge_inputs:
        print("No valid predictions to evaluate!")
        return skipped_samples

    # Create judge with Curator
    judge = GSWJudge(
        model_name="gpt-5-mini",
        response_format=Judge_Format,
        batch=False,
    )

    # Run batch evaluation
    print(f"Processing {len(judge_inputs)} samples with Curator...")
    judge_response = judge(judge_inputs)

    # Extract judgements from response
    all_judgements = []
    for response in judge_response.dataset:
        try:
            # Response already has the parsed format from GSWJudge.parse()
            all_judgements.append({
                "sample_id": response.get("sample_id", 0),
                "global_id": response.get("global_id", "unknown"),
                "judgement": response.get("judgement", {}),
                "overall_score": response.get("overall_score", None)
            })
        except Exception as e:
            print(f"\nError processing result for sample {response.get('sample_id', '?')}: {e}")
            all_judgements.append({
                "sample_id": response.get("sample_id", 0),
                "global_id": response.get("global_id", "unknown"),
                "error": str(e),
                "overall_score": None
            })

    # Include skipped samples in the results
    all_judgements.extend(skipped_samples)

    # Sort by sample_id to maintain original order
    all_judgements.sort(key=lambda x: x.get("sample_id", 0))

    return all_judgements


def save_judgements(judgements: List[Dict], output_path: str):
    """Save judgements to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(judgements, f, indent=2)

    print(f"\nSaved judgements to {output_file}")


def save_pred_gsws(
    corpus_sample: List[Dict],
    pred_gsws_dict: Dict[str, GSWStructure],
    golden_gsws: List[GSWStructure],
    output_path: str
):
    """Save predicted GSWs with idx matching golden GSWs for easy comparison."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results = []
    for idx, (sample, gold_gsw) in enumerate(zip(corpus_sample, golden_gsws)):
        global_id = sample.get("global_id", f"sample_{idx}")
        pred_gsw = pred_gsws_dict.get(global_id)

        results.append({
            "idx": idx,
            "global_id": global_id,
            "text": sample.get("text", ""),
            "gold_gsw": gold_gsw.model_dump() if gold_gsw else None,
            "pred_gsw": pred_gsw.model_dump() if pred_gsw else None
        })

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved predicted GSWs to {output_file}")


def calculate_statistics(judgements: List[Dict], print_stats: bool = True) -> Dict:
    """Calculate statistics from judgements and optionally print them."""
    scores = [j["overall_score"] for j in judgements if j["overall_score"] is not None]

    if not scores:
        if print_stats:
            print("\nNo valid scores found!")
        return {"avg_score": 0, "min_score": 0, "max_score": 0, "usable_rate": 0}

    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)

    usable_count = sum(1 for j in judgements
                      if j.get("judgement") and j["judgement"].get("usable_for_QA", False))
    total_count = len([j for j in judgements if "error" not in j])
    usable_rate = usable_count / total_count * 100 if total_count > 0 else 0

    stats = {
        "total_samples": len(judgements),
        "successful_evals": len(scores),
        "failed_evals": len(judgements) - len(scores),
        "avg_score": avg_score,
        "min_score": min_score,
        "max_score": max_score,
        "usable_count": usable_count,
        "total_count": total_count,
        "usable_rate": usable_rate
    }

    if print_stats:
        print("\n" + "="*60)
        print("EVALUATION STATISTICS")
        print("="*60)
        print(f"Total samples evaluated: {stats['total_samples']}")
        print(f"Successful evaluations: {stats['successful_evals']}")
        print(f"Failed evaluations: {stats['failed_evals']}")
        print(f"\nAverage overall score: {stats['avg_score']:.2f}")
        print(f"Min score: {stats['min_score']:.2f}")
        print(f"Max score: {stats['max_score']:.2f}")
        print(f"\nUsable for QA: {stats['usable_count']}/{stats['total_count']} ({stats['usable_rate']:.1f}%)")
        print("="*60)

    return stats


# ============================================================================
# Parameter Search Functions
# ============================================================================

def generate_param_grid(
    temperatures: List[float],
    top_ps: List[float],
    top_ks: List[int],
    min_ps: List[float]
) -> List[Dict]:
    """Generate all combinations of parameters."""
    param_combinations = []
    for temp, top_p, top_k, min_p in itertools.product(temperatures, top_ps, top_ks, min_ps):
        param_combinations.append({
            "temperature": temp,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": 1.1,
            "presence_penalty": 0.3,
            "frequency_penalty": 0.3,
            "max_tokens": 4096 * 3,
        })
    return param_combinations


def run_param_search(
    corpus_sample: List[Dict],
    golden_gsws: List[GSWStructure],
    param_combinations: List[Dict],
    vllm_url: str,
    output_dir: str
) -> List[Dict]:
    """Run parameter search over all combinations."""
    all_results = []

    print(f"\n{'='*60}")
    print(f"PARAMETER SEARCH: {len(param_combinations)} combinations")
    print(f"{'='*60}\n")

    for i, params in enumerate(param_combinations):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(param_combinations)}] Testing: {params}")
        print(f"{'='*60}")

        # Generate GSWs with current params (returns dict keyed by global_id)
        pred_gsws_dict = generate_pred_gsws(corpus_sample, vllm_url, params)

        # Convert dict to list matching corpus_sample order for evaluation
        pred_gsws_list = []
        for sample in corpus_sample:
            global_id = sample.get("global_id")
            pred_gsw = pred_gsws_dict.get(global_id)
            pred_gsws_list.append(pred_gsw)

        # Evaluate
        judgements = evaluate_gsws(
            corpus_sample, pred_gsws_list, golden_gsws,
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )

        # Calculate stats
        stats = calculate_statistics(judgements, print_stats=True)

        # Save individual results
        param_str = f"t{params['temperature']}_p{params['top_p']}_k{params['top_k']}_m{params['min_p']}"
        judgements_file = os.path.join(output_dir, f"judgements_{param_str}.json")
        pred_gsws_file = os.path.join(output_dir, f"pred_gsws_{param_str}.json")

        save_judgements(judgements, judgements_file)
        save_pred_gsws(corpus_sample, pred_gsws_dict, golden_gsws, pred_gsws_file)

        # Store result
        result = {
            "params": params,
            "stats": stats,
            "judgements_file": judgements_file,
            "pred_gsws_file": pred_gsws_file
        }
        all_results.append(result)

    return all_results


def print_search_summary(results: List[Dict]):
    """Print summary of parameter search results."""
    print("\n" + "="*80)
    print("PARAMETER SEARCH SUMMARY")
    print("="*80)

    # Sort by average score
    valid_results = [r for r in results if "error" not in r]
    sorted_results = sorted(valid_results, key=lambda x: x["stats"]["avg_score"], reverse=True)

    print(f"\n{'Rank':<6} {'Temp':<8} {'Top-P':<8} {'Top-K':<8} {'Min-P':<8} {'Avg Score':<12} {'Usable %':<10}")
    print("-" * 80)

    for i, result in enumerate(sorted_results):
        params = result["params"]
        stats = result["stats"]
        print(f"{i+1:<6} {params['temperature']:<8.2f} {params['top_p']:<8.2f} {params['top_k']:<8} {params['min_p']:<8.2f} {stats['avg_score']:<12.2f} {stats['usable_rate']:<10.1f}")

    if sorted_results:
        best = sorted_results[0]
        print("\n" + "="*80)
        print("BEST PARAMETERS:")
        print(f"  temperature: {best['params']['temperature']}")
        print(f"  top_p: {best['params']['top_p']}")
        print(f"  top_k: {best['params']['top_k']}")
        print(f"  min_p: {best['params']['min_p']}")
        print(f"  Average Score: {best['stats']['avg_score']:.2f}")
        print(f"  Usable for QA: {best['stats']['usable_rate']:.1f}%")
        print("="*80)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GSW Parameter Search - Find optimal generation parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default parameter grid
    uv run python playground/gsw_creation_local/gsw_param_search.py

    # Run with custom temperatures and top_p values
    uv run python playground/gsw_creation_local/gsw_param_search.py \\
        --temperatures 0.1 0.3 0.5 \\
        --top-ps 0.9 0.95

    # Quick test with fewer samples
    uv run python playground/gsw_creation_local/gsw_param_search.py \\
        --num-samples 20 --temperatures 0.3 0.6
        """
    )

    parser.add_argument("--num-samples", type=int, default=50,
                       help="Number of samples to process (default: 50)")
    parser.add_argument("--musique-json", type=str,
                       default="/mnt/SSD3/chenda/gsw/musique.json",
                       help="Path to musique.json")
    parser.add_argument("--golden-gsw-dir", type=str,
                       default="/mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini",
                       help="Directory containing golden GSW networks")
    parser.add_argument("--vllm-url", type=str,
                       default="http://127.0.0.1:6379/v1",
                       help="VLLM base URL")
    parser.add_argument("--output-dir", type=str,
                       default=None,
                       help="Output directory for results (default: param_search_TIMESTAMP/)")

    # Parameter grid arguments
    parser.add_argument("--temperatures", type=float, nargs="+",
                       default=[0.1, 0.3, 0.5, 0.7],
                       help="Temperature values to search (default: 0.1 0.3 0.5 0.7)")
    parser.add_argument("--top-ps", type=float, nargs="+",
                       default=[0.9, 0.95],
                       help="Top-p values to search (default: 0.9 0.95)")
    parser.add_argument("--top-ks", type=int, nargs="+",
                       default=[20],
                       help="Top-k values to search (default: 20)")
    parser.add_argument("--min-ps", type=float, nargs="+",
                       default=[0.0],
                       help="Min-p values to search (default: 0.0)")

    args = parser.parse_args()

    # Set default output directory with timestamp
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"playground/gsw_creation_local/param_search_{timestamp}"

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Generate parameter grid
    param_combinations = generate_param_grid(
        args.temperatures, args.top_ps, args.top_ks, args.min_ps
    )

    print("="*60)
    print("GSW PARAMETER SEARCH")
    print("="*60)
    print(f"Number of samples: {args.num_samples}")
    print(f"Output directory: {args.output_dir}")
    print(f"Parameter combinations: {len(param_combinations)}")
    print(f"  Temperatures: {args.temperatures}")
    print(f"  Top-p values: {args.top_ps}")
    print(f"  Top-k values: {args.top_ks}")
    print(f"  Min-p values: {args.min_ps}")
    print("="*60 + "\n")

    # Load data (once, reuse for all param combinations)
    corpus_sample = load_musique_corpus(args.musique_json, args.num_samples)
    golden_gsws = load_golden_gsws(args.golden_gsw_dir, args.num_samples)

    # Run parameter search
    results = run_param_search(
        corpus_sample, golden_gsws, param_combinations,
        args.vllm_url, args.output_dir
    )

    # Save overall results
    summary_file = os.path.join(args.output_dir, "search_summary.json")
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_samples": args.num_samples,
            "param_grid": {
                "temperatures": args.temperatures,
                "top_ps": args.top_ps,
                "top_ks": args.top_ks,
                "min_ps": args.min_ps
            },
            "results": results
        }, f, indent=2)
    print(f"\nSaved search summary to {summary_file}")

    # Print summary
    print_search_summary(results)


if __name__ == "__main__":
    main()
