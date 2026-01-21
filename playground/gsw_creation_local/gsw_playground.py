#!/usr/bin/env python3
"""
GSW Playground Script - Convert notebook to runnable Python script
Generates GSWs from Musique corpus and evaluates them using LLM-as-a-judge.
"""

from openai import OpenAI
import json
import os
import random
import sys
import glob
import re
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from tqdm import tqdm
from pathlib import Path
from bespokelabs import curator

# os.environ["CURATOR_DISABLE_CACHE"] = "1"

# Display Curator cache configuration
curator_cache_dir = os.environ.get('CURATOR_CACHE_DIR')
if not curator_cache_dir:
    # Curator default: ~/.cache/curator
    curator_cache_dir = str(Path.home() / '.cache' / 'curator')

cache_disabled = os.environ.get('CURATOR_DISABLE_CACHE', '0')

print(f"{'='*60}")
print(f"CURATOR CACHE CONFIGURATION")
print(f"{'='*60}")
print(f"Cache Directory: {curator_cache_dir}")
print(f"Cache Disabled: {cache_disabled == '1'}")
print(f"Cache Exists: {Path(curator_cache_dir).exists()}")
if Path(curator_cache_dir).exists():
    try:
        num_cache_entries = len(list(Path(curator_cache_dir).glob('*')))
        cache_size = sum(f.stat().st_size for f in Path(curator_cache_dir).rglob('*') if f.is_file()) / (1024**3)
        print(f"Number of Cache Entries: {num_cache_entries}")
        print(f"Total Cache Size: {cache_size:.2f} GB")
    except Exception as e:
        print(f"Could not calculate cache stats: {e}")
print(f"{'='*60}\n")

# Fix sys.path and import paths
sys.path.append("/home/yigit/codebase/gsw-memory/src/gsw_memory/")
from gsw_memory.memory.models import GSWStructure
from gsw_memory.memory.operator_utils import parse_gsw, GSWOperator
from gsw_memory.prompts.operator_prompts import PromptType, FactualExtractionPrompts
os.environ["OPENAI_API_KEY"] = "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"

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
    subject: str = Field(description="Subject entity")
    predicate: str = Field(description="Predicate/verb phrase")
    object: str = Field(description="Object entity")


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


def parse_thinking_trace(response_text: str) -> Tuple[str, str]:
    """Parse thinking trace and content from model response.

    Args:
        response_text: Full model response potentially containing <think> tags

    Returns:
        Tuple of (thinking_trace, content_without_thinking)
    """
    # Match <think>...</think> tags (case insensitive, multiline)
    thinking_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL | re.IGNORECASE)

    if thinking_match:
        thinking_trace = thinking_match.group(1).strip()
        # Remove the thinking tags from the content
        content = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL | re.IGNORECASE).strip()
    else:
        thinking_trace = ""
        content = response_text.strip()

    return thinking_trace, content


def load_musique_corpus(musique_path: str, num_samples: int = 100, is_train: bool = False):
    """Load and sample Musique corpus data.

    Args:
        musique_path: Path to musique JSON or JSONL file
        num_samples: Number of samples to load
        is_train: Whether this is training data (JSONL) or test data (JSON)

    Returns:
        List of document dictionaries
    """
    print(f"Loading Musique data from {musique_path}...")
    print(f"  Source: {'Training set' if is_train else 'Test set'}")

    # Load data based on format
    if is_train:
        # Load JSONL format for training data
        musique_data = []
        with open(musique_path) as f:
            for line in f:
                musique_data.append(json.loads(line))
    else:
        # Load JSON format for test data
        with open(musique_path) as f:
            musique_data = json.load(f)

    # Build corpus dictionaries
    corpus = {}
    for data in musique_data:
        paragraphs = data["paragraphs"]
        for doc_idx, paragraph in enumerate(paragraphs):
            global_id = f"{data['id']}_{paragraph['idx']}"
            corpus[global_id] = {
                "global_id": global_id,
                "title": paragraph["title"],
                "text": paragraph["title"] + "\n" + paragraph["paragraph_text"],
                "id": data["id"],
                "idx": paragraph["idx"]
            }

    # Sample corpus
    corpus_sample = list(corpus.values())[:num_samples]

    print(f"Loaded {len(corpus_sample)} samples")
    return corpus_sample[:500]


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


def generate_pred_gsws(corpus_sample: List[Dict], vllm_base_url: str = "http://127.0.0.1:6380/v1"):
    """Generate predicted GSWs using GSWOperator.

    Returns:
        List of dicts with format: {'global_id': str, 'gsw': GSWStructure, 'thinking_trace': str}
    """
    print("Generating predicted GSWs with GSWOperator...")

    os.environ["HOSTED_VLLM_API_KEY"] = "token-abc123"

    gsw_model = GSWOperator(
        model_name="hosted_vllm/Qwen/Qwen3-14B",
        backend_params={
            "base_url": vllm_base_url,
            "request_timeout": 600.0,
            "max_concurrent_requests": 64,
            "max_requests_per_minute": 120,
            "max_tokens_per_minute": 200000,
            "seconds_to_pause_on_rate_limit": 5,
            "require_all_responses": False,
        },
        generation_params={
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0,
                "max_tokens": 4096 * 3,
                # "repetition_penalty": 1.1,
                # "presence_penalty": 0.3,
                # "frequency_penalty": 0.3,
                            },
        prompt_type=PromptType.FACTUAL,
        backend="litellm",
        response_format=GSWStructure,
        batch=False,
    )

    # Display GSWOperator cache information
    print(f"\n{'='*60}")
    print(f"GSWOperator Curator Instance Info:")
    if hasattr(gsw_model, '_cache_dir'):
        print(f"  Cache Directory: {gsw_model._cache_dir}")
    elif hasattr(gsw_model, 'cache_dir'):
        print(f"  Cache Directory: {gsw_model.cache_dir}")
    else:
        print(f"  Cache Directory: {Path.home() / '.cache' / 'curator'} (default)")
    print(f"  Model: {gsw_model.model_name if hasattr(gsw_model, 'model_name') else 'Unknown'}")
    print(f"  Backend: {gsw_model.backend if hasattr(gsw_model, 'backend') else 'Unknown'}")
    print(f"{'='*60}\n")

    # Capture cache directory before and after call to detect which is used
    curator_cache_base = Path(curator_cache_dir)
    # Only track directories, not files like metadata.db
    before_cache_dirs = set(p for p in curator_cache_base.glob('*') if p.is_dir()) if curator_cache_base.exists() else set()

    gsw_response = gsw_model(corpus_sample)

    # Display the actual cache directory used for this run
    print(f"\n{'='*60}")
    print(f"ACTIVE CURATOR CACHE FOR THIS RUN:")
    print(f"{'='*60}")

    # Capture cache directory for return
    detected_cache_dir = None

    # Method 1: Try to get from response object
    if hasattr(gsw_response, 'cache_dir') and gsw_response.cache_dir:
        actual_cache_dir = gsw_response.cache_dir
        detected_cache_dir = str(actual_cache_dir)
        cache_hash = os.path.basename(actual_cache_dir)
        print(f"Full Path: {actual_cache_dir}")
        print(f"Cache Hash: {cache_hash}")
        print(f"Exists: {Path(actual_cache_dir).exists()}")
        if Path(actual_cache_dir).exists():
            cache_files = list(Path(actual_cache_dir).glob('*'))
            print(f"Files: {[f.name for f in cache_files[:10]]}")  # Show first 10 files
    else:
        # Method 2: Detect from filesystem changes (only directories)
        after_cache_dirs = set(p for p in curator_cache_base.glob('*') if p.is_dir()) if curator_cache_base.exists() else set()
        new_cache_dirs = after_cache_dirs - before_cache_dirs

        if new_cache_dirs:
            newest_cache = max(new_cache_dirs, key=lambda p: p.stat().st_mtime)
            detected_cache_dir = str(newest_cache)
            print(f"NEW Cache Created: {newest_cache}")
            print(f"Cache Hash: {newest_cache.name}")
        else:
            # Using existing cache - show most recent directory only
            all_cache_dirs = [p for p in curator_cache_base.glob('*') if p.is_dir()]
            all_cache_dirs = sorted(all_cache_dirs,
                                   key=lambda p: p.stat().st_mtime,
                                   reverse=True)
            if all_cache_dirs:
                most_recent = all_cache_dirs[0]
                detected_cache_dir = str(most_recent)
                from datetime import datetime
                print(f"EXISTING Cache Used: {most_recent}")
                print(f"Cache Hash: {most_recent.name}")
                print(f"Last Modified: {datetime.fromtimestamp(most_recent.stat().st_mtime)}")
            else:
                print("⚠️  Could not determine cache directory")

    print(f"{'='*60}\n")

    # Parse responses and extract thinking traces
    pred_results = []
    for response in gsw_response.dataset:
        try:
            # Extract GSW
            if response["gsw"]:
                gsw_dict = response["gsw"]
                gsw = GSWStructure(**gsw_dict)
            else:
                gsw = parse_gsw(response["graph"])

            global_id = response["global_id"]

            # Extract thinking trace from reasoning_content field
            thinking_trace = response.get("reasoning_content", "")
            raw_text = response.get("text", "")

            pred_results.append({
                "global_id": global_id,
                "gsw": gsw,
                "raw_text": raw_text,
                "thinking_trace": thinking_trace
            })
        except Exception as e:
            print(f"Error parsing GSW for chunk {response.get('global_id', 'unknown')}: {e}")
            continue

    print(f"Generated {len(pred_results)} predicted GSWs")
    return pred_results, detected_cache_dir


def generate_pred_gsws_with_openai(corpus_sample: List[Dict], vllm_base_url: str = "http://127.0.0.1:6380/v1"):
    """Generate predicted GSWs using OpenAI client directly (no Curator).

    This approach uses the OpenAI client directly to maintain access to reasoning_content,
    which contains the model's thinking traces. This is slower than Curator but guarantees
    access to thinking traces.

    Args:
        corpus_sample: List of documents to process
        vllm_base_url: Base URL for vLLM server

    Returns:
        List of dicts with format: {'global_id': str, 'gsw': GSWStructure, 'thinking_trace': str}
    """
    print("Generating predicted GSWs with OpenAI client (sequential, no batching)...")

    # Initialize OpenAI client
    client = OpenAI(base_url=vllm_base_url, api_key="token-abc123")

    # Get prompts
    SYSTEM_PROMPT = FactualExtractionPrompts.SYSTEM_PROMPT
    USER_PROMPT_TEMPLATE = FactualExtractionPrompts.USER_PROMPT_TEMPLATE

    # Process each document sequentially
    pred_results = []

    for doc in tqdm(corpus_sample, desc="Generating GSWs"):
        try:
            # Format prompt
            user_prompt = USER_PROMPT_TEMPLATE.format(
                input_text=doc["text"],
                background_context=doc.get("context", "")
            )

            # Call OpenAI API with structured output
            completion = client.chat.completions.parse(
                model="Qwen/Qwen3-14B",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,
                response_format=GSWStructure,
                extra_body={
                    "top_p": 0.95,
                    "top_k": 20,
                    "min_p": 0,
                    "max_tokens": 4096 * 3,
                    # "repetition_penalty": 1.1,
                    # "presence_penalty": 0.3,
                    # "frequency_penalty": 0.3,
                }
            )

            # Extract GSW and reasoning_content
            gsw = completion.choices[0].message.parsed  # Already a GSWStructure object
            reasoning_content = completion.choices[0].message.reasoning_content or ""

            pred_results.append({
                "global_id": doc["global_id"],
                "gsw": gsw,
                "thinking_trace": reasoning_content
            })

        except Exception as e:
            print(f"\nError processing {doc.get('global_id', 'unknown')}: {e}")
            continue

    print(f"Generated {len(pred_results)} predicted GSWs")
    return pred_results, None  # No cache directory for OpenAI client approach


def evaluate_gsws(corpus_sample: List[Dict], pred_gsws: List[GSWStructure],
                  golden_gsws: List[GSWStructure], openai_api_key: str = None):
    """Evaluate predicted GSWs against golden GSWs using LLM-as-a-judge with Curator."""
    print("Evaluating GSWs with LLM-as-a-judge (Curator + GPT-4.1-mini)...")

    # Prepare input dataset for Curator
    num_samples = min(len(corpus_sample), len(pred_gsws), len(golden_gsws))
    judge_inputs = []

    for i in range(num_samples):
        judge_inputs.append({
            "sample_id": i,
            "global_id": corpus_sample[i].get("global_id", f"sample_{i}"),
            "text": corpus_sample[i]["text"],
            "gold_gsw_json": golden_gsws[i].model_dump_json(indent=2),
            "pred_gsw_json": pred_gsws[i].model_dump_json(indent=2)
        })

    # Create judge with Curator
    judge = GSWJudge(
        model_name="gpt-4.1-mini",
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

    return all_judgements


def save_pred_gsws(pred_results: List[Dict], output_path: str,
                   save_format: str = "consolidated", metadata: Dict = None):
    """Save predicted GSWs with thinking traces to file(s).

    Args:
        pred_results: List of dicts with format {'global_id': str, 'gsw': GSWStructure, 'thinking_trace': str}
        output_path: Base output path
        save_format: One of "individual", "consolidated", or "both"
        metadata: Optional metadata to include (model, params, etc.)
    """
    print(f"\nSaving {len(pred_results)} predicted GSWs with thinking traces...")
    print(f"  Format: {save_format}")

    # Serialize results to dicts
    results_dicts = []
    for result in pred_results:
        results_dicts.append({
            "global_id": result["global_id"],
            "raw_text": result["raw_text"],
            "gsw": result["gsw"].model_dump(),
            "thinking_trace": result["thinking_trace"]
        })

    if save_format in ["consolidated", "both"]:
        # Save as single consolidated JSON file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data_to_save = {
            "metadata": metadata or {},
            "gsws": results_dicts
        }

        with open(output_file, 'w') as f:
            json.dump(data_to_save, f, indent=2)

        print(f"  ✓ Saved consolidated file: {output_file}")

    if save_format in ["individual", "both"]:
        # Save as individual files (like golden GSWs)
        base_dir = Path(output_path).parent / "pred_gsws_individual"
        base_dir.mkdir(parents=True, exist_ok=True)

        for idx, result_dict in enumerate(results_dicts):
            doc_dir = base_dir / f"doc_{idx}"
            doc_dir.mkdir(exist_ok=True)

            # Save GSW and thinking trace together
            gsw_file = doc_dir / "gsw.json"
            with open(gsw_file, 'w') as f:
                json.dump(result_dict, f, indent=2)

        print(f"  ✓ Saved individual files: {base_dir}/")

    print(f"  Total: {len(pred_results)} GSWs saved")


def load_pred_gsws(input_path: str) -> List[GSWStructure]:
    """Load previously saved predicted GSWs.

    Args:
        input_path: Path to consolidated JSON file or directory of individual files

    Returns:
        List of GSWStructure objects (for backward compatibility, thinking traces are not returned)
    """
    print(f"\nLoading predicted GSWs from {input_path}...")
    input_path = Path(input_path)

    pred_gsws = []

    if input_path.is_file():
        # Load from consolidated JSON file
        with open(input_path) as f:
            data = json.load(f)

        # Handle different formats
        if isinstance(data, list):
            # Old format: list of GSW dicts
            gsws_dicts = data
        elif isinstance(data, dict) and "gsws" in data:
            # New format with metadata and potentially thinking traces
            gsws_list = data["gsws"]
            if "metadata" in data:
                print(f"  Metadata: {json.dumps(data['metadata'], indent=2)}")

            # Extract GSWs from new format (may have thinking_trace and global_id)
            gsws_dicts = []
            for item in gsws_list:
                if isinstance(item, dict) and "gsw" in item:
                    # New format: {'global_id': ..., 'gsw': {...}, 'thinking_trace': ...}
                    gsws_dicts.append(item["gsw"])
                else:
                    # Old format: just the GSW dict
                    gsws_dicts.append(item)
        else:
            raise ValueError(f"Invalid file format: {input_path}")

        # Convert dicts to GSWStructure objects
        for gsw_dict in gsws_dicts:
            try:
                pred_gsws.append(GSWStructure(**gsw_dict))
            except Exception as e:
                print(f"  Warning: Failed to load GSW: {e}")

    elif input_path.is_dir():
        # Load from individual files
        doc_dirs = sorted(input_path.glob("doc_*"), key=lambda x: int(x.name.split("_")[1]))

        for doc_dir in doc_dirs:
            gsw_file = doc_dir / "gsw.json"
            if gsw_file.exists():
                try:
                    with open(gsw_file) as f:
                        data = json.load(f)

                    # Handle new format with thinking traces
                    if isinstance(data, dict) and "gsw" in data:
                        gsw_dict = data["gsw"]
                    else:
                        gsw_dict = data

                    pred_gsws.append(GSWStructure(**gsw_dict))
                except Exception as e:
                    print(f"  Warning: Failed to load {gsw_file}: {e}")

    else:
        raise FileNotFoundError(f"Input path not found: {input_path}")

    print(f"  Loaded {len(pred_gsws)} predicted GSWs")
    return pred_gsws


def extract_thinking_traces_from_cache(cache_dir: str, output_file: str = None, corpus_sample: List[Dict] = None):
    """Extract all thinking traces from Curator cache responses file.

    Args:
        cache_dir: Path to cache directory (e.g., ~/.cache/curator/<hash>)
        output_file: Optional path to save extracted thinking traces as JSON
        corpus_sample: Optional list of corpus samples to include raw input text

    Returns:
        List of dicts with format: {'index': int, 'raw_text': str, 'thinking_trace': str, 'gsw': dict, 'token_usage': dict}
    """
    cache_path = Path(cache_dir)
    responses_file = cache_path / "responses_0.jsonl"

    if not responses_file.exists():
        print(f"\n❌ No responses file found at: {responses_file}")
        print(f"   This cache may have been reused (hit) with no new responses saved.")
        return []

    thinking_traces = []

    print(f"\nReading responses from: {responses_file}")
    with open(responses_file, 'r') as f:
        for idx, line in enumerate(f):
            try:
                response = json.loads(line)

                # Extract reasoning_content from raw_response
                raw_response = response.get('raw_response', {})
                choices = raw_response.get('choices', [])
                if choices:
                    message = choices[0].get('message', {})
                    reasoning_content = message.get('reasoning_content', '')

                    # Also get the parsed GSW
                    response_message = response.get('response_message', {})

                    # Get raw input text if corpus_sample is provided
                    raw_text = ""
                    if corpus_sample and idx < len(corpus_sample):
                        raw_text = corpus_sample[idx].get('text', '')

                    thinking_traces.append({
                        'index': idx,
                        'raw_text': raw_text,
                        'thinking_trace': reasoning_content,
                        'gsw': response_message,
                        'token_usage': response.get('token_usage', {})
                    })

                    if reasoning_content:
                        print(f"  ✓ Entry {idx}: Found thinking trace ({len(reasoning_content)} chars)")
                    else:
                        print(f"  ⚠ Entry {idx}: No thinking trace")

            except json.JSONDecodeError as e:
                print(f"  ❌ Entry {idx}: Failed to parse JSON: {e}")
            except Exception as e:
                print(f"  ❌ Entry {idx}: Error: {e}")

    print(f"\nExtracted {len(thinking_traces)} entries")
    print(f"  With thinking traces: {sum(1 for t in thinking_traces if t['thinking_trace'])}")
    print(f"  Without thinking traces: {sum(1 for t in thinking_traces if not t['thinking_trace'])}")

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                'cache_dir': str(cache_dir),
                'num_entries': len(thinking_traces),
                'entries': thinking_traces
            }, f, indent=2)

        print(f"\n✓ Saved to: {output_path}")

    return thinking_traces


def save_judgements(judgements: List[Dict], output_path: str):
    """Save judgements to JSON file."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(judgements, f, indent=2)

    print(f"\nSaved judgements to {output_file}")


def calculate_statistics(judgements: List[Dict]):
    """Calculate and print statistics from judgements."""
    scores = [j["overall_score"] for j in judgements if j["overall_score"] is not None]

    if not scores:
        print("\nNo valid scores found!")
        return

    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)

    usable_count = sum(1 for j in judgements
                      if j.get("judgement") and j["judgement"].get("usable_for_QA", False))
    total_count = len([j for j in judgements if "error" not in j])

    print("\n" + "="*60)
    print("EVALUATION STATISTICS")
    print("="*60)
    print(f"Total samples evaluated: {len(judgements)}")
    print(f"Successful evaluations: {len(scores)}")
    print(f"Failed evaluations: {len(judgements) - len(scores)}")
    print(f"\nAverage overall score: {avg_score:.2f}")
    print(f"Min score: {min_score:.2f}")
    print(f"Max score: {max_score:.2f}")
    print(f"\nUsable for QA: {usable_count}/{total_count} ({usable_count/total_count*100:.1f}%)")
    print("="*60)


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="GSW Playground - Generate and evaluate GSWs")

    # Data source arguments
    parser.add_argument("--num-samples", type=int, default=100,
                       help="Number of samples to process (default: 100)")
    parser.add_argument("--use-train-set", action="store_true",
                       help="Use training set instead of test set")
    parser.add_argument("--musique-json", type=str,
                       default="/home/yigit/codebase/gsw-memory/playground_data/musique.json",
                       help="Path to musique.json (test set)")
    parser.add_argument("--train-jsonl", type=str,
                       default="/home/yigit/codebase/gsw-memory/playground_data/musique_full_v1.0_train.jsonl",
                       help="Path to musique train JSONL")
    parser.add_argument("--golden-gsw-dir", type=str,
                       default="/mnt/SSD1/shreyas/SM_GSW/musique/networks_4_1_mini",
                       help="Directory containing golden GSW networks")

    # Generation arguments
    parser.add_argument("--vllm-url", type=str,
                       default="http://127.0.0.1:6380/v1",
                       help="VLLM base URL")
    parser.add_argument("--skip-generation", action="store_true",
                       help="Skip GSW generation and load from file")
    parser.add_argument("--load-pred-gsws", type=str,
                       default=None,
                       help="Path to pre-saved predicted GSWs (file or directory)")
    parser.add_argument("--use-openai-client", action="store_true",
                       help="Use OpenAI client directly instead of Curator (slower but guarantees thinking trace capture)")
    parser.add_argument("--extract-thinking-traces", action="store_true",
                       help="Extract thinking traces from Curator cache after generation (only applies when using Curator)")

    # Output arguments
    parser.add_argument("--output", type=str,
                       default=None,
                       help="Output path for judgements JSON (default: judgements_output_TIMESTAMP.json)")
    parser.add_argument("--pred-gsw-output", type=str,
                       default=None,
                       help="Output path for predicted GSWs (default: pred_gsws_TIMESTAMP.json)")
    parser.add_argument("--save-format", type=str,
                       choices=["individual", "consolidated", "both"],
                       default="consolidated",
                       help="Format for saving predicted GSWs (default: consolidated)")

    # Evaluation arguments
    parser.add_argument("--skip-evaluation", action="store_true",
                       help="Skip LLM-as-a-judge evaluation")

    args = parser.parse_args()

    # Set default output paths with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source = "train" if args.use_train_set else "test"

    if args.output is None:
        args.output = f"playground/gsw_creation_local/judgements_output_{timestamp}.json"
    if args.pred_gsw_output is None:
        args.pred_gsw_output = f"playground/gsw_creation_local/pred_gsws_{source}_{timestamp}.json"

    # Determine musique data path
    musique_path = args.train_jsonl if args.use_train_set else args.musique_json

    print("="*60)
    print("GSW PLAYGROUND - EVALUATION PIPELINE")
    print("="*60)
    print(f"Data source: {'Training set' if args.use_train_set else 'Test set'}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Pred GSW output: {args.pred_gsw_output}")
    print(f"Save format: {args.save_format}")
    if not args.skip_evaluation:
        print(f"Judgements output: {args.output}")
    print("="*60 + "\n")

    # Load corpus data
    corpus_sample = load_musique_corpus(
        musique_path,
        args.num_samples,
        is_train=args.use_train_set
    )

    # Load golden GSWs
    golden_gsws = load_golden_gsws(
        args.golden_gsw_dir,
        args.num_samples
    )

    # Generate or load predicted GSWs
    if args.load_pred_gsws:
        # Load pre-saved predicted GSWs
        pred_gsws = load_pred_gsws(args.load_pred_gsws)

        # Ensure we have the right number of samples
        if len(pred_gsws) < args.num_samples:
            print(f"Warning: Loaded {len(pred_gsws)} GSWs, but requested {args.num_samples}")
            print(f"  Adjusting num_samples to {len(pred_gsws)}")
            args.num_samples = len(pred_gsws)
            corpus_sample = corpus_sample[:args.num_samples]
            golden_gsws = golden_gsws[:args.num_samples]
        elif len(pred_gsws) > args.num_samples:
            pred_gsws = pred_gsws[:args.num_samples]

    elif not args.skip_generation:
        # Choose generation method based on flag
        if args.use_openai_client:
            print("\n*** Using OpenAI client (direct, sequential, guaranteed thinking traces) ***")
            pred_results, cache_dir_used = generate_pred_gsws_with_openai(corpus_sample, args.vllm_url)
            generation_method = "openai_client"
        else:
            print("\n*** Using Curator (batched, parallel, faster) ***")
            pred_results, cache_dir_used = generate_pred_gsws(corpus_sample, args.vllm_url)
            generation_method = "curator"

        # Save predicted GSWs with thinking traces
        metadata = {
            "model": "Qwen/Qwen3-14B",
            "generation_method": generation_method,
            "timestamp": timestamp,
            "num_samples": len(pred_results),
            "source": source,
            "vllm_url": args.vllm_url,
            "generation_params": {
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0,
                "max_tokens": 4096 * 3,
                # "repetition_penalty": 1.1,
                # "presence_penalty": 0.3,
                # "frequency_penalty": 0.3,
            }
        }
        save_pred_gsws(pred_results, args.pred_gsw_output, args.save_format, metadata)

        # Extract thinking traces from cache if requested (only for Curator method)
        if args.extract_thinking_traces and generation_method == "curator" and cache_dir_used:
            print(f"\n{'='*60}")
            print("EXTRACTING THINKING TRACES FROM CACHE")
            print(f"{'='*60}")

            # Construct output filename based on pred_gsw_output
            if args.pred_gsw_output:
                output_path = Path(args.pred_gsw_output)
                thinking_traces_output = str(output_path.parent / f"{output_path.stem}_thinking_traces.json")
            else:
                thinking_traces_output = f"thinking_traces_{timestamp}.json"

            # Extract and save thinking traces (including raw input text)
            thinking_traces = extract_thinking_traces_from_cache(cache_dir_used, thinking_traces_output, corpus_sample)

            if thinking_traces:
                print(f"\n✓ Extracted {len(thinking_traces)} thinking traces")
                print(f"✓ Saved to: {thinking_traces_output}")
            else:
                print(f"\n⚠ No thinking traces found in cache")
        elif args.extract_thinking_traces and generation_method == "openai_client":
            print("\n⚠ --extract-thinking-traces is not needed with --use-openai-client")
            print("  Thinking traces are already included in the output when using OpenAI client")

        # Extract GSWs for evaluation
        pred_gsws = [result["gsw"] for result in pred_results]

    else:
        print("ERROR: --skip-generation requires --load-pred-gsws")
        print("  Please specify a path to load predicted GSWs from")
        return

    # Evaluate GSWs (if not skipped)
    if not args.skip_evaluation:
        judgements = evaluate_gsws(corpus_sample, pred_gsws, golden_gsws)
        save_judgements(judgements, args.output)
        calculate_statistics(judgements)
    else:
        print("\nSkipping evaluation (--skip-evaluation flag set)")

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
