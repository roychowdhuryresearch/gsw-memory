"""
GSW Coverage Repair Operator.

Second-pass operator that detects dangling entities (entities not referenced
as an answer in any question) and asks the LLM to generate additional
verb phrases + questions to cover them.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set, Tuple

from bespokelabs import curator

from ...prompts.operator_prompts import PromptType
from ..models import GSWStructure, VerbPhraseNode
from .gsw_operator import get_prompt_class


def find_dangling_entities(gsw: GSWStructure) -> List[Dict[str, Any]]:
    """Return entity dicts that are never referenced as an answer in any question."""
    referenced: Set[str] = set()
    for vp in gsw.verb_phrase_nodes:
        for q in vp.questions:
            referenced.update(q.answers)

    dangling = []
    for entity in gsw.entity_nodes:
        if entity.id not in referenced:
            dangling.append({
                "id": entity.id,
                "name": entity.name,
                "roles": [r.model_dump() for r in entity.roles],
            })
    return dangling


COVERAGE_REPAIR_SYSTEM_PROMPT = """\
You are an expert at generating factual QA pairs for a semantic knowledge graph.

You will receive:
1. The original source text
2. A list of entities that currently have NO question pointing to them as an answer
3. The existing verb phrases and questions (so you don't duplicate)

Your job: generate additional verb_phrase_nodes so that every dangling entity \
appears as an answer (by entity ID) in at least one question.

Rules:
- Each verb phrase must have exactly 2 bidirectional questions (A→B and B→A)
- Answers must be entity IDs (e.g. "e5"), NOT text
- Only use facts stated in the source text — do not fabricate
- Do not duplicate existing verb phrases / questions
- Use the most specific relationship from the text
- VP ids should continue from the last existing VP id (e.g. if last is v32, start from v33)
- Question ids should continue from the last existing question id
- You may reference any existing entity by ID in your questions (not just dangling ones)

Output ONLY a JSON array of verb_phrase_node objects. No markdown, no explanation."""


COVERAGE_REPAIR_USER_TEMPLATE = """\
<source_text>
{source_text}
</source_text>

<dangling_entities>
{dangling_entities_json}
</dangling_entities>

<existing_verb_phrases>
{existing_vps_json}
</existing_verb_phrases>

<existing_entity_map>
{entity_map_json}
</existing_entity_map>

Generate verb_phrase_nodes to cover all dangling entities. Output a JSON array:
```json
[
  {{
    "id": "v{next_vp_id}",
    "phrase": "<relationship>",
    "questions": [
      {{"id": "q{next_q_id}", "text": "<A→B question>", "answers": ["<entity_id>"]}},
      {{"id": "q{next_q_id_plus_1}", "text": "<B→A question>", "answers": ["<entity_id>"]}}
    ]
  }}
]
```"""


def _build_repair_messages(
    source_text: str,
    gsw: GSWStructure,
    dangling: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Build chat messages for the coverage repair call."""
    entity_map = {e.id: e.name for e in gsw.entity_nodes}

    existing_vps = []
    max_vp_num = 0
    max_q_num = 0
    for vp in gsw.verb_phrase_nodes:
        vp_num = int(vp.id.lstrip("v")) if vp.id.startswith("v") else 0
        max_vp_num = max(max_vp_num, vp_num)
        vp_summary: Dict[str, Any] = {"id": vp.id, "phrase": vp.phrase, "questions": []}
        for q in vp.questions:
            q_num = int(q.id.lstrip("q")) if q.id.startswith("q") else 0
            max_q_num = max(max_q_num, q_num)
            vp_summary["questions"].append({
                "id": q.id,
                "text": q.text,
                "answers": q.answers,
            })
        existing_vps.append(vp_summary)

    user_content = COVERAGE_REPAIR_USER_TEMPLATE.format(
        source_text=source_text,
        dangling_entities_json=json.dumps(dangling, indent=2),
        existing_vps_json=json.dumps(existing_vps, indent=2),
        entity_map_json=json.dumps(entity_map, indent=2),
        next_vp_id=max_vp_num + 1,
        next_q_id=max_q_num + 1,
        next_q_id_plus_1=max_q_num + 2,
    )

    return [
        {"role": "system", "content": COVERAGE_REPAIR_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_repair_vps(raw_text: str) -> List[Dict[str, Any]]:
    """Parse the LLM response into a list of VP dicts."""
    text = raw_text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # skip ```json
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    parsed = json.loads(text)
    if isinstance(parsed, dict) and "verb_phrase_nodes" in parsed:
        parsed = parsed["verb_phrase_nodes"]
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON array, got {type(parsed)}")
    return parsed


def merge_repair_vps(gsw: GSWStructure, new_vps: List[Dict[str, Any]]) -> GSWStructure:
    """Merge repair VPs into an existing GSWStructure, returning a new copy."""
    merged_vps = list(gsw.verb_phrase_nodes)
    for vp_dict in new_vps:
        vp = VerbPhraseNode(**vp_dict)
        merged_vps.append(vp)
    return GSWStructure(
        entity_nodes=list(gsw.entity_nodes),
        verb_phrase_nodes=merged_vps,
    )


class CoverageRepairOperator(curator.LLM):
    """Curator operator that generates additional VPs for dangling entities."""

    return_completions_object = True

    def prompt(self, input_row):
        return input_row["messages"]

    def parse(self, input_row, response):
        content = response["choices"][0]["message"]["content"]
        return {
            "idx": input_row["idx"],
            "doc_idx": input_row["doc_idx"],
            "global_id": input_row["global_id"],
            "repair_content": content,
        }


def repair_dangling_entities(
    all_documents_data: List[Dict[str, Dict[str, Any]]],
    model_name: str,
    backend: str = "openai",
    backend_params: Optional[Dict[str, Any]] = None,
    generation_params: Optional[Dict[str, Any]] = None,
    min_dangling: int = 1,
) -> Tuple[int, int]:
    """Run coverage repair on all chunks that have dangling entities.

    Modifies all_documents_data in-place (updates gsw fields).

    Args:
        all_documents_data: The pipeline's document data (list of doc dicts).
        model_name: Model to use for repair.
        backend: Curator backend ("openai", "litellm", etc.)
        backend_params: Backend params for Curator.
        generation_params: Generation params (temperature, etc.)
        min_dangling: Minimum dangling entities to trigger repair for a chunk.

    Returns:
        (chunks_repaired, entities_covered): counts of what was fixed.
    """
    if generation_params is None:
        generation_params = {"temperature": 0.0}
    if backend_params is None:
        backend_params = {"require_all_responses": False}

    # Collect chunks that need repair
    repair_inputs = []
    chunk_refs = []  # (doc_idx, global_id, gsw, dangling)

    for doc_idx, doc_chunks in enumerate(all_documents_data):
        for global_id, chunk_data in doc_chunks.items():
            gsw = chunk_data.get("gsw")
            if gsw is None:
                continue
            if not isinstance(gsw, GSWStructure):
                gsw = GSWStructure(**gsw) if isinstance(gsw, dict) else gsw

            dangling = find_dangling_entities(gsw)
            if len(dangling) < min_dangling:
                continue

            messages = _build_repair_messages(
                source_text=chunk_data.get("text", ""),
                gsw=gsw,
                dangling=dangling,
            )
            repair_inputs.append({
                "idx": len(repair_inputs),
                "doc_idx": doc_idx,
                "global_id": global_id,
                "messages": messages,
            })
            chunk_refs.append((doc_idx, global_id, gsw, dangling))

    if not repair_inputs:
        return 0, 0

    print(f"--- Coverage Repair: {len(repair_inputs)} chunks with dangling entities ---")

    operator = CoverageRepairOperator(
        model_name=model_name,
        backend=backend,
        backend_params=backend_params,
        generation_params=generation_params,
    )

    dataset = operator(repair_inputs)
    rows = [dict(row) for row in (dataset.dataset if hasattr(dataset, "dataset") else dataset)]

    rows_by_idx = {row["idx"]: row for row in rows}

    chunks_repaired = 0
    entities_covered = 0

    for i, (doc_idx, global_id, gsw, dangling) in enumerate(chunk_refs):
        row = rows_by_idx.get(i)
        if row is None:
            print(f"  {global_id}: no repair response")
            continue

        try:
            new_vps = _parse_repair_vps(row["repair_content"])
            merged = merge_repair_vps(gsw, new_vps)

            # Count how many previously-dangling entities are now covered
            still_dangling = find_dangling_entities(merged)
            covered = len(dangling) - len(still_dangling)

            all_documents_data[doc_idx][global_id]["gsw"] = merged
            chunks_repaired += 1
            entities_covered += covered
            print(f"  {global_id}: +{len(new_vps)} VPs, covered {covered}/{len(dangling)} dangling entities")
            if still_dangling:
                names = [d["name"] for d in still_dangling[:5]]
                print(f"    still dangling: {names}")
        except Exception as e:
            print(f"  {global_id}: repair parse failed: {e}")

    print(f"--- Coverage Repair complete: {chunks_repaired} chunks, {entities_covered} entities covered ---")
    return chunks_repaired, entities_covered
