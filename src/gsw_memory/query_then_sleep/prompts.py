from __future__ import annotations

from typing import Any, Dict, Iterable, List


QUERY_DECOMPOSITION_SYSTEM_PROMPT = """You decompose multi-hop questions into atomic retrieval steps.
Return strict JSON only with key 'sub_questions'.
Each item in 'sub_questions' must contain:
- sub_question: short atomic question
- relation_hint: short snake_case relation guess
- entity_hint: main entity to search first, if obvious
Prefer 2-4 sub-questions. Do not answer the original question here."""


QUERY_TOOL_AGENT_INSTRUCTIONS = """You answer questions over a document memory using 3 tools:
- search_entities(query): PRIMARY TOOL — find entities by name, returns their QA pairs (facts like nationality, birth date, spouse, etc.), roles, and states
- search_qa_pairs(query): search QA pairs across all documents — less precise, use only when you don't know the entity name
- get_bridge_evidence(query): search pre-computed multi-hop bridge evidence from prior analysis

## Decision flow (check these in order, stop at the first match)

**Step 0 — Direct bridge hit (answer in 0 additional calls):**
If a bridge question in the provided evidence matches your question (same entities, same relation, even with paraphrasing) AND the bridge answer is a concrete value (named entity, date, place, nationality, number) — **answer immediately, do NOT make any tool calls.**

Examples of direct bridge hits (answer with 0 extra calls):
- Q: "What is the date of death of Elizabeth Capell, Countess of Essex's father?"
  Bridge: "When did Elizabeth Capell, Countess of Essex's father die? → 26 May 1711"
  → Answer: "26 May 1711". STOP. Do not search.
- Q: "Who is the spouse of the director of film West of Shanghai?"
  Bridge: "Who was married to the director of West of Shanghai? → Maureen O'Sullivan"
  → Answer: "Maureen O'Sullivan". STOP.
- Q: "Who is the paternal grandfather of Anna Dorothea, Abbess of Quedlinburg?"
  Bridge: "Who is the paternal grandfather of Duchess Anna Dorothea of Saxe-Weimar? → William, Duke of Saxe-Weimar"
  → Answer: "William, Duke of Saxe-Weimar". STOP.

**Step 1 — Bridge gives the intermediate but not the answer (1-2 calls):**
If a bridge provides the first hop (e.g., names the intermediate entity) but not the final answer, use search_entities on the intermediate entity to retrieve the missing fact.

Example:
- Q: "What nationality is the director of film Daytime Wives?"
  Bridge: "Who directed Daytime Wives? → Émile Chautard" (gives intermediate, not nationality)
  Step 1: search_entities("Émile Chautard")
  → QA pairs include: "What is the nationality of Émile Chautard?" → "French-American"
  → Answer: "French-American" (1 call)

**Step 2 — No useful bridge, follow the chain with search_entities (2-4 calls):**
If bridges don't help, start from the question's root entity and chain through search_entities calls. Each entity result includes inline QA pairs — read them carefully.

Example — Question: "What nationality is the director of film Daytime Wives?"
Step 1: search_entities("Daytime Wives")
→ QA pairs include: "Who directed Daytime Wives?" → "Émile Chautard"
Step 2: search_entities("Émile Chautard")
→ QA pairs include: "What is the nationality of Émile Chautard?" → "French-American"
→ Answer: "French-American" (2 calls)

## Rules
- **Bridges are ground truth.** They were pre-computed from the same documents. If a bridge matches your question, use its answer without verification.
- **Do NOT search for verification.** If you already have the answer from a bridge or entity QA pairs, stop. No confirmatory searches.
- **search_entities is your primary fallback.** Use it when bridges don't cover the question.
- **Avoid search_qa_pairs** unless you truly don't know which entity to look for.
- **Never repeat the same query.** If a search didn't help, try a different query or stop."""


QUERY_FINAL_SYNTHESIS_SYSTEM_PROMPT = """You turn a question, a decomposition, and tool traces into a structured answer summary.
Return strict JSON only with keys:
- answer
- reasoning_chain
- sub_questions
- found_relations
- missing_relations
- bridge_ids_used

CRITICAL — the 'answer' field MUST be a short entity, name, place, date, or number ONLY.
Rules for 'answer':
- NEVER return a full sentence. NEVER include explanations, markdown, or references in the answer.
- For "Who is X?" → return just the name: "Johann of Limburg" (NOT "X's grandfather was Johann of Limburg")
- For "Where was X born?" → return just the place: "Paris" (NOT "X was born in Paris")
- For "What nationality?" → return just the nationality: "French" (NOT "The nationality is French")
- For boolean questions → return exactly "yes" or "no"
- If unsupported → return empty string ""
- Strip any doc references like 【doc_123::e4】 from the answer
- Do not restate the question in the answer.
- Do not prefix the answer with "Answer:", "The answer is", or similar filler.

Example:
Question: "Who is the maternal grandmother of Margaret of Valois?"
Good answer: "Madeleine de La Tour d'Auvergne"
Bad answer: "The maternal grandmother of Margaret of Valois was Madeleine de La Tour d'Auvergne【doc_123::e4】"

Example:
Question: "Who is the spouse of the director of film Pratisodh (2004 Film)?"
Good answer: "Piya Sengupta"
Bad answer: "The spouse of the director of Pratisodh (2004 Film) is Piya Sengupta."

Example:
Question: "Where did John Ii, Duke Of Cleves's wife die?"
Good answer: "Cologne"
Bad answer: "John II's wife Mathilde of Hesse died in Cologne【doc_4626::e5】"

Example:
Question: "Were Jimmy Santiago Baca and Duane Armstrong from the same country?"
Good answer: "yes"
Bad answer: "Yes, they were both from the United States."

Example:
Question: "Who is Eochu Uairches's paternal grandfather?"
Good answer: "Énna Derg"
Bad answer: "Énna Derg.【doc_4577::e2}"

Each item in sub_questions must include:
- sub_question
- answer_found
- answer
- searched_for
- found_instead
- notes"""


INTERACTION_ANALYSIS_SYSTEM_PROMPT = """You normalize relation demand from query traces.
Return strict JSON only with keys:
- relation_label
- relation_family
- evidence_type
Use short snake_case relation labels."""


INTERACTION_GUIDANCE_SUMMARY_SYSTEM_PROMPT = """You write detailed guidance for a sleep-time bridge-generation agent based on what went wrong in the previous query batch. Your guidance will be read by the LLM that generates bridges — it must be specific, actionable, and grounded in real failures.

Return strict JSON with key: natural_language_summary

The guidance MUST have five sections with concrete specifics. No abstract categories. No hallucinated bridge types. Every claim must be grounded in the inputs (failed_questions, prior_guidance_summary, supporting_examples).

You may also receive `prior_guidance_summary` — the guidance written after the previous batch. Use it to build cumulative knowledge across batches rather than starting fresh every time. Carry forward lessons that are still relevant, drop lessons that have been addressed, and add new lessons from this batch's failures.

## Section 1 — Failed Questions (name them specifically)
For each failed or partial question from `failed_questions` (up to 5), write 1-2 sentences that name:
- The question text
- What the query agent answered vs the gold answer
- Which bridge was retrieved (if any) and why it was wrong or missing
- The EXACT bridge question that SHOULD have been generated to answer it (phrased as a retrievable forward question with a named root entity)

Example:
"Question 'What is the date of death of Ferdinand, Prince of Solms-Braunfels's father?' failed (got '13 April 1814', gold '24 February 1761'). The registry had the correct fact but encoded as 'When did the person who died in Sławięcice die?' — unretrievable because the anchor is a place. Generate instead: 'When did the father of Ferdinand, Prince of Solms-Braunfels die?'"

## Section 2 — Coverage Gaps (name entities and docs)
List 3-5 specific entity-relation pairs that need new bridges. For each:
- Name the root entity by its proper name (from a failed question's packet)
- Name the relation chain
- Name the source docs from the failed question's packet (if available in failed_questions entries)

Example:
"- Anna Dorothea of Saxe-Weimar → paternal grandfather (docs doc_2454, doc_2456). Generate: 'Who is the paternal grandfather of Anna Dorothea of Saxe-Weimar?'
- Margaret of Baux → father-in-law (docs doc_4580, doc_4582). Generate: 'Who is the father-in-law of Margaret of Baux?'
- Neferneferure → mother's place of death (docs doc_3901, doc_3903). Generate: 'Where did the mother of Neferneferure die?'"

## Section 3 — Anchor Rule Violations (judge retrieved bridges)
For each failed question in `failed_questions` that has a `top_bridge_retrieved` entry, look at that retrieved bridge and decide whether it had anchor issues. A bridge has anchor issues when its root is a date, place, year, number, or an unnamed descriptive clause ("the person who...") instead of a proper named entity (person name, film title, organization). When you find offenders, cite them by bridge_id and give the corrected bridge form anchored on a named entity.

Reason directly from the retrieved bridges — do not invent bridges that aren't in the inputs.

Example:
"For question `2e294eec` (date of death of Ferdinand's father), the retrieved bridge was `bridge_ed0e1a97` 'On what date did the predecessor of Wilhelm Christian Karl, 3rd Prince of Solms Braunfels, die?' Its root is 'the predecessor of...' — an unnamed descriptive clause, not a named person — so it didn't match a user query starting from 'Ferdinand, Prince of Solms-Braunfels'. Corrected form: 'When did the father of Ferdinand, Prince of Solms-Braunfels die?'"

If none of the retrieved bridges have anchor issues, write "No anchor rule violations detected in retrieved bridges."

## Section 4 — Format Alerts
If any failed question had the right entity but a format mismatch (GSW stores 'French-American' but gold was 'French', or 'Apopka, Florida' vs 'Apopka', or 'Henry McLaren, 2nd Baron Aberconway' vs 'Henry McLaren'), flag it here with the specific examples so bridges include both forms as alternate answers.

Example:
"Question `c9238c7c` expected 'French' but the GSW entity is 'French-American'. When generating nationality bridges, include both the hyphenated GSW form and the country-level form as alternate answers. Question `899ecc9e` expected 'Henry McLaren' but the GSW entity is 'Henry McLaren, 2nd Baron Aberconway' — store both the full titled form and the bare name."

## Section 5 — Carry-forward Lessons (from prior_guidance_summary)
If `prior_guidance_summary` is non-empty, review it and decide what to carry forward into this batch's guidance:
- **Keep** persistent lessons that are still relevant — anchor-rule reminders, format-mismatch patterns, entity-name conventions that the sleep agent should remember across batches.
- **Drop** lessons that have been addressed (e.g. a gap from batch 2 that batch 3's sleep phase filled).
- **Merge** overlapping lessons from prior and current batches into a single cumulative statement.
- **Add** new lessons from this batch's failures.

Write this section as 2-4 short paragraphs that the sleep agent can treat as "running notes" — persistent reminders it should apply on top of the batch-specific guidance above.

Example (when prior guidance exists):
"Carry-forward reminders from prior batches: nationality bridges must include both the hyphenated GSW form (e.g. 'French-American') and the base form ('French') as alternate answers — this was flagged in the previous batch and remains unresolved. Place-of-birth bridges should include both the full form ('Apopka, Florida') and the bare city name ('Apopka'). The 'father-in-law' relation is still under-covered — prior batch's spouse-chain bridges did not propagate to in-law lookups, and batch 4 added two more failures in this family."

If `prior_guidance_summary` is empty (first batch), write "No prior guidance — this is the first batch."

## Hard requirements
- No abstract phrases like "parent chains saturated" or "grandparent bridges missing" unless you immediately follow with the exact question/entity/doc that exhibits the gap.
- No hallucinated bridge categories (producer→award, sibling bridges, performer-to-award-to-year, etc.) that don't appear in the current packet's failed_questions.
- Every suggestion must cite a specific failed question_id or bridge_id from the inputs.
- If a section has no grounded data (e.g. no failed_questions), write "No data for this section." — do not hallucinate."""


def render_interaction_guidance_block(guidance: Dict[str, Any] | None) -> str:
    if not isinstance(guidance, dict) or not guidance:
        return ""

    interaction_guidance = guidance.get("interaction_guidance")
    if isinstance(interaction_guidance, dict):
        payload = interaction_guidance
    else:
        payload = guidance

    lines: List[str] = ["Interaction guidance (advisory only):"]

    summary = str(payload.get("natural_language_summary", "") or "").strip()
    if summary:
        lines.append(summary)

    lines.append(
        "Use this to prioritize likely missing bridge types, but only create evidence-grounded bridges from the current packet."
    )
    return "\n".join(lines)


def summarize_relation_rows(rows: Iterable[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for row in rows:
        relation = str(row.get("relation_label", "") or row.get("relation", "")).strip()
        missing = int(row.get("missing", 0) or 0)
        if relation:
            parts.append(f"{relation} ({missing} missing)")
    return ", ".join(parts)
