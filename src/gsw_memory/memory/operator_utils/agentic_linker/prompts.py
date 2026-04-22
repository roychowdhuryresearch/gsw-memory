"""Prompts for the linker pipeline. Four agents, four prompt classes."""

from __future__ import annotations

import json
from typing import Dict, Optional, Sequence

from .schemas import CanonicalEntity, Link, VerbPhraseSlot


def _json_block(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# EntityListAgent
# ---------------------------------------------------------------------------


class EntityListPrompts:
    SYSTEM = (
        "You are an exhaustive entity extractor. You read a document and "
        "produce a flat list of every canonical named entity it discusses. "
        "Each entity must have a non-empty role label. Return only valid JSON."
    )

    @classmethod
    def user(cls, text: str) -> str:
        return (
            "TASK: List every canonical entity in the document below.\n\n"
            "RULES:\n"
            "- Capture EVERY person, organization, place, country, work title, "
            "date, year, event, and named factual object.\n"
            "- Use the canonical/most complete name as `name`. Other surface "
            "variants go in `aliases`.\n"
            "- Assign each entity a sequential id starting at e1.\n"
            "- Every entity MUST have at least one `role` object with a short "
            "semantic label (e.g. 'film', 'director', 'actor', 'date', "
            "'location', 'organization', 'person', 'country', 'work', 'event'). "
            "`states` may be empty but `role` itself is required.\n"
            "- Do NOT include verb phrases, mention offsets, or any relation data. "
            "Only entities.\n"
            "- Do NOT return placeholder names like 'A', 'The', 'Generic', or "
            "single letters.\n"
            "- Returning a near-empty entity list on a non-empty document is "
            "almost always wrong.\n\n"
            "Example output shape:\n"
            '{"entities": [\n'
            '  {"id": "e1", "name": "Ada Lovelace", "aliases": [], '
            '"roles": [{"role": "mathematician", "states": ["English", "19th century"]}]},\n'
            '  {"id": "e2", "name": "Charles Babbage", "aliases": [], '
            '"roles": [{"role": "inventor", "states": []}]},\n'
            '  {"id": "e3", "name": "Analytical Engine", "aliases": [], '
            '"roles": [{"role": "machine", "states": []}]}\n'
            "]}\n\n"
            "<document>\n"
            f"{text}\n"
            "</document>\n\n"
            'Return JSON with shape {"entities": [...]}.'
        )


# ---------------------------------------------------------------------------
# VerbPhraseListAgent
# ---------------------------------------------------------------------------


class VerbPhraseListPrompts:
    SYSTEM = (
        "You are a verb phrase extractor. You receive a document AND the "
        "canonical entity list that was just extracted from it. Your goal is "
        "to produce a flat list of relation slots such that **every entity "
        "in the list would be wired into at least one relation** by the "
        "downstream linker. Under-extraction leaves orphan entities; "
        "over-extraction wastes compute. Aim for coverage, not a character-"
        "count target.\n\n"
        "Duplicates are allowed and expected when a verb phrase has multiple "
        "distinct instances (e.g. `released in 2016` and `released in the "
        "United States` are two distinct `released in` slots). Return only "
        "valid JSON."
    )

    @classmethod
    def user(
        cls,
        text: str,
        *,
        entities: Optional[Sequence["CanonicalEntity"]] = None,
    ) -> str:
        entity_block = ""
        target_block = ""
        if entities:
            entity_rows = [
                {
                    "id": e.id,
                    "name": e.name,
                    "role": e.roles[0].role if e.roles else "",
                }
                for e in entities
            ]
            # Rule of thumb: ~1.5 verb phrases per entity covers the graph
            # on most articles. Fact-dense articles (sports stats, scientific
            # papers) may need 2–3× per entity because each entity appears in
            # multiple relations.
            target = int(len(entities) * 1.5)
            entity_block = (
                f"\n\n**Entity list to cover ({len(entities)} entities)**:\n"
                f"{_json_block(entity_rows)}\n\n"
                "Every entity in this list will be wired into the final GSW "
                "by the downstream linker IF you produce a verb phrase that "
                "can serve as one of its relations. Your verb phrases don't "
                "reference entities by id — they're just relation labels — "
                "but your mental model should be: **'do I have enough verb "
                "phrases to give every entity at least one relation?'**\n\n"
                f"**Rough target**: about {target} verb phrases "
                f"(~1.5 × the {len(entities)} entities). Fact-dense articles "
                "(sports stats, scientific results, financial reports) may "
                "need 2–3× that. Narrative articles may need fewer."
            )
            target_block = (
                "TARGET: aim for coverage, not a fixed count. Every entity "
                "above should be something a reader could answer a question "
                "about using at least one of your verb phrases. If an entity "
                "is a date, you need a `released in` / `occurred in` / "
                "`born in` style relation. If it's a place, a `located in` / "
                "`held in` / `filmed in` style relation. If it's a person, "
                "an `authored` / `directed` / `played for` style relation.\n\n"
            )

        return (
            "TASK: Extract relation slots that cover every entity in the "
            "entity list below.\n\n"
            f"{target_block}"
            "RULES:\n"
            "- DUPLICATES ARE INTENTIONAL when a verb phrase has multiple "
            "slots. Example: a film 'released in 2016 in the United States' "
            "should emit `released in` TWICE as two distinct vp_ids, one "
            "for the year and one for the country. The downstream linker "
            "uses each entry as one binary relation slot.\n"
            "- Use a sequential vp_id starting at vp1.\n"
            "- `label` is a concise verb phrase (1–4 words).\n"
            "- Optional `hint` may briefly say which slot/instance this "
            "entry covers (e.g. 'year', 'country', 'lead actor', 'source book').\n"
            "- Do NOT emit entities, mentions, or offsets. Only verb phrases.\n\n"
            "RELATION FAMILIES TO SWEEP FOR (enumerate explicitly — do not "
            "rely on intuition alone):\n\n"
            "**Authorship / attribution / creation** — for every creative work "
            "or utterance, who is responsible?\n"
            "- `wrote`, `authored`, `co-wrote`, `composed`, `co-composed`, "
            "`designed`, `invented`, `illustrated`, `edited`, `translated`, "
            "`founded`, `built`, `painted`, `filmed`, `photographed`, "
            "`produced`, `directed`, `developed`, `drafted`, `said`, "
            "`argued`, `claimed`, `reported`, `wrote about`.\n"
            "- If the article says `based on the 2014 book X by Y`, that is "
            "TWO relations: `based on` (→ book) AND `wrote` / `authored` "
            "(book → Y).\n\n"
            "**Cast / performance / role** — every actor↔character and "
            "every player↔team / position / tournament:\n"
            "- `stars`, `starred`, `plays`, `portrayed`, `cast as`, "
            "`appears as`, `voices`, `played for`, `captained`, "
            "`managed`, `coached`, `signed with`, `transferred to`.\n\n"
            "**Temporal** — every date/year needs a relation:\n"
            "- `released in`, `born in`, `died in`, `occurred in`, "
            "`took place in`, `published in`, `founded in`, `began in`, "
            "`ended in`, `aired in`, `held in`.\n\n"
            "**Spatial** — every location/venue/country needs a relation:\n"
            "- `set in`, `located in`, `based in`, `hosted in`, `shot in`, "
            "`recorded in`, `filmed at`, `played at`.\n\n"
            "**Causal / event** — who did what to whom:\n"
            "- `killed`, `defeated`, `won`, `lost to`, `attacked`, "
            "`defended`, `arrested`, `elected`, `signed`, `awarded`, "
            "`nominated for`, `scored`, `assisted`, `fouled`.\n\n"
            "**Organizational** — membership, ownership, affiliation:\n"
            "- `member of`, `worked for`, `owned by`, `sponsored by`, "
            "`distributed by`, `published by`, `broadcast by`, "
            "`associated with` (only when specific — not as filler).\n\n"
            "**Attribute / measurement** — the entity HAS a value:\n"
            "- `has budget`, `earned`, `grossed`, `lasted`, `rated`, "
            "`scored`, `measured`, `priced at`.\n\n"
            "EXTRACTION PROCEDURE:\n"
            "1. Walk through the document paragraph by paragraph.\n"
            "2. In each paragraph, list every subject→verb→object triple "
            "that could form a binary relation, even if it feels minor.\n"
            "3. For each triple, add one entry to `verb_phrases` using the "
            "verb as the label.\n"
            "4. If one sentence expresses multiple relations "
            "(e.g. 'directed and produced by Bay'), emit one entry per "
            "relation.\n"
            "5. After a first pass, check: for every entity in the entity "
            "list above, is there at least one verb phrase that could wire "
            "it into a relation? If not, go back and find one. An entity "
            "with zero relations will be dumped into a slower fallback "
            "stage downstream — it's much cheaper to catch it here.\n\n"
            "Example output shape:\n"
            '{"verb_phrases": [\n'
            '  {"vp_id": "vp1", "label": "directed", "hint": ""},\n'
            '  {"vp_id": "vp2", "label": "produced", "hint": ""},\n'
            '  {"vp_id": "vp3", "label": "written by", "hint": "screenwriter"},\n'
            '  {"vp_id": "vp4", "label": "based on", "hint": "source book"},\n'
            '  {"vp_id": "vp5", "label": "wrote", "hint": "book author"},\n'
            '  {"vp_id": "vp6", "label": "released in", "hint": "year"},\n'
            '  {"vp_id": "vp7", "label": "released in", "hint": "country"},\n'
            '  {"vp_id": "vp8", "label": "starred", "hint": "lead"},\n'
            '  {"vp_id": "vp9", "label": "set in", "hint": "city"},\n'
            '  {"vp_id": "vp10", "label": "grossed", "hint": "worldwide box office"},\n'
            '  {"vp_id": "vp11", "label": "nominated for", "hint": "Oscar"}\n'
            "]}\n"
            f"{entity_block}\n\n"
            "<document>\n"
            f"{text}\n"
            "</document>\n\n"
            'Return JSON with shape {"verb_phrases": [...]}.'
        )


# ---------------------------------------------------------------------------
# LinkAgent — fan-out, one call per verb phrase
# ---------------------------------------------------------------------------


class LinkPrompts:
    SYSTEM = (
        "You are wiring ONE verb phrase slot into a binary relation between "
        "two canonical entities. You receive the verb phrase, the full entity "
        "list, and the source document. You return ONE structured JSON object "
        "containing the link plus its forward and reverse questions.\n\n"
        "PRINCIPLE: a Generative Semantic Workspace is a faithful encoding of "
        "the source document. Every named thing in the text deserves to "
        "participate in at least one binary relation. You are NOT an editor — "
        "importance and centrality are downstream concerns.\n\n"
        "YOUR TASK FOR THIS CALL:\n"
        "1. Read the verb phrase label and hint. Find where in the source "
        "text this relation occurs.\n"
        "2. Identify the subject and object — the two participants.\n"
        "3. Look up each participant in the entity list:\n"
        "   - If a canonical entity already represents it, use that entity's "
        "id (e.g. `\"e3\"`).\n"
        "   - If NO existing entity represents it (e.g. the verb phrase is "
        "`\"received mixed reviews\"` and \"mixed reviews\" isn't in the "
        "entity list), add a new entity to `proposed_entities` and reference "
        "it via `\"NEW:<exact name>\"` in the subject_id/object_id field. "
        "The orchestrator will assign a real id after this call.\n"
        "4. Write `supporting_text` as a direct quote from the source.\n\n"
        # ============================================================
        # Subject / object / forward / reverse — the part models get wrong
        # ============================================================
        "=== SUBJECT / OBJECT CONVENTION (the most important rule) ===\n\n"
        "A binary relation has a direction. If the source says `\"Honduras "
        "held Spain to a 1-1 draw\"`, the structure is:\n\n"
        "    Honduras (SUBJECT)  —[held]→  Spain (OBJECT)\n\n"
        "The SUBJECT is the actor/agent — the one performing the action. "
        "The OBJECT is the patient/recipient — the one the action is done "
        "to. For `\"A wrote B\"`, A is subject, B is object. For "
        "`\"A directed B\"`, A is subject, B is object. For `\"A won B\"`, "
        "A is subject, B is object. Active voice, left-to-right.\n\n"
        "Fill the fields like this:\n"
        "    subject_id = Honduras's entity id (e.g. \"e12\")\n"
        "    object_id  = Spain's entity id (e.g. \"e2\")\n\n"
        "=== FORWARD AND REVERSE QUESTIONS ===\n\n"
        "The forward and reverse questions are MIRROR IMAGES of each other. "
        "Each one HIDES exactly one participant and ASKS for it.\n\n"
        "**Forward question** hides the OBJECT. The subject's name IS "
        "visible in the question. The answer WILL BE the object.\n"
        "  Forward question: \"Which team did Honduras hold to a 1-1 draw?\"\n"
        "  Expected answer:  Spain (the object)\n\n"
        "**Reverse question** hides the SUBJECT. The object's name IS "
        "visible in the question. The answer WILL BE the subject.\n"
        "  Reverse question: \"Which team held Spain to a 1-1 draw?\"\n"
        "  Expected answer:  Honduras (the subject)\n\n"
        "The downstream assembler **automatically** fixes answers as:\n"
        "    forward.answers = [object_id]   (Spain)\n"
        "    reverse.answers = [subject_id]  (Honduras)\n\n"
        "So your ONLY job is to write question texts that match this "
        "hiding convention. Do NOT set answers yourself — the pipeline "
        "does that from subject_id/object_id.\n\n"
        "=== THE MOST COMMON BUG (AVOID THIS) ===\n\n"
        "Models frequently write a forward question that accidentally "
        "contains the object's name. Example of the bug:\n\n"
        "  ❌ WRONG:\n"
        "    subject_id = \"e2\"     (Spain — wrong, Spain is the object)\n"
        "    object_id  = \"e12\"    (Honduras — wrong, Honduras is subject)\n"
        "    forward_question = \"Which team held Spain to a 1-1 draw?\"\n\n"
        "  This is broken because:\n"
        "  (a) subject and object are swapped from the natural relation.\n"
        "  (b) The forward question contains the word \"Spain\" — but "
        "\"Spain\" is what the forward answer is supposed to be! The "
        "forward answer is forced to be [object_id], which here is "
        "Honduras — but the question asks \"which team held Spain?\" and "
        "Honduras IS the right answer to that English sentence, except "
        "the question is supposed to be the FORWARD question, which hides "
        "the OBJECT, not the subject.\n\n"
        "  ✓ CORRECT:\n"
        "    subject_id = \"e12\"    (Honduras)\n"
        "    object_id  = \"e2\"     (Spain)\n"
        "    forward_question = \"Which team did Honduras hold to a 1-1 draw?\"\n"
        "    reverse_question = \"Which team held Spain to a 1-1 draw?\"\n\n"
        "=== SELF-CHECK BEFORE EMITTING ===\n\n"
        "Before returning, verify BOTH of these:\n"
        "  (1) Your forward_question text does NOT contain the OBJECT "
        "entity's canonical name. If the object is Spain, the forward "
        "question must not say \"Spain\" anywhere.\n"
        "  (2) Your reverse_question text does NOT contain the SUBJECT "
        "entity's canonical name. If the subject is Honduras, the reverse "
        "question must not say \"Honduras\" anywhere.\n\n"
        "If either check fails, you have the convention backwards — swap "
        "subject_id and object_id, then rewrite the questions.\n\n"
        "RULES:\n"
        "- ALWAYS produce a link. `action=\"skip\"` is reserved for verb "
        "phrases that genuinely have no source-text grounding (e.g. an "
        "upstream hallucination). Use it sparingly with `skip_reason`.\n"
        "- If a participant slot has a vague answer (a concept, a quality, "
        "a value), MINT A NEW ENTITY for it via `proposed_entities`. Do not "
        "skip just because the slot is abstract.\n"
        "- **Role labels for proposed entities must be CONCRETE.** Choose "
        "from this list whenever it fits: person, character, actor, "
        "director, writer, producer, organization, company, team, country, "
        "city, location, venue, date, year, period, event, film, book, "
        "work, song, album, award, score, rating, amount, currency, "
        "percentage, quote, quantity, statistic, role_title, language, "
        "ethnicity, religion, sport, position, tournament, season, match, "
        "game, weapon, vehicle, species, substance, document, law, treaty.\n"
        "- **NEVER use the role `concept`** unless the entity is genuinely "
        "an abstract idea (e.g. `freedom`, `democracy`, `offside`). Things "
        "like `mixed reviews`, `approval rating`, `budget`, `runtime`, "
        "`genre`, `storyline`, `sequel` are all concrete roles — pick the "
        "matching label above. `mixed reviews` → `rating` or `reception`. "
        "`$50 million budget` → `amount`. `6 goals` → `statistic`.\n"
        "- **When minting a new entity, the `name` field should be a direct "
        "substring of the source document** if possible. Do not combine "
        "phrases or add parenthetical disambiguators. `US diplomatic "
        "compound` is fine; `US diplomatic compound (Benghazi)` is not — "
        "drop the parenthetical.\n"
        "- Use entity NAMES in the question text, not ids. The model never "
        "writes `\"e3\"` in the question itself — that's an answer-id, not "
        "question text.\n"
        "- Do not use pronouns (it/he/she) when an entity name fits.\n"
        "- Do not invent facts beyond the supporting_text.\n\n"
        "Return only the structured JSON. No commentary."
    )

    @classmethod
    def user(
        cls,
        *,
        verb_phrase: VerbPhraseSlot,
        entities: Sequence[CanonicalEntity],
        text: str,
        retry_hint: str = "",
    ) -> str:
        entity_payload = [
            {
                "id": e.id,
                "name": e.name,
                "role": e.roles[0].role if e.roles else "",
                "aliases": list(e.aliases),
            }
            for e in entities
        ]
        retry_block = (
            f"\nVerifier rejected the previous attempt with this hint:\n{retry_hint}\n\n"
            if retry_hint
            else ""
        )
        return (
            f"Verb phrase to link:\n{_json_block(verb_phrase.model_dump(mode='json'))}\n\n"
            f"Existing canonical entities ({len(entities)}):\n{_json_block(entity_payload)}\n\n"
            f"{retry_block}"
            "<source_text>\n"
            f"{text}\n"
            "</source_text>\n\n"
            "Return JSON matching this shape exactly:\n"
            "{\n"
            '  "action": "link",\n'
            '  "skip_reason": "",\n'
            '  "subject_id": "e3",\n'
            '  "object_id": "e7",\n'
            '  "supporting_text": "direct quote from source",\n'
            '  "forward_question": "Question with subject name visible, object hidden",\n'
            '  "reverse_question": "Question with object name visible, subject hidden",\n'
            '  "proposed_entities": []\n'
            "}\n\n"
            "If you mint a new entity, set the placeholder in subject_id "
            "and/or object_id like `\"NEW:Mitchell Zuckoff\"` and add a "
            "matching entry to proposed_entities."
        )

    @classmethod
    def batch_user(
        cls,
        *,
        verb_phrases: Sequence[VerbPhraseSlot],
        entities: Sequence[CanonicalEntity],
        text: str,
        retry_hints: Optional[Dict[str, str]] = None,
    ) -> str:
        """User prompt for a batched LinkAgent call. The model receives N
        verb phrases and must return N LinkAgentOutput objects, each keyed
        by the input ``vp_id``. All other conventions from the single-call
        version apply (subject/object rules, question hiding, proposed
        entities, role taxonomy)."""
        retry_hints = retry_hints or {}
        entity_payload = [
            {
                "id": e.id,
                "name": e.name,
                "role": e.roles[0].role if e.roles else "",
                "aliases": list(e.aliases),
            }
            for e in entities
        ]
        vp_payload = []
        for vp in verb_phrases:
            row = {"vp_id": vp.vp_id, "label": vp.label}
            if vp.hint:
                row["hint"] = vp.hint
            if retry_hints.get(vp.vp_id):
                row["retry_hint"] = retry_hints[vp.vp_id]
            vp_payload.append(row)
        return (
            f"Verb phrases to link ({len(verb_phrases)}):\n"
            f"{_json_block(vp_payload)}\n\n"
            f"Existing canonical entities ({len(entities)}):\n"
            f"{_json_block(entity_payload)}\n\n"
            "<source_text>\n"
            f"{text}\n"
            "</source_text>\n\n"
            "Produce ONE link object per verb phrase in the input list. "
            "Emit them in the same order, keyed by `vp_id`. Each link object "
            "follows the same shape as the single-call version: "
            "`{action, subject_id, object_id, supporting_text, "
            "forward_question, reverse_question, proposed_entities, "
            "skip_reason}`.\n\n"
            "Return JSON with this shape:\n"
            "{\n"
            '  "items": [\n'
            '    {"vp_id": "vp1", "output": {"action": "link", '
            '"subject_id": "e3", "object_id": "e7", '
            '"supporting_text": "...", "forward_question": "...", '
            '"reverse_question": "...", "proposed_entities": []}},\n'
            '    {"vp_id": "vp2", "output": {...}}\n'
            "  ]\n"
            "}\n\n"
            "All rules from the single-call link prompt apply (subject/object "
            "direction, forward hides object, reverse hides subject, role "
            "taxonomy, supporting_text as direct quote). The self-check is "
            "especially important here — for EACH output, verify that the "
            "forward_question does NOT contain the object's name and the "
            "reverse_question does NOT contain the subject's name."
        )


# ---------------------------------------------------------------------------
# LinkVerifier — judges one link
# ---------------------------------------------------------------------------


class LinkVerifierPrompts:
    SYSTEM = (
        "You judge a single binary relation produced by a link agent. Given "
        "the verb phrase, the canonical subject and object entities, the "
        "supporting text quote, the two questions, and the source document, "
        "you return a structured verdict.\n\n"
        "VALID iff ALL of these hold:\n"
        "- The supporting_text appears verbatim (or with trivial whitespace "
        "differences) in the source document.\n"
        "- The supporting_text actually expresses the verb phrase between "
        "the subject and object.\n"
        "- The subject and object are correct for that relation.\n"
        "- The forward question is answerable by the object entity name and "
        "doesn't reveal the answer.\n"
        "- The reverse question is answerable by the subject entity name and "
        "doesn't reveal the answer.\n\n"
        "If anything is wrong, set `valid=false`, write a one-sentence `issue` "
        "describing the problem, and a one-sentence `hint` telling the link "
        "agent what to fix on retry.\n\n"
        "Return only the structured JSON. No commentary."
    )

    @classmethod
    def user(
        cls,
        *,
        link: Link,
        subject_entity: CanonicalEntity,
        object_entity: CanonicalEntity,
        text: str,
    ) -> str:
        return (
            f"Link to verify:\n{_json_block(link.model_dump(mode='json'))}\n\n"
            f"Subject entity:\n{_json_block(subject_entity.model_dump(mode='json'))}\n\n"
            f"Object entity:\n{_json_block(object_entity.model_dump(mode='json'))}\n\n"
            "<source_text>\n"
            f"{text}\n"
            "</source_text>\n\n"
            "Return JSON: "
            '{"valid": true|false, "confidence": 0.0-1.0, '
            '"issue": "", "hint": ""}.'
        )

    @classmethod
    def batch_user(
        cls,
        *,
        links: Sequence[Link],
        entity_by_id: Dict[str, CanonicalEntity],
        text: str,
    ) -> str:
        """User prompt for a batched LinkVerifier call. The model judges N
        links at once and returns N verdicts keyed by ``link_id``."""
        link_rows = []
        for link in links:
            subject = entity_by_id.get(link.subject_id)
            obj = entity_by_id.get(link.object_id)
            link_rows.append(
                {
                    "link": link.model_dump(mode="json"),
                    "subject_entity": subject.model_dump(mode="json") if subject else None,
                    "object_entity": obj.model_dump(mode="json") if obj else None,
                }
            )
        return (
            f"Links to verify ({len(links)}):\n{_json_block(link_rows)}\n\n"
            "<source_text>\n"
            f"{text}\n"
            "</source_text>\n\n"
            "For each link, return a verdict. Emit them in the same order, "
            "keyed by `link_id`. Each verdict follows the single-call "
            "shape: `{valid, confidence, issue, hint}`.\n\n"
            "Return JSON:\n"
            "{\n"
            '  "items": [\n'
            '    {"link_id": "l_vp1", "verdict": {"valid": true, '
            '"confidence": 0.95, "issue": "", "hint": ""}},\n'
            '    {"link_id": "l_vp2", "verdict": {"valid": false, '
            '"confidence": 0.4, "issue": "supporting_text not in source", '
            '"hint": "pick a direct quote from the article"}}\n'
            "  ]\n"
            "}"
        )


# ---------------------------------------------------------------------------
# FillerAgent — for orphan entities, mints a new verb phrase + link + QA
# ---------------------------------------------------------------------------


class FillerPrompts:
    SYSTEM = (
        "You are wiring ONE orphan canonical entity into the GSW. The entity "
        "exists in the entity list but no current link references it as "
        "subject or object. Your job is to mint a new binary relation that "
        "involves this entity, drawing on the source document.\n\n"
        "YOUR TASK FOR THIS CALL:\n"
        "1. Find where the orphan entity appears in the source text.\n"
        "2. Find another entity in its surrounding context that has a "
        "grounded relation to it.\n"
        "3. Pick a concise, SEMANTICALLY MEANINGFUL verb phrase label "
        "(e.g. `\"located in\"`, `\"founded by\"`, `\"scored\"`, "
        "`\"directed\"`, `\"played for\"`, `\"released in\"`).\n\n"
        "FORBIDDEN verb phrase labels — these are filler sludge, not "
        "relations. If you find yourself wanting one of these, stop and "
        "look again for a real semantic relation in the text:\n"
        "- `mentioned in article`, `appears in`, `listed in`, `included in`\n"
        "- `is a`, `has a`, `related to`, `associated with`, `linked to`\n"
        "- `part of article`, `referenced in`, `covered in`, `discussed in`\n\n"
        "4. Output a structured response as if you were the LinkAgent: "
        "subject_id, object_id (one of which must be the orphan entity), "
        "supporting_text, forward_question, reverse_question, and an empty "
        "proposed_entities list (don't propose new entities here — link to "
        "what already exists).\n"
        "5. supporting_text must be a direct verbatim quote from the source "
        "document — not a summary and not a fragment truncated with '...'.\n\n"
        # ============================================================
        # Subject / object / forward / reverse — the part models get wrong
        # ============================================================
        "=== SUBJECT / OBJECT CONVENTION (critical) ===\n\n"
        "A binary relation has a direction. If the source says `\"Honduras "
        "held Spain to a 1-1 draw\"`, the structure is:\n\n"
        "    Honduras (SUBJECT)  —[held]→  Spain (OBJECT)\n\n"
        "The SUBJECT is the actor/agent. The OBJECT is the patient/recipient. "
        "For `\"A wrote B\"`, A is subject, B is object. Active voice, "
        "left-to-right.\n\n"
        "Fill the fields like this:\n"
        "    subject_id = Honduras's entity id (e.g. \"e12\")\n"
        "    object_id  = Spain's entity id (e.g. \"e2\")\n\n"
        "=== FORWARD AND REVERSE QUESTIONS ===\n\n"
        "**Forward question** HIDES the OBJECT. The subject's name is "
        "visible in the question; the answer will be the object.\n"
        "  Forward: \"Which team did Honduras hold to a 1-1 draw?\"\n"
        "  Answer:  Spain (the object)\n\n"
        "**Reverse question** HIDES the SUBJECT. The object's name is "
        "visible in the question; the answer will be the subject.\n"
        "  Reverse: \"Which team held Spain to a 1-1 draw?\"\n"
        "  Answer:  Honduras (the subject)\n\n"
        "The downstream assembler AUTOMATICALLY fixes answers as:\n"
        "    forward.answers = [object_id]\n"
        "    reverse.answers = [subject_id]\n\n"
        "Your only job is to write question texts that match the hiding "
        "convention. Do NOT set answers — the pipeline does that.\n\n"
        "=== COMMON BUG TO AVOID ===\n\n"
        "  ❌ WRONG: writing a forward question that contains the object's "
        "name, e.g.\n"
        "      subject_id = \"e2\" (Spain), object_id = \"e12\" (Honduras)\n"
        "      forward_question = \"Which team held Spain to a 1-1 draw?\"\n"
        "    This swaps subject and object AND leaks the object name into "
        "the forward question. Downstream, this produces nonsense answers.\n\n"
        "  ✓ CORRECT:\n"
        "      subject_id = \"e12\" (Honduras), object_id = \"e2\" (Spain)\n"
        "      forward_question = \"Which team did Honduras hold to a 1-1 draw?\"\n"
        "      reverse_question = \"Which team held Spain to a 1-1 draw?\"\n\n"
        "=== SELF-CHECK BEFORE EMITTING ===\n\n"
        "Before returning, verify BOTH:\n"
        "  (1) Your forward_question does NOT contain the OBJECT entity's "
        "canonical name.\n"
        "  (2) Your reverse_question does NOT contain the SUBJECT entity's "
        "canonical name.\n\n"
        "If either check fails, swap subject_id and object_id and rewrite "
        "the questions.\n\n"
        "**Role labels (for the subject/object entity roles you reference) "
        "must be concrete.** Never use `concept` when something more specific "
        "applies: person, organization, team, country, city, venue, date, "
        "year, event, film, book, award, score, rating, amount, statistic, "
        "role_title, etc.\n\n"
        "If the orphan entity genuinely has no grounded relation to any "
        "other entity in the source text, set `action=\"skip\"` with a "
        "`skip_reason` that cites the specific sentence(s) you examined.\n\n"
        "Return only the structured JSON. No commentary."
    )

    @classmethod
    def user(
        cls,
        *,
        orphan_entity: CanonicalEntity,
        entities: Sequence[CanonicalEntity],
        text: str,
    ) -> str:
        entity_payload = [
            {
                "id": e.id,
                "name": e.name,
                "role": e.roles[0].role if e.roles else "",
                "aliases": list(e.aliases),
            }
            for e in entities
            if e.id != orphan_entity.id
        ]
        return (
            f"Orphan entity to wire in:\n{_json_block(orphan_entity.model_dump(mode='json'))}\n\n"
            f"Other canonical entities in scope ({len(entity_payload)}):\n"
            f"{_json_block(entity_payload)}\n\n"
            "<source_text>\n"
            f"{text}\n"
            "</source_text>\n\n"
            "Return JSON matching the LinkAgent output shape:\n"
            "{\n"
            '  "action": "link",\n'
            '  "skip_reason": "",\n'
            '  "subject_id": "<eN, must be the orphan or one of the other entities>",\n'
            '  "object_id": "<the other side of the relation>",\n'
            '  "supporting_text": "direct quote",\n'
            '  "forward_question": "...",\n'
            '  "reverse_question": "...",\n'
            '  "proposed_entities": []\n'
            "}\n\n"
            "Note: the verb phrase you pick will be added to the verb phrase "
            "list automatically; there is no vp_id yet."
        )
