from __future__ import annotations

import json
from typing import Iterable, Sequence

from .schemas import AtomicFrame, CanonicalEntity, MentionSpan, TextWindow


def _json_block(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _slim_frame_view(frame: AtomicFrame) -> dict:
    """Compact view of an AtomicFrame for prompts that only need the shape of
    the relation, not its evidence. Drops ``supporting_text``, doc offsets,
    window_id, and the full participant objects in favour of just the refs.
    Cuts per-frame payload size by ~70% — enough to keep the refiner and
    verifier inputs under the reasoning-model token budget on long docs."""

    return {
        "frame_id": frame.frame_id,
        "vp_phrase": frame.vp_phrase,
        "slot_kind": frame.slot_kind,
        "subject_ref": frame.subject.ref,
        "subject_role": frame.subject.role,
        "object_ref": frame.object.ref,
        "object_role": frame.object.role,
    }


class EntitySeedPrompts:
    SYSTEM = (
        "You are an exhaustive entity mention extractor. "
        "You list every named thing in a document. "
        "Return only valid JSON. No prose. No commentary. No code fences."
    )

    EXAMPLE_DOC = (
        "Ada Lovelace (1815–1852), daughter of Lord Byron, wrote the first "
        "published algorithm intended for Charles Babbage's Analytical Engine. "
        "In 1843 she published her notes in Taylor's Scientific Memoirs in London."
    )
    EXAMPLE_OUTPUT = {
        "mentions": [
            {"text": "Ada Lovelace", "aliases": []},
            {"text": "1815", "aliases": []},
            {"text": "1852", "aliases": []},
            {"text": "Lord Byron", "aliases": []},
            {"text": "Charles Babbage", "aliases": []},
            {"text": "Analytical Engine", "aliases": []},
            {"text": "1843", "aliases": []},
            {"text": "Taylor's Scientific Memoirs", "aliases": []},
            {"text": "London", "aliases": []},
        ]
    }

    @classmethod
    def user(cls, text: str, *, hint: str = "") -> str:
        hint_block = f"\nRepair hint:\n{hint}\n" if hint else ""
        example_block = (
            "=== WORKED EXAMPLE ===\n"
            "Example document:\n"
            f'"""{cls.EXAMPLE_DOC}"""\n\n'
            "Example output:\n"
            f"{_json_block(cls.EXAMPLE_OUTPUT)}\n"
            "=== END EXAMPLE ===\n\n"
        )
        return (
            "TASK: List every entity mention in the document below. You MUST "
            "extract at minimum every proper noun, person name, organization, "
            "place, country, date, year, work title, event name, and named "
            "factual object that appears.\n\n"
            "OUTPUT FORMAT: JSON object with a single key `mentions` whose "
            "value is a list of `{\"text\": <exact surface string>, "
            "\"aliases\": []}` objects. You do NOT emit character offsets — "
            "just copy the exact surface string as it appears in the document. "
            "Downstream code computes the offsets deterministically.\n\n"
            "RULES:\n"
            "- Copy the EXACT surface string. Do not normalize capitalization, "
            "spacing, punctuation, or accents. Do not expand abbreviations.\n"
            "- List every DISTINCT mention. If the same entity appears five "
            "times, list its surface string five times (or once — duplicates "
            "are harmless, omissions are not).\n"
            "- Err on the side of over-extraction. Returning an empty or "
            "near-empty list on a non-empty document is always wrong.\n"
            "- `aliases` only contains aliases explicitly introduced in the "
            "text (e.g. `\"CIA (Central Intelligence Agency)\"` → alias `\"CIA\"`).\n"
            "- Do not output relations, verb phrases, or questions.\n"
            f"{hint_block}"
            f"{example_block}"
            "Now extract mentions from this document:\n\n"
            "<document>\n"
            f"{text}\n"
            "</document>\n\n"
            'Respond with `{"mentions": [...]}` only.'
        )


class AtomicFramePrompts:
    SYSTEM = (
        "You extract atomic binary relation frames from a window of factual text. "
        "Return only valid JSON."
    )

    @classmethod
    def user(
        cls,
        window: TextWindow,
        mentions: Sequence[MentionSpan],
        *,
        hint: str = "",
    ) -> str:
        mention_payload = [
            {
                "ref": mention.ref,
                "text": mention.text,
                "char_start": mention.char_start,
                "char_end": mention.char_end,
            }
            for mention in mentions
        ]
        hint_block = f"\nRepair hint:\n{hint}\n" if hint else ""
        return (
            "Extract atomic binary frames from the window.\n"
            "Rules:\n"
            "- Each frame must represent exactly one slot relation: recipient, content, time, location, agent, patient, manner, purpose, or other.\n"
            "- Split multi-slot events into separate frames.\n"
            "- Use mention refs when possible.\n"
            "- If a required participant is missing from the mention catalog, use ref='span_text:<exact surface form>'.\n"
            "- subject and object must each contain a role and a ref.\n"
            "- supporting_text must be a direct quote from the window.\n"
            "- support_char_start and support_char_end are absolute offsets in the full document; if unknown, set them to 0.\n"
            "- Do not emit questions.\n"
            f"{hint_block}"
            f"Window metadata:\n{_json_block({'id': window.id, 'char_start': window.char_start, 'char_end': window.char_end})}\n"
            f"Mention catalog:\n{_json_block(mention_payload)}\n"
            "<window_text>\n"
            f"{window.text}\n"
            "</window_text>\n"
            "Return JSON with shape {\"frames\": [...]}."
        )


class EntityRefinerPrompts:
    SYSTEM = (
        "You canonicalize mention-level entities into final entities grounded "
        "in the text. Return only valid JSON. You MUST emit BOTH `entities` "
        "and `mention_map`, AND every entity MUST have at least one non-empty "
        "`role` object with a meaningful role label. Empty `roles: []` is "
        "invalid output — it makes the entity unusable downstream."
    )

    @classmethod
    def user(
        cls,
        mentions: Sequence[MentionSpan],
        frames: Sequence[AtomicFrame],
        *,
        hint: str = "",
    ) -> str:
        mention_payload = [mention.model_dump(mode="json") for mention in mentions]
        # Slim frame view: refiner only needs ref → participant mapping,
        # not supporting_text / offsets. Shrinks refiner input from ~141k to
        # ~30k chars on doc-0-sized FRAMES articles.
        frame_payload = [_slim_frame_view(frame) for frame in frames]
        hint_block = f"\nRepair hint:\n{hint}\n" if hint else ""
        return (
            "Merge mentions into canonical entities.\n"
            "Rules:\n"
            "- Reuse evidence from mentions and frames only.\n"
            "- Every canonical entity MUST have at least one `role` object. "
            "The `role` field is a short semantic label (e.g. 'director', "
            "'actor', 'film', 'location', 'date', 'organization', 'event', "
            "'person', 'work'). Use the frame participants and mention "
            "context to infer the right role. Never return `roles: []`.\n"
            "- Each role can have a `states` list for attributes (e.g. "
            "director → states=['American','Academy-Award-nominated']). "
            "states may be empty, but the role itself must be present.\n"
            "- You MUST emit a `mention_map` dict. Every input mention ref "
            "(m1, m2, ...) that canonicalizes to an entity you produced MUST "
            "appear as a key in `mention_map`, with the value set to the "
            "canonical entity's id (e1, e2, ...).\n"
            "- `mention_map` is NOT optional. Returning an empty object when "
            "mentions were given is invalid output. Pipeline stages downstream "
            "depend on it for routing questions to the correct entity IDs.\n"
            "- source_mentions on each canonical entity must list the mention "
            "refs merged into it, and each of those refs must also appear as "
            "a key in `mention_map`.\n"
            "- Do not invent unsupported aliases.\n"
            "\n"
            "Example output shape (note: `roles` is always non-empty):\n"
            '{"entities": [{"id": "e1", "name": "Ada Lovelace", "aliases": [], '
            '"roles": [{"role": "mathematician", "states": ["English", "19th century"]}], '
            '"source_mentions": ["m1", "m4"]}], '
            '"mention_map": {"m1": "e1", "m4": "e1"}}\n'
            f"{hint_block}"
            f"Mentions:\n{_json_block(mention_payload)}\n"
            f"Frames:\n{_json_block(frame_payload)}\n"
            "Return JSON with shape {\"entities\": [...], \"mention_map\": {...}}."
        )


class QuestionPairPrompts:
    SYSTEM = (
        "You generate one forward and one reverse question for a single atomic factual frame. "
        "Return only valid JSON."
    )

    @classmethod
    def user(
        cls,
        frame: AtomicFrame,
        entities: Sequence[CanonicalEntity],
        *,
        supporting_text: str,
        hint: str = "",
    ) -> str:
        entity_payload = [entity.model_dump(mode="json") for entity in entities]
        hint_block = f"\nRepair hint:\n{hint}\n" if hint else ""
        return (
            "Generate exactly one forward question and one reverse question for this atomic frame.\n"
            "Follow the factual QA convention: every relation is answerable by "
            "canonical entities. Both question sides MUST have at least one "
            "entity-id answer — this frame's subject and object are both "
            "guaranteed to exist as canonical entities in the scope list below.\n\n"
            "Rules:\n"
            "- Questions must be grounded in the supporting text.\n"
            "- The forward question should leave the frame object unknown; its "
            "answer(s) must be the canonical entity id(s) of the frame object "
            "(and any additional in-scope entities that also satisfy the question).\n"
            "- The reverse question should leave the frame subject unknown; its "
            "answer(s) must be the canonical entity id(s) of the frame subject "
            "(and any additional in-scope entities that also satisfy it).\n"
            "- answers MUST contain ONLY canonical entity ids (e.g. \"e3\") from "
            "the 'Canonical entities in scope' list below. Nothing else is allowed.\n"
            "- Do NOT emit \"TEXT:...\" prefixed answers.\n"
            "- Do NOT emit \"None\", \"null\", \"Unknown\", or any placeholder string.\n"
            "- Do NOT emit empty answer lists — every side has at least one "
            "entity answer because the frame's participants are canonicalized.\n"
            "- Do not use pronouns like it/he/she when a grounded entity name can be used.\n"
            "\n"
            "Valid answer shapes (and ONLY these shapes):\n"
            "  \"answers\": [\"e3\"]          # single entity answer\n"
            "  \"answers\": [\"e3\", \"e7\"]   # multiple entity answers\n"
            f"{hint_block}"
            f"Frame:\n{_json_block(frame.model_dump(mode='json'))}\n"
            f"Canonical entities in scope:\n{_json_block(entity_payload)}\n"
            "<supporting_text>\n"
            f"{supporting_text}\n"
            "</supporting_text>\n"
            "Return JSON with shape {\"pair\": {\"frame_id\": ..., \"forward\": {...}, \"reverse\": {...}}}."
        )


class CoverageQAPrompts:
    """Prompt used to repair one orphan entity (an entity that no question
    answer currently references). The agent receives the target entity, its
    local paragraph, and a narrow scope of nearby canonical entities, and
    is asked to mint one or more grounded atomic frames + QA pairs that wire
    the target entity into the graph as an answer id."""

    SYSTEM = (
        "You repair QA coverage for one uncovered grounded entity. You may "
        "create missing semantic relation frames only when the local source "
        "text directly supports them. Return only valid JSON."
    )

    @classmethod
    def user(
        cls,
        *,
        target_entity: CanonicalEntity,
        target_mentions: Sequence[MentionSpan],
        available_mentions: Sequence[MentionSpan],
        local_window: TextWindow,
        nearby_entities: Sequence[CanonicalEntity],
        existing_frames: Sequence[AtomicFrame],
        hint: str = "",
    ) -> str:
        hint_block = f"\nRepair hint:\n{hint}\n" if hint else ""
        return (
            "The target entity is currently uncovered: it does not appear as "
            "an answer ID in any QA pair. Create one or more grounded atomic "
            "relation frames plus forward/reverse QA pairs that cover this "
            "target entity as an answer ID.\n\n"
            "Hard rules:\n"
            "- Every item must be supported by the local source text below.\n"
            "- Do NOT create weak meta relations such as 'mentioned in article', "
            "'appears in the text', 'is listed in', or 'is included in'.\n"
            "- If no grounded semantic relation exists in this local text, "
            "return items=[] and explain briefly in skipped_reason.\n"
            "- Each frame is binary and atomic: exactly one subject, one verb "
            "phrase, and one object/attribute slot.\n"
            "- Frame subject.ref and object.ref should use mention refs from "
            "the available mention catalog. If a required participant is not "
            "cataloged, use ref='span_text:<exact surface form>'.\n"
            "- Question answers MUST use canonical entity IDs only, never "
            "mention refs or free text.\n"
            "- At least one question in every returned pair MUST include the "
            f"target entity id `{target_entity.id}` in its answers.\n"
            "- Generate exactly one forward and one reverse question per frame.\n"
            f"{hint_block}"
            f"Target entity:\n{_json_block(target_entity.model_dump(mode='json'))}\n"
            f"Target source mentions:\n{_json_block([m.model_dump(mode='json') for m in target_mentions])}\n"
            f"Available mention catalog:\n{_json_block([m.model_dump(mode='json') for m in available_mentions])}\n"
            f"Nearby canonical entities:\n{_json_block([e.model_dump(mode='json') for e in nearby_entities])}\n"
            f"Existing frames in this local context:\n{_json_block([f.model_dump(mode='json') for f in existing_frames])}\n"
            f"Local window metadata:\n{_json_block({'id': local_window.id, 'char_start': local_window.char_start, 'char_end': local_window.char_end})}\n"
            "<local_source_text>\n"
            f"{local_window.text}\n"
            "</local_source_text>\n\n"
            "Return JSON with shape "
            "{\"items\": [{\"frame\": {...}, \"pair\": {\"frame_id\": \"\", "
            "\"forward\": {\"text\": \"...\", \"answers\": [\"e...\"]}, "
            "\"reverse\": {\"text\": \"...\", \"answers\": [\"e...\"]}}}], "
            "\"skipped_reason\": \"\"}."
        )


class VerifierPrompts:
    COVERAGE_SYSTEM = (
        "You verify factual extraction coverage against source text. Return only valid JSON."
    )
    GROUNDING_SYSTEM = (
        "You verify whether extracted entities and frames are grounded in source text. Return only valid JSON."
    )
    QA_SYSTEM = (
        "You verify question-pair completeness and answer resolution. Return only valid JSON."
    )

    @classmethod
    def coverage_user(
        cls,
        *,
        source_text: str,
        windows: Sequence[TextWindow],
        mentions: Sequence[MentionSpan],
        frames: Sequence[AtomicFrame],
    ) -> str:
        return (
            "Check whether important entities or atomic relations were missed.\n"
            "Rules:\n"
            "- If an entity seems missing, emit a failure targeting producer_stage='entity_seed' and item_type='window'.\n"
            "- If a relation seems missing for a window, emit a failure targeting producer_stage='relation' and item_type='window'.\n"
            "- item_id should be a window id.\n"
            "- hint should briefly say what appears to be missing.\n"
            f"Windows:\n{_json_block([window.model_dump(mode='json') for window in windows])}\n"
            f"Mentions:\n{_json_block([mention.model_dump(mode='json') for mention in mentions])}\n"
            f"Frames:\n{_json_block([_slim_frame_view(frame) for frame in frames])}\n"
            "<source_text>\n"
            f"{source_text}\n"
            "</source_text>\n"
            "Return JSON with shape {\"failures\": [...]}."
        )

    @classmethod
    def grounding_user(
        cls,
        *,
        source_text: str,
        entities: Sequence[CanonicalEntity],
        frames: Sequence[AtomicFrame],
    ) -> str:
        return (
            "Check whether each entity and frame is grounded in the source text.\n"
            "Rules:\n"
            "- Unsupported canonical entities target producer_stage='entity_refiner' and item_type='entity'.\n"
            "- Unsupported frames target producer_stage='relation' and item_type='frame'.\n"
            "- hint should explain the grounding issue.\n"
            f"Entities:\n{_json_block([entity.model_dump(mode='json') for entity in entities])}\n"
            f"Frames:\n{_json_block([_slim_frame_view(frame) for frame in frames])}\n"
            "<source_text>\n"
            f"{source_text}\n"
            "</source_text>\n"
            "Return JSON with shape {\"failures\": [...]}."
        )

    @classmethod
    def qa_user(
        cls,
        *,
        frames: Sequence[AtomicFrame],
        question_pairs: Iterable[dict],
        entity_ids: Sequence[str],
    ) -> str:
        return (
            "Check whether each frame has a complete forward/reverse question pair and whether answers resolve.\n"
            "Rules:\n"
            "- Missing or invalid pairs target producer_stage='question' and item_type='question_pair'.\n"
            "- item_id should be the frame_id.\n"
            "- hint should explain what is wrong.\n"
            f"Frames:\n{_json_block([_slim_frame_view(frame) for frame in frames])}\n"
            f"Question pairs:\n{_json_block(list(question_pairs))}\n"
            f"Known entity ids:\n{_json_block(list(entity_ids))}\n"
            "Return JSON with shape {\"failures\": [...]}."
        )
