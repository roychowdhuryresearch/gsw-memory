"""Pydantic schemas for the linker pipeline.

These are pipeline-internal types; the final emitted format is still
``GSWStructure`` from ``gsw_memory.memory.models``.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from ...models import Role


# ---------------------------------------------------------------------------
# Stage 1 — EntityListAgent output
# ---------------------------------------------------------------------------


class CanonicalEntity(BaseModel):
    """One canonical entity emitted by EntityListAgent. The agent assigns
    the id directly (``e1``, ``e2``, ...) so the linker can reference it
    without a translation step."""

    id: str
    name: str
    aliases: List[str] = Field(default_factory=list)
    roles: List[Role] = Field(default_factory=list)


class EntityListOutput(BaseModel):
    entities: List[CanonicalEntity] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 2 — VerbPhraseListAgent output
# ---------------------------------------------------------------------------


class VerbPhraseSlot(BaseModel):
    """One verb phrase slot. Duplicates are intentional: a doc that says
    "released in 2016 in the U.S." emits ``released`` twice as two distinct
    ``vp_id``s, one for the year and one for the country."""

    vp_id: str
    label: str
    hint: str = ""  # optional one-line hint about which slot/instance this is


class VerbPhraseListOutput(BaseModel):
    verb_phrases: List[VerbPhraseSlot] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 3 — LinkerAgent state + output
# ---------------------------------------------------------------------------


class ProposedEntity(BaseModel):
    """An entity the LinkAgent wants to add to the canonical list because no
    existing entity fits a participant slot. The orchestrator deduplicates
    these by name (case-insensitive) and assigns the next available id."""

    name: str
    role: str = "concept"
    aliases: List[str] = Field(default_factory=list)


class LinkAgentOutput(BaseModel):
    """One per-VP LinkAgent call returns this in a single structured response.

    ``subject_id`` and ``object_id`` may be one of:
      - an existing canonical entity id (``"e3"``)
      - a placeholder ``"NEW:<exact name>"`` whose <exact name> matches an
        entry in ``proposed_entities``. The orchestrator rewrites these
        placeholders to real ids after entity expansion.
    """

    action: Literal["link", "skip"] = "link"
    skip_reason: str = ""
    subject_id: str = ""
    object_id: str = ""
    supporting_text: str = ""
    forward_question: str = ""
    reverse_question: str = ""
    proposed_entities: List[ProposedEntity] = Field(default_factory=list)


class Link(BaseModel):
    """A single binary relation + its forward/reverse QA pair, fully
    canonicalized. Both subject_id and object_id reference real entity ids by
    the time we build a Link (placeholders have been rewritten)."""

    link_id: str
    vp_id: str
    vp_label: str
    subject_id: str
    object_id: str
    supporting_text: str = ""
    forward_question: str = ""
    reverse_question: str = ""


class LinkVerifierOutput(BaseModel):
    """LinkVerifier per-link output. ``hint`` is fed back to a LinkAgent
    retry when ``valid`` is False."""

    valid: bool = True
    confidence: float = 1.0
    issue: str = ""
    hint: str = ""


# ---------------------------------------------------------------------------
# Batched schemas — the fan-out cost fix
# ---------------------------------------------------------------------------


class BatchedLinkItem(BaseModel):
    """One input-keyed entry in a batched LinkAgent response. The agent
    produces one of these per verb phrase in the batch; the orchestrator
    matches them back to the input by ``vp_id``."""

    vp_id: str
    output: LinkAgentOutput = Field(default_factory=LinkAgentOutput)


class BatchedLinkAgentOutput(BaseModel):
    items: List[BatchedLinkItem] = Field(default_factory=list)


class BatchedLinkVerifierItem(BaseModel):
    link_id: str
    verdict: LinkVerifierOutput = Field(default_factory=LinkVerifierOutput)


class BatchedLinkVerifierOutput(BaseModel):
    items: List[BatchedLinkVerifierItem] = Field(default_factory=list)
