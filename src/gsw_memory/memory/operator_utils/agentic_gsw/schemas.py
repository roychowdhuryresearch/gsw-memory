from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from ...models import Role


class TextWindow(BaseModel):
    id: str
    text: str
    char_start: int
    char_end: int
    core_char_start: int
    core_char_end: int


class MentionSpan(BaseModel):
    ref: str = ""
    text: str
    aliases: List[str] = Field(default_factory=list)
    char_start: int
    char_end: int
    window_id: Optional[str] = None


class ProvisionalParticipant(BaseModel):
    role: str = "other"
    ref: str = ""


class AtomicFrame(BaseModel):
    frame_id: str = ""
    vp_phrase: str = ""
    subject: ProvisionalParticipant = Field(default_factory=ProvisionalParticipant)
    object: ProvisionalParticipant = Field(default_factory=ProvisionalParticipant)
    slot_kind: Literal[
        "recipient",
        "content",
        "time",
        "location",
        "agent",
        "patient",
        "manner",
        "purpose",
        "other",
    ] = "other"
    window_id: str = ""
    supporting_text: str = ""
    support_char_start: int = 0
    support_char_end: int = 0


class CanonicalEntity(BaseModel):
    id: str = ""
    name: str
    aliases: List[str] = Field(default_factory=list)
    roles: List[Role] = Field(default_factory=list)
    source_mentions: List[str] = Field(default_factory=list)


class QuestionDraft(BaseModel):
    text: str
    answers: List[str] = Field(default_factory=list)


class QuestionPairDraft(BaseModel):
    frame_id: str
    forward: QuestionDraft
    reverse: QuestionDraft


class VerifierFailure(BaseModel):
    # All routing fields are optional so a single malformed item returned by the
    # model cannot invalidate the rest of the batch. Post-parse filtering in
    # VerifierAgent.run_all drops items that lack item_id or producer_stage
    # (un-routable) and logs a WARNING — this preserves the other failures.
    check: Optional[Literal["coverage", "grounding", "qa_completeness"]] = None
    item_type: Optional[Literal["window", "mention", "frame", "entity", "question_pair"]] = None
    item_id: Optional[str] = None
    producer_stage: Optional[Literal["entity_seed", "relation", "entity_refiner", "question"]] = None
    window_id: Optional[str] = None
    hint: str = ""


class VerifierReport(BaseModel):
    failures: List[VerifierFailure] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


class MentionSeed(BaseModel):
    """Model-emitted mention. The LLM only provides surface text + aliases;
    absolute char offsets are computed in Python by searching the document.
    This decouples the hardest constraint (exact offsets) from the easiest
    judgment (is this a mention?)."""

    text: str
    aliases: List[str] = Field(default_factory=list)


class EntitySeedOutput(BaseModel):
    mentions: List[MentionSeed] = Field(default_factory=list)


class RelationFrameOutput(BaseModel):
    frames: List[AtomicFrame] = Field(default_factory=list)


class EntityRefinerOutput(BaseModel):
    entities: List[CanonicalEntity] = Field(default_factory=list)
    mention_map: Dict[str, str] = Field(default_factory=dict)


class QuestionPairOutput(BaseModel):
    pair: QuestionPairDraft


class VerifierOutput(BaseModel):
    failures: List[VerifierFailure] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Coverage repair (Run N+4 — Phase 1)
# ---------------------------------------------------------------------------


class CoverageLedger(BaseModel):
    """Deterministic snapshot of which entities/frames are wired into the
    question graph. Computed in Python after each pipeline pass — never
    asked from an LLM."""

    covered_entity_ids: List[str] = Field(default_factory=list)
    uncovered_entity_ids: List[str] = Field(default_factory=list)
    covered_frame_ids: List[str] = Field(default_factory=list)
    uncovered_frame_ids: List[str] = Field(default_factory=list)


class CoverageFrameQuestionDraft(BaseModel):
    """One (frame + question pair) produced by the CoverageAgent when
    repairing an orphan entity."""

    frame: AtomicFrame = Field(default_factory=AtomicFrame)
    pair: QuestionPairDraft = Field(default_factory=QuestionPairDraft)


class CoverageQAOutput(BaseModel):
    """Raw CoverageAgent model output. `items` may be empty and
    `skipped_reason` populated when no grounded relation exists locally."""

    items: List[CoverageFrameQuestionDraft] = Field(default_factory=list)
    skipped_reason: str = ""


class CoverageQAResult(BaseModel):
    """Normalized CoverageAgent output consumed by the orchestrator."""

    frames: List[AtomicFrame] = Field(default_factory=list)
    question_pairs: List[QuestionPairDraft] = Field(default_factory=list)
    skipped_reason: str = ""
