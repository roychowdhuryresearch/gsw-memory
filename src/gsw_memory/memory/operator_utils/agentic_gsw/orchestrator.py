from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

from ...models import GSWStructure
from .assembler import assemble_gsw
from .coverage_agent import CoverageAgent
from .entity_refiner_agent import EntityRefinerAgent
from .entity_seed_agent import EntitySeedAgent
from .question_agent import QuestionAgent
from .relation_agent import RelationAgent
from .schemas import (
    AtomicFrame,
    CanonicalEntity,
    CoverageLedger,
    CoverageQAResult,
    MentionSpan,
    QuestionPairDraft,
    TextWindow,
    VerifierFailure,
)
from .transport_shim import TransportShim
from .verifier_agent import VerifierAgent
from .windowing import assign_primary_window_id, overlaps_window, split_into_windows


class AgenticGSWPipeline:
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        reasoning_effort: str = "medium",
        max_retries: int = 1,
        window_strategy: str = "paragraph",
        sem_relation: int = 8,
        sem_question: int = 16,
        sem_verifier: int = 3,
        sem_coverage: int = 8,
        enable_entity_coverage: bool = True,
        max_coverage_passes: int = 1,
        debug: bool = False,
        trace_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.max_retries = max(0, int(max_retries))
        self.window_strategy = window_strategy
        self.enable_entity_coverage = bool(enable_entity_coverage)
        self.max_coverage_passes = max(0, int(max_coverage_passes))
        self.debug = debug
        self.trace_dir = Path(trace_dir) if trace_dir else None

        transport = TransportShim(
            model_name,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
        )
        self.entity_seed_agent = EntitySeedAgent(transport)
        self.relation_agent = RelationAgent(transport)
        self.entity_refiner_agent = EntityRefinerAgent(transport)
        self.question_agent = QuestionAgent(transport)
        self.verifier_agent = VerifierAgent(transport)
        self.coverage_agent = CoverageAgent(transport)

        self._relation_semaphore = asyncio.Semaphore(max(1, int(sem_relation)))
        self._question_semaphore = asyncio.Semaphore(max(1, int(sem_question)))
        self._verifier_semaphore = asyncio.Semaphore(max(1, int(sem_verifier)))
        self._coverage_semaphore = asyncio.Semaphore(max(1, int(sem_coverage)))

    async def process_document_async(self, text: str) -> GSWStructure:
        gsw, _ = await self._process_document_state(text, doc_index=None)
        return gsw

    def process_document(self, text: str) -> GSWStructure:
        return asyncio.run(self.process_document_async(text))

    def process_documents(
        self,
        docs: List[str],
        *,
        on_document_done=None,
    ) -> List[Dict]:
        """Process a list of documents through the full pipeline.

        ``on_document_done(doc_idx, results_so_far)`` — optional sync callback
        fired after each document's assembly step. Use it to persist output
        incrementally so a mid-run crash doesn't lose completed docs. The
        ``results_so_far`` argument is the cumulative list of chunk dicts in
        the same shape returned at the end of the call.
        """

        async def _runner() -> List[Dict]:
            results: List[Dict] = []
            for doc_idx, doc in enumerate(docs):
                try:
                    gsw, trace_payload = await self._process_document_state(doc, doc_index=doc_idx)
                except Exception as exc:
                    logger.error(
                        "doc=%d pipeline_error error=%r — skipping this document; "
                        "other docs will continue",
                        doc_idx,
                        exc,
                    )
                    continue
                global_id = f"{doc_idx}_0"
                results.append(
                    {
                        global_id: {
                            "text": doc,
                            "doc_idx": doc_idx,
                            "chunk_idx": 0,
                            "global_id": global_id,
                            "start_sentence": 0,
                            "end_sentence": 0,
                            "gsw": gsw,
                            "context": "",
                            "spacetime": {},
                            "agentic_trace": trace_payload,
                        }
                    }
                )
                if on_document_done is not None:
                    try:
                        on_document_done(doc_idx, results)
                    except Exception as exc:
                        logger.warning(
                            "doc=%d on_document_done callback raised error=%r — "
                            "continuing with next document",
                            doc_idx,
                            exc,
                        )
            return results

        return asyncio.run(_runner())

    async def _process_document_state(
        self,
        text: str,
        *,
        doc_index: Optional[int],
    ) -> tuple[GSWStructure, dict]:
        doc_label = f"doc={doc_index if doc_index is not None else 0}"
        doc_start = time.time()
        logger.info("=" * 72)
        logger.info("%s PROCESS START chars=%d", doc_label, len(text))
        logger.info("=" * 72)

        windows = split_into_windows(text, strategy=self.window_strategy)
        self._log_step(doc_label, 0, "windowing", "OK", extra=f"windows={len(windows)} strategy={self.window_strategy}")

        # STEP 1/6 — entity seed
        mentions = await self._run_stage(
            doc_label,
            step_num=1,
            step_name="entity_seed",
            coro_factory=lambda: self.entity_seed_agent.run(text, windows),
            summary=lambda m: f"mentions={len(m)}",
        )

        # STEP 2/6 — atomic relation frames (fan-out, per-window isolation)
        relation_hints: dict[str, str] = {}
        frames, rel_errors = await self._run_stage(
            doc_label,
            step_num=2,
            step_name="relations",
            coro_factory=lambda: self._run_relations(windows, mentions, relation_hints),
            summary=lambda pair: f"frames={len(pair[0])} failed_windows={len(pair[1])} total_windows={len(windows)}",
        )
        if rel_errors:
            for window_id, err in rel_errors.items():
                logger.warning("%s   relation_window_error window=%s error=%s", doc_label, window_id, err)

        mentions, frames = self._promote_span_refs(text, mentions, frames, windows)

        # STEP 3/6 — entity refiner
        entities, mention_map = await self._run_stage(
            doc_label,
            step_num=3,
            step_name="entity_refiner",
            coro_factory=lambda: self.entity_refiner_agent.run(mentions, frames),
            summary=lambda pair: f"entities={len(pair[0])} mention_map={len(pair[1])}",
        )
        entities, mention_map = self._stabilize_entities(entities, mention_map, mentions)
        # Factual-convention invariant: every relation must be answerable.
        # This means every participant ref used by any frame MUST resolve to
        # a canonical entity. If the refiner missed a ref, we deterministically
        # create a lightweight canonical entity from that mention's surface
        # text so the invariant holds by construction — unanswerable relations
        # never exist in this pipeline.
        entities, mention_map, backfilled = self._backfill_canonicalization(
            entities, mention_map, mentions, frames
        )
        if backfilled:
            logger.warning(
                "%s   refiner_backfilled_entities count=%d — refiner missed "
                "mentions used by frame participants; pipeline created "
                "lightweight canonical entities to preserve answerability",
                doc_label,
                backfilled,
            )
        logger.info(
            "%s   refiner_stabilized entities=%d mention_map=%d",
            doc_label,
            len(entities),
            len(mention_map),
        )

        # STEP 4/6 — question pairs (fan-out, per-frame isolation)
        question_pairs, q_errors = await self._run_stage(
            doc_label,
            step_num=4,
            step_name="questions",
            coro_factory=lambda: self._run_questions(
                text=text,
                frames=frames,
                mentions=mentions,
                entities=entities,
                mention_map=mention_map,
                question_hints={},
            ),
            summary=lambda pair: f"pairs={len(pair[0])} failed_frames={len(pair[1])} total_frames={len(frames)}",
        )
        if q_errors:
            for frame_id, err in q_errors.items():
                logger.warning("%s   question_frame_error frame=%s error=%s", doc_label, frame_id, err)

        # STEP 5/6 — verifier
        verifier_report = await self._run_stage(
            doc_label,
            step_num=5,
            step_name="verifier",
            coro_factory=lambda: self.verifier_agent.run_all(
                source_text=text,
                windows=windows,
                mentions=mentions,
                entities=entities,
                frames=frames,
                question_pairs=question_pairs,
                semaphore=self._verifier_semaphore,
            ),
            summary=lambda report: f"failures={len(report.failures)}",
        )

        retry_counts: dict[tuple[str, str], int] = {}
        if verifier_report.failures and self.max_retries > 0:
            logger.info(
                "%s repair_start failures=%d stages=%s",
                doc_label,
                len(verifier_report.failures),
                sorted({f.producer_stage for f in verifier_report.failures}),
            )
            mentions, frames, entities, mention_map, question_pairs = await self._repair_once(
                text=text,
                windows=windows,
                mentions=mentions,
                frames=frames,
                entities=entities,
                mention_map=mention_map,
                question_pairs=question_pairs,
                failures=verifier_report.failures,
                retry_counts=retry_counts,
            )
            verifier_report = await self.verifier_agent.run_all(
                source_text=text,
                windows=windows,
                mentions=mentions,
                entities=entities,
                frames=frames,
                question_pairs=question_pairs,
                semaphore=self._verifier_semaphore,
            )

        # STEP 5b/6 — coverage repair (deterministic orphan detection +
        # targeted CoverageAgent calls per orphan entity). Only runs if
        # the user hasn't disabled it via --disable-entity-coverage.
        coverage_ledger = self._compute_coverage_ledger(entities, frames, question_pairs)
        if self.enable_entity_coverage and self.max_coverage_passes > 0:
            (
                frames,
                question_pairs,
                coverage_ledger,
            ) = await self._run_coverage_repair(
                doc_label=doc_label,
                text=text,
                windows=windows,
                mentions=mentions,
                entities=entities,
                mention_map=mention_map,
                frames=frames,
                question_pairs=question_pairs,
                initial_ledger=coverage_ledger,
            )

        # STEP 6/6 — deterministic assembly
        chunk_id = f"{doc_index if doc_index is not None else 0}_0"
        try:
            t0 = time.time()
            gsw = assemble_gsw(
                entities=entities,
                mention_map=mention_map,
                frames=frames,
                question_pairs=question_pairs,
                chunk_id=chunk_id,
            )
            gsw = GSWStructure.model_validate(gsw.model_dump(mode="json"))
            self._log_step(
                doc_label,
                6,
                "assemble",
                "OK",
                extra=f"entity_nodes={len(gsw.entity_nodes)} vp_nodes={len(gsw.verb_phrase_nodes)} duration={time.time() - t0:.2f}s",
            )
        except Exception as exc:
            self._log_step(doc_label, 6, "assemble", "FAIL", extra=f"error={exc!r}")
            raise

        trace_payload = {
            "windows": [window.model_dump(mode="json") for window in windows],
            "mentions": [mention.model_dump(mode="json") for mention in mentions],
            "frames": [frame.model_dump(mode="json") for frame in frames],
            "entities": [entity.model_dump(mode="json") for entity in entities],
            "mention_map": dict(mention_map),
            "question_pairs": {
                frame_id: pair.model_dump(mode="json") for frame_id, pair in question_pairs.items()
            },
            "verifier_failures": [failure.model_dump(mode="json") for failure in verifier_report.failures],
            "coverage_ledger": coverage_ledger.model_dump(mode="json"),
        }
        self._write_trace(doc_index, trace_payload)
        logger.info(
            "%s process_done total_duration=%.2fs entities=%d frames=%d question_pairs=%d failures=%d",
            doc_label,
            time.time() - doc_start,
            len(entities),
            len(frames),
            len(question_pairs),
            len(verifier_report.failures),
        )
        return gsw, trace_payload

    @staticmethod
    def _log_step(doc_label: str, step_num: int, step_name: str, status: str, *, extra: str = "") -> None:
        marker = "✓" if status == "OK" else ("✗" if status == "FAIL" else "•")
        logger.info(
            "%s %s STEP %d/6 %-15s %s%s",
            doc_label,
            marker,
            step_num,
            step_name,
            status,
            f" | {extra}" if extra else "",
        )

    @staticmethod
    def _backfill_canonicalization(
        entities: Sequence[CanonicalEntity],
        mention_map: dict[str, str],
        mentions: Sequence[MentionSpan],
        frames: Sequence[AtomicFrame],
    ) -> tuple[list[CanonicalEntity], dict[str, str], int]:
        """Guarantee that every mention ref used by any frame participant
        resolves to a canonical entity in mention_map.

        If the refiner left any participant ref uncanonicalized, this helper
        creates a lightweight CanonicalEntity from the corresponding
        MentionSpan's surface text, assigns it a fresh ``eN`` id, and adds
        the ref → id link to mention_map. No refiner LLM call needed —
        deterministic so the factual-convention invariant "every relation is
        answerable" holds unconditionally by the time Stage 4 runs.
        """
        entity_list = [entity.model_copy(deep=True) for entity in entities]
        mention_map = dict(mention_map)
        mention_by_ref = {mention.ref: mention for mention in mentions}
        existing_ids = {entity.id for entity in entity_list}
        name_to_id = {entity.name.casefold(): entity.id for entity in entity_list}

        # Start numbering after the highest existing entity id.
        next_idx = 1
        for entity in entity_list:
            match = re.match(r"e(\d+)$", entity.id or "")
            if match:
                next_idx = max(next_idx, int(match.group(1)) + 1)

        backfilled = 0
        for frame in frames:
            for participant in (frame.subject, frame.object):
                ref = str(participant.ref or "").strip()
                if not ref:
                    continue
                if ref in mention_map and mention_map[ref] in existing_ids:
                    continue

                mention = mention_by_ref.get(ref)
                if mention is None:
                    # The ref doesn't correspond to any known mention — skip;
                    # this is a much deeper bug (relation agent hallucinated
                    # a ref not in the catalog). Fall through and let the
                    # question/verifier stages surface it.
                    continue

                surface = mention.text.strip()
                if not surface:
                    continue

                canonical_key = surface.casefold()
                if canonical_key in name_to_id:
                    # A canonical entity with this surface already exists —
                    # just link the orphan mention to it.
                    mention_map[ref] = name_to_id[canonical_key]
                    continue

                new_id = f"e{next_idx}"
                next_idx += 1
                aliases = sorted(
                    {alias.strip() for alias in mention.aliases if alias and alias.strip() != surface}
                )
                entity_list.append(
                    CanonicalEntity(
                        id=new_id,
                        name=surface,
                        aliases=aliases,
                        roles=[],
                        source_mentions=[ref],
                    )
                )
                existing_ids.add(new_id)
                name_to_id[canonical_key] = new_id
                mention_map[ref] = new_id
                backfilled += 1

        return entity_list, mention_map, backfilled

    async def _run_stage(
        self,
        doc_label: str,
        *,
        step_num: int,
        step_name: str,
        coro_factory,
        summary,
    ):
        t0 = time.time()
        try:
            result = await coro_factory()
        except Exception as exc:
            self._log_step(
                doc_label,
                step_num,
                step_name,
                "FAIL",
                extra=f"duration={time.time() - t0:.2f}s error={exc!r}",
            )
            raise
        extra = f"{summary(result)} duration={time.time() - t0:.2f}s"
        self._log_step(doc_label, step_num, step_name, "OK", extra=extra)
        return result

    async def _run_relations(
        self,
        windows: Sequence[TextWindow],
        mentions: Sequence[MentionSpan],
        hints_by_window: dict[str, str],
    ) -> tuple[list[AtomicFrame], dict[str, str]]:
        tasks = []
        ordered_window_ids: list[str] = []
        for window in windows:
            mentions_in_window = [
                mention
                for mention in mentions
                if overlaps_window(mention.char_start, mention.char_end, window)
            ]
            tasks.append(
                self.relation_agent.run(
                    window,
                    mentions_in_window,
                    hint=hints_by_window.get(window.id, ""),
                    semaphore=self._relation_semaphore,
                )
            )
            ordered_window_ids.append(window.id)
        per_window = (
            await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        )
        frames: list[AtomicFrame] = []
        errors: dict[str, str] = {}
        for window_id, result in zip(ordered_window_ids, per_window):
            if isinstance(result, Exception):
                errors[window_id] = repr(result)
                continue
            frames.extend(result)
        return frames, errors

    async def _run_questions(
        self,
        *,
        text: str,
        frames: Sequence[AtomicFrame],
        mentions: Sequence[MentionSpan],
        entities: Sequence[CanonicalEntity],
        mention_map: dict[str, str],
        question_hints: dict[str, str],
        only_frame_ids: Optional[set[str]] = None,
        existing_pairs: Optional[dict[str, QuestionPairDraft]] = None,
    ) -> tuple[dict[str, QuestionPairDraft], dict[str, str]]:
        entity_by_id = {entity.id: entity for entity in entities}
        mention_by_ref = {mention.ref: mention for mention in mentions}
        pairs = dict(existing_pairs or {})
        tasks = []
        task_frame_ids: list[str] = []
        for frame in frames:
            if only_frame_ids is not None and frame.frame_id not in only_frame_ids:
                continue
            support_entities = self._entities_for_frame_scope(frame, mention_by_ref, mention_map, entity_by_id)
            # Factual-convention guarantee: the frame's subject and object
            # refs are canonicalized (orchestrator back-fills any gap before
            # this stage), so these fallback ids are always resolvable. The
            # question agent uses them as a last-resort fallback when the
            # model's own answer drifts from the scope list.
            forward_fallback = mention_map.get(frame.object.ref, "") or ""
            reverse_fallback = mention_map.get(frame.subject.ref, "") or ""
            tasks.append(
                self.question_agent.run(
                    frame,
                    support_entities,
                    supporting_text=text[frame.support_char_start:frame.support_char_end] or frame.supporting_text,
                    forward_fallback_id=forward_fallback,
                    reverse_fallback_id=reverse_fallback,
                    hint=question_hints.get(frame.frame_id, ""),
                    semaphore=self._question_semaphore,
                )
            )
            task_frame_ids.append(frame.frame_id)
        errors: dict[str, str] = {}
        if not tasks:
            return pairs, errors
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for frame_id, pair in zip(task_frame_ids, results):
            if isinstance(pair, Exception):
                errors[frame_id] = repr(pair)
                continue
            pairs[frame_id] = pair
        return pairs, errors

    async def _repair_once(
        self,
        *,
        text: str,
        windows: Sequence[TextWindow],
        mentions: list[MentionSpan],
        frames: list[AtomicFrame],
        entities: list[CanonicalEntity],
        mention_map: dict[str, str],
        question_pairs: dict[str, QuestionPairDraft],
        failures: Sequence[VerifierFailure],
        retry_counts: dict[tuple[str, str], int],
    ) -> tuple[list[MentionSpan], list[AtomicFrame], list[CanonicalEntity], dict[str, str], dict[str, QuestionPairDraft]]:
        relation_hints: dict[str, str] = {}
        question_hints: dict[str, str] = {}
        rerun_entity_refiner_hints: list[str] = []
        affected_windows: set[str] = set()
        affected_frames: set[str] = set()

        for failure in failures:
            key = (failure.producer_stage, failure.item_id)
            if retry_counts.get(key, 0) >= self.max_retries:
                continue
            retry_counts[key] = retry_counts.get(key, 0) + 1
            if failure.producer_stage == "entity_seed":
                # entity_seed failures are handled by Stage 5b (CoverageAgent).
                # We do NOT re-run the entity_seed → relations → refiner →
                # questions cascade here — that was the source of silent
                # regressions. Coverage repair creates new frames + QA pairs
                # directly for orphan entities instead.
                continue
            elif failure.producer_stage == "relation":
                window_id = failure.window_id or failure.item_id
                affected_windows.add(window_id)
                relation_hints[window_id] = self._merge_hint(relation_hints.get(window_id), failure.hint)
            elif failure.producer_stage == "entity_refiner":
                rerun_entity_refiner_hints.append(failure.hint)
            elif failure.producer_stage == "question":
                affected_frames.add(failure.item_id)
                question_hints[failure.item_id] = self._merge_hint(question_hints.get(failure.item_id), failure.hint)


        if affected_windows:
            window_map = {window.id: window for window in windows}
            repaired_frames_by_window: dict[str, list[AtomicFrame]] = {}
            relation_tasks = []
            relation_window_ids = []
            for window_id in sorted(affected_windows):
                window = window_map.get(window_id)
                if window is None:
                    continue
                mentions_in_window = [
                    mention
                    for mention in mentions
                    if overlaps_window(mention.char_start, mention.char_end, window)
                ]
                relation_tasks.append(
                    self.relation_agent.run(
                        window,
                        mentions_in_window,
                        hint=relation_hints.get(window_id, ""),
                        semaphore=self._relation_semaphore,
                    )
                )
                relation_window_ids.append(window_id)
            repaired_batches = (
                await asyncio.gather(*relation_tasks, return_exceptions=True)
                if relation_tasks
                else []
            )
            repaired_frames_by_window: dict[str, list[AtomicFrame]] = {}
            for window_id, batch in zip(relation_window_ids, repaired_batches):
                if isinstance(batch, Exception):
                    logger.warning(
                        "repair: relation retry for window=%s failed error=%r",
                        window_id,
                        batch,
                    )
                    repaired_frames_by_window[window_id] = []
                    continue
                repaired_frames_by_window[window_id] = batch
            frames = [
                frame for frame in frames if frame.window_id not in repaired_frames_by_window
            ]
            for window_id in relation_window_ids:
                frames.extend(repaired_frames_by_window.get(window_id, []))
            frames.sort(key=lambda item: (item.window_id, item.frame_id))
            mentions, frames = self._promote_span_refs(text, mentions, frames, windows)
            previous_entities = list(entities)
            entities, mention_map = await self.entity_refiner_agent.run(
                mentions,
                frames,
                hint=" ".join(rerun_entity_refiner_hints),
            )
            entities, mention_map = self._stabilize_entities(
                entities,
                mention_map,
                mentions,
                previous_entities=previous_entities,
            )
            affected_frames.update(
                frame.frame_id for frame in frames if frame.window_id in affected_windows
            )

        if rerun_entity_refiner_hints and not affected_windows:
            previous_entities = list(entities)
            entities, mention_map = await self.entity_refiner_agent.run(
                mentions,
                frames,
                hint=" ".join(rerun_entity_refiner_hints),
            )
            entities, mention_map = self._stabilize_entities(
                entities,
                mention_map,
                mentions,
                previous_entities=previous_entities,
            )
            affected_frames.update(frame.frame_id for frame in frames)

        if affected_frames or question_hints:
            question_pairs, q_errors = await self._run_questions(
                text=text,
                frames=frames,
                mentions=mentions,
                entities=entities,
                mention_map=mention_map,
                question_hints=question_hints,
                only_frame_ids=affected_frames or set(question_hints),
                existing_pairs=question_pairs,
            )
            if q_errors:
                logger.warning(
                    "repair: question retry had %d frame failures", len(q_errors)
                )

        return mentions, frames, entities, mention_map, question_pairs

    @staticmethod
    def _merge_hint(current: Optional[str], new_hint: str) -> str:
        merged = " ".join(part for part in [current or "", new_hint or ""] if part).strip()
        return merged

    def _promote_span_refs(
        self,
        text: str,
        mentions: Sequence[MentionSpan],
        frames: Sequence[AtomicFrame],
        windows: Sequence[TextWindow],
    ) -> tuple[list[MentionSpan], list[AtomicFrame]]:
        mention_list = [mention.model_copy(deep=True) for mention in mentions]
        mention_by_ref = {mention.ref: mention for mention in mention_list}
        next_index = self._next_mention_index(mention_list)

        def _existing_ref_for_span(char_start: int, char_end: int, surface: str) -> Optional[str]:
            for mention in mention_list:
                if mention.char_start == char_start and mention.char_end == char_end:
                    return mention.ref
                if mention.text.casefold() == surface.casefold() and mention.char_start == char_start:
                    return mention.ref
            return None

        rewritten_frames: list[AtomicFrame] = []
        for frame in frames:
            new_frame = frame.model_copy(deep=True)
            for participant_name in ("subject", "object"):
                participant = getattr(new_frame, participant_name)
                ref = str(participant.ref or "").strip()
                if ref in mention_by_ref:
                    continue
                span = self._resolve_span_reference(
                    ref=ref,
                    frame=new_frame,
                    text=text,
                )
                if span is None:
                    continue
                char_start, char_end = span
                if char_end <= char_start:
                    continue
                surface = text[char_start:char_end].strip()
                if not surface:
                    continue
                existing_ref = _existing_ref_for_span(char_start, char_end, surface)
                if existing_ref is not None:
                    participant.ref = existing_ref
                    continue
                new_ref = f"m{next_index}"
                next_index += 1
                mention = MentionSpan(
                    ref=new_ref,
                    text=surface,
                    aliases=[],
                    char_start=char_start,
                    char_end=char_end,
                    window_id=assign_primary_window_id(char_start, windows),
                )
                mention_list.append(mention)
                mention_by_ref[new_ref] = mention
                participant.ref = new_ref
            rewritten_frames.append(new_frame)

        mention_list.sort(key=lambda item: (item.char_start, item.char_end, item.ref))
        return mention_list, rewritten_frames

    @staticmethod
    def _resolve_span_reference(
        *,
        ref: str,
        frame: AtomicFrame,
        text: str,
    ) -> Optional[tuple[int, int]]:
        match = re.match(r"span:(\d+):(\d+)$", ref)
        if match:
            return int(match.group(1)), int(match.group(2))

        if ref.startswith("span_text:"):
            needle = ref.split(":", 1)[1].strip()
            if not needle:
                return None
            support_slice = text[frame.support_char_start:frame.support_char_end]
            relative_index = support_slice.find(needle)
            if relative_index >= 0:
                char_start = frame.support_char_start + relative_index
                return char_start, char_start + len(needle)
            absolute_index = text.find(needle)
            if absolute_index >= 0:
                return absolute_index, absolute_index + len(needle)
        return None

    @staticmethod
    def _next_mention_index(mentions: Sequence[MentionSpan]) -> int:
        max_idx = 0
        for mention in mentions:
            match = re.match(r"m(\d+)$", mention.ref)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        return max_idx + 1

    @staticmethod
    def _stabilize_entities(
        entities: Sequence[CanonicalEntity],
        mention_map: dict[str, str],
        mentions: Sequence[MentionSpan],
        previous_entities: Optional[Sequence[CanonicalEntity]] = None,
    ) -> tuple[list[CanonicalEntity], dict[str, str]]:
        mention_by_ref = {mention.ref: mention for mention in mentions}
        previous_entities = list(previous_entities or [])
        old_by_id = {entity.id: entity for entity in previous_entities if entity.id}
        used_old_ids: set[str] = set()
        next_idx = 1
        for entity in previous_entities:
            match = re.match(r"e(\d+)$", entity.id)
            if match:
                next_idx = max(next_idx, int(match.group(1)) + 1)

        normalized = sorted(
            entities,
            key=lambda entity: (
                min((mention_by_ref[ref].char_start for ref in entity.source_mentions if ref in mention_by_ref), default=10**9),
                entity.name.casefold(),
            ),
        )
        stabilized: list[CanonicalEntity] = []
        for entity in normalized:
            matched_old_id = AgenticGSWPipeline._match_previous_entity_id(entity, old_by_id, used_old_ids)
            if matched_old_id:
                entity.id = matched_old_id
                used_old_ids.add(matched_old_id)
            elif not re.match(r"e\d+$", entity.id or ""):
                entity.id = f"e{next_idx}"
                next_idx += 1
            stabilized.append(entity.model_copy(deep=True))

        valid_ids = {entity.id for entity in stabilized}
        cleaned_map: dict[str, str] = {}
        for mention_ref, entity_id in mention_map.items():
            mapped_id = entity_id
            if entity_id not in valid_ids:
                entity_name_map = {
                    candidate.name.casefold(): candidate.id for candidate in stabilized
                }
                mapped_id = entity_name_map.get(str(entity_id).casefold(), entity_id)
            if mapped_id in valid_ids:
                cleaned_map[mention_ref] = mapped_id
        for entity in stabilized:
            for mention_ref in entity.source_mentions:
                if mention_ref not in cleaned_map and mention_ref in mention_by_ref:
                    cleaned_map[mention_ref] = entity.id
        return stabilized, cleaned_map

    @staticmethod
    def _match_previous_entity_id(
        entity: CanonicalEntity,
        previous_by_id: dict[str, CanonicalEntity],
        used_old_ids: set[str],
    ) -> Optional[str]:
        wanted_mentions = set(entity.source_mentions)
        wanted_names = {entity.name.casefold(), *(alias.casefold() for alias in entity.aliases)}
        for old_id, old_entity in previous_by_id.items():
            if old_id in used_old_ids:
                continue
            if wanted_mentions and wanted_mentions.intersection(old_entity.source_mentions):
                return old_id
            old_names = {old_entity.name.casefold(), *(alias.casefold() for alias in old_entity.aliases)}
            if wanted_names.intersection(old_names):
                return old_id
        return None

    @staticmethod
    def _entities_for_frame_scope(
        frame: AtomicFrame,
        mention_by_ref: dict[str, MentionSpan],
        mention_map: dict[str, str],
        entity_by_id: dict[str, CanonicalEntity],
    ) -> list[CanonicalEntity]:
        entity_ids: set[str] = set()
        for participant_ref in [frame.subject.ref, frame.object.ref]:
            entity_id = mention_map.get(participant_ref)
            if entity_id:
                entity_ids.add(entity_id)
        for mention in mention_by_ref.values():
            if mention.char_start < frame.support_char_end and mention.char_end > frame.support_char_start:
                entity_id = mention_map.get(mention.ref)
                if entity_id:
                    entity_ids.add(entity_id)
        return [entity_by_id[entity_id] for entity_id in sorted(entity_ids) if entity_id in entity_by_id]

    # ------------------------------------------------------------------
    # Coverage repair (Stage 5b)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_coverage_ledger(
        entities: Sequence[CanonicalEntity],
        frames: Sequence[AtomicFrame],
        question_pairs: dict[str, QuestionPairDraft],
    ) -> CoverageLedger:
        """Deterministic coverage check. An entity is covered iff its id
        appears in at least one question answer (forward or reverse) in any
        existing pair. A frame is covered iff its frame_id has a question
        pair at all."""
        entity_ids = {entity.id for entity in entities}
        referenced: set[str] = set()
        for pair in question_pairs.values():
            for answer in pair.forward.answers:
                if answer in entity_ids:
                    referenced.add(answer)
            for answer in pair.reverse.answers:
                if answer in entity_ids:
                    referenced.add(answer)
        covered_entities = sorted(referenced)
        uncovered_entities = sorted(entity_ids - referenced)
        covered_frames = sorted(frame.frame_id for frame in frames if frame.frame_id in question_pairs)
        uncovered_frames = sorted(frame.frame_id for frame in frames if frame.frame_id not in question_pairs)
        return CoverageLedger(
            covered_entity_ids=covered_entities,
            uncovered_entity_ids=uncovered_entities,
            covered_frame_ids=covered_frames,
            uncovered_frame_ids=uncovered_frames,
        )

    async def _run_coverage_repair(
        self,
        *,
        doc_label: str,
        text: str,
        windows: Sequence[TextWindow],
        mentions: Sequence[MentionSpan],
        entities: Sequence[CanonicalEntity],
        mention_map: dict[str, str],
        frames: list[AtomicFrame],
        question_pairs: dict[str, QuestionPairDraft],
        initial_ledger: CoverageLedger,
    ) -> tuple[list[AtomicFrame], dict[str, QuestionPairDraft], CoverageLedger]:
        """Run up to ``self.max_coverage_passes`` coverage-repair passes.
        Each pass: fan out one CoverageAgent call per orphan entity (in
        parallel, capped by ``self._coverage_semaphore``), validate that
        returned pairs reference the target entity, and merge new frames
        + pairs into the running state. Rebuild the ledger after every
        pass; stop early when there are no orphans."""

        entity_by_id = {entity.id: entity for entity in entities}
        mention_by_ref = {mention.ref: mention for mention in mentions}
        valid_entity_ids = set(entity_by_id)

        # Reverse index: entity_id -> list of mention refs that map to it.
        entity_to_mentions: dict[str, list[str]] = {}
        for mention_ref, entity_id in mention_map.items():
            entity_to_mentions.setdefault(entity_id, []).append(mention_ref)

        ledger = initial_ledger
        for pass_idx in range(1, self.max_coverage_passes + 1):
            orphans = [entity_by_id[eid] for eid in ledger.uncovered_entity_ids if eid in entity_by_id]
            if not orphans:
                logger.info(
                    "%s   coverage_repair_pass=%d no orphans — skipping",
                    doc_label,
                    pass_idx,
                )
                break

            t0 = time.time()
            tasks = []
            task_targets: list[CanonicalEntity] = []
            for entity in orphans:
                target_mentions, local_window, nearby_entities = self._coverage_context_for_entity(
                    target_entity=entity,
                    entity_to_mentions=entity_to_mentions,
                    mention_by_ref=mention_by_ref,
                    mention_map=mention_map,
                    entity_by_id=entity_by_id,
                    windows=windows,
                )
                if local_window is None:
                    # No mention for this entity — can't ground it. Skip.
                    logger.debug(
                        "%s   coverage_skip_no_window entity=%s",
                        doc_label,
                        entity.id,
                    )
                    continue
                existing_local_frames = [
                    frame
                    for frame in frames
                    if frame.window_id == local_window.id
                ]
                tasks.append(
                    self.coverage_agent.run(
                        target_entity=entity,
                        target_mentions=target_mentions,
                        available_mentions=[
                            m for m in mentions
                            if m.window_id == local_window.id
                        ],
                        local_window=local_window,
                        nearby_entities=nearby_entities,
                        existing_frames=existing_local_frames,
                        semaphore=self._coverage_semaphore,
                    )
                )
                task_targets.append(entity)

            results: list[CoverageQAResult] = []
            if tasks:
                gathered = await asyncio.gather(*tasks, return_exceptions=True)
                for target, result in zip(task_targets, gathered):
                    if isinstance(result, Exception):
                        logger.warning(
                            "%s   coverage_call_error entity=%s error=%r",
                            doc_label,
                            target.id,
                            result,
                        )
                        continue
                    results.append(result)

            filled = 0
            skipped = 0
            for target, result in zip(task_targets, results):
                added_any = False
                for frame, pair in zip(result.frames, result.question_pairs):
                    # Reject items that don't actually cover the target entity.
                    fwd = [a for a in pair.forward.answers if a in valid_entity_ids]
                    rev = [a for a in pair.reverse.answers if a in valid_entity_ids]
                    if target.id not in fwd and target.id not in rev:
                        skipped += 1
                        logger.debug(
                            "%s   coverage_item_rejected entity=%s frame=%s "
                            "reason=target_not_in_answers",
                            doc_label,
                            target.id,
                            frame.frame_id,
                        )
                        continue
                    pair_copy = pair.model_copy(deep=True)
                    pair_copy.forward.answers = fwd or [target.id]
                    pair_copy.reverse.answers = rev or [target.id]
                    frames.append(frame)
                    question_pairs[frame.frame_id] = pair_copy
                    added_any = True
                    filled += 1
                if not added_any:
                    skipped += 1

            ledger = self._compute_coverage_ledger(entities, frames, question_pairs)
            orphans_after = len(ledger.uncovered_entity_ids)
            self._log_step(
                doc_label,
                step_num=5,
                step_name=f"coverage_repair_{pass_idx}",
                status="OK",
                extra=(
                    f"orphans_before={len(orphans)} filled={filled} "
                    f"skipped={skipped} orphans_after={orphans_after} "
                    f"duration={time.time() - t0:.2f}s"
                ),
            )
            if orphans_after == 0:
                break

        return frames, question_pairs, ledger

    @staticmethod
    def _coverage_context_for_entity(
        *,
        target_entity: CanonicalEntity,
        entity_to_mentions: dict[str, list[str]],
        mention_by_ref: dict[str, MentionSpan],
        mention_map: dict[str, str],
        entity_by_id: dict[str, CanonicalEntity],
        windows: Sequence[TextWindow],
    ) -> tuple[list[MentionSpan], Optional[TextWindow], list[CanonicalEntity]]:
        """Return (target_mentions, local_window, nearby_entities) for a
        coverage-repair call. Picks the earliest mention of the target entity
        and finds the window containing it. Nearby entities are all canonical
        entities that have at least one mention inside the same window."""
        mention_refs = entity_to_mentions.get(target_entity.id, []) + list(target_entity.source_mentions)
        # Deduplicate while preserving order.
        seen_refs: set[str] = set()
        ordered_refs: list[str] = []
        for ref in mention_refs:
            if ref in seen_refs:
                continue
            seen_refs.add(ref)
            ordered_refs.append(ref)

        target_mentions = [mention_by_ref[ref] for ref in ordered_refs if ref in mention_by_ref]
        if not target_mentions:
            return [], None, []
        target_mentions.sort(key=lambda m: (m.char_start, m.char_end))
        anchor = target_mentions[0]

        # Find the window containing the anchor mention.
        local_window: Optional[TextWindow] = None
        for window in windows:
            if window.char_start <= anchor.char_start < window.char_end:
                local_window = window
                break
        if local_window is None and windows:
            local_window = windows[0]
        if local_window is None:
            return target_mentions, None, []

        # Nearby entities = any canonical entity with ≥1 mention in this window.
        nearby_ids: set[str] = set()
        for mention in mention_by_ref.values():
            if mention.char_start >= local_window.char_start and mention.char_end <= local_window.char_end:
                entity_id = mention_map.get(mention.ref)
                if entity_id and entity_id in entity_by_id and entity_id != target_entity.id:
                    nearby_ids.add(entity_id)
        nearby_entities = [entity_by_id[eid] for eid in sorted(nearby_ids)]

        return target_mentions, local_window, nearby_entities

    def _write_trace(self, doc_index: Optional[int], payload: dict) -> None:
        if not self.debug or self.trace_dir is None:
            return
        doc_label = f"doc_{doc_index if doc_index is not None else 0}"
        doc_dir = self.trace_dir / doc_label
        doc_dir.mkdir(parents=True, exist_ok=True)
        trace_path = doc_dir / "trace.json"
        trace_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
