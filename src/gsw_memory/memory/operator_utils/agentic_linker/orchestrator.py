"""LinkerPipeline — fan-out agentic GSW orchestrator.

Stages:
  STEP 1: EntityListAgent          (1 LLM call, parallel with STEP 2)
  STEP 2: VerbPhraseListAgent      (1 LLM call, parallel with STEP 1)
  STEP 3: LinkAgent fan-out        (1 LLM call per VP, parallel)
  STEP 3b: entity expansion        (Python — dedupe + assign new ids)
  STEP 3c: rewrite NEW: placeholders → canonical ids in pending links
  STEP 4: LinkVerifier fan-out     (1 LLM call per link, parallel)
  STEP 4b: LinkAgent retries       (rejected links, up to max_retries)
  STEP 5: orphan detection (Python)
  STEP 6: FillerAgent fan-out      (1 LLM call per orphan, parallel)
  STEP 7: deterministic assembly into GSWStructure
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from ...models import GSWStructure
from ..agentic_gsw.transport_shim import TransportShim
from .assembler import assemble_gsw
from .entity_list_agent import EntityListAgent
from .filler_agent import FillerAgent
from .link_agent import LinkAgent
from .link_verifier_agent import LinkVerifierAgent
from .pricing import estimate_cost_usd
from .schemas import (
    CanonicalEntity,
    Link,
    LinkAgentOutput,
    LinkVerifierOutput,
    ProposedEntity,
    VerbPhraseSlot,
)
from .verb_phrase_list_agent import VerbPhraseListAgent

logger = logging.getLogger(__name__)


_NEW_PREFIX = "NEW:"


class LinkerPipeline:
    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        reasoning_effort: str = "medium",
        sem_link: int = 8,
        link_batch_size: int = 8,
        max_link_retries: int = 2,
        debug: bool = False,
        trace_dir: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.link_batch_size = max(1, int(link_batch_size))
        self.max_link_retries = max(0, int(max_link_retries))
        self.debug = debug
        self.trace_dir = Path(trace_dir) if trace_dir else None

        transport = TransportShim(
            model_name,
            base_url=base_url,
            reasoning_effort=reasoning_effort,
        )
        self.transport = transport
        self.entity_list_agent = EntityListAgent(transport)
        self.verb_phrase_list_agent = VerbPhraseListAgent(transport)
        self.link_agent = LinkAgent(transport)
        self.link_verifier_agent = LinkVerifierAgent(transport)
        self.filler_agent = FillerAgent(transport)

        self._link_semaphore = asyncio.Semaphore(max(1, int(sem_link)))

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

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
        async def _runner() -> List[Dict]:
            results: List[Dict] = []
            for doc_idx, doc in enumerate(docs):
                try:
                    gsw, trace_payload = await self._process_document_state(doc, doc_index=doc_idx)
                except Exception as exc:
                    logger.error(
                        "doc=%d pipeline_error error=%r — skipping",
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
                            "linker_trace": trace_payload,
                        }
                    }
                )
                if on_document_done is not None:
                    try:
                        on_document_done(doc_idx, results)
                    except Exception as exc:
                        logger.warning(
                            "doc=%d on_document_done callback raised error=%r",
                            doc_idx,
                            exc,
                        )
            return results

        return asyncio.run(_runner())

    # ------------------------------------------------------------------
    # Per-document pipeline
    # ------------------------------------------------------------------

    async def _process_document_state(
        self,
        text: str,
        *,
        doc_index: Optional[int],
    ) -> tuple[GSWStructure, dict]:
        doc_label = f"doc={doc_index if doc_index is not None else 0}"
        doc_start = time.time()
        logger.info("=" * 72)
        logger.info("%s LINKER PIPELINE START chars=%d", doc_label, len(text))
        logger.info("=" * 72)

        # STEP 1 — entity list first. STEP 2 depends on its output for
        # entity-coverage-driven extraction, so these stages are sequential
        # now (was parallel in V1 of the linker pipeline).
        t0 = time.time()
        try:
            entities = await self.entity_list_agent.run(text)
        except Exception as exc:
            self._log_step(doc_label, 1, "entity_list", "FAIL", extra=f"error={exc!r}")
            raise
        self._log_step(
            doc_label, 1, "entity_list", "OK",
            extra=f"entities={len(entities)} duration={time.time() - t0:.2f}s",
        )

        # STEP 2 — verb phrase list. Receives the entity list so it can aim
        # for coverage rather than a fixed character-count target.
        t0 = time.time()
        try:
            verb_phrases = await self.verb_phrase_list_agent.run(text, entities=entities)
        except Exception as exc:
            self._log_step(doc_label, 2, "verb_phrase_list", "FAIL", extra=f"error={exc!r}")
            raise
        self._log_step(
            doc_label, 2, "verb_phrase_list", "OK",
            extra=(
                f"verb_phrases={len(verb_phrases)} target=~{int(len(entities) * 1.5)} "
                f"duration={time.time() - t0:.2f}s"
            ),
        )

        # STEP 3 — LinkAgent fan-out: one call per verb phrase.
        t0 = time.time()
        link_outputs, link_errors = await self._run_links(verb_phrases, entities, text, retry_hints={})
        self._log_step(
            doc_label, 3, "link_fanout", "OK",
            extra=(
                f"vps={len(verb_phrases)} ok={len(link_outputs)} "
                f"errors={len(link_errors)} duration={time.time() - t0:.2f}s"
            ),
        )

        # STEP 3b — entity expansion: dedupe proposed entities, assign ids,
        # rewrite NEW:<name> placeholders in each LinkAgentOutput.
        entities, name_to_id, expansion_count = self._expand_entities_from_proposals(
            entities, link_outputs.values(),
        )
        self._log_step(
            doc_label, 3, "entity_expansion", "OK",
            extra=f"new_entities={expansion_count} total_entities={len(entities)}",
        )

        # STEP 3c — convert LinkAgentOutputs into Link objects.
        links: list[Link] = []
        for vp_id, output in link_outputs.items():
            link = self._materialize_link(
                vp=self._vp_by_id(verb_phrases, vp_id),
                output=output,
                name_to_id=name_to_id,
                entity_ids={e.id for e in entities},
            )
            if link is not None:
                links.append(link)

        # STEP 4 — LinkVerifier fan-out.
        t0 = time.time()
        verdicts = await self._run_verifier(links, entities, text)
        valid_links = [l for l in links if verdicts.get(l.link_id, _DEFAULT_VALID).valid]
        invalid_links = [l for l in links if not verdicts.get(l.link_id, _DEFAULT_VALID).valid]
        self._log_step(
            doc_label, 4, "verifier", "OK",
            extra=(
                f"links={len(links)} valid={len(valid_links)} invalid={len(invalid_links)} "
                f"duration={time.time() - t0:.2f}s"
            ),
        )

        # STEP 4b — retry rejected links with verifier hints.
        retry_attempt = 0
        while invalid_links and retry_attempt < self.max_link_retries:
            retry_attempt += 1
            t0 = time.time()
            retry_hints = {
                l.vp_id: verdicts.get(l.link_id, _DEFAULT_VALID).hint or verdicts.get(l.link_id, _DEFAULT_VALID).issue
                for l in invalid_links
            }
            retry_vps = [self._vp_by_id(verb_phrases, l.vp_id) for l in invalid_links]
            retry_vps = [vp for vp in retry_vps if vp is not None]
            retry_outputs, retry_errors = await self._run_links(
                retry_vps, entities, text, retry_hints=retry_hints,
            )
            entities, name_to_id, _ = self._expand_entities_from_proposals(
                entities, retry_outputs.values(), name_to_id=name_to_id,
            )

            new_links: list[Link] = []
            for vp_id, output in retry_outputs.items():
                rebuilt = self._materialize_link(
                    vp=self._vp_by_id(verb_phrases, vp_id),
                    output=output,
                    name_to_id=name_to_id,
                    entity_ids={e.id for e in entities},
                )
                if rebuilt is not None:
                    new_links.append(rebuilt)

            new_verdicts = await self._run_verifier(new_links, entities, text)
            verdicts.update(new_verdicts)

            # Replace the invalid links with their rebuilt versions in the
            # main pool.
            new_link_by_vp = {l.vp_id: l for l in new_links}
            updated_links: list[Link] = []
            for link in links:
                replacement = new_link_by_vp.get(link.vp_id)
                if replacement is not None:
                    updated_links.append(replacement)
                else:
                    updated_links.append(link)
            links = updated_links
            valid_links = [l for l in links if verdicts.get(l.link_id, _DEFAULT_VALID).valid]
            invalid_links = [l for l in links if not verdicts.get(l.link_id, _DEFAULT_VALID).valid]
            self._log_step(
                doc_label, 4, f"link_retry_{retry_attempt}", "OK",
                extra=(
                    f"retried={len(retry_vps)} now_valid={len(valid_links)} "
                    f"still_invalid={len(invalid_links)} duration={time.time() - t0:.2f}s"
                ),
            )

        # Drop the still-invalid links — they were chronically broken.
        if invalid_links:
            logger.warning(
                "%s   dropping_invalid_links count=%d after %d retries",
                doc_label, len(invalid_links), self.max_link_retries,
            )
        links = valid_links

        # STEP 5 — orphan detection (pure Python).
        orphan_entities = self._detect_orphans(entities, links)
        self._log_step(
            doc_label, 5, "orphan_check", "OK",
            extra=f"orphans={len(orphan_entities)}/{len(entities)}",
        )

        # STEP 6 — FillerAgent fan-out for orphan entities.
        if orphan_entities:
            t0 = time.time()
            filler_outputs = await self._run_filler(orphan_entities, entities, text)
            entities, name_to_id, _ = self._expand_entities_from_proposals(
                entities, filler_outputs.values(), name_to_id=name_to_id,
            )
            filler_links: list[Link] = []
            for orphan_id, output in filler_outputs.items():
                if output.action != "link":
                    continue
                synthetic_vp_id = f"vp_filler_{orphan_id}"
                synthetic_vp = VerbPhraseSlot(
                    vp_id=synthetic_vp_id,
                    label=self._infer_filler_vp_label(output),
                    hint="auto-generated by FillerAgent",
                )
                rebuilt = self._materialize_link(
                    vp=synthetic_vp,
                    output=output,
                    name_to_id=name_to_id,
                    entity_ids={e.id for e in entities},
                    link_id_override=f"l_filler_{orphan_id}",
                )
                if rebuilt is not None:
                    filler_links.append(rebuilt)
            links.extend(filler_links)
            self._log_step(
                doc_label, 6, "filler", "OK",
                extra=(
                    f"orphans={len(orphan_entities)} filled={len(filler_links)} "
                    f"skipped={len(orphan_entities) - len(filler_links)} "
                    f"duration={time.time() - t0:.2f}s"
                ),
            )
        else:
            self._log_step(doc_label, 6, "filler", "OK", extra="no orphans — skipped")

        # STEP 7 — deterministic assembly.
        chunk_id = f"{doc_index if doc_index is not None else 0}_0"
        try:
            t0 = time.time()
            gsw = assemble_gsw(entities=entities, links=links, chunk_id=chunk_id)
            gsw = GSWStructure.model_validate(gsw.model_dump(mode="json"))
            self._log_step(
                doc_label, 7, "assemble", "OK",
                extra=(
                    f"entity_nodes={len(gsw.entity_nodes)} "
                    f"vp_nodes={len(gsw.verb_phrase_nodes)} "
                    f"duration={time.time() - t0:.2f}s"
                ),
            )
        except Exception as exc:
            self._log_step(doc_label, 7, "assemble", "FAIL", extra=f"error={exc!r}")
            raise

        # Drain the per-call ledger from TransportShim so the markdown
        # trace can render every agent call's full reasoning + output.
        # Drained per-doc to avoid bleeding across docs.
        call_log = self.transport.drain_call_log()

        # Compute per-doc cost from the call log. Each entry already has
        # prompt/completion tokens captured by TransportShim._record_call.
        cost_summary = self._compute_cost_summary(call_log)

        trace_payload = {
            "entities": [e.model_dump(mode="json") for e in entities],
            "verb_phrases": [v.model_dump(mode="json") for v in verb_phrases],
            "links": [l.model_dump(mode="json") for l in links],
            "verdicts": {lid: v.model_dump(mode="json") for lid, v in verdicts.items()},
            "call_log": call_log,
            "cost": cost_summary,
        }
        self._write_trace(doc_index, trace_payload)
        self._write_markdown_trace(
            doc_index=doc_index,
            doc_label=doc_label,
            text=text,
            entities=entities,
            verb_phrases=verb_phrases,
            links=links,
            verdicts=verdicts,
            call_log=call_log,
            cost_summary=cost_summary,
            total_duration=time.time() - doc_start,
        )
        logger.info(
            "%s pipeline_done total_duration=%.2fs entities=%d links=%d "
            "calls=%d prompt_tokens=%d completion_tokens=%d cost_usd=$%.4f",
            doc_label,
            time.time() - doc_start,
            len(entities),
            len(links),
            cost_summary["calls"],
            cost_summary["prompt_tokens"],
            cost_summary["completion_tokens"],
            cost_summary["total_cost_usd"],
        )
        return gsw, trace_payload

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    async def _run_links(
        self,
        verb_phrases: Sequence[VerbPhraseSlot],
        entities: Sequence[CanonicalEntity],
        text: str,
        *,
        retry_hints: Dict[str, str],
    ) -> tuple[Dict[str, LinkAgentOutput], Dict[str, str]]:
        """Fan-out LinkAgent calls in BATCHES of ``self.link_batch_size``.

        Each batched call ships the full source text + entity list once and
        processes N verb phrases in one LLM round-trip, cutting prompt-token
        cost by ~Nx compared to single-VP-per-call fan-out.
        """
        results: Dict[str, LinkAgentOutput] = {}
        errors: Dict[str, str] = {}
        if not verb_phrases:
            return results, errors

        batches = [
            list(verb_phrases[i : i + self.link_batch_size])
            for i in range(0, len(verb_phrases), self.link_batch_size)
        ]
        tasks = [
            self.link_agent.run_batch(
                verb_phrases=batch,
                entities=entities,
                text=text,
                retry_hints=retry_hints,
                semaphore=self._link_semaphore,
            )
            for batch in batches
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for batch, result in zip(batches, gathered):
            if isinstance(result, Exception):
                logger.warning(
                    "link_batch_error batch_size=%d error=%r — marking all "
                    "vps in this batch as errored",
                    len(batch), result,
                )
                for vp in batch:
                    errors[vp.vp_id] = repr(result)
                continue
            results.update(result)
        logger.info(
            "link_batch_done batches=%d vps=%d results=%d errors=%d",
            len(batches), len(verb_phrases), len(results), len(errors),
        )
        return results, errors

    async def _run_verifier(
        self,
        links: Sequence[Link],
        entities: Sequence[CanonicalEntity],
        text: str,
    ) -> Dict[str, LinkVerifierOutput]:
        """Fan-out LinkVerifier calls in BATCHES of ``self.link_batch_size``.

        Same cost-savings shape as ``_run_links``: each batched call ships
        the full source text once and judges N links in one LLM round-trip.
        """
        out: Dict[str, LinkVerifierOutput] = {}
        if not links:
            return out
        entity_by_id = {e.id: e for e in entities}
        # Skip links whose subject or object entity is missing — they can't
        # be verified meaningfully.
        valid_links = [
            link for link in links
            if link.subject_id in entity_by_id and link.object_id in entity_by_id
        ]

        batches = [
            valid_links[i : i + self.link_batch_size]
            for i in range(0, len(valid_links), self.link_batch_size)
        ]
        tasks = [
            self.link_verifier_agent.run_batch(
                links=batch,
                entity_by_id=entity_by_id,
                text=text,
                semaphore=self._link_semaphore,
            )
            for batch in batches
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for batch, result in zip(batches, gathered):
            if isinstance(result, Exception):
                logger.warning(
                    "link_verify_batch_error batch_size=%d error=%r — "
                    "defaulting batch to valid",
                    len(batch), result,
                )
                for link in batch:
                    out[link.link_id] = _DEFAULT_VALID
                continue
            out.update(result)
        logger.info(
            "link_verify_batch_done batches=%d links=%d results=%d",
            len(batches), len(valid_links), len(out),
        )
        return out

    async def _run_filler(
        self,
        orphans: Sequence[CanonicalEntity],
        entities: Sequence[CanonicalEntity],
        text: str,
    ) -> Dict[str, LinkAgentOutput]:
        tasks = []
        order: list[str] = []
        for orphan in orphans:
            order.append(orphan.id)
            tasks.append(
                self.filler_agent.run(
                    orphan_entity=orphan,
                    entities=entities,
                    text=text,
                    semaphore=self._link_semaphore,
                )
            )
        out: Dict[str, LinkAgentOutput] = {}
        if not tasks:
            return out
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for orphan_id, result in zip(order, gathered):
            if isinstance(result, Exception):
                logger.warning("filler_call orphan=%s error=%r", orphan_id, result)
                continue
            out[orphan_id] = result
            logger.info(
                "filler_call orphan=%s action=%s subject=%s object=%s",
                orphan_id, result.action, result.subject_id, result.object_id,
            )
        return out

    # ------------------------------------------------------------------
    # Pure Python helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _vp_by_id(vps: Sequence[VerbPhraseSlot], vp_id: str) -> Optional[VerbPhraseSlot]:
        for vp in vps:
            if vp.vp_id == vp_id:
                return vp
        return None

    @staticmethod
    def _expand_entities_from_proposals(
        entities: Sequence[CanonicalEntity],
        outputs: Sequence[LinkAgentOutput],
        *,
        name_to_id: Optional[Dict[str, str]] = None,
    ) -> tuple[list[CanonicalEntity], Dict[str, str], int]:
        """Add new entities for each unique proposed name; return the
        expanded list plus a name→id map keyed on casefold-normalized name."""
        from ...models import Role

        expanded = list(entities)
        name_to_id = dict(name_to_id) if name_to_id is not None else {}
        # Seed name_to_id with existing entities so we never duplicate.
        for entity in expanded:
            name_to_id.setdefault(entity.name.casefold(), entity.id)
            for alias in entity.aliases:
                name_to_id.setdefault(alias.casefold(), entity.id)

        max_idx = 0
        for entity in expanded:
            match = re.match(r"e(\d+)$", entity.id or "")
            if match:
                max_idx = max(max_idx, int(match.group(1)))

        added = 0
        for output in outputs:
            for proposal in output.proposed_entities:
                name = (proposal.name or "").strip()
                if not name:
                    continue
                key = name.casefold()
                if key in name_to_id:
                    continue
                max_idx += 1
                new_id = f"e{max_idx}"
                expanded.append(
                    CanonicalEntity(
                        id=new_id,
                        name=name,
                        aliases=[a.strip() for a in proposal.aliases if a and a.strip()],
                        roles=[Role(role=proposal.role or "concept")],
                    )
                )
                name_to_id[key] = new_id
                added += 1
        return expanded, name_to_id, added

    @staticmethod
    def _resolve_id_or_placeholder(
        raw: str,
        name_to_id: Dict[str, str],
        entity_ids: set[str],
    ) -> Optional[str]:
        """Convert a `subject_id`/`object_id` field into a real entity id.
        Accepts:
          - existing canonical id (`"e3"`)
          - `"NEW:<name>"` placeholder (resolved via name_to_id)
          - a bare entity name that matches an entity by name or alias
        """
        raw = (raw or "").strip()
        if not raw:
            return None
        if raw in entity_ids:
            return raw
        if raw.startswith(_NEW_PREFIX):
            name = raw[len(_NEW_PREFIX):].strip()
            mapped = name_to_id.get(name.casefold())
            if mapped in entity_ids:
                return mapped
            return None
        # Treat as a bare name and look it up.
        mapped = name_to_id.get(raw.casefold())
        if mapped in entity_ids:
            return mapped
        return None

    @classmethod
    def _materialize_link(
        cls,
        *,
        vp: Optional[VerbPhraseSlot],
        output: LinkAgentOutput,
        name_to_id: Dict[str, str],
        entity_ids: set[str],
        link_id_override: Optional[str] = None,
    ) -> Optional[Link]:
        if vp is None:
            return None
        if output.action != "link":
            return None
        subject_id = cls._resolve_id_or_placeholder(output.subject_id, name_to_id, entity_ids)
        object_id = cls._resolve_id_or_placeholder(output.object_id, name_to_id, entity_ids)
        if not subject_id or not object_id or subject_id == object_id:
            return None
        link_id = link_id_override or f"l_{vp.vp_id}"
        return Link(
            link_id=link_id,
            vp_id=vp.vp_id,
            vp_label=vp.label,
            subject_id=subject_id,
            object_id=object_id,
            supporting_text=(output.supporting_text or "")[:1000],
            forward_question=output.forward_question or "",
            reverse_question=output.reverse_question or "",
        )

    @staticmethod
    def _detect_orphans(
        entities: Sequence[CanonicalEntity],
        links: Sequence[Link],
    ) -> list[CanonicalEntity]:
        referenced: set[str] = set()
        for link in links:
            referenced.add(link.subject_id)
            referenced.add(link.object_id)
        return [e for e in entities if e.id not in referenced]

    @staticmethod
    def _infer_filler_vp_label(output: LinkAgentOutput) -> str:
        """The FillerAgent doesn't get a verb phrase as input; it picks one
        and embeds it implicitly. We try to pull a verb-phrase-shaped string
        out of the supporting_text, falling back to a generic label."""
        text = (output.supporting_text or "").strip()
        if text:
            words = text.split()
            if 3 <= len(words) <= 8:
                return text
            if len(words) > 8:
                return " ".join(words[:6])
        return "related to"

    # ------------------------------------------------------------------
    # Logging + tracing
    # ------------------------------------------------------------------

    @staticmethod
    def _log_step(doc_label: str, step_num: int, step_name: str, status: str, *, extra: str = "") -> None:
        marker = "✓" if status == "OK" else ("✗" if status == "FAIL" else "•")
        logger.info(
            "%s %s STEP %d/7 %-20s %s%s",
            doc_label,
            marker,
            step_num,
            step_name,
            status,
            f" | {extra}" if extra else "",
        )

    def _compute_cost_summary(self, call_log: Sequence[dict]) -> dict:
        """Aggregate token counts and dollar cost from the call log.

        Returns a dict shaped like:
          {
            "model": "...",
            "calls": N,
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int,
            "total_cost_usd": float,
            "by_stage": {
                "entity_list": {"calls": 1, "prompt_tokens": ..., "completion_tokens": ..., "cost_usd": ...},
                ...
            }
          }
        """
        by_stage: dict[str, dict] = {}
        total_prompt = 0
        total_completion = 0
        total_cost = 0.0

        for entry in call_log:
            stage = (entry.get("stage") or "unknown").strip() or "unknown"
            # Group repair / fallback variants under their base stage.
            base = stage
            for suffix in ("_repair_text", "_repair", "_structured_fallback"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break

            prompt_tokens = int(entry.get("prompt_tokens", 0) or 0)
            completion_tokens = int(entry.get("completion_tokens", 0) or 0)
            cost = estimate_cost_usd(self.model_name, prompt_tokens, completion_tokens)

            slot = by_stage.setdefault(
                base,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                },
            )
            slot["calls"] += 1
            slot["prompt_tokens"] += prompt_tokens
            slot["completion_tokens"] += completion_tokens
            slot["cost_usd"] += cost

            total_prompt += prompt_tokens
            total_completion += completion_tokens
            total_cost += cost

        return {
            "model": self.model_name,
            "calls": len(call_log),
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost_usd": total_cost,
            "by_stage": by_stage,
        }

    def _write_trace(self, doc_index: Optional[int], payload: dict) -> None:
        if not self.debug or self.trace_dir is None:
            return
        doc_label = f"doc_{doc_index if doc_index is not None else 0}"
        doc_dir = self.trace_dir / doc_label
        doc_dir.mkdir(parents=True, exist_ok=True)
        trace_path = doc_dir / "linker_trace.json"
        trace_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_markdown_trace(
        self,
        *,
        doc_index: Optional[int],
        doc_label: str,
        text: str,
        entities: Sequence[CanonicalEntity],
        verb_phrases: Sequence[VerbPhraseSlot],
        links: Sequence[Link],
        verdicts: Dict[str, LinkVerifierOutput],
        call_log: Sequence[dict],
        cost_summary: dict,
        total_duration: float,
    ) -> None:
        """Render a human-readable per-doc markdown trace.

        Full fidelity, no truncation. Groups every agent call by stage
        (entity_list, verb_phrase_list, link, link_verify, filler) and
        includes the model's raw reasoning, raw content, and parsed payload
        for each call. Designed for presentation / post-hoc inspection.
        """
        if not self.debug or self.trace_dir is None:
            return
        doc_dir = self.trace_dir / f"doc_{doc_index if doc_index is not None else 0}"
        doc_dir.mkdir(parents=True, exist_ok=True)
        path = doc_dir / "linker_trace.md"

        # Group calls by stage prefix. Stages used by the linker pipeline:
        # entity_list, verb_phrase_list, link, link_verify, filler
        # (plus *_repair / *_structured_fallback variants from transport_shim).
        groups: Dict[str, list[dict]] = {}
        for entry in call_log:
            stage = entry.get("stage", "unknown")
            base = stage
            for suffix in ("_repair_text", "_repair", "_structured_fallback"):
                if base.endswith(suffix):
                    base = base[: -len(suffix)]
                    break
            groups.setdefault(base, []).append(entry)

        lines: list[str] = []
        lines.append(f"# {doc_label} — LinkerPipeline trace")
        lines.append("")
        lines.append(f"- **Total duration**: {total_duration:.2f}s")
        lines.append(f"- **Source chars**: {len(text)}")
        lines.append(f"- **Final entities**: {len(entities)}")
        lines.append(f"- **Final links**: {len(links)}")
        lines.append(f"- **Total agent calls**: {len(call_log)}")
        lines.append(
            f"- **Tokens**: prompt={cost_summary['prompt_tokens']:,} "
            f"completion={cost_summary['completion_tokens']:,} "
            f"total={cost_summary['total_tokens']:,}"
        )
        lines.append(
            f"- **Total cost**: ${cost_summary['total_cost_usd']:.4f} "
            f"_(model: `{cost_summary['model']}`)_"
        )
        lines.append("")
        # Per-stage cost breakdown table
        if cost_summary.get("by_stage"):
            lines.append("### Cost by stage")
            lines.append("")
            lines.append("| Stage | Calls | Prompt tokens | Completion tokens | Cost (USD) |")
            lines.append("|---|---:|---:|---:|---:|")
            for stage_name in sorted(cost_summary["by_stage"]):
                slot = cost_summary["by_stage"][stage_name]
                lines.append(
                    f"| `{stage_name}` | {slot['calls']} | "
                    f"{slot['prompt_tokens']:,} | {slot['completion_tokens']:,} | "
                    f"${slot['cost_usd']:.4f} |"
                )
            lines.append("")
        lines.append("## Source document")
        lines.append("")
        lines.append("```")
        lines.append(text)
        lines.append("```")
        lines.append("")

        # Per-stage sections in pipeline order.
        stage_order = [
            ("entity_list", "STEP 1 — EntityListAgent"),
            ("verb_phrase_list", "STEP 2 — VerbPhraseListAgent"),
            ("link", "STEP 3 — LinkAgent fan-out"),
            ("link_verify", "STEP 4 — LinkVerifier fan-out"),
            ("filler", "STEP 6 — FillerAgent fan-out"),
        ]
        for stage_key, heading in stage_order:
            calls = groups.get(stage_key, [])
            if not calls:
                continue
            lines.append(f"## {heading}")
            lines.append("")
            lines.append(f"_{len(calls)} call(s)_")
            lines.append("")
            for idx, entry in enumerate(calls, start=1):
                lines.append(f"### Call {idx} (stage=`{entry.get('stage', stage_key)}`, path=`{entry.get('path', '')}`)")
                lines.append("")
                reasoning = (entry.get("reasoning") or "").strip()
                content = (entry.get("content") or "").strip()
                payload = entry.get("payload")
                if reasoning:
                    lines.append("**Reasoning**")
                    lines.append("")
                    lines.append("```")
                    lines.append(reasoning)
                    lines.append("```")
                    lines.append("")
                if content:
                    lines.append("**Raw content**")
                    lines.append("")
                    lines.append("```")
                    lines.append(content)
                    lines.append("```")
                    lines.append("")
                if payload is not None:
                    lines.append("**Parsed payload**")
                    lines.append("")
                    lines.append("```json")
                    lines.append(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
                    lines.append("```")
                    lines.append("")

        # Final assembled state (entities + links) for quick inspection.
        lines.append("## STEP 7 — Final assembled state")
        lines.append("")
        lines.append(f"### Entities ({len(entities)})")
        lines.append("")
        for entity in entities:
            roles = ", ".join(r.role for r in entity.roles) or "—"
            aliases = ", ".join(entity.aliases) if entity.aliases else ""
            alias_block = f" _(aliases: {aliases})_" if aliases else ""
            lines.append(f"- **{entity.id}** `{entity.name}` — roles: {roles}{alias_block}")
        lines.append("")
        lines.append(f"### Links ({len(links)})")
        lines.append("")
        entity_by_id = {e.id: e for e in entities}
        for link in links:
            verdict = verdicts.get(link.link_id)
            verdict_block = ""
            if verdict is not None:
                vstatus = "✓" if verdict.valid else "✗"
                verdict_block = (
                    f" — verifier {vstatus} (conf={verdict.confidence:.2f}"
                    f"{', issue=' + verdict.issue if verdict.issue else ''})"
                )
            subj_name = entity_by_id.get(link.subject_id, type("E", (), {"name": link.subject_id})).name
            obj_name = entity_by_id.get(link.object_id, type("E", (), {"name": link.object_id})).name
            lines.append(f"#### `{link.link_id}` — {subj_name} → **{link.vp_label}** → {obj_name}{verdict_block}")
            lines.append("")
            if link.supporting_text:
                lines.append("**Supporting text**")
                lines.append("")
                lines.append("> " + link.supporting_text.replace("\n", "\n> "))
                lines.append("")
            lines.append(f"- forward Q: {link.forward_question}")
            lines.append(f"  - answer: `{link.object_id}` ({obj_name})")
            lines.append(f"- reverse Q: {link.reverse_question}")
            lines.append(f"  - answer: `{link.subject_id}` ({subj_name})")
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")


_DEFAULT_VALID = LinkVerifierOutput(valid=True, confidence=1.0)
