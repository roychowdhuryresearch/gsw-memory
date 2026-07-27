"""Shared retrieval result types and fusion utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SearchHit:
    """A retrieval result independent of the underlying search backend."""

    item_id: str
    score: float
    source: str
    rank: int
    metadata: Mapping[str, object] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: Sequence[Iterable[SearchHit]],
    *,
    rank_constant: float = 60.0,
    top_k: int | None = None,
) -> list[SearchHit]:
    """Fuse ranked lists with Reciprocal Rank Fusion.

    Every input hit contributes ``1 / (rank_constant + rank)``. Duplicate item
    IDs are merged, and their source labels are preserved in the result
    metadata.
    """

    if rank_constant < 0:
        raise ValueError("rank_constant must be non-negative")
    fused_scores: dict[str, float] = {}
    source_names: dict[str, list[str]] = {}
    metadata_by_id: dict[str, Mapping[str, object]] = {}

    for ranking in rankings:
        for fallback_rank, hit in enumerate(ranking, start=1):
            rank = hit.rank if hit.rank > 0 else fallback_rank
            fused_scores[hit.item_id] = fused_scores.get(hit.item_id, 0.0) + (
                1.0 / (rank_constant + rank)
            )
            source_names.setdefault(hit.item_id, []).append(hit.source)
            metadata_by_id.setdefault(hit.item_id, hit.metadata)

    ordered = sorted(
        fused_scores,
        key=lambda item_id: (-fused_scores[item_id], item_id),
    )
    if top_k is not None:
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        ordered = ordered[:top_k]

    return [
        SearchHit(
            item_id=item_id,
            score=fused_scores[item_id],
            source="rrf",
            rank=rank,
            metadata={
                **metadata_by_id[item_id],
                "fusion_sources": tuple(source_names[item_id]),
            },
        )
        for rank, item_id in enumerate(ordered, start=1)
    ]


class DualRetriever:
    """Paper-style entity expansion plus direct QA retrieval.

    ``entity_index`` may be BM25, TF-IDF, or dense. ``qa_index`` may likewise
    use any backend as long as its ``search`` method returns ``SearchHit``
    records. The caller supplies a query vector only when a dense backend needs
    it.
    """

    def __init__(
        self,
        *,
        entity_index,
        qa_index,
        entity_rows: Sequence[Mapping[str, object]],
        qa_rows: Sequence[Mapping[str, object]],
    ):
        self.entity_index = entity_index
        self.qa_index = qa_index
        self.entity_rows = {
            str(row["entity_uid"]): row for row in entity_rows
        }
        self.qa_rows = {str(row["qa_uid"]): row for row in qa_rows}

        qa_by_verb: dict[tuple[str, str, str], list[str]] = defaultdict(list)
        verbs_by_entity: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        for row in qa_rows:
            verb_key = (
                str(row["document_id"]),
                str(row["gsw_file"]),
                str(row["verb_phrase_id"]),
            )
            qa_uid = str(row["qa_uid"])
            qa_by_verb[verb_key].append(qa_uid)
            for local_answer_id in row.get("answer_local_ids", []):
                entity_uid = (
                    f"{row['document_id']}::{row['gsw_file']}::"
                    f"{local_answer_id}"
                )
                verbs_by_entity[entity_uid].add(verb_key)

        self.qa_ids_by_entity: dict[str, tuple[str, ...]] = {}
        for entity_uid, verb_keys in verbs_by_entity.items():
            attached = {
                qa_uid
                for verb_key in verb_keys
                for qa_uid in qa_by_verb[verb_key]
            }
            self.qa_ids_by_entity[entity_uid] = tuple(sorted(attached))

    @staticmethod
    def _search(index, query: str, query_vector, top_k: int):
        try:
            return index.search(query, top_k)
        except (AttributeError, TypeError, ValueError):
            if query_vector is None:
                raise ValueError(
                    "This retrieval backend requires a query embedding"
                )
            return index.search(query_vector, top_k)

    def search(
        self,
        query: str,
        *,
        query_vector=None,
        entity_top_k: int = 20,
        qa_top_k: int = 15,
        fused_top_k: int = 30,
        rank_constant: float = 60.0,
    ) -> list[SearchHit]:
        entity_hits = self._search(
            self.entity_index, query, query_vector, entity_top_k
        )
        direct_qa_hits = self._search(
            self.qa_index, query, query_vector, qa_top_k
        )

        expanded_scores: dict[str, float] = {}
        expanded_sources: dict[str, str] = {}
        for entity_hit in entity_hits:
            for qa_uid in self.qa_ids_by_entity.get(entity_hit.item_id, ()):
                if entity_hit.score > expanded_scores.get(qa_uid, float("-inf")):
                    expanded_scores[qa_uid] = entity_hit.score
                    expanded_sources[qa_uid] = entity_hit.item_id
        ordered_expanded = sorted(
            expanded_scores,
            key=lambda qa_uid: (-expanded_scores[qa_uid], qa_uid),
        )
        entity_qa_hits = [
            SearchHit(
                item_id=qa_uid,
                score=expanded_scores[qa_uid],
                source="entity_expansion",
                rank=rank,
                metadata={"source_entity_uid": expanded_sources[qa_uid]},
            )
            for rank, qa_uid in enumerate(ordered_expanded, start=1)
        ]

        return reciprocal_rank_fusion(
            [entity_qa_hits, direct_qa_hits],
            rank_constant=rank_constant,
            top_k=fused_top_k,
        )
