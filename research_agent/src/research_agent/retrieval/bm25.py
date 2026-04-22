"""Simple BM25 retriever for the pilot.

BM25 is fine over the FRAMES corpus (~2k chunks) — avoids pulling
down a ~1GB embedding model during pilot setup. Swapping to dense
retrieval (BGE-large) later is a drop-in replacement under the same
``search(query, top_k)`` API.

Implementation note: we avoid the external ``rank_bm25`` dependency
and ship a self-contained Okapi-BM25 here so the workspace stays
minimal. For ~2-5k short documents this is sub-10ms per query.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from research_agent.retrieval.corpus import ArticleCorpus, Chunk


_TOK_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOK_RE.findall(text)]


@dataclass(slots=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


class BM25Retriever:
    """Okapi BM25 over an ArticleCorpus.

    Parameters
    ----------
    corpus:
        The article corpus to index.
    k1, b:
        Standard BM25 hyperparameters.
    """

    def __init__(self, corpus: ArticleCorpus, *, k1: float = 1.5, b: float = 0.75):
        self.corpus = corpus
        self.k1 = k1
        self.b = b

        self._tokenized: list[list[str]] = [_tokenize(c.text) for c in corpus.chunks]
        self._doc_lens = [len(d) for d in self._tokenized]
        self._avg_dl = sum(self._doc_lens) / max(len(self._doc_lens), 1)

        # Term frequencies per doc, document frequencies.
        self._tf: list[Counter[str]] = [Counter(doc) for doc in self._tokenized]
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())

        n = len(self._tokenized)
        # IDF = ln((N - df + 0.5) / (df + 0.5) + 1) — Robertson/Okapi form
        self._idf: dict[str, float] = {
            term: math.log((n - freq + 0.5) / (freq + 0.5) + 1) for term, freq in df.items()
        }

    def _score_doc(self, doc_idx: int, query_terms: Iterable[str]) -> float:
        tf = self._tf[doc_idx]
        dl = self._doc_lens[doc_idx] or 1
        score = 0.0
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            f = tf.get(term, 0)
            if f == 0:
                continue
            numer = f * (self.k1 + 1)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self._avg_dl)
            score += idf * numer / denom
        return score

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        title_bias: dict[str, float] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-k ``RetrievedChunk``s for a query.

        ``title_bias`` multiplies the score of chunks whose title matches
        a key (useful when the benchmark tells us which articles matter).
        """
        terms = _tokenize(query)
        if not terms:
            return []

        scored: list[tuple[float, int]] = []
        for idx, chunk in enumerate(self.corpus.chunks):
            s = self._score_doc(idx, terms)
            if title_bias:
                bias = title_bias.get(chunk.title, 1.0)
                s *= bias
            if s > 0:
                scored.append((s, idx))

        scored.sort(reverse=True)
        out: list[RetrievedChunk] = []
        for s, idx in scored[:top_k]:
            out.append(RetrievedChunk(chunk=self.corpus.chunks[idx], score=s))
        return out
