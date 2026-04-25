"""Dense retrieval + hybrid (BM25 + dense RRF) for the FRAMES corpus.

The adapters expect a retriever with::

    .search(query: str, *, top_k: int) -> list[RetrievedChunk]

where ``RetrievedChunk`` has ``.chunk`` (with ``chunk_id``, ``title``,
``text``) and ``.score``. ``BM25Retriever`` already matches this shape;
this module adds two drop-in alternatives:

- :class:`DenseRetriever` — Qwen3-Embedding-8B (default) encoded chunks,
  FAISS ``IndexFlatIP`` over L2-normalized vectors, disk-cached so the
  ~3-minute index build only happens once per corpus.
- :class:`HybridRetriever` — reciprocal-rank-fusion of BM25 + dense.

Both preserve ``RetrievedChunk`` so no adapter code changes beyond
choosing which retriever to instantiate.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from research_agent.retrieval.bm25 import BM25Retriever, RetrievedChunk
from research_agent.retrieval.corpus import ArticleCorpus

_log = logging.getLogger(__name__)


# Default embedding model — the top-tier Qwen3 family member the user used
# before. ~16GB FP16 on GPU. Swap via DenseRetriever(model_name=...).
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"


# Cache lives under the GSW-memory data dir so it survives repo rebuilds.
_DEFAULT_CACHE_ROOT = Path(
    os.environ.get("GSW_MEMORY_ROOT", "/home/yigit/codebase/gsw-memory")
) / "data" / "retrieval_cache"


# ---------------------------------------------------------------------------
# DenseRetriever
# ---------------------------------------------------------------------------


class DenseRetriever:
    """Encode corpus chunks with Qwen3-Embedding, search via FAISS.

    The encoder is a :class:`sentence_transformers.SentenceTransformer`.
    Chunks are encoded once at init and cached to disk; subsequent
    constructor calls with the same ``(corpus fingerprint, model_name)``
    load the cached index in <1s.
    """

    def __init__(
        self,
        corpus: ArticleCorpus,
        *,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
        query_prompt: Optional[str] = None,
        torch_dtype: Optional[str] = None,
    ) -> None:
        self.corpus = corpus
        self.model_name = model_name
        self.batch_size = batch_size
        # Qwen3 embedding models use an instruction-style query prompt.
        self.query_prompt = (
            query_prompt
            if query_prompt is not None
            else (
                "Instruct: Given a web search query, retrieve relevant "
                "passages that answer the query\nQuery: "
            )
        )

        cache_dir = cache_dir or _DEFAULT_CACHE_ROOT / self._fingerprint()
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_dir = cache_dir

        self._device = device or _pick_device()
        # Auto-pick torch_dtype: FP16 on GPU (2× faster + half memory),
        # FP32 on CPU (half-precision on CPU is usually slower).
        if torch_dtype is None:
            torch_dtype = "float16" if self._device.startswith("cuda") else "float32"
        self._torch_dtype = torch_dtype
        _log.info(
            f"DenseRetriever: model={model_name} device={self._device} "
            f"dtype={torch_dtype} batch_size={batch_size}"
        )

        emb_path = cache_dir / "chunk_embeddings.npy"
        ids_path = cache_dir / "chunk_ids.pkl"
        index_path = cache_dir / "faiss.index"

        # Lazy-import so this module stays importable even if torch is missing.
        import numpy as np

        self._embeddings: np.ndarray
        self._chunk_ids: list[str]

        if emb_path.exists() and ids_path.exists() and index_path.exists():
            _log.info(f"DenseRetriever: loading cached index from {cache_dir}")
            self._embeddings = np.load(emb_path)
            with ids_path.open("rb") as f:
                self._chunk_ids = pickle.load(f)
            import faiss

            self._index = faiss.read_index(str(index_path))
        else:
            _log.info(
                f"DenseRetriever: building index for {len(corpus.chunks)} chunks "
                f"(this takes a few minutes for 8B on first run)"
            )
            t0 = time.time()
            self._build_index(emb_path, ids_path, index_path)
            _log.info(f"DenseRetriever: index built in {time.time() - t0:.1f}s")

        # The encoder is only needed for queries after the build.
        self._model = None  # lazy — loaded on first search

    # ------------------------------------------------------------------
    # Index build
    # ------------------------------------------------------------------

    def _build_index(self, emb_path: Path, ids_path: Path, index_path: Path) -> None:
        import faiss
        import numpy as np
        import torch
        from sentence_transformers import SentenceTransformer

        dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
        model_kwargs = {"torch_dtype": dtype_map.get(self._torch_dtype, torch.float32)}
        self._model = SentenceTransformer(
            self.model_name, device=self._device, model_kwargs=model_kwargs
        )
        chunks = self.corpus.chunks
        texts = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]

        # Encode in batches to stay within GPU memory.
        embs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype(np.float32)

        dim = embs.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embs)

        np.save(emb_path, embs)
        with ids_path.open("wb") as f:
            pickle.dump(chunk_ids, f)
        faiss.write_index(index, str(index_path))

        self._embeddings = embs
        self._chunk_ids = chunk_ids
        self._index = index

    def _fingerprint(self) -> str:
        """A stable key for (corpus, model) cache bucketing."""
        titles_hash = hashlib.sha256(
            "\n".join(sorted(self.corpus._by_title.keys())).encode()
        ).hexdigest()[:12]
        model_slug = self.model_name.replace("/", "__")
        return f"{model_slug}__{titles_hash}"

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _ensure_model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            dtype_map = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }
            model_kwargs = {
                "torch_dtype": dtype_map.get(self._torch_dtype, torch.float32)
            }
            self._model = SentenceTransformer(
                self.model_name, device=self._device, model_kwargs=model_kwargs
            )
        return self._model

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        import numpy as np

        model = self._ensure_model()
        q_emb = model.encode(
            [self.query_prompt + query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        scores, idxs = self._index.search(q_emb, top_k)
        out: list[RetrievedChunk] = []
        id_to_chunk = self.corpus._by_id  # ArticleCorpus maintains this
        for s, i in zip(scores[0].tolist(), idxs[0].tolist()):
            if i < 0:
                continue
            chunk_id = self._chunk_ids[i]
            chunk = id_to_chunk.get(chunk_id)
            if chunk is None:
                continue
            out.append(RetrievedChunk(chunk=chunk, score=float(s)))
        return out


# ---------------------------------------------------------------------------
# HybridRetriever — reciprocal rank fusion
# ---------------------------------------------------------------------------


class HybridRetriever:
    """Reciprocal Rank Fusion (RRF) over BM25 + dense.

    RRF score for a chunk c is::

        score(c) = sum_r (1 / (k + rank_r(c)))

    where r ranges over the member retrievers and ``rank_r(c)`` is c's
    1-based rank in retriever r (or infinity if not in r's top_k).
    Industry-standard fusion; no hyperparameter tuning beyond ``k``.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        *,
        k: int = 60,
        per_retriever_top_k: int = 50,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.k = k
        self.per_top_k = per_retriever_top_k

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        bm25_hits = self.bm25.search(query, top_k=self.per_top_k)
        dense_hits = self.dense.search(query, top_k=self.per_top_k)

        scores: dict[str, float] = {}
        chunk_by_id: dict[str, Any] = {}

        for rank, h in enumerate(bm25_hits, start=1):
            cid = h.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.k + rank)
            chunk_by_id[cid] = h.chunk
        for rank, h in enumerate(dense_hits, start=1):
            cid = h.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.k + rank)
            chunk_by_id.setdefault(cid, h.chunk)

        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        out: list[RetrievedChunk] = []
        for cid, s in ranked[:top_k]:
            out.append(RetrievedChunk(chunk=chunk_by_id[cid], score=s))
        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_retriever(
    retriever_type: str,
    corpus: ArticleCorpus,
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    cache_dir: Optional[Path] = None,
    device: Optional[str] = None,
) -> Any:
    """Construct the requested retriever. Shared entry point for adapters."""
    rt = (retriever_type or "bm25").lower()
    if rt == "bm25":
        return BM25Retriever(corpus)
    if rt == "dense":
        return DenseRetriever(
            corpus, model_name=model_name, cache_dir=cache_dir, device=device
        )
    if rt == "hybrid":
        bm25 = BM25Retriever(corpus)
        dense = DenseRetriever(
            corpus, model_name=model_name, cache_dir=cache_dir, device=device
        )
        return HybridRetriever(bm25, dense)
    raise ValueError(
        f"unknown retriever_type {retriever_type!r}; expected bm25|dense|hybrid"
    )


def _pick_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            # Pick the GPU with the most free memory.
            free_mems = []
            for i in range(torch.cuda.device_count()):
                free, _total = torch.cuda.mem_get_info(i)
                free_mems.append((free, i))
            free_mems.sort(reverse=True)
            return f"cuda:{free_mems[0][1]}"
        return "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"
