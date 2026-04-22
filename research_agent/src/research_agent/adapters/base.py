"""Adapter contract — the one interface every competitor system + our own implements.

An ``Adapter`` wraps a specific agentic-search framework (ASearcher /
Search-R1 / Search-o1 / Context-1 / Tongyi DR / plain RAG+ReAct / GSW)
so the eval harness can treat all of them uniformly. The only method
adapters MUST implement is :meth:`Adapter.run_question` — return a
``Trajectory`` for the given question.

Adapters are registered by id. The harness loads them by (system_id,
model_id) and calls ``run_question`` per question. No adapter should
know about benchmarks — the harness passes in the specific question
and expects a trajectory back.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from research_agent.models.trace import Trajectory


class RegistryError(KeyError):
    """Raised when an adapter id is missing from the registry."""


@dataclass
class AdapterContext:
    """Runtime context handed to every adapter ``run_question`` call.

    The harness populates this before calling the adapter. Adapters
    use it to reach the model endpoint, the retriever (if any), the
    output dir for raw traces, and any extra config.
    """

    system_id: str
    model_id: str

    # Model access: either an OpenAI-compatible base_url + model_name,
    # or a LiteLLM / Bedrock string via the gsw-memory TransportShim.
    base_url: str = ""
    model_name: str = ""
    api_key: str = ""

    # Retrieval access for adapters that delegate retrieval to us
    # (vanilla RAG+ReAct, Search-o1 wrapper, etc.). Adapters that ship
    # their own retriever (e.g. Search-R1's E5 index) ignore this.
    retriever: Any = None

    # Where to dump raw trajectory JSON, tool logs, model outputs. One
    # subdir per adapter call — the harness sets this.
    output_dir: str = ""

    # Hard caps the harness enforces on every adapter.
    # max_completion_tokens is generous because reasoning models (GPT-5,
    # o-series) consume hidden reasoning inside this budget — 2-4k is not
    # enough for them on multi-hop FRAMES questions.
    max_turns: int = 16
    max_prompt_tokens: int = 32000
    max_completion_tokens: int = 50000

    extra: dict[str, Any] = field(default_factory=dict)


class Adapter(ABC):
    """Base class for all adapters (competitor frameworks + ours)."""

    #: Canonical identifier used by the harness registry.
    system_id: ClassVar[str] = ""

    #: Human-readable label for reports.
    display_name: ClassVar[str] = ""

    #: Short description shown in the grid.
    description: ClassVar[str] = ""

    def __init__(self, ctx: AdapterContext) -> None:
        self.ctx = ctx

    @abstractmethod
    def run_question(self, question: str, *, question_id: str, articles: list[dict[str, Any]] | None = None) -> Trajectory:
        """Run the adapter on one question; return its full Trajectory.

        Parameters
        ----------
        question:
            The raw user question as it appears in the benchmark.
        question_id:
            Benchmark-assigned id, forwarded into the returned Trajectory.
        articles:
            Optional per-question article hints (e.g. FRAMES lists the
            Wikipedia titles needed). Adapters can use these as search
            targets or ignore them entirely.
        """

    def close(self) -> None:
        """Optional resource cleanup. Default is a no-op."""


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, type[Adapter]] = {}


def register_adapter(cls: type[Adapter]) -> type[Adapter]:
    """Class decorator that registers an adapter under its ``system_id``."""
    if not cls.system_id:
        raise ValueError(f"Adapter {cls.__name__} must declare system_id")
    _REGISTRY[cls.system_id] = cls
    return cls


def get_adapter(system_id: str) -> type[Adapter]:
    try:
        return _REGISTRY[system_id]
    except KeyError as exc:
        raise RegistryError(
            f"No adapter registered under system_id={system_id!r}. "
            f"Known: {sorted(_REGISTRY)}"
        ) from exc
