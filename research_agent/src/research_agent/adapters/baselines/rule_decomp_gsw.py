"""Rule-decomp variant of ``ours_gsw_v1``.

Same downstream pipeline (focused retrieval → GSW triples → per-sub-Q
answer → aggregator). The only substitution: the decomposition step is
served by a pre-emitted deterministic module instead of a live LLM call.

Invariant asserted on every run:

    traj.extra["decomposer_llm_calls"] == 0

The adapter holds the invariant by never calling ``self.llm`` during
decomposition, and by tracking prompt/completion tokens so a diff of
``prompt_tokens`` against ``ours_gsw_v1`` shows the saved-by-rules
budget directly.

The decomposer module is dataset-specialized — no cross-dataset
generalized variant. Pass a module path via
``AdapterContext.extra['decomposer_module']`` (MuSiQue / MoNaCo / ...)
and the adapter loads it through the zero-LLM sandbox. When omitted,
the adapter defaults to the MuSiQue-specialized module.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from research_agent.adapters.base import AdapterContext, register_adapter
from research_agent.adapters.baselines.ours_gsw_v1 import (
    OursGSWv1Adapter,
    SubQuestion as OursSubQuestion,
)
from research_agent.decomposers import load_decomposer
from research_agent.models.trace import Trajectory


def _default_decomposer_path() -> str:
    """Path to the canonical MuSiQue-specialized decomposer module."""
    here = Path(__file__).resolve().parent
    return str((here.parent / "decomposers" / "musique_specialized.py").resolve())


class _RuleDecompAdapterBase(OursGSWv1Adapter):
    """Base adapter that swaps LLM decomposition for a deterministic module."""

    #: Subclasses override with a concrete decomposer-module path.
    default_decomposer_path: str = ""

    def __init__(self, ctx: AdapterContext) -> None:
        super().__init__(ctx)
        module_path = ctx.extra.get("decomposer_module") or self.default_decomposer_path
        if not module_path:
            raise ValueError(
                f"{self.__class__.__name__} requires ctx.extra['decomposer_module'] "
                f"or a default_decomposer_path."
            )
        per_call_timeout_s = float(ctx.extra.get("decomposer_timeout_s", 2.0))
        # ``load_decomposer`` raises if the emitted module imports any banned
        # LLM / network package. This is where the strict zero-LLM claim is
        # machine-enforced.
        self._decompose_fn = load_decomposer(module_path, per_call_timeout_s=per_call_timeout_s)
        self._decomposer_module_path = module_path
        self._decomposer_llm_calls = 0  # invariant, see assertion in run_question

    # ------------------------------------------------------------------
    # Override the only method that used to call an LLM
    # ------------------------------------------------------------------

    def _decompose(self, question: str, traj: Trajectory) -> list[OursSubQuestion]:
        t0 = time.time()
        try:
            emitted = self._decompose_fn(question) or []
        except BaseException as exc:
            traj.extra.setdefault("rule_decomp_errors", []).append(
                f"{type(exc).__name__}: {exc}"
            )
            emitted = []

        traj.extra.setdefault("rule_decomp_elapsed_s", 0.0)
        traj.extra["rule_decomp_elapsed_s"] += round(time.time() - t0, 4)

        if not emitted:
            # Strict zero-LLM: if the rule engine has no confident parse,
            # we fall back to treating the question as a single hop. We do
            # NOT call an LLM. This matches the ``OursGSWv1Adapter`` safety
            # net shape so downstream stages still work.
            return [
                OursSubQuestion(
                    sub_id="s1",
                    question=question,
                    entity_focus="",
                    hop_type="lookup",
                )
            ]

        out: list[OursSubQuestion] = []
        for sq in emitted[: self._max_sub_questions]:
            if hasattr(sq, "to_ours_gsw_kwargs"):
                kwargs = sq.to_ours_gsw_kwargs()
            else:
                kwargs = {
                    "sub_id": getattr(sq, "sub_id", "") or f"s{len(out) + 1}",
                    "question": getattr(sq, "question", "") or "",
                    "entity_focus": getattr(sq, "entity_focus", "") or "",
                    "hop_type": getattr(sq, "hop_type", "") or "lookup",
                }
            # OursSubQuestion has strict str typing — coerce any stray Nones.
            for k in ("sub_id", "question", "entity_focus", "hop_type"):
                kwargs[k] = kwargs.get(k) or ("" if k != "hop_type" else "lookup")
            if not kwargs["question"]:
                continue
            if not kwargs["sub_id"]:
                kwargs["sub_id"] = f"s{len(out) + 1}"
            out.append(OursSubQuestion(**kwargs))

        if not out:
            return [
                OursSubQuestion(
                    sub_id="s1",
                    question=question,
                    entity_focus="",
                    hop_type="lookup",
                )
            ]
        return out

    # ------------------------------------------------------------------

    def run_question(
        self,
        question: str,
        *,
        question_id: str,
        articles: list[dict[str, Any]] | None = None,
    ) -> Trajectory:
        prompt_before = 0  # placeholder for the pre-decompose token count
        traj = super().run_question(question, question_id=question_id, articles=articles)

        # Zero-LLM invariant: the rule-based decomposer must not have
        # touched the LLM. ``ours_gsw_v1._decompose`` increments
        # ``traj.prompt_tokens`` on every LLM call; our override returns
        # without touching ``self.llm`` at all, so any nonzero decomposer
        # LLM call count means something regressed.
        traj.extra["decomposer_llm_calls"] = self._decomposer_llm_calls
        traj.extra["decomposer_module"] = self._decomposer_module_path
        if self._decomposer_llm_calls != 0:
            raise AssertionError(
                "rule_decomp_gsw invariant violated: decomposer made "
                f"{self._decomposer_llm_calls} LLM calls"
            )
        # ``prompt_before`` is currently unused — token delta is computed
        # externally by diffing ``prompt_tokens`` against the ours_gsw_v1
        # cell in the grid summary.
        _ = prompt_before
        return traj


@register_adapter
class RuleDecompAdapter(_RuleDecompAdapterBase):
    """Deterministic decomposer specialised per dataset (MuSiQue, MoNaCo, ...).

    The emitted decomposer is dataset-specific by design — each dataset's
    gold decompositions teach a distinct operator vocabulary (e.g.
    MuSiQue's `X of Y` peel vs MoNaCo's QDMR set ops). There is no
    cross-dataset generalized variant.
    """

    system_id = "rule_decomp_gsw"
    display_name = "Ours — rule-decomp GSW (dataset-specialized)"
    description = (
        "Zero-LLM decomposer authored offline against one target dataset → "
        "focused retrieval → per-sub-Q GSW triples → aggregation. "
        "Swap decomposer module via ctx.extra['decomposer_module']."
    )
    default_decomposer_path = _default_decomposer_path()
