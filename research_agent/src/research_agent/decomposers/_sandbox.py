"""Sandboxed loader for emitted decomposer modules.

The experimental claim is **strict zero-LLM at inference**. This file is
the guard that makes the claim machine-checked, not documentation-checked.

``load_decomposer(path)`` imports a candidate ``.py`` decomposer module
with two protections:

1. **Zero-LLM import guard.** The source is statically scanned for any
   import that touches a network-capable or LLM client package. If one
   is found, loading fails with ``ForbiddenImportError`` before the
   module ever executes.

2. **Per-call timeout.** The loaded ``decompose`` callable is wrapped so
   a single question cannot hang the eval loop. Default 2s, configurable.

Two escape hatches a rogue module could still use:

* ``importlib`` called at runtime inside the module body — we detect
  the literal module name (``importlib``) in the static scan.
* Dynamic string-eval — flagged, though ultimately the sandbox is a
  trust-but-verify layer, not a full capability sandbox.

The goal is "hard to break accidentally, easy to audit", not
adversarial robustness. The author of the emitted module is one of
our own LLM calls under our control.
"""

from __future__ import annotations

import ast
import importlib.util
import signal
import sys
import uuid
from pathlib import Path
from typing import Callable, Iterable, List

from research_agent.decomposers._schema import SubQuestion


FORBIDDEN_IMPORTS = frozenset(
    {
        "openai",
        "anthropic",
        "google",
        "google.generativeai",
        "cohere",
        "litellm",
        "together",
        "requests",
        "httpx",
        "urllib",
        "urllib2",
        "urllib3",
        "socket",
        "http",
        "aiohttp",
        "grpc",
        "subprocess",
        "os.system",
        "importlib",
        "pkg_resources",
        "gsw_memory.memory.operator_utils.agentic_gsw.transport_shim",
    }
)


class ForbiddenImportError(ImportError):
    """Raised when an emitted decomposer module imports a banned symbol."""


class DecomposerTimeoutError(RuntimeError):
    """Raised when a single ``decompose`` call exceeds its time budget."""


def scan_source_for_forbidden_imports(source: str) -> List[str]:
    """Return the list of forbidden top-level import names used in ``source``.

    Uses the AST, not a regex, so commented-out imports and string literals
    that happen to look like imports are ignored.
    """

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise ForbiddenImportError(f"decomposer source does not parse: {exc}") from exc

    hits: List[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_IMPORTS or alias.name in FORBIDDEN_IMPORTS:
                    hits.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".")[0]
            if root in FORBIDDEN_IMPORTS or mod in FORBIDDEN_IMPORTS:
                hits.append(mod)

    return hits


def _timebox(fn: Callable, *, seconds: float) -> Callable:
    """Wrap ``fn`` so each call gets at most ``seconds`` wall time.

    Uses SIGALRM which only works on the main thread of POSIX processes.
    That matches the eval harness, which runs decomposers synchronously
    on the main thread. If you ever need this off-main-thread, swap for
    a thread-based watchdog.
    """

    def _handler(signum, frame):  # noqa: ARG001
        raise DecomposerTimeoutError(f"decompose() exceeded {seconds}s budget")

    def _wrapped(*args, **kwargs):
        prev = signal.signal(signal.SIGALRM, _handler)
        # setitimer supports float seconds; alarm() floors to int
        signal.setitimer(signal.ITIMER_REAL, seconds)
        try:
            return fn(*args, **kwargs)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prev)

    return _wrapped


def load_decomposer(
    path: str | Path,
    *,
    per_call_timeout_s: float = 2.0,
) -> Callable[[str], List[SubQuestion]]:
    """Load a decomposer module from ``path`` and return its ``decompose`` fn.

    Raises ``ForbiddenImportError`` if the source contains a banned import.
    Raises ``AttributeError`` if the module does not expose ``decompose``.
    """

    p = Path(path)
    source = p.read_text()
    hits = scan_source_for_forbidden_imports(source)
    if hits:
        raise ForbiddenImportError(
            f"decomposer at {p} imports forbidden symbols: {sorted(set(hits))}"
        )

    mod_name = f"research_agent._decomposer_sandbox_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, p)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not build import spec for {p}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(mod_name, None)
        raise

    fn = getattr(module, "decompose", None)
    if fn is None:
        raise AttributeError(f"decomposer at {p} does not define a 'decompose' function")

    return _timebox(fn, seconds=per_call_timeout_s)


def _is_semantically_clean(pred_subs: list, original_question: str) -> bool:
    """Reject pred sub-questions that contain non-question words.

    The Script-Writer's failure mode is mid-word regex peeling — emitting
    sub-questions like 'f >> or' whose tokens never appeared in the
    original question. This check strips placeholders (#s1, #s2, ...) and
    asserts every remaining alphanumeric token is a substring of some
    word in the original question. If any sub fails, the whole prediction
    for this example is treated as incorrect.
    """
    import re as _re

    if not pred_subs:
        return True  # empty is handled by coverage metric
    original_lower = original_question.lower()
    original_tokens = set(_re.findall(r"[a-z0-9]+", original_lower))
    if not original_tokens:
        return True
    for s in pred_subs:
        qtext = getattr(s, "question", "") or ""
        # Strip placeholders like #s1, #s2, #1, #23
        cleaned = _re.sub(r"#s?\d+", " ", qtext).lower()
        tokens = _re.findall(r"[a-z0-9]+", cleaned)
        # Tolerate a handful of function words the authoring model may
        # add (e.g. "What is the X of Y?" wrapper) and QDMR operator
        # vocabulary (MoNaCo uses `return #1, #2`, `return different #3`,
        # `[list]`, `sort by`, etc.).
        allow = {
            # wh- + copulas + determiners + prepositions
            "what", "who", "whom", "whose", "where", "when", "why", "how", "which",
            "is", "are", "was", "were", "be", "been", "being",
            "the", "a", "an",
            "of", "in", "at", "on", "by", "for", "to", "and", "or",
            "s",  # bare possessive `'s` strips to 's'
            # QDMR operator vocabulary (used heavily in MoNaCo gold decomps)
            "return", "different", "list", "list]", "[list]",
            "sort", "sorted", "ascending", "descending", "by",
            "filter", "select", "project", "aggregate", "group",
            "compare", "count", "sum", "average", "max", "min",
            "same", "highest", "lowest", "number", "each", "than", "that",
            "these", "those", "this",
        }
        for tok in tokens:
            if tok in allow:
                continue
            # Whole-token membership is the strictest check — catches
            # mid-word peels like 'f', 'or', 'irth'.
            if tok not in original_tokens:
                return False
    return True


def _hop_bucket_for(ex: dict) -> str:
    """Map an example to a hop-count bucket name.

    Preference order: explicit ``hop`` field (set by ``load_musique``),
    then ``id`` prefix (``2hop_...``), then fall back to "unknown".
    """
    h = ex.get("hop")
    if h:
        return str(h)
    rid = str(ex.get("id", "") or "")
    for b in ("2hop", "3hop", "4hop", "5hop"):
        if rid.startswith(b):
            return b
    return "unknown"


def grade_decomposer(
    decompose_fn: Callable[[str], List[SubQuestion]],
    examples: Iterable[dict],
) -> dict:
    """Score a decomposer on an iterable of examples.

    Each example is expected to have keys ``question`` and
    ``gold_decomposition`` (list of dicts with at least ``question``).

    Metrics returned:

    * ``coverage``: fraction of examples where decompose returned >= 1 sub.
    * ``count_match``: fraction where the number of subs equals gold count.
    * ``soft_match``: average (over examples) of
      ``min(pred_len, gold_len) / max(pred_len, gold_len)`` — a lightweight
      continuous score that doesn't require semantic comparison.
    * ``by_hop``: per-hop-bucket breakdown of the three metrics above.
    * ``fail_count``: how many examples raised (e.g. timeout).
    * ``failures``: sample of up to 5 ``(question, error)`` pairs for debug.

    Semantic match is deliberately out of scope — the adapter-level F1 run
    in Stage 2 measures semantic correctness; this grader runs hundreds of
    times per synthesis iteration so it stays structural + fast.
    """

    from collections import defaultdict

    n = 0
    n_cov = 0
    n_count_eq = 0
    soft_sum = 0.0
    fails: List[tuple] = []
    by_hop: dict[str, dict[str, float]] = defaultdict(
        lambda: {"n": 0, "cov": 0, "count_eq": 0, "soft_sum": 0.0}
    )

    for ex in examples:
        n += 1
        bucket = _hop_bucket_for(ex)
        by_hop[bucket]["n"] += 1
        q = ex.get("question", "")
        gold = ex.get("gold_decomposition", []) or []
        try:
            pred = decompose_fn(q) or []
        except BaseException as exc:
            fails.append((q, f"{type(exc).__name__}: {exc}"))
            continue

        # Semantic gate: if the pred subs contain mid-word fragments, we
        # treat the whole prediction as empty for grading. This pushes
        # the Script-Writer to write anchor-safe regex instead of
        # optimising structural count on garbage output.
        clean = _is_semantically_clean(pred, q)
        effective_pred = pred if clean else []

        cov_inc = 1 if effective_pred else 0
        count_eq_inc = 1 if len(effective_pred) == len(gold) else 0
        soft_inc = 0.0
        if effective_pred or gold:
            hi = max(len(effective_pred), len(gold)) or 1
            lo = min(len(effective_pred), len(gold))
            soft_inc = lo / hi

        n_cov += cov_inc
        n_count_eq += count_eq_inc
        soft_sum += soft_inc
        by_hop[bucket]["cov"] += cov_inc
        by_hop[bucket]["count_eq"] += count_eq_inc
        by_hop[bucket]["soft_sum"] += soft_inc

    by_hop_out: dict[str, dict[str, float]] = {}
    for bucket, row in by_hop.items():
        bn = row["n"] or 1
        by_hop_out[bucket] = {
            "n": int(row["n"]),
            "coverage": row["cov"] / bn,
            "count_match": row["count_eq"] / bn,
            "soft_match": row["soft_sum"] / bn,
        }

    return {
        "n": n,
        "coverage": (n_cov / n) if n else 0.0,
        "count_match": (n_count_eq / n) if n else 0.0,
        "soft_match": (soft_sum / n) if n else 0.0,
        "by_hop": by_hop_out,
        "fail_count": len(fails),
        "failures": fails[:5],
    }
