"""LLM-authored deterministic decomposers.

At inference time, an adapter imports the emitted module here and calls
``decompose(question) -> list[SubQuestion]``. Zero LLM calls at inference.

The decomposer modules themselves are emitted offline by the
Script-Writer pipeline in ``_synthesis.py``. Every emitted module is
loaded through ``_sandbox.load_decomposer`` so the zero-LLM guarantee is
machine-checked.

Modules:

* ``_schema``    — canonical ``SubQuestion`` dataclass.
* ``_sandbox``   — import guard + timeout wrapper.
* ``_synthesis`` — draft / grade / repair loop that authors decomposers.
* ``<dataset>_specialized.py`` — emitted per-dataset modules.
* ``generalized.py``           — emitted single cross-dataset module.
"""

from research_agent.decomposers._schema import SubQuestion
from research_agent.decomposers._sandbox import (
    ForbiddenImportError,
    DecomposerTimeoutError,
    load_decomposer,
    grade_decomposer,
    scan_source_for_forbidden_imports,
)

__all__ = [
    "SubQuestion",
    "ForbiddenImportError",
    "DecomposerTimeoutError",
    "load_decomposer",
    "grade_decomposer",
    "scan_source_for_forbidden_imports",
]
