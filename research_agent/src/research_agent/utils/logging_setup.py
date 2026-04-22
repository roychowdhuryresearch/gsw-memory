"""Run-scoped logging setup.

All research-agent code logs through the ``research_agent`` logger
hierarchy. ``setup_run_logger(run_dir)`` wires it up with two handlers:

- A ``StreamHandler`` on stderr so progress shows live in the terminal.
- A ``FileHandler`` writing to ``run_dir / run.log`` so we keep every
  line of terminal output alongside the per-question trace JSONs.

The caller decides the console level via ``verbose``:
``False`` → INFO, ``True`` → DEBUG (includes per-turn tool calls).
The file handler always writes at DEBUG so the full story survives.
"""

from __future__ import annotations

import logging
from pathlib import Path

_ROOT_NAME = "research_agent"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a research-agent-scoped logger. ``name`` is appended to the root."""
    if not name:
        return logging.getLogger(_ROOT_NAME)
    if name.startswith(_ROOT_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_NAME}.{name}")


def setup_run_logger(
    run_dir: Path | str,
    *,
    verbose: bool = False,
    log_filename: str = "run.log",
) -> logging.Logger:
    """Configure the root ``research_agent`` logger with stream + file handlers.

    Idempotent — safe to call multiple times in one process. The previous
    run's handlers are removed first so ``logs/<cell_A>/run.log`` doesn't
    silently receive output from ``<cell_B>``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / log_filename

    logger = logging.getLogger(_ROOT_NAME)
    logger.setLevel(logging.DEBUG)

    # Drop any pre-existing handlers from a prior cell.
    for h in list(logger.handlers):
        logger.removeHandler(h)
        try:
            h.close()
        except Exception:  # noqa: BLE001
            pass

    fmt = logging.Formatter("%(asctime)s %(levelname)-5s %(name)s | %(message)s", datefmt="%H:%M:%S")

    stream = logging.StreamHandler()
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # Quiet noisy libraries at INFO level so the terminal stays readable.
    for noisy in ("httpx", "urllib3", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.propagate = False
    logger.info("log file: %s (console=%s, file=DEBUG)", log_path, "DEBUG" if verbose else "INFO")
    return logger
