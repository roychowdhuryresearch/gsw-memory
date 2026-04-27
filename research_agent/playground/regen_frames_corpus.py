"""Regenerate the FRAMES articles cache with table-aware parsing.

Reads the titles from the existing
``data/sleep_time/frames/articles/articles_cache.json`` and re-fetches
each one via ``research_agent.retrieval.wiki_parse.parse_article``,
which preserves tables / lists / infoboxes that the original
``wikipedia`` library dropped.

Output is written to ``articles_cache_v2.json`` next to the original
(non-destructive — the v1 cache stays put). The new corpus is loaded
by passing ``cache_path=…/articles_cache_v2.json`` to
``load_frames_corpus``.

Usage::

    .venv/bin/python playground/regen_frames_corpus.py
        [--limit N]   # only re-parse the first N titles (smoke test)
        [--titles t1,t2,...]  # only re-parse these specific titles
        [--cache-dir PATH]  # disk-cache for HTML responses
        [--out PATH]  # output path; default = articles_cache_v2.json next to v1

The HTML disk cache (default
``data/sleep_time/frames/articles/wiki_html_cache/``) makes re-runs
cheap — once an article's HTML is on disk, no HTTP call is made.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Make src importable when launched without `pip install -e .`
_HERE = Path(__file__).resolve()
_SRC = _HERE.parent.parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from research_agent.retrieval.wiki_parse import parse_article  # noqa: E402


_GSW_ROOT = Path(os.environ.get("GSW_MEMORY_ROOT", "/home/yigit/codebase/gsw-memory"))
_DEFAULT_CACHE_V1 = _GSW_ROOT / "data" / "sleep_time" / "frames" / "articles" / "articles_cache.json"
_DEFAULT_CACHE_V2 = _GSW_ROOT / "data" / "sleep_time" / "frames" / "articles" / "articles_cache_v2.json"
_DEFAULT_HTML_CACHE = _GSW_ROOT / "data" / "sleep_time" / "frames" / "articles" / "wiki_html_cache"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-v1", type=Path, default=_DEFAULT_CACHE_V1,
                   help="Existing articles_cache.json (used for the title list).")
    p.add_argument("--out", type=Path, default=_DEFAULT_CACHE_V2,
                   help="Output path for the v2 cache.")
    p.add_argument("--cache-dir", type=Path, default=_DEFAULT_HTML_CACHE,
                   help="Disk cache for raw Wikipedia HTML responses.")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, only re-parse the first N titles.")
    p.add_argument("--titles", type=str, default="",
                   help="Comma-separated list of titles to re-parse. "
                        "If set, --limit and the v1 cache title list are ignored.")
    p.add_argument("--resume", action="store_true",
                   help="If --out exists, load it and only fetch missing titles.")
    p.add_argument("--checkpoint-every", type=int, default=25,
                   help="Save partial progress to --out every N articles.")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    # Build the list of titles to re-parse.
    if args.titles.strip():
        titles = [t.strip() for t in args.titles.split(",") if t.strip()]
    else:
        if not args.cache_v1.exists():
            print(f"v1 cache not found: {args.cache_v1}", file=sys.stderr)
            return 2
        with args.cache_v1.open() as f:
            v1: dict[str, str] = json.load(f)
        titles = list(v1.keys())
        if args.limit > 0:
            titles = titles[: args.limit]

    out_data: dict[str, str] = {}
    if args.resume and args.out.exists():
        try:
            with args.out.open() as f:
                out_data = json.load(f)
            print(f"resuming: {len(out_data)} titles already in {args.out.name}")
        except json.JSONDecodeError:
            out_data = {}

    todo = [t for t in titles if t not in out_data or not out_data[t]]
    print(f"plan: {len(titles)} total, {len(todo)} to fetch "
          f"(skipped {len(titles) - len(todo)} already cached in --out)")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    started = time.time()
    n_done = 0
    n_empty = 0
    for i, title in enumerate(todo, start=1):
        text = parse_article(title, cache_dir=args.cache_dir)
        out_data[title] = text
        if not text:
            n_empty += 1
        n_done += 1
        if args.verbose or i % 10 == 0:
            elapsed = time.time() - started
            avg = elapsed / max(n_done, 1)
            eta = avg * (len(todo) - i)
            print(
                f"  [{i}/{len(todo)}] {title}  "
                f"len={len(text):>6} chars  "
                f"({n_empty} empties so far)  "
                f"avg={avg:.2f}s  eta={eta/60:.1f}min"
            )
        if i % args.checkpoint_every == 0:
            args.out.write_text(json.dumps(out_data, indent=0))
            print(f"  ✓ checkpoint saved ({len(out_data)} titles)")

    args.out.write_text(json.dumps(out_data, indent=0))
    elapsed = time.time() - started
    print(
        f"\ndone in {elapsed:.1f}s. wrote {len(out_data)} titles to {args.out}. "
        f"{n_empty} fetches returned empty text."
    )
    avg_len = (
        sum(len(t) for t in out_data.values()) / max(len(out_data), 1)
    )
    print(f"avg article length: {avg_len:.0f} chars")
    return 0


if __name__ == "__main__":
    sys.exit(main())
