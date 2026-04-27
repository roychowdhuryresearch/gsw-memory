"""Table-aware Wikipedia article parser.

The original FRAMES corpus generator used the ``wikipedia`` Python
library, which discards tables / lists / infoboxes when serialising
articles to plain text. That stripped the actual data out of
list-of-X articles (e.g., "List of WNBA career scoring leaders" lost
the player names, leaving only the section headers).

This module fetches the Wikipedia REST HTML for an article and
converts it to a plaintext representation that PRESERVES:

- Section headings (with their level).
- Paragraphs (as before).
- **Tables** — flattened to one-row-per-line, columns separated by
  ``|``, with a header row and a separator.
- **Lists** — flattened to ``- item`` lines.
- **Infoboxes** — flattened to ``key: value`` lines, prefixed with
  ``[INFOBOX]``.

Output is meant to be searchable / readable plaintext — not a perfect
Wikipedia round-trip. The retrieval pipeline reads this through a
chunker that splits on paragraphs.

Disk caching is built in: each fetched article is written to a
``cache_dir/<sha256(title)>.html`` sidecar so re-runs are fast and
Wikipedia rate-limits aren't tripped.
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import quote

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

_log = logging.getLogger(__name__)


# Wikipedia REST API endpoint.
_REST_HTML = "https://en.wikipedia.org/api/rest_v1/page/html/{title}"

# Minimum delay between successive remote fetches (seconds). The REST
# API allows ~200 req/sec but we stay polite.
_FETCH_DELAY_S = 0.05

_DEFAULT_HEADERS = {
    "User-Agent": (
        "gsw-memory/research-agent FRAMES-corpus-regenerator "
        "(noreply@example.org)"
    ),
    "Accept": "text/html",
}


# ---------------------------------------------------------------------------
# HTTP fetch with disk cache
# ---------------------------------------------------------------------------


def _cache_key(title: str) -> str:
    return hashlib.sha256(title.encode("utf-8")).hexdigest()[:32]


def fetch_html(
    title: str,
    *,
    cache_dir: Optional[Path] = None,
    timeout: float = 30.0,
    session: Optional[requests.Session] = None,
) -> str:
    """Fetch the Wikipedia REST HTML for an article.

    Returns the raw HTML body. Uses ``cache_dir`` for disk caching;
    if a cached file exists for the title, it is returned without an
    HTTP call.
    """
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{_cache_key(title)}.html"
        if cache_file.exists():
            return cache_file.read_text(encoding="utf-8")

    url = _REST_HTML.format(title=quote(title.replace(" ", "_"), safe=""))
    sess = session or requests
    resp = sess.get(url, headers=_DEFAULT_HEADERS, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    if cache_dir is not None:
        cache_file = cache_dir / f"{_cache_key(title)}.html"
        cache_file.write_text(html, encoding="utf-8")

    time.sleep(_FETCH_DELAY_S)
    return html


# ---------------------------------------------------------------------------
# HTML → plaintext, preserving structured content
# ---------------------------------------------------------------------------


def _clean_text(text: str) -> str:
    """Squash whitespace; strip Wikipedia-style citation markers like '[1]'."""
    text = re.sub(r"\[\d+\]", "", text)  # citation refs
    text = re.sub(r"\[edit\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _row_text(row: Tag) -> list[str]:
    """Return one cell-text per <th>/<td> in a row, in order."""
    cells = row.find_all(["th", "td"], recursive=False)
    return [_clean_text(c.get_text(" ", strip=True)) for c in cells]


def _parse_infobox(table: Tag) -> str:
    """An infobox is rendered as ``[INFOBOX]\\nkey: value\\n…``.

    Wikipedia infoboxes are <table class="infobox …"> with rows of
    <th> (label) + <td> (value).
    """
    lines = ["[INFOBOX]"]
    # Caption (the article-title-ish header at the top of the box).
    caption = table.find("caption")
    if caption:
        cap_text = _clean_text(caption.get_text(" ", strip=True))
        if cap_text:
            lines.append(f"  caption: {cap_text}")
    for row in table.find_all("tr"):
        cells = row.find_all(["th", "td"], recursive=False)
        if len(cells) == 2:
            k = _clean_text(cells[0].get_text(" ", strip=True))
            v = _clean_text(cells[1].get_text(" ", strip=True))
            if k and v:
                lines.append(f"  {k}: {v}")
        elif len(cells) == 1:
            t = _clean_text(cells[0].get_text(" ", strip=True))
            if t:
                lines.append(f"  {t}")
    return "\n".join(lines)


def _parse_table(table: Tag) -> str:
    """Plain wikitable → header row + separator + body rows.

    Output shape (Markdown-ish):
        | col1 | col2 | col3 |
        | --- | --- | --- |
        | a | b | c |
        | d | e | f |
    """
    rows = table.find_all("tr")
    if not rows:
        return ""
    parsed_rows = [_row_text(r) for r in rows]
    parsed_rows = [r for r in parsed_rows if any(c for c in r)]
    if not parsed_rows:
        return ""

    # Header is the first row that contains any <th>; if none, use row 0
    # as a "data-only" header.
    header_idx = 0
    for i, r in enumerate(rows):
        if r.find("th"):
            header_idx = i
            break
    n_cols = max(len(r) for r in parsed_rows)
    # Normalize widths.
    parsed_rows = [r + [""] * (n_cols - len(r)) for r in parsed_rows]

    out: list[str] = []
    out.append("| " + " | ".join(parsed_rows[header_idx]) + " |")
    out.append("| " + " | ".join(["---"] * n_cols) + " |")
    for i, r in enumerate(parsed_rows):
        if i == header_idx:
            continue
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _parse_list(ul_or_ol: Tag, *, ordered: bool = False) -> str:
    """Bullet/numbered list to lines. Nested lists indent."""
    out: list[str] = []
    for i, li in enumerate(ul_or_ol.find_all("li", recursive=False), start=1):
        # Capture the immediate li text without nested lists' contents
        # twice.
        nested = li.find_all(["ul", "ol"], recursive=False)
        for n in nested:
            n.extract()
        body = _clean_text(li.get_text(" ", strip=True))
        marker = f"{i}." if ordered else "-"
        out.append(f"{marker} {body}")
        # Recurse into nested.
        for n in nested:
            sub = _parse_list(n, ordered=(n.name == "ol"))
            for line in sub.splitlines():
                out.append("  " + line)
    return "\n".join(out)


def _is_infobox(t: Tag) -> bool:
    cls = t.get("class") or []
    return "infobox" in cls or any("infobox" in c for c in cls)


def _is_navbox(t: Tag) -> bool:
    cls = t.get("class") or []
    skip = {"navbox", "sidebar", "metadata", "ambox", "vertical-navbox"}
    return any(c in skip for c in cls)


def _heading_level(t: Tag) -> int:
    if t.name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        return int(t.name[1])
    return 0


def parse_html_to_text(html: str) -> str:
    """Convert Wikipedia REST HTML into structured plaintext.

    Walks top-level body content elements in order, dispatching to
    handlers per type. Skips navboxes, references, edit-section markers.
    """
    soup = BeautifulSoup(html, "html.parser")
    # Wikipedia REST returns the article body inside a <body> or <section>.
    root = soup.body or soup

    out: list[str] = []
    # Collect top-level descendants of the article body.
    for el in root.descendants:
        # Skip text nodes and most non-tags.
        if not isinstance(el, Tag):
            continue
        # Skip elements within infoboxes/tables/lists when we'll grab
        # them at the container level — to do this we use a "consumed"
        # set tracked via element ids.
        # Simpler: only process top-level structural elements; skip
        # nested ones.
    # The above descendant walk is too noisy. Instead, walk the
    # children of the article body.

    # Find the article container — Wikipedia REST wraps content in
    # <section data-mw-section-id="..."> tags.
    sections = root.find_all("section", recursive=False)
    targets: list[Tag] = list(sections) if sections else [root]

    for section in targets:
        for el in section.children:
            if not isinstance(el, Tag):
                continue
            text = _render_element(el)
            if text:
                out.append(text)

    return "\n\n".join(out).strip()


def _render_element(el: Tag) -> str:
    """Dispatch one top-level element to the right renderer."""
    if not isinstance(el, Tag):
        return ""

    name = el.name
    if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
        text = _clean_text(el.get_text(" ", strip=True))
        if not text or text.lower() in {
            "see also", "references", "external links", "notes",
            "further reading", "bibliography",
        }:
            # Skip Wikipedia housekeeping sections.
            return f"\n## {text}\n" if text else ""
        prefix = "#" * _heading_level(el)
        return f"{prefix} {text}"

    if name == "p":
        text = _clean_text(el.get_text(" ", strip=True))
        return text

    if name == "table":
        if _is_navbox(el):
            return ""
        if _is_infobox(el):
            return _parse_infobox(el)
        return _parse_table(el)

    if name in ("ul", "ol"):
        return _parse_list(el, ordered=(name == "ol"))

    if name == "section":
        # Recurse into nested sections.
        out: list[str] = []
        for child in el.children:
            if isinstance(child, Tag):
                t = _render_element(child)
                if t:
                    out.append(t)
        return "\n\n".join(out)

    if name in ("div", "figure", "blockquote"):
        # Skip navboxes / sidebars / metadata.
        if _is_navbox(el):
            return ""
        # Recurse — sometimes a div wraps real content.
        out: list[str] = []
        for child in el.children:
            if isinstance(child, Tag):
                t = _render_element(child)
                if t:
                    out.append(t)
        return "\n\n".join(out)

    return ""


# ---------------------------------------------------------------------------
# Top-level helper
# ---------------------------------------------------------------------------


def parse_article(
    title: str,
    *,
    cache_dir: Optional[Path] = None,
    session: Optional[requests.Session] = None,
) -> str:
    """Fetch a Wikipedia article + return its table-aware plaintext.

    Returns empty string on fetch failure (logs a warning).
    """
    try:
        html = fetch_html(title, cache_dir=cache_dir, session=session)
    except requests.HTTPError as exc:
        _log.warning(f"wiki fetch failed for {title!r}: {exc}")
        return ""
    except requests.RequestException as exc:
        _log.warning(f"wiki fetch transport error for {title!r}: {exc}")
        return ""
    try:
        return parse_html_to_text(html)
    except Exception as exc:  # noqa: BLE001
        _log.warning(f"wiki parse failed for {title!r}: {exc}")
        return ""


def parse_articles(
    titles: Iterable[str],
    *,
    cache_dir: Optional[Path] = None,
) -> dict[str, str]:
    """Convenience: parse a sequence of titles, returning ``{title: text}``."""
    sess = requests.Session()
    out: dict[str, str] = {}
    for title in titles:
        out[title] = parse_article(title, cache_dir=cache_dir, session=sess)
    return out
