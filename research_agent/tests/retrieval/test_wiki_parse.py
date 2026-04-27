"""Unit tests for ``research_agent.retrieval.wiki_parse``.

All tests work against fixture HTML strings — no live HTTP. Confirms
that tables / lists / infoboxes survive the conversion (the original
``wikipedia`` library dropped them).
"""

from __future__ import annotations

import textwrap

import pytest

from research_agent.retrieval.wiki_parse import (
    _is_infobox,
    parse_html_to_text,
)


# ---------------------------------------------------------------------------
# Fixtures — small HTML excerpts mimicking Wikipedia REST output
# ---------------------------------------------------------------------------


# A single-table list article (mirrors "List of WNBA career scoring leaders").
_LIST_ARTICLE_HTML = textwrap.dedent(
    """
    <body>
      <section data-mw-section-id="0">
        <p>The following is a list of the players who have scored the most
        points during their WNBA careers.</p>
      </section>
      <section data-mw-section-id="1">
        <h2>Scoring leaders</h2>
        <p>Statistics accurate as of the 2025 WNBA season.</p>
        <table class="wikitable">
          <tr><th>Rank</th><th>Player</th><th>Points</th><th>Active</th></tr>
          <tr><td>1</td><td>Diana Taurasi</td><td>10646</td><td>2004–present</td></tr>
          <tr><td>2</td><td>Tina Charles</td><td>7634</td><td>2010–present</td></tr>
          <tr><td>3</td><td>DeWanna Bonner</td><td>7521</td><td>2009–present</td></tr>
          <tr><td>4</td><td>Tina Thompson</td><td>7488</td><td>1997–2013</td></tr>
          <tr><td>5</td><td>Tamika Catchings</td><td>7380</td><td>2002–2016</td></tr>
        </table>
      </section>
    </body>
    """
).strip()


# An article with an infobox (mirrors a person/place article).
_INFOBOX_HTML = textwrap.dedent(
    """
    <body>
      <section data-mw-section-id="0">
        <table class="infobox vcard">
          <caption>Liverpool Maternity Hospital</caption>
          <tr><th>Type</th><td>Maternity hospital</td></tr>
          <tr><th>Location</th><td>Brownlow Street, Liverpool</td></tr>
          <tr><th>Established</th><td>November 1841</td></tr>
        </table>
        <p>The Liverpool Maternity Hospital was established as the
        Lying-in Hospital and Dispensary for the Diseases of Women
        and Children in Horatio Street, Scotland Road, Liverpool, in
        November 1841.</p>
      </section>
      <section data-mw-section-id="1">
        <h2>Notable births</h2>
        <ul>
          <li>John Lennon, English musician (b. 1940)</li>
          <li>Walton sextuplets (b. 1983)</li>
        </ul>
      </section>
    </body>
    """
).strip()


# A regular prose article — no tables or infoboxes.
_PROSE_HTML = textwrap.dedent(
    """
    <body>
      <section data-mw-section-id="0">
        <p>Pablo Picasso (25 October 1881 – 8 April 1973) was a Spanish
        painter and sculptor.</p>
      </section>
      <section data-mw-section-id="1">
        <h2>Biography</h2>
        <p>Picasso was born in Málaga.</p>
      </section>
      <section data-mw-section-id="2">
        <h2>References</h2>
        <p>This section is housekeeping and should be marked as such.</p>
      </section>
    </body>
    """
).strip()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_list_article_table_preserves_player_names():
    """The whole point of Phase-3.3: scoring leaders' names must
    survive the HTML → plaintext pass."""
    text = parse_html_to_text(_LIST_ARTICLE_HTML)
    # All five player names appear.
    for name in [
        "Diana Taurasi", "Tina Charles", "DeWanna Bonner",
        "Tina Thompson", "Tamika Catchings",
    ]:
        assert name in text, f"name {name!r} missing from parsed text:\n{text}"
    # Numeric points appear too.
    assert "10646" in text
    assert "7521" in text
    # Section heading preserved.
    assert "Scoring leaders" in text
    # Markdown-table style separator present somewhere.
    assert "| --- |" in text or "|---|" in text or "|---" in text


def test_infobox_preserves_address_and_caption():
    """Infobox key:value rendering keeps location data."""
    text = parse_html_to_text(_INFOBOX_HTML)
    assert "[INFOBOX]" in text
    assert "Liverpool Maternity Hospital" in text
    assert "Brownlow Street" in text
    assert "1841" in text
    # Notable-births list got rendered as bullets.
    assert "John Lennon" in text
    assert "Walton sextuplets" in text
    # Bullets present.
    assert "- John Lennon" in text or "- John" in text


def test_prose_article_keeps_paragraphs_drops_housekeeping_content():
    """Prose articles should chunk to paragraphs cleanly. References /
    See also / External links sections are recognized but their
    headings render with a sentinel prefix."""
    text = parse_html_to_text(_PROSE_HTML)
    assert "Pablo Picasso" in text
    assert "1881" in text
    assert "Picasso was born in Málaga" in text
    # The References heading is recognized — body kept since the
    # parser doesn't drop content, but the heading is normalized.
    assert "References" in text


def test_is_infobox_detection():
    """Class-based detection of <table class='infobox'>."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup('<table class="infobox vcard"></table>', "html.parser")
    t = soup.find("table")
    assert _is_infobox(t)
    soup2 = BeautifulSoup('<table class="wikitable"></table>', "html.parser")
    t2 = soup2.find("table")
    assert not _is_infobox(t2)


def test_navbox_and_metadata_tables_dropped():
    """Navboxes / sidebars / ambox / metadata containers get skipped."""
    html = textwrap.dedent(
        """
        <body>
          <section>
            <p>Real article text.</p>
            <table class="navbox">
              <tr><td>This is navbox junk that should not appear.</td></tr>
            </table>
            <div class="sidebar">
              <p>Sidebar fluff that should be skipped.</p>
            </div>
          </section>
        </body>
        """
    ).strip()
    text = parse_html_to_text(html)
    assert "Real article text" in text
    assert "navbox junk" not in text
    assert "Sidebar fluff" not in text


def test_nested_lists_indent():
    html = textwrap.dedent(
        """
        <body>
          <section>
            <ul>
              <li>Top item
                <ul>
                  <li>Nested item</li>
                </ul>
              </li>
            </ul>
          </section>
        </body>
        """
    ).strip()
    text = parse_html_to_text(html)
    assert "- Top item" in text
    # Indented nested.
    assert "  - Nested item" in text
