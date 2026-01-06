#!/usr/bin/env python3
"""
Create Notion pages from GSW parameter search results.

Visualizes pred_gsws_{params}.json files with:
- Text (original input)
- Gold GSW (with nested toggles for entity_nodes, verb_phrase_nodes)
- Pred GSW (with nested toggles for entity_nodes, verb_phrase_nodes)
- Judgement results (optional)

Usage:
    uv run python playground/gsw_creation_local/notion_param_search_viz.py \
        --pred-gsws-file playground/gsw_creation_local/param_search_20251230_141159/pred_gsws_t0.6_p0.9_k20_m0.0.json \
        --judgements-file playground/gsw_creation_local/param_search_20251230_141159/judgements_t0.6_p0.9_k20_m0.0.json
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from notion_client import Client

MAX_RICH_TEXT_CHARS = 2000
MAX_RICH_TEXT_ITEMS = 100


# --- Notion helpers ---
def rt(text: str) -> dict:
    return {"type": "text", "text": {"content": text, "link": None}}


def paragraph(text: str) -> dict:
    return {"type": "paragraph", "paragraph": {"rich_text": [rt(text)]}}


def chunk_text(s: str, maxlen: int = MAX_RICH_TEXT_CHARS):
    chunks = []
    start = 0
    while start < len(s):
        end = min(start + maxlen, len(s))
        nl = s.rfind("\n", start, end)
        if nl > -1 and nl > start:
            end = nl + 1
        chunks.append(s[start:end])
        start = end
    return chunks


def code_blocks_for_text(text: str, language: str):
    segments = chunk_text(text)
    blocks = []
    for i in range(0, len(segments), MAX_RICH_TEXT_ITEMS):
        piece = segments[i : i + MAX_RICH_TEXT_ITEMS]
        blocks.append(
            {
                "type": "code",
                "code": {
                    "rich_text": [rt(seg) for seg in piece],
                    "language": language,
                    "caption": [],
                },
            }
        )
    return blocks


def toggle_block(title: str, children):
    if not children:
        children = [paragraph("No content found.")]
    return {"type": "toggle", "toggle": {"rich_text": [rt(title)], "children": children}}


def heading_block(text: str, level: int = 2) -> dict:
    """Create a heading block (level 1, 2, or 3)."""
    heading_type = f"heading_{level}"
    return {
        "type": heading_type,
        heading_type: {"rich_text": [rt(text)], "is_toggleable": False}
    }


def divider_block() -> dict:
    return {"type": "divider", "divider": {}}


# --- file reading ---
def load_json(p: Path):
    """Load JSON and return data and an error string if any."""
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f), None
    except Exception as e:
        return None, f"Error reading {p.name}: {e}"


# --- Notion page creation ---
def create_page(notion, parent_page_id, title, children, icon_emoji=None):
    payload = {
        "parent": {"page_id": parent_page_id},
        "properties": {"title": {"title": [rt(title)]}},
        "children": children,
    }
    if icon_emoji:
        payload["icon"] = {"type": "emoji", "emoji": icon_emoji}
    return notion.pages.create(**payload)


# --- GSW nested toggle builder ---
def build_gsw_toggle(gsw_data: dict, title: str, emoji: str = "") -> dict:
    """
    Build a toggle for GSW data with nested toggles for:
    - entity_nodes
    - verb_phrase_nodes
    """
    if gsw_data is None:
        return toggle_block(f"{emoji} {title}".strip(), [paragraph("No GSW data (prediction failed).")])

    nested = []

    # Entity nodes
    if "entity_nodes" in gsw_data:
        pretty = json.dumps(gsw_data["entity_nodes"], indent=2, ensure_ascii=False)
        nested.append(toggle_block("entity_nodes", code_blocks_for_text(pretty, "json")))
    else:
        nested.append(toggle_block("entity_nodes", [paragraph("Not found.")]))

    # Verb phrase nodes
    if "verb_phrase_nodes" in gsw_data:
        pretty = json.dumps(gsw_data["verb_phrase_nodes"], indent=2, ensure_ascii=False)
        nested.append(toggle_block("verb_phrase_nodes", code_blocks_for_text(pretty, "json")))
    else:
        nested.append(toggle_block("verb_phrase_nodes", [paragraph("Not found.")]))

    # Full GSW JSON
    full_pretty = json.dumps(gsw_data, indent=2, ensure_ascii=False)
    nested.append(toggle_block("Full JSON", code_blocks_for_text(full_pretty, "json")))

    return toggle_block(f"{emoji} {title}".strip(), nested)


def build_judgement_toggle(judgement_data: dict) -> dict:
    """Build a toggle for judgement results."""
    if judgement_data is None:
        return toggle_block("Judgement", [paragraph("No judgement data.")])

    nested = []

    # Summary stats
    if "judgement" in judgement_data and judgement_data["judgement"]:
        j = judgement_data["judgement"]
        summary_lines = [
            f"Overall Score: {j.get('overall_score', 'N/A')}",
            f"Usable for QA: {j.get('usable_for_QA', 'N/A')}",
        ]
        if "subscores" in j:
            ss = j["subscores"]
            summary_lines.extend([
                f"Coverage: {ss.get('coverage', 'N/A')}",
                f"Precision: {ss.get('precision', 'N/A')}",
                f"Format Compliance: {ss.get('format_compliance', 'N/A')}",
            ])
        nested.append(toggle_block("Summary", [paragraph("\n".join(summary_lines))]))

        # Critical violations
        if j.get("critical_violations"):
            violations_text = json.dumps(j["critical_violations"], indent=2, ensure_ascii=False)
            nested.append(toggle_block("Critical Violations", code_blocks_for_text(violations_text, "json")))

        # Fact comparison
        if j.get("fact_comparison"):
            fact_text = json.dumps(j["fact_comparison"], indent=2, ensure_ascii=False)
            nested.append(toggle_block("Fact Comparison", code_blocks_for_text(fact_text, "json")))

        # Full judgement
        full_pretty = json.dumps(j, indent=2, ensure_ascii=False)
        nested.append(toggle_block("Full Judgement JSON", code_blocks_for_text(full_pretty, "json")))
    elif "error" in judgement_data:
        nested.append(paragraph(f"Error: {judgement_data['error']}"))
    else:
        nested.append(paragraph("No judgement details available."))

    return toggle_block("Judgement", nested)


def main():
    parser = argparse.ArgumentParser(description="Visualize GSW param search results in Notion")
    parser.add_argument("--parent-page-id", default="2da40a7c71d2801fa4deea00af23ed6b",
                        help="Notion parent page ID")
    parser.add_argument("--pred-gsws-file", type=Path, required=True,
                        help="Path to pred_gsws JSON file")
    parser.add_argument("--judgements-file", type=Path, default=None,
                        help="Path to judgements JSON file (optional)")
    parser.add_argument("--page-title", type=str, default=None,
                        help="Custom title for the container page")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to visualize")
    parser.add_argument("--sleep", type=float, default=0.3,
                        help="Sleep between API calls")
    parser.add_argument("--notion-token", default="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",
                        help="Notion API token")
    args = parser.parse_args()

    if not args.notion_token:
        print("Missing Notion token (set --notion-token).", file=sys.stderr)
        sys.exit(1)

    notion = Client(auth=args.notion_token)

    # Load pred_gsws
    print(f"Loading pred_gsws from {args.pred_gsws_file}...")
    pred_gsws_data, err = load_json(args.pred_gsws_file)
    if err:
        print(f"Error: {err}", file=sys.stderr)
        sys.exit(1)

    # Load judgements if provided
    judgements_map = {}
    if args.judgements_file and args.judgements_file.exists():
        print(f"Loading judgements from {args.judgements_file}...")
        judgements_data, err = load_json(args.judgements_file)
        if err:
            print(f"Warning: {err}")
        else:
            # Build map by global_id for easy lookup
            for j in judgements_data:
                judgements_map[j.get("global_id", j.get("sample_id"))] = j

    # Limit samples if requested
    if args.max_samples:
        pred_gsws_data = pred_gsws_data[:args.max_samples]

    # Create container page title
    if args.page_title:
        container_title = args.page_title
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        param_str = args.pred_gsws_file.stem.replace("pred_gsws_", "")
        container_title = f"ParamSearch-{param_str}-{timestamp}"

    # Calculate summary stats
    total_samples = len(pred_gsws_data)
    failed_preds = sum(1 for d in pred_gsws_data if d.get("pred_gsw") is None)

    if judgements_map:
        scores = [j.get("overall_score") for j in judgements_map.values() if j.get("overall_score") is not None]
        avg_score = sum(scores) / len(scores) if scores else 0
        usable_count = sum(1 for j in judgements_map.values()
                          if j.get("judgement", {}).get("usable_for_QA", False))
    else:
        avg_score = None
        usable_count = None

    # Build summary block
    summary_lines = [
        f"Total samples: {total_samples}",
        f"Failed predictions: {failed_preds}",
    ]
    if avg_score is not None:
        summary_lines.append(f"Average score: {avg_score:.2f}")
        summary_lines.append(f"Usable for QA: {usable_count}/{total_samples}")

    summary_block = toggle_block("Summary Statistics", [paragraph("\n".join(summary_lines))])

    # Create container page
    container = create_page(
        notion,
        parent_page_id=args.parent_page_id,
        title=container_title,
        children=[
            paragraph(f"Parameter search results from: {args.pred_gsws_file.name}"),
            summary_block,
            divider_block(),
        ],
        icon_emoji="🔬",
    )
    container_id = container["id"]
    print(f"Container page created: {container.get('url')}")

    # Create per-sample pages
    for entry in pred_gsws_data:
        idx = entry.get("idx", "?")
        global_id = entry.get("global_id", f"sample_{idx}")
        text = entry.get("text", "")
        gold_gsw = entry.get("gold_gsw")
        pred_gsw = entry.get("pred_gsw")

        # Get judgement if available
        judgement = judgements_map.get(global_id)

        blocks = []

        # Text
        blocks.append(toggle_block("Text", code_blocks_for_text(text, "plain text")))

        # Gold GSW
        blocks.append(build_gsw_toggle(gold_gsw, "Gold GSW", ""))

        # Pred GSW
        blocks.append(build_gsw_toggle(pred_gsw, "Pred GSW", ""))

        # Judgement
        if judgement:
            blocks.append(build_judgement_toggle(judgement))

        # Add score indicator to page title
        score_str = ""
        if judgement and judgement.get("overall_score") is not None:
            score = judgement["overall_score"]
            score_str = f" [{score:.1f}]"

        page_title = f"{idx:03d} - {global_id}{score_str}"
        page = create_page(notion, container_id, page_title, blocks, icon_emoji="📄")
        print(f"  Created page: {page_title}")
        time.sleep(args.sleep)

    print(f"\nDone! Created {len(pred_gsws_data)} sample pages.")


if __name__ == "__main__":
    main()
