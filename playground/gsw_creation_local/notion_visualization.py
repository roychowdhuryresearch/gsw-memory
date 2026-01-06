#!/usr/bin/env python3
"""
Create Notion pages from three aligned folders:
- story_to_raw_dir: JSON files (keep only 'text' field) → toggle 'Raw chunk'
- raw_to_inter_dir: TXT files → toggle 'Intermediate'
- inter_to_gsw_dir: JSON files (full pretty print) → toggle 'GSW'
  - Inside 'GSW', add four nested toggles: 'spaces', 'times', 'entities', 'events'

Additionally:
- Adds top-level toggles for prompts (raw→inter, inter→gsw system/user)
- Names the container page as 'Gsw-visualization-<timestamp>'
"""

import argparse
import json
import os
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


# --- file reading ---
def read_file_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading {p.name}: {e}"


def read_story_to_raw_json(p: Path) -> str:
    """Return only the 'text' field from story_to_raw JSON."""
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return str(data.get("text", ""))
    except Exception as e:
        return f"Error reading {p.name}: {e}"


def load_json(p: Path):
    """Load JSON and return dict (or None) and an error string if any."""
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


# --- filename parsing ---
def extract_prefix(filename: str) -> str:
    """
    Extract base prefix like 'substory_000_chunk_000' from filenames:
    - substory_000_chunk_000.json
    - substory_000_chunk_000.json.txt
    - substory_000_chunk_000.json_output.json
    """
    base = filename.split(".")[0]
    if "_output" in base:
        base = base.split("_output")[0]
    return base


def map_files_by_prefix(folder: Path, exts):
    mapping = {}
    for p in folder.rglob("*"):
        if p.is_file() and any(p.name.endswith(e) for e in exts):
            prefix = extract_prefix(p.name)
            mapping[prefix] = p
    return mapping


# --- GSW nested toggle builder ---
def build_gsw_toggle_from_path(json_path: Path) -> dict:
    """
    Build the top-level 'GSW' toggle, with four nested toggles:
    - 'spaces', 'times', 'entities', 'events'
    Each nested toggle shows the pretty-printed JSON of that key.
    """
    data, err = load_json(json_path)
    if err:
        return toggle_block("GSW", [paragraph(err)])

    nested = []
    for key in ["spaces", "times", "entities", "events"]:
        if key in data:
            pretty = json.dumps(data[key], indent=2, ensure_ascii=False)
            nested.append(toggle_block(key, code_blocks_for_text(pretty, "json")))
        else:
            nested.append(toggle_block(key, [paragraph("Not found.")]))

    # Wrap the four key toggles inside the 'GSW' toggle
    return toggle_block("GSW", nested)


# --- main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-page-id", default="28d40a7c71d28075b4e2fb1f5161e343")
    parser.add_argument("--story-to-raw-dir", default="/mnt/SSD1/chenda/gsw-memory/improving_prompts/story_to_raw/chunks", type=Path)
    parser.add_argument("--raw-to-inter-dir", default="/mnt/SSD1/chenda/gsw-memory/improving_prompts/raw_to_inter/output_2level_manual_full", type=Path)
    parser.add_argument("--inter-to-gsw-dir", default="/mnt/SSD1/chenda/gsw-memory/improving_prompts/inter_to_gsw/output_2level_manual_full/batch_20251010_162500/individual_outputs", type=Path)
    parser.add_argument("--raw_to_inter_prompt", default="/mnt/SSD1/chenda/gsw-memory/improving_prompts/iterations_gpt-5/iteration_004_20250915_183945/current_prompt_exp.txt", type=Path)
    parser.add_argument("--inter_to_gsw_system_prompt", default="/mnt/SSD1/chenda/gsw-memory/improving_prompts/inter_to_gsw/prompt/system_prompt.txt", type=Path)
    parser.add_argument("--inter_to_gsw_user_prompt", default="/mnt/SSD1/chenda/gsw-memory/improving_prompts/inter_to_gsw/prompt/user_prompt.txt", type=Path)
    parser.add_argument("--sleep", type=float, default=0.2)
    # Use env var by default; you can also pass --notion-token "secret_xxx"
    parser.add_argument("--notion-token", default="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")
    args = parser.parse_args()

    if not args.notion_token:
        print("❌ Missing Notion token (set NOTION_TOKEN or pass --notion-token).", file=sys.stderr)
        sys.exit(1)

    notion = Client(auth=args.notion_token)

    # --- timestamp for container ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    container_title = f"Gsw-visualization-{timestamp}"

    # --- map all files ---
    story_map = map_files_by_prefix(args.story_to_raw_dir, [".json"])
    raw_map = map_files_by_prefix(args.raw_to_inter_dir, [".txt"])
    gsw_map = map_files_by_prefix(args.inter_to_gsw_dir, [".json"])
    all_prefixes = sorted(set(story_map) | set(raw_map) | set(gsw_map))
    if not all_prefixes:
        print("No matching files found.")
        return

    # --- top toggles for prompts ---
    top_blocks = []

    if args.raw_to_inter_prompt.exists():
        raw_prompt = read_file_safe(args.raw_to_inter_prompt)
        top_blocks.append(toggle_block("🟢 Raw→Inter prompt", code_blocks_for_text(raw_prompt, "plain text")))

    if args.inter_to_gsw_system_prompt.exists():
        sys_prompt = read_file_safe(args.inter_to_gsw_system_prompt)
        top_blocks.append(toggle_block("🟠 Inter→GSW system prompt", code_blocks_for_text(sys_prompt, "plain text")))

    if args.inter_to_gsw_user_prompt.exists():
        user_prompt = read_file_safe(args.inter_to_gsw_user_prompt)
        top_blocks.append(toggle_block("🔵 Inter→GSW user prompt", code_blocks_for_text(user_prompt, "plain text")))

    # --- create container page ---
    container = create_page(
        notion,
        parent_page_id=args.parent_page_id,
        title=container_title,
        children=[paragraph("Automatically imported pipeline results.")] + top_blocks,
        icon_emoji="📦",
    )
    container_id = container["id"]
    print(f"✅ Container page created: {container.get('url')}")

    # --- per prefix pages ---
    for prefix in all_prefixes:
        blocks = []

        # Raw chunk
        if prefix in story_map:
            text = read_story_to_raw_json(story_map[prefix])
            blocks.append(toggle_block("Raw chunk", code_blocks_for_text(text, "plain text")))
        else:
            blocks.append(toggle_block("Raw chunk", [paragraph("Not found.")]))

        # Intermediate
        if prefix in raw_map:
            txt = read_file_safe(raw_map[prefix])
            blocks.append(toggle_block("Intermediate", code_blocks_for_text(txt, "plain text")))
        else:
            blocks.append(toggle_block("Intermediate", [paragraph("Not found.")]))

        # GSW (with nested toggles for spaces/times/entities/events)
        if prefix in gsw_map:
            blocks.append(build_gsw_toggle_from_path(gsw_map[prefix]))
        else:
            blocks.append(toggle_block("GSW", [paragraph("Not found.")]))

        page = create_page(notion, container_id, prefix, blocks, icon_emoji="📄")
        print(f"  - Created page for {prefix}: {page.get('url')}")
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
