#!/usr/bin/env python3
"""
Extract rejected questions from musique_platinum_dataset.json and format them
like musique.json with answer set to "No Answer".
"""

import json
import re
from pathlib import Path


def parse_oracle_context(oracle_context: str, supporting_facts: list) -> list:
    """
    Parse oracle_context text into structured paragraphs.

    oracle_context format:
    "Wikipedia Title: Title1\nParagraph text...\n\nWikipedia Title: Title2\n..."

    supporting_facts format: [[title, idx], ...]
    """
    # Get supporting titles for marking is_supporting
    supporting_titles = {fact[0] for fact in supporting_facts}

    # Split by "Wikipedia Title: " pattern
    # First split may have empty string if context starts with it
    parts = re.split(r'(?:^|\n\n)Wikipedia Title: ', oracle_context)

    paragraphs = []
    idx = 0

    for part in parts:
        part = part.strip()
        if not part:
            continue

        # First line is the title, rest is paragraph text
        lines = part.split('\n', 1)
        title = lines[0].strip()
        paragraph_text = lines[1].strip() if len(lines) > 1 else ""

        # Check if this paragraph is supporting
        is_supporting = title in supporting_titles

        paragraphs.append({
            "idx": idx,
            "title": title,
            "paragraph_text": paragraph_text,
            "is_supporting": is_supporting
        })
        idx += 1

    return paragraphs


def main():
    # Paths
    base_path = Path(__file__).parent.parent / "playground_data"
    input_path = base_path / "musique_platinum_dataset.json"
    output_path = base_path / "musique_unanswerable.json"

    # Load platinum dataset
    print(f"Loading {input_path}...")
    with open(input_path, 'r') as f:
        platinum = json.load(f)

    # Filter rejected questions
    rejected = [q for q in platinum['questions'] if q['review_status'] == 'rejected']
    print(f"Found {len(rejected)} rejected questions")

    # Transform to musique.json format
    output = []
    for q in rejected:
        paragraphs = parse_oracle_context(q['oracle_context'], q['supporting_facts'])

        output.append({
            'id': q['question_id'],
            'paragraphs': paragraphs,
            'question': q['question'],
            'answer': 'No Answer',
            'answer_aliases': [],
            'question_decomposition': [],
            'answerable': False
        })

    # Save
    print(f"Writing {len(output)} questions to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

    print("Done!")


if __name__ == "__main__":
    main()
