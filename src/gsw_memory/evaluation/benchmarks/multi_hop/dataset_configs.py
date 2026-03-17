"""
Dataset configuration registry for multi-hop QA evaluation.

Each config specifies the answer field name, whether JSON parsing is needed,
and whether "No Answer" responses should be allowed.
"""

from typing import Any, Dict

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    "2wiki": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": False,
    },
    "2wiki_platinum": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": True,
    },
    "2wiki_unanswerable": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": True,
    },
    "musique": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": False,
    },
    "musique_platinum": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": True,
    },
    "musique_unanswerable": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": True,
    },
    "hotpotqa": {
        "answer_field": "answer",
        "parse_json": False,
        "allow_no_answer": False,
    },
    "popqa": {
        "answer_field": "possible_answers",
        "parse_json": True,
        "allow_no_answer": False,
    },
    "nq_rear": {
        "answer_field": "reference",
        "parse_json": False,
        "allow_no_answer": False,
    },
    "lveval": {
        "answer_field": "answers",
        "parse_json": False,
        "allow_no_answer": False,
    },
}
