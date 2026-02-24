"""
LongMemEval data ingestion.

Parses LongMemEval JSON format into:
1. Per-question chat histories as documents for GSW operator input
2. QA pairs for evaluation
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ChatTurn:
    role: str
    content: str
    has_answer: bool = False


@dataclass
class ChatSession:
    session_id: str
    date: str
    turns: List[ChatTurn]

    def to_document(self, include_metadata: bool = True) -> str:
        """Convert session to a text document for GSW operator input."""
        lines = []
        if include_metadata and self.date:
            lines.append(f"[{self.date}]")
            lines.append("")

        for turn in self.turns:
            role_label = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{role_label}: {turn.content}")

        return "\n".join(lines)


@dataclass
class LongMemEvalInstance:
    question_id: str
    question_type: str
    question: str
    answer: str
    question_date: str
    sessions: List[ChatSession]
    answer_session_ids: List[str]

    def to_documents(self, **kwargs) -> List[str]:
        """Convert all sessions to documents for GSW operator input."""
        return [session.to_document(**kwargs) for session in self.sessions]


class LongMemEvalLoader:
    """Load and parse LongMemEval benchmark data."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self._raw_data = None
        self._instances = None

    def load(self) -> List[LongMemEvalInstance]:
        """Load and parse all question instances."""
        if self._instances is not None:
            return self._instances

        with open(self.data_path) as f:
            self._raw_data = json.load(f)

        self._instances = [self._parse_instance(item) for item in self._raw_data]
        return self._instances

    def _parse_instance(self, raw: dict) -> LongMemEvalInstance:
        sessions = []
        for i, (session_turns, session_id, date) in enumerate(
            zip(
                raw["haystack_sessions"],
                raw["haystack_session_ids"],
                raw["haystack_dates"],
            )
        ):
            turns = [
                ChatTurn(
                    role=t["role"],
                    content=t["content"],
                    has_answer=t.get("has_answer", False),
                )
                for t in session_turns
            ]
            sessions.append(ChatSession(session_id=session_id, date=date, turns=turns))

        return LongMemEvalInstance(
            question_id=raw["question_id"],
            question_type=raw["question_type"],
            question=raw["question"],
            answer=raw["answer"],
            question_date=raw["question_date"],
            sessions=sessions,
            answer_session_ids=raw.get("answer_session_ids", []),
        )