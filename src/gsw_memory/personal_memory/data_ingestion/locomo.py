"""
LoCoMo data ingestion.

Parses LoCoMo JSON format into:
1. Documents suitable for GSW operator input (one document per session)
2. QA pairs for evaluation
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class Turn:
    speaker: str
    text: str
    dia_id: str
    query: Optional[str] = None       # semantic label of shared image (if any)
    blip_caption: Optional[str] = None  # AI vision caption of shared image (stored, not rendered)


@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]

    def to_document(self, include_speaker_labels: bool = True, include_metadata: bool = True) -> str:
        """Convert session to a text document for GSW operator input.

        Args:
            include_speaker_labels: Prepend speaker name to each turn.
            include_metadata: Include session date/time header.
        """
        lines = []
        if include_metadata:
            lines.append(f"[Session {self.session_id} — {self.date_time}]")
            lines.append("")

        for turn in self.turns:
            line = f"{turn.speaker}: {turn.text}" if include_speaker_labels else turn.text
            if turn.query:
                line += f" [shared: {turn.query}]"
            lines.append(line)

        return "\n".join(lines)


@dataclass
class QAPair:
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: int
    adversarial_answer: Optional[str] = None


@dataclass
class Conversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: List[Session]
    qa_pairs: List[QAPair]
    event_summaries: dict = field(default_factory=dict)
    session_summaries: dict = field(default_factory=dict)

    def to_documents(self, **kwargs) -> List[str]:
        """Convert all sessions to documents for GSW operator input."""
        return [session.to_document(**kwargs) for session in self.sessions]


class LoCoMoLoader:
    """Load and parse LoCoMo benchmark data."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        self._raw_data = None
        self._conversations = None

    def load(self) -> List[Conversation]:
        """Load and parse all conversations."""
        if self._conversations is not None:
            return self._conversations

        with open(self.data_path) as f:
            self._raw_data = json.load(f)

        self._conversations = [self._parse_conversation(c) for c in self._raw_data]
        return self._conversations

    def _parse_conversation(self, raw: dict) -> Conversation:
        conv_data = raw["conversation"]
        speaker_a = conv_data["speaker_a"]
        speaker_b = conv_data["speaker_b"]

        # Parse sessions
        session_keys = sorted(
            [k for k in conv_data if k.startswith("session_") and not k.endswith("_date_time") and k[-1].isdigit()],
            key=lambda k: int(k.split("_")[1]),
        )

        sessions = []
        for key in session_keys:
            session_num = int(key.split("_")[1])
            date_key = f"{key}_date_time"
            date_time = conv_data.get(date_key, "")

            turns = [
                Turn(
                    speaker=t["speaker"],
                    text=t["text"],
                    dia_id=t["dia_id"],
                    query=t.get("query"),
                    blip_caption=t.get("blip_caption"),
                )
                for t in conv_data[key]
            ]
            sessions.append(Session(session_id=session_num, date_time=date_time, turns=turns))

        # Parse QA pairs
        qa_pairs = []
        for q in raw.get("qa", []):
            qa_pairs.append(
                QAPair(
                    question=q["question"],
                    answer=q.get("answer"),
                    evidence=q.get("evidence", []),
                    category=q["category"],
                    adversarial_answer=q.get("adversarial_answer"),
                )
            )

        return Conversation(
            sample_id=raw.get("sample_id", ""),
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
            qa_pairs=qa_pairs,
            event_summaries=raw.get("event_summary", {}),
            session_summaries=raw.get("session_summary", {}),
        )