"""Test configuration for the standalone course-project package."""

from __future__ import annotations

import sys
from pathlib import Path


COURSE_SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(COURSE_SRC))
