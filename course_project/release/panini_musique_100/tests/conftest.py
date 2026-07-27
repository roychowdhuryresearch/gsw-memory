"""Test configuration for the standalone course-project package."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COURSE_SRC = PROJECT_ROOT / "src"
sys.path.insert(0, str(COURSE_SRC if COURSE_SRC.exists() else PROJECT_ROOT))
