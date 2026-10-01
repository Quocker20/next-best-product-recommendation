"""Single source of truth for project paths."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
CONFIGS = ROOT / "configs"
EXPERIMENTS = ROOT / "experiments"
REPORTS = ROOT / "reports"

EXPEDIA_RAW = RAW / "hospitality" / "expedia"
