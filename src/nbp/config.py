"""YAML config loading into dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


@dataclass
class DataConfig:
    """Data-pipeline parameters (configs/data.yaml)."""

    chunk_size: int = 1_000_000
    max_history: int = 20
    seed: int = 42
    device: str = "auto"  # auto | cpu | cuda


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML file into a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(cls: type[T], path: str | Path) -> T:
    """Build dataclass `cls` from a YAML file; unknown keys raise, missing keys use defaults."""
    raw = load_yaml(path)
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"Unknown config keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**raw)


def resolve_device(device: str = "auto") -> str:
    """Return 'cuda' or 'cpu'; 'auto' picks cuda when available."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"
