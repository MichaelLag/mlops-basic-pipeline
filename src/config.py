# src/config.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    if path is None:
        # Resolve repo root: src/ -> project root
        path = Path(__file__).resolve().parents[1] / "config.yaml"
    else:
        path = Path(path)

    with path.open("r") as f:
        return yaml.safe_load(f)
