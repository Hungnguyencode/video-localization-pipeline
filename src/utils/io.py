from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(data: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_json(input_path: str | Path) -> Any:
    path = Path(input_path)

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)