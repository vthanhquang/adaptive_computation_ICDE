"""JSON Lines helpers. Files ending in ``.gz`` are read and written transparently compressed."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable, Iterator


def _open(path: Path, mode: str):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode + "t", encoding="utf8")
    return open(path, mode, encoding="utf8")


def resolve(path: Path) -> Path:
    """Return ``path`` if it exists, else its ``.gz`` sibling (released artifacts are compressed)."""
    path = Path(path)
    if path.exists() or path.suffix == ".gz":
        return path
    gz = path.with_name(path.name + ".gz")
    return gz if gz.exists() else path


def iter_jsonl(path: Path) -> Iterator[dict]:
    with _open(resolve(path), "r") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_jsonl(path: Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: Path, rows: Iterable[dict], append: bool = False) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _open(path, "a" if append else "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
