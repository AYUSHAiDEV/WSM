"""Module 1 - Document loading.

Reads raw documents from disk (or from in-memory text, e.g. Streamlit uploads)
and turns them into `Document` objects with a stable integer `doc_id`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass(frozen=True)
class Document:
    doc_id: int
    title: str
    text: str
    source: str = ""


def load_directory(directory: str | Path) -> List[Document]:
    """Load every supported file under `directory`, sorted by path for determinism."""
    root = Path(directory)
    if not root.is_dir():
        raise FileNotFoundError(f"Document directory not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS)
    return [
        Document(doc_id=i, title=f.stem, text=f.read_text(encoding="utf-8", errors="ignore"), source=str(f))
        for i, f in enumerate(files)
    ]


def load_from_texts(items: Iterable[Tuple[str, str]]) -> List[Document]:
    """Build documents from (title, text) pairs."""
    return [Document(doc_id=i, title=title, text=text, source=title) for i, (title, text) in enumerate(items)]
