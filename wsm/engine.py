"""Facade tying all modules together: documents -> index -> query -> formatted results."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from .document_loader import Document, load_directory
from .inverted_index import InvertedIndex
from .normalizer import Normalizer
from .query_processor import QueryProcessor
from .tokenizer import tokenize_with_spans
from .wildcard import PermutermIndex


@dataclass
class SearchHit:
    doc_id: int
    title: str
    source: str
    snippet: str


@dataclass
class SearchResponse:
    query: str
    hits: List[SearchHit]
    expansions: Dict[str, List[str]] = field(default_factory=dict)
    elapsed_ms: float = 0.0

    def to_text(self) -> str:
        lines = [f"Query: {self.query}", f"Matched documents: {len(self.hits)}   ({self.elapsed_ms:.2f} ms)", ""]
        if self.expansions:
            lines.append("Wildcard expansions:")
            for pattern, terms in self.expansions.items():
                shown = ", ".join(terms[:15]) + (" ..." if len(terms) > 15 else "")
                lines.append(f"  {pattern}  ->  {shown or '(no matching terms)'}")
            lines.append("")
        if not self.hits:
            lines.append("No documents matched.")
        for rank, hit in enumerate(self.hits, 1):
            lines.append(f"[{rank}] {hit.title}   (doc {hit.doc_id})")
            lines.append(f"    {hit.snippet}")
            lines.append("")
        return "\n".join(lines).rstrip()


class SearchEngine:
    def __init__(self, documents: Sequence[Document], remove_stopwords: bool = False):
        self.normalizer = Normalizer(remove_stopwords)
        self.index = InvertedIndex(self.normalizer).build(documents)
        self.wildcard = PermutermIndex(self.index.vocabulary())
        self.processor = QueryProcessor(self.index, self.wildcard)

    @classmethod
    def from_directory(cls, directory: str | Path, remove_stopwords: bool = False) -> "SearchEngine":
        return cls(load_directory(directory), remove_stopwords)

    def search(self, query: str) -> SearchResponse:
        start = time.perf_counter()
        result = self.processor.search(query)
        hits = [
            SearchHit(d, self.index.documents[d].title, self.index.documents[d].source,
                      self._snippet(self.index.documents[d], result.highlight_terms))
            for d in result.doc_ids
        ]
        return SearchResponse(query, hits, result.expansions, (time.perf_counter() - start) * 1000)

    def _snippet(self, doc: Document, terms, width: int = 70) -> str:
        text = doc.text
        for token, s, e in tokenize_with_spans(text):
            if self.normalizer.normalize_term(token) in terms:
                a, b = max(0, s - width), min(len(text), e + width)
                return ("..." if a else "") + " ".join(text[a:b].split()) + ("..." if b < len(text) else "")
        return " ".join(text[: 2 * width].split()) + ("..." if len(text) > 2 * width else "")
