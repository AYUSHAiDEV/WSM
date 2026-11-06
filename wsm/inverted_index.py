"""Module 4 - Inverted index construction.

A positional inverted index:  term -> { doc_id -> [positions] }.
Positions enable phrase queries; the doc_id keys form the postings list.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

from .document_loader import Document
from .normalizer import Normalizer


class InvertedIndex:
    def __init__(self, normalizer: Normalizer | None = None):
        self.normalizer = normalizer or Normalizer()
        self.postings: Dict[str, Dict[int, List[int]]] = {}
        self.documents: Dict[int, Document] = {}
        self.total_tokens = 0

    def add_document(self, doc: Document) -> None:
        self.documents[doc.doc_id] = doc
        for pos, term in self.normalizer.process(doc.text):
            self.postings.setdefault(term, {}).setdefault(doc.doc_id, []).append(pos)
            self.total_tokens += 1

    def build(self, docs: Iterable[Document]) -> "InvertedIndex":
        for doc in docs:
            self.add_document(doc)
        return self

    # --- accessors -------------------------------------------------------
    @property
    def doc_ids(self) -> List[int]:
        return sorted(self.documents)

    def vocabulary(self) -> List[str]:
        return sorted(self.postings)

    def get_doc_ids(self, term: str) -> List[int]:
        """Sorted postings list (document ids only) for a normalized term."""
        return sorted(self.postings.get(term, {}))

    def get_positions(self, term: str) -> Dict[int, List[int]]:
        return self.postings.get(term, {})

    def stats(self) -> Dict[str, int]:
        return {
            "documents": len(self.documents),
            "vocabulary": len(self.postings),
            "tokens": self.total_tokens,
            "postings": sum(len(p) for p in self.postings.values()),
        }
