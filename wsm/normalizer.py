"""Module 3 - Text normalization / case folding.

Pipeline applied identically to documents and queries (a core IR requirement):
    1. case folding      ("Computer" -> "computer")
    2. accent stripping  ("café" -> "cafe")
    3. punctuation removal inside the token ("don't" -> "dont")
    4. optional stop-word removal

Token positions are taken from the ORIGINAL token stream, so removing a stop
word leaves a gap instead of shifting later positions. This keeps phrase
queries correct.
"""
from __future__ import annotations

import unicodedata
from typing import List, Tuple

from .tokenizer import tokenize

STOP_WORDS = frozenset(
    "a an and are as at be by for from has he in is it its of on that the to was were will with".split()
)


def case_fold(text: str) -> str:
    return text.casefold()


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


class Normalizer:
    def __init__(self, remove_stopwords: bool = False):
        self.remove_stopwords = remove_stopwords

    def normalize_term(self, token: str) -> str:
        folded = strip_accents(case_fold(token))
        return "".join(ch for ch in folded if ch.isalnum())

    def normalize_pattern(self, pattern: str) -> str:
        """Normalize a wildcard pattern, preserving the '*' characters."""
        return "*".join(self.normalize_term(piece) for piece in pattern.split("*"))

    def process(self, text: str) -> List[Tuple[int, str]]:
        """Return [(original_position, normalized_term), ...]."""
        out: List[Tuple[int, str]] = []
        for pos, token in enumerate(tokenize(text)):
            term = self.normalize_term(token)
            if not term:
                continue
            if self.remove_stopwords and term in STOP_WORDS:
                continue
            out.append((pos, term))
        return out
