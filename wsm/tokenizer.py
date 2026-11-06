"""Module 2 - Tokenization.

Splits raw text into word tokens. A token is a maximal run of letters/digits,
optionally joined by internal apostrophes (so "don't" stays one token).
Punctuation, underscores and whitespace are delimiters.
"""
from __future__ import annotations

import re
from typing import List, Tuple

_TOKEN_RE = re.compile(r"[^\W_]+(?:['\u2019][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text)


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Tokens with (start, end) character offsets; used for result snippets."""
    return [(m.group(), m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
