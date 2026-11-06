"""Module 5 - Query processing.

Supported syntax (operators must be UPPERCASE):
    term                 single word
    "exact phrase"       positional phrase match
    comp*  *ing  c*r     wildcard terms (delegated to the wildcard module)
    A AND B, A OR B, NOT A, ( ... )
    A B                  implicit AND

Grammar (precedence NOT > AND > OR):
    or_expr  := and_expr ( "OR"  and_expr )*
    and_expr := not_expr ( ["AND"] not_expr )*
    not_expr := "NOT" not_expr | primary
    primary  := "(" or_expr ")" | PHRASE | WORD

Boolean operators are evaluated with the classic linear-time two-pointer
merges over sorted postings lists.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .inverted_index import InvertedIndex
from .wildcard import PermutermIndex

_LEX_RE = re.compile(r'"[^"]*"|\(|\)|[^\s()"]+')


class QuerySyntaxError(ValueError):
    pass


# --- postings-list algorithms (inputs are sorted lists of doc ids) ---------
def intersect(a: List[int], b: List[int]) -> List[int]:
    i = j = 0
    out: List[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def union(a: List[int], b: List[int]) -> List[int]:
    i = j = 0
    out: List[int] = []
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i]); i += 1; j += 1
        elif a[i] < b[j]:
            out.append(a[i]); i += 1
        else:
            out.append(b[j]); j += 1
    out.extend(a[i:]); out.extend(b[j:])
    return out


def difference(a: List[int], b: List[int]) -> List[int]:
    """a AND NOT b."""
    i = j = 0
    out: List[int] = []
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i]); i += 1
        elif a[i] == b[j]:
            i += 1; j += 1
        else:
            j += 1
    return out


@dataclass
class QueryResult:
    doc_ids: List[int] = field(default_factory=list)
    expansions: Dict[str, List[str]] = field(default_factory=dict)  # wildcard -> matched terms
    highlight_terms: Set[str] = field(default_factory=set)


class _Evaluator:
    def __init__(self, index: InvertedIndex, wildcard: PermutermIndex, tokens: List[str]):
        self.index, self.wildcard, self.tokens = index, wildcard, tokens
        self.norm = index.normalizer
        self.pos = 0
        self.universe = index.doc_ids
        self.result = QueryResult()

    def peek(self) -> Optional[str]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def next(self) -> Optional[str]:
        tok = self.peek()
        self.pos += 1
        return tok

    def run(self) -> QueryResult:
        docs = self.or_expr()
        if self.pos < len(self.tokens):
            raise QuerySyntaxError(f"Unexpected token {self.tokens[self.pos]!r}.")
        self.result.doc_ids = docs
        return self.result

    def or_expr(self) -> List[int]:
        left = self.and_expr()
        while self.peek() == "OR":
            self.next()
            left = union(left, self.and_expr())
        return left

    def and_expr(self) -> List[int]:
        left = self.not_expr()
        while True:
            tok = self.peek()
            if tok == "AND":
                self.next()
                left = intersect(left, self.not_expr())
            elif tok is not None and tok not in ("OR", ")"):
                left = intersect(left, self.not_expr())   # implicit AND
            else:
                return left

    def not_expr(self) -> List[int]:
        if self.peek() == "NOT":
            self.next()
            return difference(self.universe, self.not_expr())
        return self.primary()

    def primary(self) -> List[int]:
        tok = self.next()
        if tok is None:
            raise QuerySyntaxError("Query ended unexpectedly (missing operand).")
        if tok == "(":
            value = self.or_expr()
            if self.next() != ")":
                raise QuerySyntaxError("Missing closing parenthesis.")
            return value
        if tok in (")", "AND", "OR"):
            raise QuerySyntaxError(f"Unexpected {tok!r}; an operand was expected.")
        if tok.startswith('"'):
            return self.phrase(tok[1:-1])
        return self.word(tok)

    # --- leaves -----------------------------------------------------------
    def word(self, raw: str) -> List[int]:
        if "*" in raw:
            pattern = self.norm.normalize_pattern(raw)
            terms = self.wildcard.match(pattern)
            self.result.expansions[raw] = terms
            self.result.highlight_terms.update(terms)
            docs: List[int] = []
            for t in terms:
                docs = union(docs, self.index.get_doc_ids(t))
            return docs
        processed = self.norm.process(raw)
        if not processed:                       # e.g. a removed stop word: neutral operand
            return self.universe
        if len(processed) == 1:
            term = processed[0][1]
            self.result.highlight_terms.add(term)
            return self.index.get_doc_ids(term)
        return self.match_phrase(processed)     # "state-of-the-art" behaves as a phrase

    def phrase(self, text: str) -> List[int]:
        if not text.strip():
            raise QuerySyntaxError("Empty phrase.")
        processed = self.norm.process(text)
        if not processed:
            return self.universe
        return self.match_phrase(processed)

    def match_phrase(self, processed: List[Tuple[int, str]]) -> List[int]:
        base = processed[0][0]
        offsets = [(p - base, t) for p, t in processed]
        self.result.highlight_terms.update(t for _, t in offsets)
        candidates = self.index.get_doc_ids(offsets[0][1])
        for _, term in offsets[1:]:
            candidates = intersect(candidates, self.index.get_doc_ids(term))
        hits = []
        for doc_id in candidates:
            lookup = {t: set(self.index.get_positions(t)[doc_id]) for _, t in offsets}
            first = self.index.get_positions(offsets[0][1])[doc_id]
            if any(all((p + off) in lookup[t] for off, t in offsets) for p in first):
                hits.append(doc_id)
        return hits


class QueryProcessor:
    def __init__(self, index: InvertedIndex, wildcard: PermutermIndex):
        self.index = index
        self.wildcard = wildcard

    def search(self, query: str) -> QueryResult:
        if not query.strip():
            raise QuerySyntaxError("Empty query.")
        if query.count('"') % 2:
            raise QuerySyntaxError('Unbalanced double quote (").')
        tokens = _LEX_RE.findall(query)
        return _Evaluator(self.index, self.wildcard, tokens).run()
