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

Boolean operators use the classic linear-time two-pointer merges over sorted
postings lists. While evaluating, the processor records a human-readable
`trace` of every step (used by the GUI's "retrieval steps" panel).
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


def classify(token: str) -> str:
    if token in ("AND", "OR", "NOT"):
        return "OPERATOR"
    if token in ("(", ")"):
        return "PAREN"
    if token.startswith('"'):
        return "PHRASE"
    if "*" in token:
        return "WILDCARD"
    return "TERM"


@dataclass
class QueryResult:
    doc_ids: List[int] = field(default_factory=list)
    expansions: Dict[str, List[str]] = field(default_factory=dict)   # wildcard -> matched terms
    highlight_terms: Set[str] = field(default_factory=set)
    tokens: List[Tuple[str, str]] = field(default_factory=list)      # (kind, text) from the lexer
    trace: List[Tuple[str, List[str]]] = field(default_factory=list)  # (title, detail lines)


def _preview(items: List[str], limit: int = 12) -> str:
    if not items:
        return "(none)"
    extra = f", ... (+{len(items) - limit} more)" if len(items) > limit else ""
    return ", ".join(items[:limit]) + extra


class _Evaluator:
    def __init__(self, index: InvertedIndex, wildcard: PermutermIndex, tokens: List[str]):
        self.index, self.wildcard, self.tokens = index, wildcard, tokens
        self.norm = index.normalizer
        self.pos = 0
        self.universe = index.doc_ids
        self.result = QueryResult(tokens=[(classify(t), t) for t in tokens])

    # --- helpers ----------------------------------------------------------
    def fmt(self, ids: List[int], limit: int = 6) -> str:
        if not ids:
            return "{ }  [0 docs]"
        names = [self.index.documents[d].title for d in ids[:limit]]
        extra = f", ... (+{len(ids) - limit} more)" if len(ids) > limit else ""
        return "{" + ", ".join(names) + extra + "}  " + f"[{len(ids)} doc(s)]"

    def step(self, title: str, *lines: str) -> None:
        self.result.trace.append((title, list(lines)))

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

    # --- grammar ----------------------------------------------------------
    def or_expr(self) -> List[int]:
        left = self.and_expr()
        while self.peek() == "OR":
            self.next()
            right = self.and_expr()
            out = union(left, right)
            self.step("OR - union of two postings lists",
                      f"left   = {self.fmt(left)}", f"right  = {self.fmt(right)}",
                      f"left ∪ right = {self.fmt(out)}",
                      "Method: two-pointer merge (both lists are sorted by doc id)")
            left = out
        return left

    def and_expr(self) -> List[int]:
        left = self.not_expr()
        while True:
            tok = self.peek()
            if tok == "AND":
                self.next()
                label = "AND"
            elif tok is not None and tok not in ("OR", ")"):
                label = "AND (implicit - adjacent terms)"
            else:
                return left
            right = self.not_expr()
            out = intersect(left, right)
            self.step(f"{label} - intersection of two postings lists",
                      f"left   = {self.fmt(left)}", f"right  = {self.fmt(right)}",
                      f"left ∩ right = {self.fmt(out)}",
                      "Method: two-pointer merge, advance the pointer with the smaller doc id")
            left = out

    def not_expr(self) -> List[int]:
        if self.peek() == "NOT":
            self.next()
            operand = self.not_expr()
            out = difference(self.universe, operand)
            self.step("NOT - complement with respect to the collection",
                      f"all documents = {self.fmt(self.universe)}", f"operand = {self.fmt(operand)}",
                      f"all − operand = {self.fmt(out)}")
            return out
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
            return self.wildcard_word(raw)
        processed = self.norm.process(raw)
        if not processed:
            self.step(f"Term {raw!r}",
                      "Removed by the normalizer as a stop word -> neutral operand (matches all documents)")
            return self.universe
        if len(processed) == 1:
            term = processed[0][1]
            self.result.highlight_terms.add(term)
            ids = self.index.get_doc_ids(term)
            positions = self.index.get_positions(term)
            lines = [f"Normalise (case folding, accent stripping): {raw!r} -> {term!r}",
                     f"Dictionary lookup, postings list of {term!r}: {self.fmt(ids)}"]
            lines += [f"  {self.index.documents[d].title}: positions {positions[d][:8]}" for d in ids[:5]]
            self.step(f"Term {raw!r}", *lines)
            return ids
        return self.match_phrase(processed, raw)   # e.g. "state-of-the-art" behaves as a phrase

    def wildcard_word(self, raw: str) -> List[int]:
        pattern = self.norm.normalize_pattern(raw)
        info = self.wildcard.explain(pattern)
        terms = info["matches"]
        self.result.expansions[raw] = terms
        self.result.highlight_terms.update(terms)
        docs: List[int] = []
        for t in terms:
            docs = union(docs, self.index.get_doc_ids(t))
        self.step(
            f"Wildcard term {raw!r}",
            f"1. Normalise pattern: {raw!r} -> {pattern!r}",
            f"2. Permuterm rotation (move '*' to the end): {info['rotated']!r}",
            f"3. Binary search in the sorted rotation list for prefix {info['prefix']!r}: "
            f"{len(info['candidates'])} candidate term(s): {_preview(info['candidates'])}",
            f"4. Regex post-filter on the full pattern: {len(terms)} matching term(s): {_preview(terms)}",
            f"5. Union of their postings lists: {self.fmt(docs)}",
        )
        return docs

    def phrase(self, text: str) -> List[int]:
        if not text.strip():
            raise QuerySyntaxError("Empty phrase.")
        processed = self.norm.process(text)
        if not processed:
            self.step(f"Phrase \"{text}\"", "All words were removed as stop words -> matches all documents")
            return self.universe
        return self.match_phrase(processed, text)

    def match_phrase(self, processed: List[Tuple[int, str]], label: str) -> List[int]:
        base = processed[0][0]
        offsets = [(p - base, t) for p, t in processed]
        self.result.highlight_terms.update(t for _, t in offsets)
        candidates = self.index.get_doc_ids(offsets[0][1])
        for _, term in offsets[1:]:
            candidates = intersect(candidates, self.index.get_doc_ids(term))
        hits: List[int] = []
        checks: List[str] = []
        for doc_id in candidates:
            lookup = {t: set(self.index.get_positions(t)[doc_id]) for _, t in offsets}
            first = self.index.get_positions(offsets[0][1])[doc_id]
            start = next((p for p in first if all((p + off) in lookup[t] for off, t in offsets)), None)
            title = self.index.documents[doc_id].title
            if start is not None:
                hits.append(doc_id)
                checks.append(f"  {title}: terms found at consecutive positions starting at {start}  ✓")
            else:
                checks.append(f"  {title}: all terms present but NOT adjacent in this order  ✗")
        self.step(
            f"Phrase \"{label}\"",
            "Normalised terms with relative offsets: " + ", ".join(f"{t}@+{o}" for o, t in offsets),
            f"Candidate documents (contain every term - intersection of postings): {self.fmt(candidates)}",
            "Positional check (term i must appear exactly i positions after the first term):",
            *(checks[:8] or ["  (no candidate documents)"]),
            f"Phrase matches: {self.fmt(hits)}",
        )
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
