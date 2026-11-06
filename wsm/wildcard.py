"""Module 6 - Wildcard processing (permuterm index).

Each term t is stored under every rotation of t + '$'. For "hel*o" we rotate
the query so the '*' lands at the end ("o$hel*") and do a PREFIX lookup on the
sorted rotation list with binary search: O(log N + k) for k candidates.

Patterns with several '*' (e.g. "a*b*c") use the prefix lookup for the first
fragment and a regex post-filter for the rest, as in Manning et al., IIR 3.2.
"""
from __future__ import annotations

import bisect
import re
from typing import Iterable, List, Set


class PermutermIndex:
    END = "$"

    def __init__(self, terms: Iterable[str]):
        self._terms_set = set(terms)
        rotations = []
        for term in self._terms_set:
            w = term + self.END
            for i in range(len(w)):
                rotations.append((w[i:] + w[:i], term))
        rotations.sort()
        self._keys = [r[0] for r in rotations]
        self._terms = [r[1] for r in rotations]

    def _prefix_lookup(self, prefix: str) -> Set[str]:
        lo = bisect.bisect_left(self._keys, prefix)
        hi = bisect.bisect_left(self._keys, prefix + "\U0010ffff")
        return set(self._terms[lo:hi])

    def explain(self, pattern: str) -> dict:
        """Like match(), but also returns the intermediate steps (for the GUI trace)."""
        if "*" not in pattern:
            found = [pattern] if pattern in self._terms_set else []
            return {"rotated": None, "prefix": None, "candidates": found, "matches": found}
        w = pattern + self.END
        k = w.rindex("*")
        rotated = w[k + 1:] + w[:k] + "*"      # '*' moved to the end
        prefix = rotated.split("*", 1)[0]      # up to the first remaining '*'
        regex = re.compile(".*".join(re.escape(p) for p in pattern.split("*")))
        candidates = sorted(self._prefix_lookup(prefix))
        matches = [t for t in candidates if regex.fullmatch(t)]
        return {"rotated": rotated, "prefix": prefix, "candidates": candidates, "matches": matches}

    def match(self, pattern: str) -> List[str]:
        """All vocabulary terms matching a normalized pattern ('*' = any substring)."""
        return self.explain(pattern)["matches"]
