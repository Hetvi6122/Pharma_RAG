"""BM25 keyword search, written from scratch with an inverted index.

Inverted index = hash map from term -> {doc_id: term_count}. A query only
touches documents that contain at least one query term, instead of scanning
every document.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import Counter, defaultdict

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is",
    "it", "of", "on", "or", "that", "the", "to", "was", "were", "with", "what",
    "which", "who", "how", "does", "do", "can", "this",
}


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.n_docs = len(docs)
        self.doc_len = []
        self.index: dict[str, dict[int, int]] = defaultdict(dict)
        for doc_id, doc in enumerate(docs):
            counts = Counter(tokenize(doc))
            self.doc_len.append(sum(counts.values()))
            for term, tf in counts.items():
                self.index[term][doc_id] = tf
        self.avg_len = sum(self.doc_len) / self.n_docs if self.n_docs else 0.0

    def idf(self, term: str) -> float:
        df = len(self.index.get(term, {}))
        # +1 inside the log keeps idf positive for very common terms.
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> dict[int, float]:
        out: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.index.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_id, tf in postings.items():
                norm = 1 - self.b + self.b * self.doc_len[doc_id] / self.avg_len
                out[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return out

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        """Return the top-k (doc_id, score) pairs. A heap gives O(n log k) top-k."""
        s = self.scores(query)
        return heapq.nlargest(k, s.items(), key=lambda x: x[1])
