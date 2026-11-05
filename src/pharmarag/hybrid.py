"""Reciprocal Rank Fusion (RRF): merge ranked lists using ranks, not raw scores.

BM25 scores and cosine similarities live on different scales, so adding them
directly is meaningless. RRF gives each doc 1 / (k + rank) from every list.
"""

from __future__ import annotations

from collections import defaultdict


def rrf(ranked_lists: list[list[int]], k: int = 60, top_n: int = 5) -> list[tuple[int, float]]:
    fused: dict[int, float] = defaultdict(float)
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] += 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_n]
