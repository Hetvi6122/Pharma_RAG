"""One place that builds the retrievers, so the CLI and the evaluator search the same way."""

from __future__ import annotations

from pathlib import Path

from pharmarag.bm25 import BM25
from pharmarag.hybrid import rrf

EMB_CACHE = Path("data/emb.npy")


def chunk_to_text(c: dict) -> str:
    """Prefix drug + section so both retrievers can match on them."""
    return f"{c['drug']} {c['generic_name']} {c['section']}: {c['text']}"


class Searcher:
    def __init__(self, chunks: list[dict], use_dense: bool = True, encoder=None):
        self.chunks = chunks
        texts = [chunk_to_text(c) for c in chunks]
        self.bm25 = BM25(texts)
        self.dense = None
        if use_dense:
            from pharmarag.dense import DenseIndex, load_encoder

            self.dense = DenseIndex(encoder or load_encoder())
            self.dense.build(texts, cache=None if encoder else EMB_CACHE)

    def search(self, query: str, mode: str = "hybrid", k: int = 5) -> list[tuple[int, float]]:
        if mode == "bm25":
            return self.bm25.search(query, k=k)
        if self.dense is None:
            raise ValueError("dense index not built; create Searcher(use_dense=True)")
        if mode == "dense":
            return self.dense.search(query, k=k)
        pool = max(k * 4, 20)  # candidates per retriever before fusion
        bm = [i for i, _ in self.bm25.search(query, k=pool)]
        de = [i for i, _ in self.dense.search(query, k=pool)]
        return rrf([bm, de], top_n=k)
