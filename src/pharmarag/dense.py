"""Dense (semantic) search with sentence embeddings and cosine similarity.

Embeddings let "heart attack" match "myocardial infarction", which BM25 misses.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_encoder(model_name: str = DEFAULT_MODEL) -> Callable[[list[str]], np.ndarray]:
    """Load MiniLM lazily so the rest of the package works without sentence-transformers."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts, batch_size=64, show_progress_bar=False)


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


class DenseIndex:
    def __init__(self, encoder: Callable[[list[str]], np.ndarray]):
        self.encoder = encoder
        self.emb: np.ndarray | None = None

    def build(self, docs: list[str], cache: Path | None = None) -> None:
        if cache and cache.exists():
            self.emb = np.load(cache)
            if len(self.emb) == len(docs):
                return
        self.emb = normalize(np.asarray(self.encoder(docs), dtype=np.float32))
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.emb)

    def search(self, query: str, k: int = 5) -> list[tuple[int, float]]:
        q = normalize(np.asarray(self.encoder([query]), dtype=np.float32))[0]
        sims = self.emb @ q  # cosine similarity, since vectors are unit length
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.argsort(-sims[top])]
        return [(int(i), float(sims[i])) for i in top]
