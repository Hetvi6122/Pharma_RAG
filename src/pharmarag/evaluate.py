"""Retrieval evaluation: Recall@k and MRR over a labeled question set.

A chunk counts as relevant to a question if it comes from the right drug
(generic name match) AND the right label section. This is section-level
relevance: it checks we found the right part of the right label.

- Recall@k: share of questions with at least one relevant chunk in the top k.
- MRR (Mean Reciprocal Rank): average of 1/rank of the first relevant chunk
  (0 if none in the top k). Rewards putting the right chunk near the top.
"""

from __future__ import annotations

import json
from pathlib import Path


def is_relevant(chunk: dict, q: dict) -> bool:
    generic = (chunk.get("generic_name") or "").lower()
    return q["generic"].lower() in generic and chunk["section"] in q["sections"]


def first_relevant_rank(ranked_ids: list[int], chunks: list[dict], q: dict) -> int | None:
    for rank, i in enumerate(ranked_ids, start=1):
        if is_relevant(chunks[i], q):
            return rank
    return None


def score(ranks: list[int | None], k: int) -> dict:
    n = len(ranks)
    hits = sum(1 for r in ranks if r is not None and r <= k)
    mrr = sum(1.0 / r for r in ranks if r is not None and r <= k)
    return {"recall@k": hits / n if n else 0.0, "mrr": mrr / n if n else 0.0, "n": n}


def answerable(questions: list[dict], chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split questions into ones the corpus can answer and ones it cannot (drug/section missing)."""
    ok, missing = [], []
    for q in questions:
        (ok if any(is_relevant(c, q) for c in chunks) else missing).append(q)
    return ok, missing


def evaluate(searcher, questions: list[dict], modes: list[str], k: int = 5) -> dict:
    results = {}
    for mode in modes:
        ranks, per_q = [], []
        for q in questions:
            ids = [i for i, _ in searcher.search(q["question"], mode=mode, k=k)]
            r = first_relevant_rank(ids, searcher.chunks, q)
            ranks.append(r)
            per_q.append({"id": q["id"], "rank": r})
        results[mode] = {**score(ranks, k), "per_question": per_q}
    return results


def to_markdown(results: dict, k: int) -> str:
    lines = [f"| Mode | Recall@{k} | MRR@{k} | Questions |", "|---|---|---|---|"]
    for mode, r in results.items():
        lines.append(f"| {mode} | {r['recall@k']:.3f} | {r['mrr']:.3f} | {r['n']} |")
    return "\n".join(lines)


def load_questions(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
