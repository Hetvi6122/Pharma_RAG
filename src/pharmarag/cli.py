"""Command line: `pharmarag ingest --n 500` and `pharmarag search "question" --mode hybrid`."""

from __future__ import annotations

import argparse
from pathlib import Path

from pharmarag.bm25 import BM25
from pharmarag.hybrid import rrf
from pharmarag.ingest import fetch_labels, label_to_chunks, read_jsonl, write_jsonl

CHUNKS = Path("data/chunks.jsonl")
EMB_CACHE = Path("data/emb.npy")


def cmd_ingest(args) -> None:
    labels = fetch_labels(n=args.n)
    chunks = [c for label in labels for c in label_to_chunks(label)]
    write_jsonl(chunks, CHUNKS)
    print(f"Fetched {len(labels)} labels -> {len(chunks)} chunks -> {CHUNKS}")


def cmd_search(args) -> None:
    chunks = read_jsonl(CHUNKS)
    texts = [f"{c['drug']} {c['section']}: {c['text']}" for c in chunks]
    pool = max(args.k * 4, 20)  # candidates per retriever before fusion

    if args.mode in ("dense", "hybrid"):
        from pharmarag.dense import DenseIndex, load_encoder

        dense = DenseIndex(load_encoder())
        dense.build(texts, cache=EMB_CACHE)

    if args.mode == "bm25":
        results = BM25(texts).search(args.query, k=args.k)
    elif args.mode == "dense":
        results = dense.search(args.query, k=args.k)
    else:
        bm = [i for i, _ in BM25(texts).search(args.query, k=pool)]
        de = [i for i, _ in dense.search(args.query, k=pool)]
        results = rrf([bm, de], top_n=args.k)

    for rank, (i, score) in enumerate(results, start=1):
        c = chunks[i]
        print(f"\n[{rank}] {c['drug']} | {c['section']} | score={score:.4f}")
        print(f"    {c['text'][:300]}...")


def main() -> None:
    p = argparse.ArgumentParser(prog="pharmarag")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="download labels from openFDA and chunk them")
    pi.add_argument("--n", type=int, default=500)
    pi.set_defaults(func=cmd_ingest)

    ps = sub.add_parser("search", help="search the chunks")
    ps.add_argument("query")
    ps.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ps.add_argument("--k", type=int, default=5)
    ps.set_defaults(func=cmd_search)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
