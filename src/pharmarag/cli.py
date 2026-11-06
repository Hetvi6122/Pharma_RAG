"""Command line.

pharmarag ingest --n 500 --with-eval      # random labels + labels for the eval drugs
pharmarag search "question" --mode hybrid
pharmarag eval --k 5                      # Recall@k and MRR for bm25, dense, hybrid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pharmarag.evaluate import answerable, evaluate, load_questions, to_markdown
from pharmarag.ingest import (
    dedupe,
    fetch_labels,
    fetch_labels_for_drugs,
    label_to_chunks,
    read_jsonl,
    write_jsonl,
)
from pharmarag.search import Searcher

CHUNKS = Path("data/chunks.jsonl")
EMB_CACHE = Path("data/emb.npy")
QUESTIONS = Path("eval/questions.jsonl")
RESULTS = Path("results")


def cmd_ingest(args) -> None:
    labels = fetch_labels(n=args.n)
    if args.with_eval:
        generics = sorted({q["generic"] for q in load_questions(QUESTIONS)})
        extra, missing = fetch_labels_for_drugs(generics)
        labels = dedupe(labels + extra)
        if missing:
            print(f"Not found on openFDA: {', '.join(missing)}")
    chunks = [c for label in labels for c in label_to_chunks(label)]
    write_jsonl(chunks, CHUNKS)
    EMB_CACHE.unlink(missing_ok=True)  # corpus changed, so old embeddings are stale
    print(f"{len(labels)} labels -> {len(chunks)} chunks -> {CHUNKS}")


def cmd_search(args) -> None:
    chunks = read_jsonl(CHUNKS)
    searcher = Searcher(chunks, use_dense=args.mode != "bm25")
    for rank, (i, score) in enumerate(searcher.search(args.query, args.mode, args.k), start=1):
        c = chunks[i]
        print(f"\n[{rank}] {c['drug']} | {c['section']} | score={score:.4f}")
        print(f"    {c['text'][:300]}...")


def cmd_eval(args) -> None:
    chunks = read_jsonl(CHUNKS)
    questions, missing = answerable(load_questions(QUESTIONS), chunks)
    if missing:
        print(f"Skipping {len(missing)} questions the corpus cannot answer: "
              f"{', '.join(q['id'] for q in missing)}")
    use_dense = any(m != "bm25" for m in args.modes)
    searcher = Searcher(chunks, use_dense=use_dense)
    results = evaluate(searcher, questions, args.modes, k=args.k)

    table = to_markdown(results, args.k)
    print(f"\nCorpus: {len(chunks)} chunks\n\n{table}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "retrieval.md").write_text(
        f"Corpus: {len(chunks)} chunks. Skipped (not in corpus): {len(missing)}.\n\n{table}\n"
    )
    (RESULTS / "retrieval.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {RESULTS}/retrieval.md and retrieval.json")


def main() -> None:
    p = argparse.ArgumentParser(prog="pharmarag")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="download labels from openFDA and chunk them")
    pi.add_argument("--n", type=int, default=500)
    pi.add_argument("--with-eval", action="store_true", help="also fetch the eval set's drugs")
    pi.set_defaults(func=cmd_ingest)

    ps = sub.add_parser("search", help="search the chunks")
    ps.add_argument("query")
    ps.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ps.add_argument("--k", type=int, default=5)
    ps.set_defaults(func=cmd_search)

    pe = sub.add_parser("eval", help="Recall@k and MRR on eval/questions.jsonl")
    pe.add_argument("--k", type=int, default=5)
    pe.add_argument("--modes", nargs="+", default=["bm25", "dense", "hybrid"])
    pe.set_defaults(func=cmd_eval)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
