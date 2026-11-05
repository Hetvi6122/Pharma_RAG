# PharmaRAG

Ask plain-English questions about prescription drugs and get answers grounded in official FDA drug labels, with citations.

![CI](https://github.com/Hetvi6122/Pharma_RAG/actions/workflows/ci.yml/badge.svg)

> **Status: v0.1 (retrieval only).** Answer generation, evaluation and the agent come next.

## Problem
Drug labels are long, dense documents. Keyword search misses paraphrases ("heart attack" vs "myocardial infarction") and pure semantic search can miss exact drug names. This project combines both and keeps every result tied to a drug and label section so answers can cite their source.

## Results
| Approach | Recall@5 | MRR | Notes |
|---|---|---|---|
| BM25 (from scratch) | [placeholder] | [placeholder] | |
| Dense (MiniLM) | [placeholder] | [placeholder] | |
| Hybrid (RRF) | [placeholder] | [placeholder] | |

Numbers will come from a hand-built eval set of ~50 questions (Week 2). Nothing here is estimated.

## Architecture
```
openFDA label API -> section-aware chunking (200 words, 40 overlap) -> BM25 index + MiniLM embeddings -> Reciprocal Rank Fusion -> top-k cited chunks
```

## Tech Stack
Python · NumPy · sentence-transformers (MiniLM) · BM25 from scratch · pytest · ruff · GitHub Actions

## Quickstart
```bash
git clone https://github.com/Hetvi6122/Pharma_RAG.git && cd Pharma_RAG
pip install -e ".[dev,dense]"
pharmarag ingest --n 500
pharmarag search "Can acetaminophen cause liver damage?" --mode hybrid
```
Use `--mode bm25` to skip the embedding model. Copy `.env.example` to `.env` for keys. Keys are never committed.

## Data
[openFDA Drug Label API](https://open.fda.gov/apis/drug/label/), public domain (U.S. FDA). Downloaded data is not committed; rebuild it with `pharmarag ingest`.

## Project Structure
```
src/pharmarag/
  ingest.py   # openFDA download + section-aware chunking
  bm25.py     # BM25 with an inverted index
  dense.py    # MiniLM embeddings + cosine search (cached)
  hybrid.py   # Reciprocal Rank Fusion
  cli.py      # ingest / search commands
tests/test_core.py
```

## Disclaimer
Not medical advice. For educational and portfolio purposes only.

---
Built by Hetvi Patel · [LinkedIn](https://linkedin.com/in/hetvi-patel-a9b6341a4)
