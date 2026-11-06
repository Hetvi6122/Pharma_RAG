# Stripped setting (DECODE-JOIN POC)

**Owner: Hetvi Patel** (stripped setting: build, verification, evaluation).

Stripped = same rows, schema/table/column names, column order and types as the
documented database. Removed: declared primary/foreign keys, UNIQUE/CHECK/NOT NULL
constraints, indexes, views, sequences, comments, and metadata schemas (e.g. `bird_metadata`).

| Dataset | Stripped DuckDB | Status |
|---|---|---|
| BIRD dev | `decode_join_bird_dev_stripped.duckdb` (build script: `data/bird_stripped/`) | done, HF PR #11 |
| KaggleDBQA | `decode_join_kaggledbqa_stripped.duckdb` | in progress |
| FIBEN | `decode_join_fiben_stripped.duckdb` | pending |

## Rebuild
```
bash data/stripped/run_stripped.sh <documented.duckdb> <stripped.duckdb>
```
`make_stripped.py` rebuilds every table with CREATE TABLE AS SELECT (keeps columns, order,
types, rows; drops constraints and comments). `verify_stripped.py` checks identical
tables, columns, row counts and per-column hashes, and that no constraints, indexes, views,
sequences, comments or dropped schemas remain.

`extract_gold_keys.py` records the declared keys from the source SQLite files as
**evaluation-only** gold. Never put its output where the pipeline can read it.
