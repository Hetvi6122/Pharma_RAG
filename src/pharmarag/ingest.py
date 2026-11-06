"""Download FDA drug labels from openFDA and split them into section-aware chunks.

Each chunk keeps its drug name and label section so answers can cite
"Drug X, Warnings" instead of a random block of text.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

OPENFDA_URL = "https://api.fda.gov/drug/label.json"

# Label sections we keep, mapped to readable names used in citations.
SECTIONS = {
    "boxed_warning": "Boxed Warning",
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "use_in_specific_populations": "Use in Specific Populations",
    "mechanism_of_action": "Mechanism of Action",
}


def fetch_labels(n: int = 500, page_size: int = 100, pause: float = 0.3) -> list[dict]:
    """Fetch up to n labels that have a generic name. openFDA allows 240 requests/min without a key."""
    labels: list[dict] = []
    skip = 0
    while len(labels) < n:
        params = {
            "search": "_exists_:openfda.generic_name",
            "limit": min(page_size, n - len(labels)),
            "skip": skip,
        }
        resp = requests.get(OPENFDA_URL, params=params, timeout=30)
        resp.raise_for_status()
        batch = resp.json().get("results", [])
        if not batch:
            break
        labels.extend(batch)
        skip += len(batch)
        time.sleep(pause)
    return labels[:n]


def pick_labels(results: list[dict], per_drug: int) -> list[dict]:
    """Prefer single-ingredient labels (no ' AND ' in the generic name) with the most sections."""
    def single(lab):
        return " AND " not in " ".join(lab.get("openfda", {}).get("generic_name", [])).upper()

    def n_sections(lab):
        return sum(1 for f in SECTIONS if lab.get(f))

    ranked = sorted(results, key=lambda lab: (single(lab), n_sections(lab)), reverse=True)
    return ranked[:per_drug]


def fetch_labels_for_drugs(
    generics: list[str], per_drug: int = 2, pause: float = 0.3
) -> tuple[list[dict], list[str]]:
    """Fetch prescription labels for specific generic names. Returns (labels, names not found)."""
    labels, missing = [], []
    for name in generics:
        params = {
            "search": f'openfda.generic_name:"{name}" AND '
            f'openfda.product_type:"HUMAN PRESCRIPTION DRUG"',
            "limit": 20,
        }
        resp = requests.get(OPENFDA_URL, params=params, timeout=30)
        if resp.status_code == 404:  # openFDA returns 404 when nothing matches
            missing.append(name)
            continue
        resp.raise_for_status()
        labels.extend(pick_labels(resp.json().get("results", []), per_drug))
        time.sleep(pause)
    return labels, missing


def dedupe(labels: list[dict]) -> list[dict]:
    seen, out = set(), []
    for lab in labels:
        if lab.get("id") not in seen:
            seen.add(lab.get("id"))
            out.append(lab)
    return out


def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    """Split text into windows of `size` words that overlap by `overlap` words."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    if not words:
        return []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks


def _first(value) -> str:
    """openFDA stores most fields as lists of strings."""
    if isinstance(value, list):
        return value[0] if value else ""
    return value or ""


def label_to_chunks(label: dict, size: int = 200, overlap: int = 40) -> list[dict]:
    """Turn one openFDA label into chunk records with citation metadata."""
    meta = label.get("openfda", {})
    drug = _first(meta.get("brand_name")) or _first(meta.get("generic_name")) or "Unknown"
    generic = _first(meta.get("generic_name"))
    label_id = label.get("id", "")
    records = []
    for field, section in SECTIONS.items():
        text = " ".join(label.get(field, []) or [])
        for i, chunk in enumerate(chunk_text(text, size, overlap)):
            records.append(
                {
                    "chunk_id": f"{label_id}:{field}:{i}",
                    "label_id": label_id,
                    "drug": drug,
                    "generic_name": generic,
                    "section": section,
                    "text": chunk,
                }
            )
    return records


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
