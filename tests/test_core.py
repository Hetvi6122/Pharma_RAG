import numpy as np
import pytest

from pharmarag.bm25 import BM25, tokenize
from pharmarag.dense import DenseIndex
from pharmarag.hybrid import rrf
from pharmarag.ingest import chunk_text, label_to_chunks


def test_chunk_text_size_and_overlap():
    text = " ".join(f"w{i}" for i in range(500))
    chunks = chunk_text(text, size=200, overlap=40)
    assert all(len(c.split()) <= 200 for c in chunks)
    # last 40 words of chunk 0 are the first 40 words of chunk 1
    assert chunks[0].split()[-40:] == chunks[1].split()[:40]
    # every word is covered
    assert chunks[-1].split()[-1] == "w499"


def test_chunk_text_edge_cases():
    assert chunk_text("") == []
    assert chunk_text("one two three", size=200, overlap=40) == ["one two three"]
    with pytest.raises(ValueError):
        chunk_text("a b c", size=10, overlap=10)


def test_label_to_chunks_metadata():
    label = {
        "id": "abc123",
        "openfda": {"brand_name": ["Tylenol"], "generic_name": ["ACETAMINOPHEN"]},
        "warnings": ["Liver warning: severe liver damage may occur."],
        "indications_and_usage": ["Temporarily relieves minor aches and pains."],
    }
    chunks = label_to_chunks(label)
    assert len(chunks) == 2
    assert {c["section"] for c in chunks} == {"Warnings", "Indications and Usage"}
    assert all(c["drug"] == "Tylenol" and c["label_id"] == "abc123" for c in chunks)
    assert chunks[0]["chunk_id"].startswith("abc123:")


def test_bm25_ranks_matching_doc_first():
    docs = [
        "ibuprofen may cause stomach bleeding",
        "acetaminophen can cause severe liver damage",
        "metformin is used for type 2 diabetes",
    ]
    bm = BM25(docs)
    top = bm.search("liver damage", k=3)
    assert top[0][0] == 1
    assert len(top) == 1  # only doc 1 contains the query terms
    assert "the" not in tokenize("the liver")


def test_rrf_rewards_docs_in_both_lists():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60, top_n=4)
    ids = [d for d, _ in fused]
    assert ids[0] == 1  # high in both lists
    assert set(ids) == {1, 2, 3, 4}


def test_dense_index_with_fake_encoder():
    # Fake encoder: one-hot vectors keyed by first word, so no model download in tests.
    vocab = {"liver": 0, "stomach": 1, "diabetes": 2}

    def enc(texts):
        out = np.zeros((len(texts), 3))
        for i, t in enumerate(texts):
            out[i, vocab[t.split()[0]]] = 1.0
        return out

    idx = DenseIndex(enc)
    idx.build(["stomach bleeding", "liver damage", "diabetes control"])
    assert idx.search("liver problems", k=1)[0][0] == 1
