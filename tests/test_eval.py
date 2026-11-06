import json
from pathlib import Path

import pytest

from pharmarag.evaluate import answerable, evaluate, first_relevant_rank, is_relevant, score
from pharmarag.ingest import dedupe, pick_labels
from pharmarag.search import Searcher

CHUNKS = [
    {"drug": "Glucophage", "generic_name": "METFORMIN HYDROCHLORIDE",
     "section": "Adverse Reactions", "text": "diarrhea nausea vomiting flatulence"},
    {"drug": "Zestril", "generic_name": "LISINOPRIL",
     "section": "Indications and Usage", "text": "treatment of hypertension and heart failure"},
    {"drug": "Glucophage", "generic_name": "METFORMIN HYDROCHLORIDE",
     "section": "Contraindications", "text": "severe renal impairment eGFR below 30"},
]
Q_SIDE = {"id": "q1", "generic": "metformin", "question": "metformin side effects diarrhea",
          "sections": ["Adverse Reactions"]}


def test_is_relevant_needs_drug_and_section():
    assert is_relevant(CHUNKS[0], Q_SIDE)
    assert not is_relevant(CHUNKS[2], Q_SIDE)  # right drug, wrong section
    assert not is_relevant(CHUNKS[1], Q_SIDE)  # wrong drug


def test_first_relevant_rank():
    assert first_relevant_rank([2, 0, 1], CHUNKS, Q_SIDE) == 2
    assert first_relevant_rank([1, 2], CHUNKS, Q_SIDE) is None


def test_score_recall_and_mrr():
    s = score([1, 2, None, 6], k=5)
    assert s["recall@k"] == pytest.approx(0.5)  # ranks 1 and 2 are within k; 6 is not
    assert s["mrr"] == pytest.approx((1 + 0.5) / 4)


def test_answerable_skips_missing_drugs():
    q_missing = {**Q_SIDE, "id": "q2", "generic": "warfarin"}
    ok, missing = answerable([Q_SIDE, q_missing], CHUNKS)
    assert [q["id"] for q in ok] == ["q1"] and [q["id"] for q in missing] == ["q2"]


def test_evaluate_bm25_end_to_end():
    res = evaluate(Searcher(CHUNKS, use_dense=False), [Q_SIDE], ["bm25"], k=2)
    assert res["bm25"]["recall@k"] == 1.0 and res["bm25"]["mrr"] == 1.0


def test_pick_labels_prefers_single_ingredient():
    combo = {"id": "a", "openfda": {"generic_name": ["LOSARTAN AND HYDROCHLOROTHIAZIDE"]},
             "warnings": ["x"], "adverse_reactions": ["y"]}
    single = {"id": "b", "openfda": {"generic_name": ["LOSARTAN POTASSIUM"]}, "warnings": ["x"]}
    assert pick_labels([combo, single], per_drug=1)[0]["id"] == "b"
    assert [lab["id"] for lab in dedupe([single, combo, single])] == ["b", "a"]


def test_question_file_is_valid():
    path = Path(__file__).parents[1] / "eval" / "questions.jsonl"
    qs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    assert len(qs) == 50
    assert len({q["id"] for q in qs}) == 50
    assert all(q["generic"] and q["question"] and q["sections"] for q in qs)
