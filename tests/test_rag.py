import json

from vidoc.evals.retrieval import GOLDEN, evaluate
from vidoc.rag import Retriever, rrf


def test_chunks_are_numbered_rules():
    r = Retriever.from_dir()
    ids = [c.id for c in r.chunks]
    assert "expense_policy#2.3" in ids and len(ids) == len(set(ids)) >= 15


def test_bm25_recall_floor():
    rows = [json.loads(l) for l in GOLDEN.read_text().splitlines() if l]
    res = evaluate(Retriever.from_dir(), rows, "bm25")
    assert res["recall@3"] >= 0.85


def test_hybrid_with_fake_embedder():
    vocab = ["taxi", "sân bay", "khách sạn", "hóa đơn", "thiết bị"]

    def embed(texts):
        return [[1.0 if w in t.lower() else 0.0 for w in vocab] for t in texts]

    r = Retriever.from_dir(embedder=embed)
    top = r.search("đi taxi ra sân bay", k=1, mode="hybrid")[0]
    assert top.id == "expense_policy#2.3"


def test_rrf():
    assert rrf([[0, 1, 2], [2, 0, 1]])[0] == 0
