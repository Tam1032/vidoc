"""Retrieval eval: recall@k and MRR of BM25, dense and hybrid on hand-written questions.

    python -m vidoc.evals.retrieval                       # BM25 only
    EMBED_MODEL=text-embedding-3-small python -m vidoc.evals.retrieval
"""

import json
from pathlib import Path

from vidoc.rag import Retriever, openai_embedder

GOLDEN = Path(__file__).resolve().parents[2] / "evals" / "rag_golden.jsonl"


def evaluate(retriever: Retriever, rows: list[dict], mode: str, ks=(1, 3, 5)) -> dict:
    hits = {k: 0 for k in ks}
    mrr = 0.0
    for r in rows:
        ranked = [c.id for c in retriever.search(r["question"], k=len(retriever.chunks), mode=mode)]
        rank = ranked.index(r["gold"]) + 1 if r["gold"] in ranked else None
        for k in ks:
            hits[k] += bool(rank and rank <= k)
        mrr += 1 / rank if rank else 0
    n = len(rows)
    return {"mode": mode, **{f"recall@{k}": round(hits[k] / n, 3) for k in ks}, "mrr": round(mrr / n, 3)}


def main() -> None:
    rows = [json.loads(line) for line in GOLDEN.read_text().splitlines() if line]
    embedder = openai_embedder()
    retriever = Retriever.from_dir(embedder=embedder)
    modes = ["bm25"] + (["dense", "hybrid"] if embedder else [])
    print("| mode | recall@1 | recall@3 | recall@5 | MRR |\n|---|---|---|---|---|")
    for m in modes:
        r = evaluate(retriever, rows, m)
        print(f"| {m} | {r['recall@1']} | {r['recall@3']} | {r['recall@5']} | {r['mrr']} |")


if __name__ == "__main__":
    main()
