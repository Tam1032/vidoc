"""Extraction eval: per-field accuracy, line-item F1, document exact match, JSON validity, latency.

    VLM_BASE_URL=http://localhost:8001/v1 VLM_MODEL=vidoc python -m vidoc.evals.extraction --split test --n 300
    VLM_MODEL=gpt-4.1-mini python -m vidoc.evals.extraction --split test --n 300
"""

import argparse
import asyncio
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from vidoc.schema import FIELDS, Invoice, check

KEY_FIELDS = ["seller_tax_code", "invoice_number", "issue_date", "subtotal", "vat_amount", "total"]


def norm(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = unicodedata.normalize("NFC", str(v)).strip().upper()
    return re.sub(r"\s+", " ", s)


def field_scores(gold: dict, pred: dict) -> dict:
    per_field = {f: norm(gold.get(f)) == norm(pred.get(f)) for f in FIELDS}
    g_items = [(norm(i["name"]), int(i["amount"])) for i in gold["items"]]
    p_items = [(norm(i.get("name")), int(i.get("amount") or 0)) for i in pred.get("items", [])]
    matched = 0
    remaining = list(p_items)
    for gi in g_items:
        if gi in remaining:
            remaining.remove(gi)
            matched += 1
    prec = matched / len(p_items) if p_items else 0.0
    rec = matched / len(g_items) if g_items else 1.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"fields": per_field, "item_f1": f1,
            "exact_match": all(per_field[f] for f in KEY_FIELDS) and f1 == 1.0}


def aggregate(rows: list[tuple[dict, Invoice | None, float]]) -> dict:
    n = len(rows)
    field_hits = defaultdict(int)
    f1s, exact, valid, consistent = [], 0, 0, 0
    for gold, inv, _ in rows:
        if inv is None:
            f1s.append(0.0)
            continue
        valid += 1
        consistent += not check(inv)
        s = field_scores(gold, inv.model_dump())
        for f, ok in s["fields"].items():
            field_hits[f] += ok
        f1s.append(s["item_f1"])
        exact += s["exact_match"]
    lat = sorted(l for _, _, l in rows)
    return {
        "n": n,
        "valid_json_rate": round(valid / n, 3),
        "doc_exact_match": round(exact / n, 3),
        "item_f1": round(sum(f1s) / n, 3),
        "arithmetic_consistent_rate": round(consistent / n, 3),
        "field_accuracy": {f: round(field_hits[f] / n, 3) for f in FIELDS},
        "p50_latency_ms": round(lat[n // 2]) if n else None,
    }


async def main_async(a) -> None:
    from vidoc.extract import Extractor

    data_dir = Path(a.data)
    rows = [json.loads(line) for line in (data_dir / f"{a.split}.jsonl").read_text(encoding="utf-8").splitlines() if line][: a.n]
    ex = Extractor.from_env("VLM")
    sem = asyncio.Semaphore(a.concurrency)

    async def one(row):
        async with sem:
            r = await ex.extract(data_dir / row["image"])
            return row["label"], r.invoice, r.latency_ms

    results = await asyncio.gather(*(one(r) for r in rows))
    report = aggregate(results)
    report["model"] = ex.model
    print(json.dumps(report, indent=2, ensure_ascii=False))
    out = Path(a.out) / f"extraction_{re.sub(r'[^\w.-]', '_', ex.model)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synth")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", default="reports")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
