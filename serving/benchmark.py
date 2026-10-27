"""Load test an extraction endpoint: latency percentiles, throughput, error rate, field accuracy.

    python serving/benchmark.py --data data/synth --split test --n 200 --concurrency 1 4 16 \
        --base-url http://localhost:8001/v1 --model vidoc
"""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

from vidoc.evals.extraction import field_scores
from vidoc.extract import Extractor


async def run_level(ex: Extractor, rows: list[dict], data_dir: Path, concurrency: int) -> dict:
    sem = asyncio.Semaphore(concurrency)
    results = []

    async def one(row):
        async with sem:
            try:
                r = await ex.extract(data_dir / row["image"])
                results.append((row, r, None))
            except Exception as exc:  # timeouts, 5xx after retries
                results.append((row, None, repr(exc)))

    t0 = time.perf_counter()
    await asyncio.gather(*(one(r) for r in rows))
    wall = time.perf_counter() - t0
    lat = sorted(r.latency_ms for _, r, e in results if r)
    ok = [(row, r) for row, r, e in results if r and r.invoice]
    scores = [field_scores(row["label"], r.invoice.model_dump())["exact_match"] for row, r in ok]
    out_tokens = sum(r.completion_tokens for _, r, e in results if r)

    def pct(p):
        return round(lat[min(len(lat) - 1, int(p * len(lat)))]) if lat else None

    return {
        "concurrency": concurrency, "requests": len(rows),
        "errors": sum(1 for _, r, e in results if e), "invalid_json": sum(1 for _, r, e in results if r and not r.invoice),
        "p50_ms": pct(0.5), "p95_ms": pct(0.95), "mean_ms": round(statistics.mean(lat)) if lat else None,
        "throughput_img_s": round(len(rows) / wall, 2), "output_tok_s": round(out_tokens / wall, 1),
        "doc_exact_match": round(sum(scores) / len(scores), 3) if scores else None,
    }


async def main_async(a) -> None:
    data_dir = Path(a.data)
    rows = [json.loads(line) for line in (data_dir / f"{a.split}.jsonl").read_text().splitlines() if line][: a.n]
    ex = Extractor(a.model, a.base_url, a.api_key, guided=not a.no_guided)
    await ex.extract(data_dir / rows[0]["image"])  # warm-up
    report = [await run_level(ex, rows, data_dir, c) for c in a.concurrency]
    print("| concurrency | p50 ms | p95 ms | img/s | out tok/s | errors | invalid JSON | doc exact |")
    print("|---|---|---|---|---|---|---|---|")
    for r in report:
        print(f"| {r['concurrency']} | {r['p50_ms']} | {r['p95_ms']} | {r['throughput_img_s']} | {r['output_tok_s']} "
              f"| {r['errors']} | {r['invalid_json']} | {r['doc_exact_match']} |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"model": a.model, "levels": report}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synth")
    p.add_argument("--split", default="test")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--base-url", default="http://localhost:8001/v1")
    p.add_argument("--model", default="vidoc")
    p.add_argument("--api-key", default=None)
    p.add_argument("--no-guided", action="store_true")
    p.add_argument("--out", default="reports/benchmark.json")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
