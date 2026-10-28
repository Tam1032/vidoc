"""Agent eval on the expense-claim scenarios; exits 1 when a threshold fails (used in CI).

    python -m vidoc.evals.agent --model rules                               # no model, smoke test
    LLM_MODEL=gpt-4.1-mini python -m vidoc.evals.agent --model env          # oracle extraction
    LLM_MODEL=... VLM_MODEL=vidoc VLM_BASE_URL=http://localhost:8001/v1 \
        python -m vidoc.evals.agent --model env --extraction vlm            # end to end from images
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from mcp import Client

from vidoc.agent import build_agent, review
from vidoc.harness.llm import LLM, OpenAICompatLLM
from vidoc.harness.trace import Tracer
from vidoc.mcp_server import build_server, oracle_reader, vlm_reader
from vidoc.rag import Retriever
from vidoc.store import ClaimStore

ROOT = Path(__file__).resolve().parents[2]
EXECUTED = {"approve": "approved", "reject": "rejected", "escalate": "escalated"}


async def run_case(llm: LLM, case: dict, reader, retriever: Retriever) -> dict:
    store = ClaimStore.load()  # fresh ledger per case
    cid = case["claim_id"]
    tracer = Tracer(run_id=cid)
    async with Client(build_server(store, reader, retriever)) as mcp:
        run = await review(await build_agent(llm, mcp, tracer=tracer), cid)
    taken = store.decisions.get(cid, {})
    cited = set(taken.get("policy_refs", [])) | set((run.final or {}).get("policy_refs", []))
    return {
        "claim_id": cid,
        "expected": case["decision"],
        "executed": taken.get("action", "none"),
        "final": run.final,
        "stop_reason": run.stop_reason,
        "gold_refs": case["policy_refs"],
        "cited": sorted(cited),
        "citation_hit": any(r in cited for r in case["policy_refs"]),
        "server_refusals": sum(1 for e in run.events if e.tool == "approve_expense" and e.outcome == "error"),
        "policy_searches": run.calls().count("search_policy"),
        "steps": run.steps,
        "tokens": run.usage.prompt_tokens + run.usage.completion_tokens,
        "latency_ms": round(run.latency_ms),
        "trace": tracer.events,
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    ok = [r for r in rows if r["executed"] == EXECUTED[r["expected"]]]
    unsafe = [r for r in rows if r["executed"] == "approved" and r["expected"] != "approve"]
    return {
        "cases": n,
        "completion_rate": round(sum(r["stop_reason"] == "final" for r in rows) / n, 3),
        "decision_accuracy": round(len(ok) / n, 3),
        "unsafe_approvals": len(unsafe),
        "citation_recall": round(sum(r["citation_hit"] for r in rows) / n, 3),
        "server_refusals": sum(r["server_refusals"] for r in rows),
        "avg_policy_searches": round(sum(r["policy_searches"] for r in rows) / n, 2),
        "avg_steps": round(sum(r["steps"] for r in rows) / n, 2),
        "avg_tokens": round(sum(r["tokens"] for r in rows) / n),
        "confusion": dict(Counter(f"{r['expected']}->{r['executed']}" for r in rows)),
    }


def gate(summary: dict, thresholds: dict) -> list[str]:
    fails = []
    for key, limit in thresholds.items():
        metric = key.removesuffix("_max")
        v = summary.get(metric)
        if v is None:
            continue
        if (key.endswith("_max") and v > limit) or (not key.endswith("_max") and v < limit):
            fails.append(f"{metric} = {v} (limit {limit})")
    return fails


def make_llm(spec: str) -> LLM:
    if spec == "rules":
        from vidoc.baseline import PolicyRules
        return PolicyRules()
    if spec == "env":
        return OpenAICompatLLM.from_env("LLM")
    return OpenAICompatLLM(model=spec, base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY"))


async def main_async(a) -> int:
    llm = make_llm(a.model)
    cases = [json.loads(l) for l in Path(a.golden).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        cases = cases[: a.limit]
    reader = oracle_reader
    if a.extraction == "vlm":
        from vidoc.extract import Extractor
        reader = vlm_reader(Extractor.from_env("VLM"))
    from vidoc.rag import openai_embedder
    retriever = Retriever.from_dir(embedder=openai_embedder())
    sem = asyncio.Semaphore(a.concurrency)

    async def one(c):
        async with sem:
            return await run_case(llm, c, reader, retriever)

    rows = await asyncio.gather(*(one(c) for c in cases))
    summary = summarize(rows)
    th = json.loads(Path(a.thresholds).read_text(encoding="utf-8"))
    fails = gate(summary, th.get(llm.name, th["default"]))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w.-]", "_", f"{llm.name}_{a.extraction}")
    (out / f"agent_{slug}.json").write_text(json.dumps(
        {"model": llm.name, "extraction": a.extraction, "summary": summary, "failures": fails,
         "cases": [{k: v for k, v in r.items() if k != "trace"} for r in rows]}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / f"agent_traces_{slug}.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            for e in r["trace"]:
                f.write(json.dumps(e, ensure_ascii=False, default=str) + "\n")
    lines = [f"## ViDoc agent eval: `{llm.name}` ({a.extraction} extraction)", "", "| metric | value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in summary.items()]
    lines += ["", "**Gate: PASS**" if not fails else "**Gate: FAIL**\n\n" + "\n".join(f"- {f}" for f in fails)]
    wrong = [r for r in rows if r["executed"] != EXECUTED[r["expected"]]]
    if wrong:
        lines += ["", "Wrong decisions:"] + [f"- {r['claim_id']}: expected {r['expected']}, executed {r['executed']}, "
                                             f"cited {r['cited']}" for r in wrong]
    md = "\n".join(lines) + "\n"
    (out / f"agent_{slug}.md").write_text(md, encoding="utf-8")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(md)
    print(md)
    return 1 if fails else 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="rules")
    p.add_argument("--extraction", choices=["oracle", "vlm"], default="oracle")
    p.add_argument("--golden", default=str(ROOT / "evals" / "claims_golden.jsonl"))
    p.add_argument("--thresholds", default=str(ROOT / "evals" / "thresholds.json"))
    p.add_argument("--out", default=str(ROOT / "reports"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=4)
    sys.exit(asyncio.run(main_async(p.parse_args())))


if __name__ == "__main__":
    main()
