import json

import httpx
import pytest

from vidoc.api import create_app
from vidoc.baseline import PolicyRules
from vidoc.evals.agent import ROOT, run_case, summarize
from vidoc.mcp_server import oracle_reader
from vidoc.rag import Retriever


async def test_rules_baseline_full_eval():
    cases = [json.loads(l) for l in (ROOT / "evals" / "claims_golden.jsonl").read_text().splitlines() if l]
    retriever = Retriever.from_dir()
    rows = [await run_case(PolicyRules(), c, oracle_reader, retriever) for c in cases]
    s = summarize(rows)
    assert s["decision_accuracy"] == 1.0 and s["unsafe_approvals"] == 0 and s["completion_rate"] == 1.0


@pytest.fixture
async def api():
    app = create_app(llm=PolicyRules())
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as c:
            yield c, app


async def test_review_waits_for_confirmation(api):
    client, app = api
    out = (await client.post("/claims/C0001/review")).json()
    assert out["decision"]["decision"] == "approve"
    [action] = out["proposed_actions"]
    assert action["tool"] == "approve_expense" and app.state.store.decisions == {}
    done = await client.post(f"/actions/{action['id']}/confirm")
    assert done.status_code == 200 and app.state.store.decisions["C0001"]["action"] == "approved"


async def test_extract_needs_vlm(api):
    client, _ = api
    r = await client.post("/extract", files={"file": ("a.jpg", b"x", "image/jpeg")})
    assert r.status_code == 503
