"""HTTP service.

    POST /extract                  upload an invoice image, get fields + arithmetic problems
    POST /claims/{id}/review       run the agent; its approve/reject/escalate call waits for a person
    GET  /actions                  proposed decisions waiting for confirmation
    POST /actions/{n}/confirm      apply a proposed decision
"""

import io
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile
from mcp import Client
from PIL import Image

from vidoc.agent import build_agent, review
from vidoc.harness.loop import DeferToQueue
from vidoc.mcp_server import build_server, oracle_reader, vlm_reader
from vidoc.rag import Retriever, openai_embedder
from vidoc.schema import check
from vidoc.store import ClaimStore

log = logging.getLogger("vidoc.api")


def create_app(llm=None, extractor=None, store: ClaimStore | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.store = store or ClaimStore.load()
        app.state.extractor = extractor
        if app.state.extractor is None and os.getenv("VLM_MODEL"):
            from vidoc.extract import Extractor
            app.state.extractor = Extractor.from_env("VLM")
        app.state.llm = llm
        if app.state.llm is None:
            if os.getenv("LLM_MODEL"):
                from vidoc.harness.llm import OpenAICompatLLM
                app.state.llm = OpenAICompatLLM.from_env("LLM")
            else:
                from vidoc.baseline import PolicyRules
                log.warning("LLM_MODEL is not set; using the hand-coded policy rules")
                app.state.llm = PolicyRules()
        reader = vlm_reader(app.state.extractor) if app.state.extractor else oracle_reader
        app.state.server = build_server(app.state.store, reader, Retriever.from_dir(embedder=openai_embedder()))
        app.state.pending = []
        yield

    app = FastAPI(title="ViDoc", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        ex = app.state.extractor
        return {"ok": True, "agent_model": app.state.llm.name, "vlm": ex.model if ex else "oracle (ground truth)"}

    @app.post("/extract")
    async def extract(file: UploadFile):
        if app.state.extractor is None:
            raise HTTPException(503, "set VLM_MODEL (and VLM_BASE_URL for vLLM) to enable extraction")
        img = Image.open(io.BytesIO(await file.read()))
        r = await app.state.extractor.extract(img)
        if r.invoice is None:
            raise HTTPException(422, f"model output did not match the schema: {r.error}")
        return {"invoice": r.invoice.model_dump(), "problems": check(r.invoice), "latency_ms": round(r.latency_ms)}

    @app.post("/claims/{claim_id}/review")
    async def review_claim(claim_id: str):
        if claim_id not in app.state.store.claims:
            raise HTTPException(404, "no such claim")
        queue = DeferToQueue()
        async with Client(app.state.server) as mcp:
            run = await review(await build_agent(app.state.llm, mcp, confirmer=queue), claim_id)
        start = len(app.state.pending)
        app.state.pending += [{"id": start + i, "tool": a.tool, "arguments": a.arguments, "status": "pending"}
                              for i, a in enumerate(queue.queued)]
        return {"claim_id": claim_id, "decision": run.final, "stop_reason": run.stop_reason,
                "proposed_actions": app.state.pending[start:]}

    @app.get("/actions")
    def actions():
        return {"items": [a for a in app.state.pending if a["status"] == "pending"]}

    @app.post("/actions/{n}/confirm")
    async def confirm(n: int):
        if n >= len(app.state.pending) or app.state.pending[n]["status"] != "pending":
            raise HTTPException(404, "no pending action with that id")
        a = app.state.pending[n]
        async with Client(app.state.server) as mcp:
            result = await mcp.call_tool(a["tool"], a["arguments"])
        if result.is_error:
            a["status"] = "failed"
            raise HTTPException(409, result.content[0].text)
        a["status"] = "done"
        return {"id": n, "status": "done", "result": result.content[0].text}

    return app


app = create_app()
