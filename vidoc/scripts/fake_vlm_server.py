"""A stand-in for `vllm serve` with the same /v1/chat/completions shape, for CI and laptops without a GPU.

It does not read the image: it answers with a fixed, schema-valid invoice after a
configurable delay. Use it to test the serving path (client, benchmark, evals),
never to report accuracy.

    uvicorn scripts.fake_vlm_server:app --port 8001
"""

import asyncio
import json
import os
import time

from fastapi import FastAPI

from vidoc.store import ClaimStore

app = FastAPI()
ANSWER = json.dumps(ClaimStore.load().claims["C0001"]["invoice"], ensure_ascii=False)
DELAY = float(os.getenv("FAKE_VLM_DELAY", "0.05"))


@app.get("/health")
def health():
    return {}


@app.post("/v1/chat/completions")
async def chat(body: dict):
    await asyncio.sleep(DELAY)
    return {"id": "fake", "object": "chat.completion", "created": int(time.time()), "model": body.get("model"),
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": ANSWER}}],
            "usage": {"prompt_tokens": 1000, "completion_tokens": 320, "total_tokens": 1320}}
