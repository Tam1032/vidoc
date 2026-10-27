import json

import httpx
import pytest

from vidoc.evals.extraction import aggregate, field_scores
from vidoc.extract import Extractor, parse_invoice
from vidoc.store import ClaimStore


def label():
    return ClaimStore.load().claims["C0001"]["invoice"]


def test_parse_tolerates_fences():
    inv = parse_invoice("```json\n" + json.dumps(label(), ensure_ascii=False) + "\n```")
    assert inv.total == label()["total"]
    with pytest.raises(ValueError):
        parse_invoice("sorry, I cannot read this")


async def test_extractor_sends_schema_and_image():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(200, json={"id": "x", "object": "chat.completion", "created": 0, "model": "vidoc",
                                         "choices": [{"index": 0, "finish_reason": "stop", "message": {
                                             "role": "assistant", "content": json.dumps(label(), ensure_ascii=False)}}],
                                         "usage": {"prompt_tokens": 900, "completion_tokens": 300, "total_tokens": 1200}})

    ex = Extractor("vidoc", "http://vllm/v1", http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    r = await ex.extract(ClaimStore.load().image_path("C0001"))
    assert r.invoice and r.invoice.total == label()["total"] and r.completion_tokens == 300
    assert seen["response_format"]["json_schema"]["schema"]["title"] == "Invoice"
    assert seen["messages"][0]["content"][0]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_field_scores_and_aggregate():
    from vidoc.schema import Invoice

    gold = label()
    pred = dict(gold, total=gold["total"] + 1000, items=gold["items"][:-1])
    s = field_scores(gold, pred)
    assert not s["fields"]["total"] and s["fields"]["seller_name"] and 0 < s["item_f1"] < 1 and not s["exact_match"]
    rep = aggregate([(gold, Invoice(**gold), 100.0), (gold, None, 200.0)])
    assert rep["valid_json_rate"] == 0.5 and rep["doc_exact_match"] == 0.5
