"""Invoice extraction with a vision-language model behind an OpenAI-compatible API.

The same code calls the self-hosted fine-tuned model on vLLM, the base model,
or a hosted model, so the eval compares them on equal terms. Output is forced
into the Invoice JSON Schema (vLLM structured outputs / OpenAI json_schema) and
then validated with pydantic.
"""

import base64
import io
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from pydantic import ValidationError

from vidoc.schema import Invoice, json_schema

PROMPT = (
    "Trích xuất thông tin từ ảnh hóa đơn này và trả về JSON đúng schema. "
    "Số tiền là số nguyên VND, không có dấu phân cách. Ngày theo dạng YYYY-MM-DD. "
    "doc_type là 'vat_invoice' nếu là hóa đơn GTGT, 'retail_receipt' nếu là hóa đơn bán lẻ hoặc biên lai. "
    "Trường nào không có trên ảnh thì để null."
)


def image_to_data_url(image: Image.Image | str | Path, max_side: int = 1280) -> str:
    img = Image.open(image) if isinstance(image, (str, Path)) else image
    img = img.convert("RGB")
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def messages_for(image: Image.Image | str | Path) -> list[dict]:
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": image_to_data_url(image)}},
        {"type": "text", "text": PROMPT},
    ]}]


def parse_invoice(text: str) -> Invoice:
    """Parse model output; tolerate code fences and text around the JSON."""
    m = re.search(r"\{.*\}", text or "", re.DOTALL)
    if not m:
        raise ValueError("no JSON object in model output")
    return Invoice.model_validate(json.loads(m.group(0)))


@dataclass
class Extraction:
    invoice: Invoice | None
    raw: str
    error: str | None
    latency_ms: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Extractor:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None,
                 guided: bool = True, max_tokens: int = 1500, http_client=None):
        from openai import AsyncOpenAI

        self.model = model
        self.guided = guided
        self.max_tokens = max_tokens
        self.client = AsyncOpenAI(base_url=base_url or None, api_key=api_key or "not-needed", max_retries=3,
                                  timeout=120, http_client=http_client)

    @classmethod
    def from_env(cls, prefix: str = "VLM") -> "Extractor":
        return cls(os.environ[f"{prefix}_MODEL"], os.getenv(f"{prefix}_BASE_URL"),
                   os.getenv(f"{prefix}_API_KEY") or os.getenv("OPENAI_API_KEY"),
                   guided=os.getenv(f"{prefix}_GUIDED", "1") == "1")

    async def extract(self, image: Image.Image | str | Path) -> Extraction:
        kwargs = {"model": self.model, "messages": messages_for(image), "temperature": 0,
                  "max_tokens": self.max_tokens}
        if self.guided:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "invoice", "schema": json_schema()}}
        t0 = time.perf_counter()
        resp = await self.client.chat.completions.create(**kwargs)
        latency = (time.perf_counter() - t0) * 1000
        raw = resp.choices[0].message.content or ""
        usage = resp.usage
        try:
            inv, err = parse_invoice(raw), None
        except (ValueError, ValidationError) as exc:
            inv, err = None, str(exc)[:300]
        return Extraction(inv, raw, err, latency,
                          usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)
