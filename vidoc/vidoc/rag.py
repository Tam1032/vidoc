"""Retrieval over policy and regulation documents: BM25 + optional dense embeddings, fused with RRF.

Markdown files are split at '###' headings so a chunk is one numbered rule
(e.g. "2.3 Đi lại bằng taxi"), and answers can cite it. Vietnamese words are
often two syllables, so BM25 indexes single syllables plus syllable bigrams.
Dense retrieval uses any OpenAI-compatible /embeddings endpoint (e.g. bge-m3
served by vLLM or Ollama) when EMBED_MODEL is set.
"""

import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent.parent / "data" / "policy"


@dataclass
class Chunk:
    id: str        # e.g. "expense_policy#2.3"
    title: str
    text: str
    source: str


def split_markdown(path: Path) -> list[Chunk]:
    chunks, title, buf = [], None, []
    for line in path.read_text().splitlines():
        if line.startswith("### "):
            if title:
                chunks.append((title, buf))
            title, buf = line[4:].strip(), []
        elif title and not line.startswith("#"):
            buf.append(line)
    if title:
        chunks.append((title, buf))
    out = []
    for title, lines in chunks:
        num = title.split()[0]
        out.append(Chunk(f"{path.stem}#{num}", title, " ".join(l.strip() for l in lines if l.strip()), path.name))
    return out


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text.lower())
    words = re.findall(r"[\wà-ỹđ]+", text)
    return words + [f"{a}_{b}" for a, b in zip(words, words[1:])]


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.docs = [Counter(d) for d in docs]
        self.lens = [len(d) for d in docs]
        self.avg = sum(self.lens) / max(1, len(docs))
        df = Counter(t for d in self.docs for t in d)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for d, ln in zip(self.docs, self.lens):
            s = 0.0
            for t in query:
                if t in d:
                    tf = d[t]
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * ln / self.avg))
            out.append(s)
        return out


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    score = Counter()
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            score[idx] += 1 / (k + rank + 1)
    return [i for i, _ in score.most_common()]


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder=None):
        self.chunks = chunks
        self.bm25 = BM25([tokens(c.title + " " + c.text) for c in chunks])
        self.embedder = embedder
        self.vectors = embedder([c.title + "\n" + c.text for c in chunks]) if embedder else None

    @classmethod
    def from_dir(cls, path: Path = POLICY_DIR, embedder=None) -> "Retriever":
        chunks = [c for p in sorted(path.glob("*.md")) for c in split_markdown(p)]
        return cls(chunks, embedder)

    def search(self, query: str, k: int = 3, mode: str = "hybrid") -> list[Chunk]:
        bm = self.bm25.scores(tokens(query))
        bm_rank = sorted(range(len(bm)), key=lambda i: -bm[i])
        if mode == "bm25" or self.vectors is None:
            order = bm_rank
        else:
            q = self.embedder([query])[0]
            sims = [sum(a * b for a, b in zip(q, v)) for v in self.vectors]
            dense_rank = sorted(range(len(sims)), key=lambda i: -sims[i])
            order = dense_rank if mode == "dense" else rrf([bm_rank, dense_rank])
        return [self.chunks[i] for i in order[:k]]


def openai_embedder(model: str | None = None, base_url: str | None = None, api_key: str | None = None):
    """Returns a function texts -> unit vectors, or None when EMBED_MODEL is not configured."""
    model = model or os.getenv("EMBED_MODEL")
    if not model:
        return None
    from openai import OpenAI

    client = OpenAI(base_url=base_url or os.getenv("EMBED_BASE_URL"),
                    api_key=api_key or os.getenv("EMBED_API_KEY") or os.getenv("OPENAI_API_KEY") or "x")

    def embed(texts: list[str]) -> list[list[float]]:
        vecs = [d.embedding for d in client.embeddings.create(model=model, input=texts).data]
        return [[x / (math.sqrt(sum(y * y for y in v)) or 1) for x in v] for v in vecs]

    return embed
