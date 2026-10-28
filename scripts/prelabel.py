"""Draft labels for real invoice photos, to be corrected by hand.

Runs the extractor (any VLM endpoint) on every image in a folder and writes one
row per image in the same format as data/synth/test.jsonl. Open the output,
fix every field against the photo, then evaluate with --data on that folder.

    VLM_MODEL=gpt-4.1-mini VLM_API_KEY=sk-... python scripts/prelabel.py data/real
"""

import argparse
import asyncio
import json
from pathlib import Path

from vidoc.extract import Extractor

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}


async def main_async(folder: Path, out_name: str) -> None:
    images = sorted(p for p in (folder / "images").iterdir() if p.suffix.lower() in IMAGE_TYPES)
    ex = Extractor.from_env("VLM")
    out = folder / out_name
    with out.open("w", encoding="utf-8") as f:
        for p in images:
            r = await ex.extract(p)
            label = r.invoice.model_dump() if r.invoice else {"error": r.error, "raw": r.raw}
            f.write(json.dumps({"image": f"images/{p.name}", "label": label}, ensure_ascii=False) + "\n")
            print(f"{p.name}: {'ok' if r.invoice else 'FAILED, fill in by hand'}")
    print(f"wrote {len(images)} drafts to {out}; correct every field before renaming it to test.jsonl")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="folder with an images/ subfolder, e.g. data/real")
    p.add_argument("--out", default="draft.jsonl")
    a = p.parse_args()
    asyncio.run(main_async(Path(a.folder), a.out))


if __name__ == "__main__":
    main()
