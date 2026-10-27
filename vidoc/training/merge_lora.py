"""Merge the LoRA adapter into the base weights, for serving without --enable-lora.

    python training/merge_lora.py --base Qwen/Qwen2.5-VL-3B-Instruct --adapter runs/vidoc-lora --out runs/vidoc-merged
"""

import argparse

import torch


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--adapter", default="runs/vidoc-lora")
    p.add_argument("--out", default="runs/vidoc-merged")
    p.add_argument("--push", default=None, help="Hugging Face repo id to push to")
    a = p.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(a.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    processor = AutoProcessor.from_pretrained(a.base)
    model.save_pretrained(a.out, safe_serialization=True)
    processor.save_pretrained(a.out)
    if a.push:
        model.push_to_hub(a.push)
        processor.push_to_hub(a.push)


if __name__ == "__main__":
    main()
