"""LoRA / QLoRA fine-tuning of a Qwen2.5-VL style model on invoice extraction.

    python training/finetune_lora.py --data data/synth --model Qwen/Qwen2.5-VL-3B-Instruct --out runs/vidoc-lora
    python training/finetune_lora.py ... --qlora          # 4-bit base, fits a 16 GB GPU

Each example is (invoice image, fixed prompt) -> compact JSON label. Only the
JSON answer tokens are trained; prompt and image tokens are masked out.
LoRA is applied to the language model's attention and MLP layers; the vision
encoder stays frozen.

Needs a CUDA GPU and `pip install -e '.[train]'`. Not run in CI.
"""

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image

from vidoc.extract import PROMPT

LM_TARGETS = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"


def load_split(data_dir: Path, split: str, limit: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in (data_dir / f"{split}.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return rows[:limit] if limit else rows


def target_text(label: dict) -> str:
    return json.dumps(label, ensure_ascii=False, separators=(",", ":"))


def find_subsequence(seq: list[int], sub: list[int]) -> int:
    for i in range(len(seq) - len(sub), -1, -1):  # last occurrence: the assistant turn
        if seq[i:i + len(sub)] == sub:
            return i
    return -1


class Collator:
    def __init__(self, processor, data_dir: Path, max_side: int):
        self.processor = processor
        self.data_dir = data_dir
        self.max_side = max_side
        tok = processor.tokenizer
        # Qwen chat template: the answer starts after "<|im_start|>assistant\n".
        self.answer_start = tok.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.pad_id = tok.pad_token_id

    def __call__(self, batch: list[dict]) -> dict:
        texts, images = [], []
        for ex in batch:
            img = Image.open(self.data_dir / ex["image"]).convert("RGB")
            img.thumbnail((self.max_side, self.max_side))
            msgs = [
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]},
                {"role": "assistant", "content": [{"type": "text", "text": target_text(ex["label"])}]},
            ]
            texts.append(self.processor.apply_chat_template(msgs, tokenize=False))
            images.append([img])
        enc = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = enc["input_ids"].clone()
        for i, ids in enumerate(enc["input_ids"].tolist()):
            start = find_subsequence(ids, self.answer_start)
            if start < 0:
                raise ValueError("assistant marker not found; is this a Qwen-style chat template?")
            labels[i, : start + len(self.answer_start)] = -100
        labels[enc["input_ids"] == self.pad_id] = -100
        enc["labels"] = labels
        return enc


def main() -> None:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig, Trainer,
                              TrainingArguments)

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/synth")
    p.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--out", default="runs/vidoc-lora")
    p.add_argument("--qlora", action="store_true")
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--epochs", type=float, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--max-side", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0, help="use only the first N train examples")
    p.add_argument("--resume", action="store_true", help="continue from the last checkpoint in --out")
    p.add_argument("--save-steps", type=int, default=100)
    a = p.parse_args()

    data_dir = Path(a.data)
    train, val = load_split(data_dir, "train", a.limit), load_split(data_dir, "val", 200)
    processor = AutoProcessor.from_pretrained(a.model, min_pixels=256 * 28 * 28, max_pixels=1024 * 28 * 28)
    processor.tokenizer.padding_side = "right"

    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16) if a.qlora else None
    model = AutoModelForImageTextToText.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                        quantization_config=quant, attn_implementation="sdpa")
    if a.qlora:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05,
                                             target_modules=LM_TARGETS, task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    # Integer warmup steps work on transformers 4.x and 5.x (5.x dropped warmup_ratio).
    steps = math.ceil(len(train) / (a.batch * a.grad_accum) * a.epochs)
    args = TrainingArguments(
        output_dir=a.out, num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine",
        warmup_steps=max(1, round(0.03 * steps)), per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        gradient_accumulation_steps=a.grad_accum, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, bf16=True, logging_steps=10,
        eval_strategy="steps", eval_steps=a.save_steps, save_strategy="steps", save_steps=a.save_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", remove_unused_columns=False,
        dataloader_num_workers=2, report_to=["mlflow"] if _has("mlflow") else [],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train, eval_dataset=val,
                      data_collator=Collator(processor, data_dir, a.max_side))
    # Checkpoints every --save-steps; after a crash (driver error, power spike) rerun with --resume.
    resume = a.resume and any(Path(a.out).glob("checkpoint-*"))
    trainer.train(resume_from_checkpoint=True if resume else None)
    trainer.save_model(a.out)
    processor.save_pretrained(a.out)
    print(f"adapter saved to {a.out}")


def _has(module: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(module) is not None


if __name__ == "__main__":
    main()
