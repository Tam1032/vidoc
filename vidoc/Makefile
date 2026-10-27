PY ?= python

.PHONY: install test data claims train merge serve bench eval-extract eval-rag eval-agent

install:
	pip install -e '.[dev]'

test:
	$(PY) -m pytest -q

data:                 ## synthetic invoices with labels (train/val/test)
	$(PY) -m vidoc.synth --n 3000 --out data/synth

claims:               ## expense-claim scenarios for the agent eval
	$(PY) scripts/make_claims.py

train:                ## needs a CUDA GPU: pip install -e '.[train]'
	$(PY) training/finetune_lora.py --data data/synth --out runs/vidoc-lora

merge:
	$(PY) training/merge_lora.py --adapter runs/vidoc-lora --out runs/vidoc-merged

serve:
	docker compose -f serving/docker-compose.vllm.yml up

bench:
	$(PY) serving/benchmark.py --data data/synth --n 200 --concurrency 1 4 16

eval-extract:
	$(PY) -m vidoc.evals.extraction --data data/synth --split test

eval-rag:
	$(PY) -m vidoc.evals.retrieval

eval-agent:
	$(PY) -m vidoc.evals.agent --model $(if $(LLM_MODEL),env,rules)
