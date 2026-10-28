#!/usr/bin/env bash
# Run by systemd at boot (deploy/vidoc-train.service): re-apply the GPU power
# limit, which a reboot resets, then resume LoRA training in a tmux session
# unless it already finished. Attach with: tmux attach -t vidoc
set -euo pipefail

REPO=${REPO:-/home/anda/projects/vidoc}
RUN_AS=${RUN_AS:-anda}
GPU=${GPU:-1}
POWER_LIMIT=${POWER_LIMIT:-250}
OUT=${OUT:-runs/vidoc-lora}

nvidia-smi -i "$GPU" -pm 1
nvidia-smi -i "$GPU" -pl "$POWER_LIMIT"

# finetune_lora.py writes the adapter to $OUT itself only after the last step.
if [ -f "$REPO/$OUT/adapter_model.safetensors" ]; then
    echo "training already finished: $REPO/$OUT"
    exit 0
fi

runuser -u "$RUN_AS" -- mkdir -p "$REPO/$OUT"
runuser -u "$RUN_AS" -- tmux new-session -d -s vidoc \
    "cd $REPO && source .venv/bin/activate && CUDA_VISIBLE_DEVICES=$GPU \
     python training/finetune_lora.py --qlora --save-steps 50 --out $OUT --resume 2>&1 | tee -a $OUT/train.log; bash"
echo "resumed training in tmux session 'vidoc'"
