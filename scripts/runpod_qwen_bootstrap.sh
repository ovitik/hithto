#!/usr/bin/env bash
set -euo pipefail

cd /workspace

export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface/transformers}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PYTHONUNBUFFERED=1

if [ ! -d llm-recursive-tokens ]; then
  if [ -z "${RUNPOD_REPO_URL:-}" ]; then
    echo "RUNPOD_REPO_URL is not set and /workspace/llm-recursive-tokens does not exist."
    echo "Upload or git clone the repo into /workspace/llm-recursive-tokens, or set RUNPOD_REPO_URL."
    exit 2
  fi
  git clone "${RUNPOD_REPO_URL}" llm-recursive-tokens
fi

cd /workspace/llm-recursive-tokens
python3 -m venv /workspace/hacot_venv
source /workspace/hacot_venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-runpod-qwen.txt
python -m pip install -e .

python scripts/qwen_hacot_pilot.py \
  --dry-run \
  --out-dir /workspace/hacot_runs/qwen_hacot_pilot_dryrun

python scripts/qwen_hacot_pilot.py \
  --model-name "${QWEN_MODEL_NAME:-Qwen/Qwen3-1.7B}" \
  --out-dir "${QWEN_HACOT_OUT_DIR:-/workspace/hacot_runs/qwen_hacot_pilot}" \
  --train-n "${QWEN_TRAIN_N:-1800}" \
  --dev-n "${QWEN_DEV_N:-240}" \
  --steps "${QWEN_STEPS:-600}" \
  --batch-size "${QWEN_BATCH_SIZE:-2}" \
  --grad-accum "${QWEN_GRAD_ACCUM:-8}" \
  --eval-n "${QWEN_EVAL_N:-120}" \
  --max-length "${QWEN_MAX_LENGTH:-512}" \
  --max-new-tokens "${QWEN_MAX_NEW_TOKENS:-160}" \
  --variants "${QWEN_VARIANTS:-flat,hacot}" \
  --lora \
  --qlora-4bit
