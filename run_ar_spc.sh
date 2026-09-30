#!/usr/bin/env bash
# Few-shot evaluation of all models on the SPC and AR tasks (settings used in the paper).
set -euo pipefail

MODELS=(
  "Qwen/Qwen2.5-0.5B-Instruct"
  "Qwen/Qwen2.5-1.5B-Instruct"
  "Qwen/Qwen2.5-3B-Instruct"
  "Qwen/Qwen2.5-7B-Instruct"
  "Qwen/Qwen2.5-14B-Instruct"
  "meta-llama/Llama-3.2-3B-Instruct"
  "meta-llama/Meta-Llama-3-8B-Instruct"
  "mistralai/Mistral-7B-Instruct-v0.2"
)
SHOTS=${SHOTS:-10}
SEED=${SEED:-1337}

python eval_spc_fewshot.py \
  --models "${MODELS[@]}" \
  --test data/spc_100.jsonl \
  --shots "$SHOTS" --seed "$SEED" \
  --per_model_out results/spc \
  --batch_size 4 \
  --max_new_tokens 8

python eval_ar_fewshot.py \
  --models "${MODELS[@]}" \
  --test data/ar_100.jsonl \
  --shots "$SHOTS" --seed "$SEED" \
  --per_model_out results/ar \
  --batch_size 4 \
  --max_new_tokens 16

# Scaling check (4-bit):
# python eval_spc_fewshot.py --models Qwen/Qwen2.5-32B-Instruct --use_4bit --test data/spc_100.jsonl --shots "$SHOTS" --seed "$SEED" --per_model_out results/spc --batch_size 4 --max_new_tokens 8
# python eval_ar_fewshot.py  --models Qwen/Qwen2.5-32B-Instruct --use_4bit --test data/ar_100.jsonl  --shots "$SHOTS" --seed "$SEED" --per_model_out results/ar  --batch_size 4 --max_new_tokens 16
