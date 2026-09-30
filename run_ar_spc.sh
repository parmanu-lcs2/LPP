python eval_spc_fewshot.py \
  --models "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct" \
  --test spc_100.jsonl \
  --shots 10 \
  --per_model_out spc_results \
  --batch_size 4 \
  --max_new_tokens 8

python eval_ar_fewshot.py \
  --models "meta-llama/Meta-Llama-3-8B-Instruct" "meta-llama/Llama-3.2-3B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2" "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct" "Qwen/Qwen2.5-1.5B-Instruct" "Qwen/Qwen2.5-0.5B-Instruct" "Qwen/Qwen2.5-3B-Instruct"  \
  --test ar_100.jsonl \
  --shots 10 \
  --per_model_out ar_results \
  --batch_size 4 \
  --max_new_tokens 16