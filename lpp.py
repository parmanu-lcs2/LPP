import argparse, json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import math

from datasets import load_dataset
from scipy.linalg import svd
from numpy.linalg import slogdet

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def build_prompt(example, text_field=None, join_output=False):
    """
    Build a simple Alpaca-style prompt from typical instruction-tuning fields.
    Set join_output=True if you want to append the output text to the prompt (usually keep False).
    """
    # If user specifies a text_field, just use that field directly when present.
    if text_field and text_field in example and isinstance(example[text_field], str) and example[text_field].strip():
        base = example[text_field].strip()
        return base

    inst = example.get("instruction", "") or example.get("prompt", "")
    ipt  = example.get("input", "")
    out  = example.get("output", "") or example.get("response", "")

    if inst and ipt:
        prompt = f"### Instruction:\n{inst}\n\n### Input:\n{ipt}\n\n### Response:\n"
    elif inst:
        prompt = f"### Instruction:\n{inst}\n\n### Response:\n"
    else:
        # Fall back to any text-like field
        for key in ["text", "question", "query", "source"]:
            if key in example and isinstance(example[key], str) and example[key].strip():
                prompt = example[key].strip()
                break
        else:
            prompt = "### Instruction:\nDescribe the topic.\n\n### Response:\n"

    if join_output and out:
        prompt = prompt + out.strip()

    return prompt


@torch.no_grad()
def last_token_entropy_for_prefix_batch(model, input_ids, attention_mask):
    """
    Compute next-token entropy at the last token position for each sequence in batch.
    input_ids, attention_mask: already truncated to the desired prefix length.
    Return: list of entropies (float) per example.
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits  # [B, T, V]
    B, T, V = logits.shape

    # Last non-padding index per sequence
    last_idx = attention_mask.sum(dim=1) - 1  # [B]
    ents = []
    for i in range(B):
        idx = int(last_idx[i].item())
        # entropy on distribution of next token from position idx
        p = F.softmax(logits[i, idx], dim=-1)
        ent = float(-(p * (p.clamp_min(1e-12)).log()).sum().item())
        ent = ent/math.log(V)

        ents.append(ent)
    return ents


@torch.no_grad()
def embeddings_for_prefix_batch(model, input_ids, attention_mask):
    """
    Return mean-pooled last-layer hidden state per example.
    Shape: [B, H]
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False
    )
    last_h = outputs.hidden_states[-1]  # [B, T, H]
    mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
    summed = (last_h * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1)
    pooled = summed / denom
    return pooled.detach().float().cpu().numpy()


def make_prefix_slices(input_ids, attention_mask, prefix_len):
    """
    Slice to first prefix_len tokens, but ensure >=1 token.
    """
    prefix_len = max(1, min(prefix_len, input_ids.shape[1]))
    return input_ids[:, :prefix_len], attention_mask[:, :prefix_len]


def compute_for_window(model, tokenizer, texts, batch_size, prefix_len, max_length):
    """
    For a given prefix length, compute:
      - mean next-token entropy across samples
      - silhouette score over pooled embeddings (KMeans k=min(8, n_samples))
    """
    all_ents = []
    all_embs = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        ids_p, mask_p = make_prefix_slices(input_ids, attention_mask, prefix_len)

        # entropy
        ents = last_token_entropy_for_prefix_batch(model, ids_p, mask_p)
        all_ents.extend(ents)

        # embeddings
        embs = embeddings_for_prefix_batch(model, ids_p, mask_p)
        all_embs.append(embs)

    all_embs = np.vstack(all_embs) if len(all_embs) else np.zeros((0, 2), dtype=np.float32)
    mean_entropy = float(np.mean(all_ents)) if all_ents else float("nan")

    return mean_entropy

# ----------------------------
# Metric helpers
# ----------------------------
def participation_ratio(eigvals):
    s = np.clip(eigvals, 0, None)
    out = float((s.sum() ** 2) / (np.square(s).sum() + 1e-12)) if s.sum() > 0 else 0.0
    out = out/eigvals.shape[-1]

    return out

def effective_rank(eigvals):
    s = np.clip(eigvals, 1e-12, None)
    p = s / s.sum()
    h = -(p * np.log(p)).sum()
    return float(np.exp(h))/eigvals.shape[-1]

# ----------------------------
# Core computation
# ----------------------------
@torch.no_grad()
def compute_metrics(model, tokenizer, texts, prefix_len, max_length, batch_size, token_cap=2048):
    ents, embs, hiddens = [], [], []

    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start+batch_size],
                        return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length)
        ids = enc["input_ids"].to(model.device)
        attn = enc["attention_mask"].to(model.device)

        # truncate to prefix_len
        ids, attn = ids[:, :prefix_len], attn[:, :prefix_len]

        out = model(input_ids=ids, attention_mask=attn,
                    output_hidden_states=True, use_cache=False)

        # Embeddings (mean pooled)
        last_h = out.hidden_states[-1]
        mask = attn.unsqueeze(-1)
        pooled = (last_h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        embs.append(pooled.detach().cpu().float().numpy())

        # Hidden states for compression/redundancy (just last layer pooled tokens)
        h = last_h.detach().cpu().float().numpy().reshape(-1, last_h.shape[-1])
        if h.shape[0] > token_cap:
            idx = np.random.choice(h.shape[0], size=token_cap, replace=False)
            h = h[idx]
        hiddens.append(h)

    # Compression + redundancy (using hidden states from last layer only)
    H = np.concatenate(hiddens, axis=0)
    H = H - H.mean(axis=0, keepdims=True)

    U, S, Vt = svd(H, full_matrices=False)
    eigvals = (S ** 2) / max(H.shape[0] - 1, 1)
    pr = participation_ratio(eigvals)
    er = effective_rank(eigvals)

    C = np.cov(H, rowvar=False)

    return pr, er

def load_model(model_id, use_4bit, device):
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Load tokenizer & model
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto", attn_implementation="eager", trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16),
        quantization_config=quant_config,
    )
    model.eval()

    return model, tokenizer

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument("--models", type=str, nargs="+", required=False, \
                    default=["Qwen/Qwen2.5-7B-Instruct", \
                             ],
                    help="List of HF model ids (e.g., meta-llama/Meta-Llama-3-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3 Qwen/Qwen2.5-7B-Instruct)")
	parser.add_argument("--dataset", type=str, default="wikitext", help="HF dataset id (e.g., wikitext)")
	parser.add_argument("--dataset_config", type=str, default="wikitext-103-v1", help="HF dataset config or None")
	parser.add_argument("--split", type=str, default="train")
	parser.add_argument("--text_field", type=str, default="text", help="Field name to read text from")
	parser.add_argument("--sample_size", type=int, default=5000)
	parser.add_argument("--min_length", type=int, default=128, help="Minimum tokenized length to keep")
	parser.add_argument("--max_length", type=int, default=1024, help="Tokenizer max_length truncation")
	parser.add_argument("--batch_size", type=int, default=1)
	parser.add_argument("--window_size", type=int, default=10, help="Prefix window step")
	parser.add_argument("--max_prefix_tokens", type=int, default=150, help="Max prefix to evaluate (inclusive)")
	parser.add_argument("--repr_on_every_window", action="store_true",
	                    help="Compute compression/redundancy on EVERY window (expensive). If not set, only on largest window.")
	parser.add_argument("--per_layer_token_cap", type=int, default=4096,
	                    help="Max token vectors per layer (across all batches) for covariance/SVD")
	parser.add_argument("--use_4bit", action="store_true", help="Quantized 4-bit load to save VRAM")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--out_csv", type=str, default="multi_model_metrics.csv")

	args, _ = parser.parse_known_args()

	all_texts = {}

	tokenizer = AutoTokenizer.from_pretrained(args.models[0], use_fast=True)
	if tokenizer.pad_token is None:
    	tokenizer.pad_token = tokenizer.eos_token

    # Load dataset & sample
	#ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:1%]")
	ds = load_dataset("tatsu-lab/alpaca", split="train[:5%]")

	min_length = 128
	def has_min_len(example):
	    return len(tokenizer(example["text"]).input_ids) >= min_length

	ds = ds.filter(has_min_len)

	if args.sample_size is not None and args.sample_size < len(ds):
	    ds = ds.select(range(args.sample_size))

	texts = []

	for ex in ds:
	    #texts.append(build_prompt(ex, text_field=args.text_field, join_output=False))
	    texts.append(ex['text'].strip())

	# Rolling windows
	all_texts['alpaca'] = texts

	sample_size = 100

	rows = []
	results = []

	for name, texts in all_texts.items():
	  if name == 'alpaca':
	    for length in [200]:
	      args.max_prefix_tokens = length
	      args.window_size = length//10

	      windows = list(range(args.window_size, args.max_prefix_tokens + 1, args.window_size))

	      for m in args.models:
	          print(f"\n=== {m} ===")
	          model, tok = load_model(m, args.use_4bit, "cuda" if torch.cuda.is_available() else "cpu")

	          for prefix_len in tqdm(windows):
	              mean_ent = compute_for_window(
	                  model=model,
	                  tokenizer=tok,
	                  texts=texts[:sample_size],
	                  batch_size=args.batch_size,
	                  prefix_len=prefix_len,
	                  max_length=args.max_length
	              )


	              pr, er = compute_metrics(model, tok, texts[:sample_size],
	                                                          prefix_len=prefix_len, max_length=args.max_length,
	                                                          batch_size=args.batch_size)

	              print (f"Entropy {mean_ent}")

	              row = {
	                  "model_id": m,
	                  "prefix_tokens": prefix_len,
	                  "dataset": name,
	                  "context_length": length,
	                  "mean_next_token_entropy": mean_ent,
	                  "participation_ratio": pr,
	                  "effective_rank": er,
	              }

	              results.append(row)

	          del model
	          torch.cuda.empty_cache()

	      df = pd.DataFrame(results)



	df.to_csv(args.out_csv, index=False)

