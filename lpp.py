"""
Latent Performance Profiling (LPP): intrinsic metrics for causal LLMs.

For every model, calibration corpus, context length and prefix length, this script computes
  * mean_next_token_entropy : next-token entropy at the last prefix position, normalised by log|V|
                              and averaged over prompts (computed from raw logits, no temperature);
  * participation_ratio     : (sum lambda)^2 / sum lambda^2 of the token-level hidden-state covariance;
  * effective_rank          : exp(Shannon entropy of the normalised eigenvalue spectrum).
PR and ER are divided by the number of non-zero-capacity eigenvalues, i.e. min(#tokens, hidden size),
exactly as in the code used for the paper.

Hidden states from all non-padding tokens of all prompts are pooled into a single covariance
(capped at --token_cap randomly selected tokens). By default the last layer is used; pass
--layers all to reproduce the layer-wise analysis.

Model-level LPP profiles (min entropy, max PR, max ER over prefix lengths) are produced by aggregate.py.

Example (paper setting, Alpaca, context length 200):
    python lpp.py --models Qwen/Qwen2.5-7B-Instruct --dataset alpaca \
                  --context_lengths 200 --sample_size 100 --out_csv results/lpp_alpaca.csv
"""
import argparse
import math
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy.linalg import svd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Calibration corpora used in the paper.
DATASETS = {
    "alpaca": dict(path="tatsu-lab/alpaca", name=None, split="train[:5%]"),
    "dolly": dict(path="databricks/databricks-dolly-15k", name=None, split="train"),
    "wikitext": dict(path="Salesforce/wikitext", name="wikitext-103-raw-v1", split="train[:1%]"),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def example_to_text(name, ex):
    """Plain-text view of one example for each supported corpus."""
    if name == "dolly":
        parts = [ex.get("instruction", ""), ex.get("context", ""), ex.get("response", "")]
        return "\n".join(p.strip() for p in parts if p and p.strip())
    return ex["text"].strip()


def load_texts(name, tokenizer, sample_size, min_length):
    """First `sample_size` examples of the corpus with at least `min_length` tokens."""
    cfg = DATASETS[name]
    ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"]) if cfg["name"] \
        else load_dataset(cfg["path"], split=cfg["split"])
    texts = []
    for ex in ds:
        t = example_to_text(name, ex)
        if t and len(tokenizer(t).input_ids) >= min_length:
            texts.append(t)
            if len(texts) >= sample_size:
                break
    return texts


def load_model(model_id, use_4bit):
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        attn_implementation="eager",
        trust_remote_code=True,
        torch_dtype=dtype,
        quantization_config=quant_config,
    )
    model.eval()
    return model, tokenizer


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def participation_ratio(eigvals):
    s = np.clip(eigvals, 0, None)
    out = float((s.sum() ** 2) / (np.square(s).sum() + 1e-12)) if s.sum() > 0 else 0.0
    return out / eigvals.shape[-1]


def effective_rank(eigvals):
    s = np.clip(eigvals, 1e-12, None)
    p = s / s.sum()
    h = -(p * np.log(p)).sum()
    return float(np.exp(h)) / eigvals.shape[-1]


def spectrum_metrics(H):
    """PR and ER of the covariance of token representations H [N, d]."""
    H = H - H.mean(axis=0, keepdims=True)
    S = svd(H, full_matrices=False, compute_uv=False)
    eigvals = (S ** 2) / max(H.shape[0] - 1, 1)
    return participation_ratio(eigvals), effective_rank(eigvals)


@torch.no_grad()
def profile_prefix(model, tokenizer, texts, prefix_len, max_length, batch_size, layers, token_cap):
    """Entropy and per-layer PR/ER for a single prefix length (one forward pass per batch)."""
    entropies = []
    hiddens = None  # layer index -> list of [n_tokens, d] arrays

    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start + batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=max_length)
        ids = enc["input_ids"][:, :prefix_len].to(model.device)
        attn = enc["attention_mask"][:, :prefix_len].to(model.device)

        out = model(input_ids=ids, attention_mask=attn, output_hidden_states=True, use_cache=False)

        # Next-token entropy at the last non-padding position, normalised by log|V|.
        logits = out.logits.float()
        V = logits.shape[-1]
        last_idx = attn.sum(dim=1) - 1
        for i in range(logits.shape[0]):
            p = F.softmax(logits[i, int(last_idx[i])], dim=-1)
            entropies.append(float(-(p * p.clamp_min(1e-12).log()).sum()) / math.log(V))

        # Token representations (padding positions excluded).
        n_layers = len(out.hidden_states) - 1
        layer_ids = [n_layers] if layers == "last" else list(range(1, n_layers + 1))
        if hiddens is None:
            hiddens = {l: [] for l in layer_ids}
        keep = attn.bool().reshape(-1).cpu().numpy()
        for l in layer_ids:
            h = out.hidden_states[l].float().cpu().numpy().reshape(-1, out.hidden_states[l].shape[-1])
            hiddens[l].append(h[keep])

    per_layer = {}
    for l, chunks in hiddens.items():
        H = np.concatenate(chunks, axis=0)
        if H.shape[0] > token_cap:
            H = H[np.random.choice(H.shape[0], size=token_cap, replace=False)]
        per_layer[l] = spectrum_metrics(H)
    return float(np.mean(entropies)), per_layer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-7B-Instruct"],
                    help="Hugging Face model ids")
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="alpaca")
    ap.add_argument("--sample_size", type=int, default=100, help="Number of prompts")
    ap.add_argument("--min_length", type=int, default=128, help="Keep prompts with at least this many tokens")
    ap.add_argument("--max_length", type=int, default=1024, help="Tokenizer truncation length")
    ap.add_argument("--context_lengths", type=int, nargs="+", default=[200],
                    help="Maximum prefix length; prefixes are evaluated in steps of context_length/10")
    ap.add_argument("--layers", choices=["last", "all"], default="last")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--token_cap", type=int, default=100_000,
                    help="Maximum number of token vectors used for the covariance")
    ap.add_argument("--use_4bit", action="store_true", help="Load the model in 4-bit (NF4)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_csv", default="results/lpp_metrics.csv")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    rows = []

    # The prompt set is selected once, with the tokenizer of the first model, so that every
    # model is profiled on exactly the same texts.
    filter_tok = AutoTokenizer.from_pretrained(args.models[0], use_fast=True)
    texts = load_texts(args.dataset, filter_tok, args.sample_size, args.min_length)

    for model_id in args.models:
        print(f"\n=== {model_id} ===")
        model, tok = load_model(model_id, args.use_4bit)
        for ctx in args.context_lengths:
            step = max(1, ctx // 10)
            for prefix_len in tqdm(range(step, ctx + 1, step), desc=f"context {ctx}"):
                ent, per_layer = profile_prefix(model, tok, texts, prefix_len, args.max_length,
                                                args.batch_size, args.layers, args.token_cap)
                for layer, (pr, er) in per_layer.items():
                    rows.append({
                        "model_id": model_id,
                        "dataset": args.dataset,
                        "sample_size": len(texts),
                        "context_length": ctx,
                        "prefix_tokens": prefix_len,
                        "layer": layer,
                        "mean_next_token_entropy": ent,
                        "participation_ratio": pr,
                        "effective_rank": er,
                    })
                pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Saved {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
