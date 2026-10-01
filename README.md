# LPP: Latent Performance Profiling for LLMs

Code and data for **Latent Performance Profiling of Large Language Models**.

Latent Performance Profiling (LPP) characterises a language model through three intrinsic properties of its forward pass on unlabeled text:

- **Entropy floor**: next-token entropy, normalised by log |V|.
- **Participation ratio (PR)**: how many dimensions of the hidden-state space are used.
- **Effective rank (ER)**: the exponential of the spectral entropy of the hidden-state covariance.

The repository also contains two synthetic diagnostic tasks introduced in the paper, **Ambiguous Reasoning (AR)** and **Symbolic Pattern Completion (SPC)**, together with the source data behind every figure.

## Repository structure

```
lpp.py                 # intrinsic LPP metrics per model / corpus / context length / prefix length / layer
aggregate.py           # per-prefix metrics -> one LPP profile per model (min entropy, max PR, max ER)
eval_ar_fewshot.py     # few-shot evaluation on AR
eval_spc_fewshot.py    # few-shot evaluation on SPC
run_ar_spc.sh          # AR + SPC evaluation for all models (paper settings)
LPP_analysis.ipynb     # correlation analysis and figures
data/                  # synthetic datasets and source data (see below)
```

## Installation

Python 3.10 or later is required. A CUDA GPU is recommended; all experiments in the paper were run on a single NVIDIA A100.

```bash
git clone https://github.com/parmanu-lcs2/lpp.git
cd LPP
pip install -r requirements.txt
```

The Llama models are gated on Hugging Face. Accept their licences and run `huggingface-cli login` before using them.

## Models

| Short name | Hugging Face id |
|---|---|
| Qwen-0.5B / 1.5B / 3B / 7B / 14B | `Qwen/Qwen2.5-{0.5B,1.5B,3B,7B,14B}-Instruct` |
| Llama-3B | `meta-llama/Llama-3.2-3B-Instruct` |
| Llama-8B | `meta-llama/Meta-Llama-3-8B-Instruct` |
| Mistral-7B | `mistralai/Mistral-7B-Instruct-v0.2` |
| Qwen-32B (scaling check, 4-bit) | `Qwen/Qwen2.5-32B-Instruct` |

## Reproducing the results

### 1. Intrinsic LPP metrics

```bash
python lpp.py \
  --models Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
           Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct meta-llama/Llama-3.2-3B-Instruct \
           meta-llama/Meta-Llama-3-8B-Instruct mistralai/Mistral-7B-Instruct-v0.2 \
  --dataset alpaca --sample_size 100 --context_lengths 200 \
  --out_csv results/lpp_alpaca.csv

python aggregate.py --in_csv results/lpp_alpaca.csv --context_length 200 \
  --out_csv results/lpp_profile.csv
```

`lpp.py` builds a prompt set once and uses it for every model. The set is the first `--sample_size` examples of the corpus that have at least `--min_length` (default 128) tokens under the first model's tokenizer.

For each context length *C*, the script evaluates prefixes of length *C*/10, 2*C*/10, …, *C*. At each prefix length it computes:

- the next-token entropy at the last prefix position, averaged over prompts;
- PR and ER of the covariance of all token representations pooled across prompts, with padding excluded.

PR and ER are divided by min(#tokens, hidden size).

Useful options:

| Option | Purpose |
|---|---|
| `--dataset {alpaca,dolly,wikitext}` | Calibration corpus |
| `--context_lengths 50 100 200 500` | Context-length sensitivity |
| `--sample_size 10 100 500 1000` | Sample-size sensitivity (one value per run) |
| `--layers all` | Layer-wise analysis instead of the last layer only |
| `--use_4bit` | NF4 loading for large models |
| `--entropy_agg`, `--rep_agg` in `aggregate.py` | Alternative aggregations (`min`, `max`, `mean`, `median`) |

### 2. Synthetic tasks (AR and SPC)

```bash
bash run_ar_spc.sh          # 10-shot, seed 1337; set SHOTS=0|1|5 for the in-context ablation
```

The script writes per-model predictions and metrics to `results/ar/` and `results/spc/`.

**AR metrics.** `accuracy` is answer-choice accuracy, which is the AR score reported in the paper. The script also reports `ambiguity_acc`, `combined` (the mean of the two) and `overconfidence`.

**SPC metrics.** The script reports exact match (`EM`) and character-level F1 (stored in the `token_f1` column).

### 3. Analysis

`LPP_analysis.ipynb` merges the LPP profiles with the extrinsic benchmark scores and the AR/SPC scores, then computes the correlations shown in Figure 2. The extrinsic scores (IFEval, BBH, MMLU-Pro) come from the Open LLM Leaderboard v2, via the `open-llm-leaderboard/contents` dataset on the Hugging Face Hub.

## Data

### Synthetic datasets

| File | Description |
|---|---|
| `data/ar_100.jsonl` | Ambiguous Reasoning: 100 items (`input`, `target`, `meta` with prefix, options A/B, hint, gold answer) |
| `data/spc_100.jsonl` | Symbolic Pattern Completion: 100 sequences from five rules: periodic, alternating, mirror, brackets, inc_mod10 (`input`, `target`, `meta`) |

### Source data for figures and tables

| File(s) | Content | Used in |
|---|---|---|
| `benchmark.csv` | Open LLM Leaderboard v2 scores (IFEval, BBH, MMLU-Pro) | Fig. 2A |
| `LPP.csv` | Per-prefix entropy, PR, ER on Alpaca (100 prompts, prefixes 10–150) | Fig. 2B |
| `final_data.csv`, `lpp_lpp_scores.csv` | Model-level LPP profiles merged with benchmark, AR and SPC scores | Figs. 2–3 |
| `ar.csv`, `spc.csv`, `lpp_new_scores.csv` | AR and SPC scores (10-shot) | Fig. 2C |
| `lpp1_res.csv`, `corr_matrix_used.csv`, `pval_matrix_used.csv` | LPP vs. benchmark correlations and p-values (Pearson) | Fig. 2D |
| `lpp3_res.csv` | LPP vs. AR/SPC correlations and p-values (Pearson) | Fig. 2E |
| `LPP_analysis1.csv`, `LPP_analysis1_derived.csv`, `LPP_analysis3.2_derived.csv` | Metrics across context and prefix lengths | Fig. 4 |
| `LPP_analysis2.csv`, `LPP_analysis2.1_derived.csv`, `LPP_analysis2.2_derived.csv` | Metrics across calibration datasets and sample sizes | Supp. Figs. 1–3 |
| `LPP_analysis3.csv`, `LPP_analysis3.1_derived.csv` | Layer-wise PR and ER | Supp. Fig. 4 |
| `lpp_lpp_scores_{min,mean,median}.csv` | Profiles under alternative aggregations | Supp. Fig. 5 |
| `all_ar.csv`, `all_spc.csv` | AR/SPC scores for k ∈ {0, 1, 5, 10} shots | Supp. Fig. 6 |
| `ext_truthfulqa.csv`, `ext_symbolic.csv`, `nb1_merged.csv`, `nb1_correlations.csv`, `nb1_partial_correlations.csv` | External validation (TruthfulQA-MC, BBH Dyck languages), including size-controlled partial correlations | Supp. §2.5 |
| `nb2_factual_judged.csv`, `nb2_merged.csv`, `nb2_correlations.csv`, `nb2_ar_regrade.csv` | LLM-judge generations and verdicts (40 TruthfulQA prompts per model; reliability 0–2) and AR re-grading | Supp. §2.5 |
| `normalized_metrics.csv`, `normalized_correlations.csv` | Outlier-dimension analysis (raw, z-scored, robust PR/ER) | Supp. §2.6 |

### Third-party data

- **Stanford Alpaca** (`tatsu-lab/alpaca`)
- **Databricks Dolly-15k** (`databricks/databricks-dolly-15k`)
- **WikiText-103** (`Salesforce/wikitext`)
- **TruthfulQA** (`truthfulqa/truthful_qa`)
- **BIG-Bench Hard**
- **Open LLM Leaderboard v2**

These datasets remain under their original licences.

## Citation

```bibtex
@article{lpp2026,
  title   = {Latent Performance Profiling of Large Language Models},
  author  = {TODO},
  journal = {Communications AI \& Computing},
  year    = {2026}
}
```

## License

Code and generated data are released under the MIT License (see `LICENSE`).
