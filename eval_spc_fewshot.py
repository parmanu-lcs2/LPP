import argparse, os, json, random, re, math, time
import pandas as pd
import torch
from datasets import load_dataset as _unused  # just to hint dependency
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_hf(model_id, use_4bit=False, trust_remote_code=True):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant = None
    if use_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                     quantization_config=quant, trust_remote_code=trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                     torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
                                                     trust_remote_code=trust_remote_code)
    model.eval()
    return tok, model

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append(obj)
    return rows

def char_f1(pred, gold):
    pred = pred.strip(); gold = gold.strip()
    if len(pred)==0 and len(gold)==0: return 1.0
    if len(pred)==0 or len(gold)==0: return 0.0
    from collections import Counter
    cp, cg = Counter(pred), Counter(gold)
    tp = sum(min(cp[c], cg[c]) for c in set(cp)|set(cg))
    prec = tp / max(1, sum(cp.values()))
    rec  = tp / max(1, sum(cg.values()))
    if prec+rec==0: return 0.0
    return 2*prec*rec/(prec+rec)

def format_fewshot(shots):
    parts = []
    for s in shots:
        parts.append(f"Sequence: {s['input'].split('Sequence:')[-1].strip()}\nAnswer: {s['target']}")
    return "\n\n".join(parts)

def build_prompt(few_shot_block, seq, k):
    instr = (f"You are given a symbolic sequence. Continue it by writing exactly the next {k} symbols, "
             f"without spaces or explanations.\n")
    return f"{instr}{few_shot_block}\n\nSequence: {seq}\nAnswer:" if few_shot_block else f"{instr}Sequence: {seq}\nAnswer:"

def generate_batch(tokenizer, model, prompts, max_new_tokens=8, temperature=0.0, top_p=1.0):
    enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(model.device) for k,v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=(temperature>0),
                             temperature=temperature, top_p=top_p, eos_token_id=tokenizer.eos_token_id)
    dec = tokenizer.batch_decode(out, skip_special_tokens=True)
    # Extract completion after the last "Answer:"
    preds = []
    for prompt, full in zip(prompts, dec):
        idx = full.rfind("Answer:")
        comp = full[idx+len("Answer:"):].strip() if idx!=-1 else full.strip()
        # keep only the first contiguous A-Z/0-9/symbols
        comp = re.findall(r"[A-Za-z0-9\(\)\[\]\{\}]+", comp)
        preds.append(comp[0] if comp else "")
    return preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--test", required=True, help="JSONL with fields: input, target")
    ap.add_argument("--shots", type=int, default=6)
    ap.add_argument("--per_model_out", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.per_model_out, exist_ok=True)

    test_rows = read_jsonl(args.test)

    # Few-shot selection: sample from test set (pragmatic ICL; alternatively pass a train file)
    fewshot_pool = random.sample(test_rows, min(args.shots, len(test_rows)))
    fewshot_block = format_fewshot(fewshot_pool)

    # Parse k from the input line (appears in instruction)
    def extract_k(text):
        m = re.search(r"next (\d+) symbols", text)
        return int(m.group(1)) if m else 3

    for mid in args.models:
        tok, model = load_hf(mid, use_4bit=args.use_4bit)
        prompts = [build_prompt(fewshot_block, r["input"].split("Sequence:")[-1].split("\n")[0].strip(),
                                extract_k(r["input"])) for r in test_rows]

        preds, all_gold = [], []
        for i in range(0, len(prompts), args.batch_size):
            batch_prompts = prompts[i:i+args.batch_size]
            batch_preds = generate_batch(tok, model, batch_prompts, args.max_new_tokens, args.temperature, args.top_p)
            preds.extend(batch_preds)
        all_gold = [r["target"].strip() for r in test_rows]

        # Metrics
        em = sum(1 for p,g in zip(preds, all_gold) if p.strip()==g.strip())/len(all_gold)
        f1s = [char_f1(p,g) for p,g in zip(preds, all_gold)]
        f1 = sum(f1s)/len(f1s)

        # Save predictions
        df = pd.DataFrame({"model_id": mid, "input": [r["input"] for r in test_rows],
                           "pred": preds, "target": all_gold})
        out_pred = os.path.join(args.per_model_out, f"spc_preds__{mid.replace('/','__')}.csv")
        df.to_csv(out_pred, index=False)

        # Save metrics
        met_path = os.path.join(args.per_model_out, f"spc_metrics__{mid.replace('/','__')}.csv")
        pd.DataFrame([{"model_id": mid, "EM": em, "token_f1": f1}]).to_csv(met_path, index=False)
        print(f"[{mid}] SPC: EM={em:.3f}  F1={f1:.3f}  -> {met_path}")

if __name__ == "__main__":
    main()
