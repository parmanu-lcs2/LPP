# eval_ar_fewshot_strict_retry.py
import argparse, os, json, random, re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_hf(model_id, use_4bit=False, trust_remote_code=True):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=trust_remote_code)
    tok.padding_side = "left"
    tok.truncation_side = "left"  # keep the tail (with 'Your response:')
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    if use_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True,
                                   bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto",
                                                     quantization_config=quant, trust_remote_code=trust_remote_code)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
            trust_remote_code=trust_remote_code
        )
    model.eval()
    return tok, model

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def parse_meta(m):
    if isinstance(m, dict): return m
    if isinstance(m, (str, bytes, bytearray)):
        try: return json.loads(m)
        except Exception: return {"raw": str(m)}
    return {"raw": str(m)}

ANS_KV = re.compile(r"answer\s*[:=]\s*([AB])\b", re.I)
AMB_RE = re.compile(r"ambiguous_status\s*[:=]\s*(AMBIGUOUS|NOT_AMBIGUOUS)", re.I)

def extract_fields(text, optA_text=None, optB_text=None):
    amb_m = AMB_RE.search(text)
    amb = amb_m.group(1).upper() if amb_m else "MISSING"

    m = ANS_KV.search(text)
    if m:
        return amb, m.group(1).upper()

    tail = text
    idx = text.lower().rfind("answer")
    if idx != -1:
        tail = text[idx:]

    patts = [
        r"answer\s*[:]\s*([AB])\b",
        r"\(([AB])\)",
        r"\boption\s*([AB])\b",
        r"\b([AB])\)",
        r"\b([AB])\.",
        r"\b([AB])\b",
    ]
    for p in patts:
        m2 = re.search(p, tail, re.I)
        if m2:
            return amb, m2.group(1).upper()

    if isinstance(optA_text, str) and optA_text.lower() in text.lower(): return amb, "A"
    if isinstance(optB_text, str) and optB_text.lower() in text.lower(): return amb, "B"

    tail2 = text.split("Your response:")[-1]
    m3 = re.search(r"\b([AB])\b", tail2, re.I)
    return amb, (m3.group(1).upper() if m3 else "M")

SYS_MSG = (
    "Follow the schema exactly. ONE LINE only:\n"
    "ambiguous_status=<AMBIGUOUS|NOT_AMBIGUOUS>; answer=<A|B>"
)

def fewshot_block_schema(rows, k):
    # Use schema-only few-shot to save tokens (avoids truncating the tail)
    exs = random.sample(rows, min(k, len(rows)))
    parts = []
    for r in exs:
        parts.append(f"Example -> {r['target']}")
    return "\n".join(parts)

def make_messages(prefix, optA, optB, hint, fewshot_text=None):
    user_core = (
        "Consider the ambiguous prefix and two possible senses. "
        "First, judge the prefix alone as AMBIGUOUS or NOT_AMBIGUOUS. "
        "Then, after reading the hint, choose the correct option (A or B).\n\n"
        f"Prefix: {prefix}\n"
        f"Options: (A) {optA} | (B) {optB}\n"
        f"Hint: {hint}\n"
        "Your response:"
    )
    user_full = (fewshot_text + "\n" + user_core) if fewshot_text else user_core
    return [
        {"role": "system", "content": SYS_MSG},
        {"role": "user",   "content": user_full},
    ]

def render_prompts(tokenizer, rows, optA_list, optB_list, prefix_list, hint_list, fewshot_text):
    prompts = []
    use_template = getattr(tokenizer, "chat_template", None) is not None
    for i in range(len(rows)):
        msgs = make_messages(prefix_list[i], optA_list[i], optB_list[i], hint_list[i], fewshot_text)
        if use_template:
            prompt = tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        else:
            prompt = (
                f"[SYSTEM]\n{SYS_MSG}\n\n"
                f"{fewshot_text+'\n' if fewshot_text else ''}"
                f"{msgs[1]['content']}"
            )
        prompts.append(prompt)
    return prompts

def collect_eos_ids(tokenizer):
    ids = set()
    if tokenizer.eos_token_id is not None: ids.add(int(tokenizer.eos_token_id))
    for tok in ["</s>", "<|eot_id|>", "<|end_of_text|>"]:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
            if isinstance(tid, int) and tid >= 0: ids.add(tid)
        except Exception:
            pass
    return sorted(list(ids)) or None

def generate_batch(tokenizer, model, prompts, max_new_tokens=12, temperature=0.0, top_p=1.0):
    enc = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(model.device) for k,v in enc.items()}
    eos_ids = collect_eos_ids(tokenizer)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=(temperature > 0),
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_ids if eos_ids is not None else tokenizer.eos_token_id,
            use_cache=False,
            no_repeat_ngram_size=0
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)

def minimal_retry_prompt(tokenizer, prefix, optA, optB, hint):
    msgs = [
        {"role": "system", "content": "Answer with only 'A' or 'B'. No other text."},
        {"role": "user", "content": f"Prefix: {prefix}\nOptions: (A) {optA} | (B) {optB}\nHint: {hint}\nA or B?"}
    ]
    if getattr(tokenizer, "chat_template", None) is not None:
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return f"[SYSTEM]\nAnswer with only 'A' or 'B'.\n\nPrefix: {prefix}\nOptions: (A) {optA} | (B) {optB}\nHint: {hint}\nA or B?"

def generate_single(tokenizer, model, prompt, max_new_tokens=4):
    enc = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(model.device) for k,v in enc.items()}
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id, use_cache=False)
    return tokenizer.decode(out[0], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--shots", type=int, default=4)
    ap.add_argument("--per_model_out", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--use_4bit", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.per_model_out, exist_ok=True)

    rows = read_jsonl(args.test)
    metas = [parse_meta(r.get("meta", {})) for r in rows]
    amb_gold = ["AMBIGUOUS"] * len(rows)
    ans_gold = [m.get("correct") for m in metas]
    optA_list = [m.get("options", {}).get("A") for m in metas]
    optB_list = [m.get("options", {}).get("B") for m in metas]
    prefix_list = [m.get("prefix") for m in metas]
    hint_list = [m.get("hint") for m in metas]

    fewshot_text = fewshot_block_schema(rows, args.shots)

    for mid in args.models:
        tok, model = load_hf(mid, use_4bit=args.use_4bit)
        prompts = render_prompts(tok, rows, optA_list, optB_list, prefix_list, hint_list, fewshot_text)

        preds_raw = []
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i:i+args.batch_size]
            preds_raw.extend(generate_batch(tok, model, batch, args.max_new_tokens, args.temperature, args.top_p))

        amb_pred, ans_pred = [], []
        for i, full in enumerate(preds_raw):
            tail = full.split("Your response:")[-1]
            a_s, ans_s = extract_fields(tail, optA_text=optA_list[i], optB_text=optB_list[i])

            # Retry if missing
            if ans_s == "M":
                rp = minimal_retry_prompt(tok, prefix_list[i], optA_list[i], optB_list[i], hint_list[i])
                rfull = generate_single(tok, model, rp, max_new_tokens=4)
                _, ans_s2 = extract_fields(rfull)
                if ans_s2 in ("A","B"):
                    ans_s = ans_s2

            amb_pred.append(a_s)
            ans_pred.append(ans_s)

        accuracy = sum(1 for a, g in zip(ans_pred, ans_gold) if a == g) / len(ans_gold)
        overconf = sum(1 for a in amb_pred if a == "NOT_AMBIGUOUS") / len(amb_pred)
        underconf = 0.0

        safe = mid.replace("/", "__")
        pd.DataFrame({
            "model_id": mid,
            "input": [r["input"] for r in rows],
            "prompt": prompts,
            "pred_full": preds_raw,
            "amb_pred": amb_pred,
            "ans_pred": ans_pred,
            "amb_gold": amb_gold,
            "ans_gold": ans_gold
        }).to_csv(os.path.join(args.per_model_out, f"ar_preds__{safe}.csv"), index=False)

        pd.DataFrame([{
            "model_id": mid,
            "accuracy": accuracy,
            "overconfidence": overconf,
            "underconfidence": underconf
        }]).to_csv(os.path.join(args.per_model_out, f"ar_metrics__{safe}.csv"), index=False)

        print(f"[{mid}] AR: ACC={accuracy:.3f}  overconf={overconf:.3f}  underconf={underconf:.3f}")

if __name__ == "__main__":
    main()
