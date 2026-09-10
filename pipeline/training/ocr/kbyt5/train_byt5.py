# Nano Machine OCR-garble corrector — ByT5-small fine-tune (Kaggle, CPU-box-target)
# Data: attached dataset with train.jsonl / val.jsonl / eval_real.jsonl  ({"src","tgt"})
import os, sys, json, subprocess, time, re

def sh(*a):
    print("$", " ".join(a), flush=True)
    subprocess.run([sys.executable, "-m", "pip"] + list(a), check=False)

print("=== env ===", flush=True)
# torchvision/torchaudio are (a) unused for ByT5 and (b) the thing that breaks
# transformers' import ("operator torchvision::nms does not exist") whenever
# torch is swapped. Remove them unconditionally.
if os.environ.get("_NM_STAGE") != "2":
    sh("uninstall", "-y", "-q", "torchvision", "torchaudio")

import torch
cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
print("gpu:", name, "cap:", cap, "torch:", torch.__version__, flush=True)
try:
    _ok = torch.cuda.is_available() and (torch.zeros(1, device="cuda") + 1).item() == 1.0
except Exception as e:
    print("cuda smoke failed:", e, flush=True); _ok = False
print("cuda usable:", _ok, flush=True)

if not _ok and os.environ.get("_NM_STAGE") != "2":
    # P100 (sm_60) — Kaggle's torch 2.10 dropped it. Pin a Pascal+Turing build.
    print(">>> pinning torch 2.4.1 + restart", flush=True)
    sh("install", "-q", "torch==2.4.1", "--index-url", "https://download.pytorch.org/whl/cu121")
    os.environ["_NM_STAGE"] = "2"
    os.execv(sys.executable, [sys.executable] + sys.argv)

sh("install", "-q", "-U", "transformers==4.45.2", "accelerate==1.0.1",
   "datasets==3.0.1", "jiwer==3.0.4", "sentencepiece")

import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments)
import jiwer

DATA = None
for root, _, files in os.walk("/kaggle/input"):
    if "train.jsonl" in files:
        DATA = root
print("data dir:", DATA, flush=True)
assert DATA, "attach the byt5 dataset"

MODEL = "google/byt5-small"
MAXLEN = 256
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

ds = load_dataset("json", data_files={
    "train": f"{DATA}/train.jsonl", "val": f"{DATA}/val.jsonl"})
evalr = [json.loads(l) for l in open(f"{DATA}/eval_real.jsonl")]

PREFIX = "fix ocr: "
def prep(b):
    x = tok([PREFIX + s for s in b["src"]], max_length=MAXLEN, truncation=True)
    y = tok(text_target=b["tgt"], max_length=MAXLEN, truncation=True)
    x["labels"] = y["input_ids"]
    return x
ds = ds.map(prep, batched=True, remove_columns=ds["train"].column_names)

coll = DataCollatorForSeq2Seq(tok, model=model)

def cer(preds, refs):
    return jiwer.cer(refs, preds)
def compute_metrics(ep):
    pr, la = ep
    pr = np.where(pr != -100, pr, tok.pad_token_id)
    la = np.where(la != -100, la, tok.pad_token_id)
    P = tok.batch_decode(pr, skip_special_tokens=True)
    R = tok.batch_decode(la, skip_special_tokens=True)
    return {"cer": cer(P, R),
            "exact": float(np.mean([p.strip() == r.strip() for p, r in zip(P, R)]))}

# mid-training eval = loss only (fast); generate-eval done once at the end.
args = Seq2SeqTrainingArguments(
    output_dir="/kaggle/working/ckpt",
    per_device_train_batch_size=16, per_device_eval_batch_size=32,
    gradient_accumulation_steps=1, learning_rate=6e-4,
    num_train_epochs=12, warmup_ratio=0.05, weight_decay=0.01,
    logging_steps=100, eval_strategy="epoch", save_strategy="epoch",
    predict_with_generate=False,
    load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
    fp16=torch.cuda.is_available() and cap and cap[0] < 8,
    bf16=torch.cuda.is_available() and cap and cap[0] >= 8,
    report_to=[], save_total_limit=1)

tr = Seq2SeqTrainer(model=model, args=args, data_collator=coll,
                    train_dataset=ds["train"], eval_dataset=ds["val"],
                    tokenizer=tok)
t0 = time.time()
tr.train()
print(f"train wall: {(time.time()-t0)/60:.1f} min", flush=True)

# ---- honest eval on REAL held-out pairs ------------------------------------
def gen(txts):
    out = []
    for i in range(0, len(txts), 32):
        b = tok([PREFIX + t for t in txts[i:i+32]], return_tensors="pt",
                padding=True, truncation=True, max_length=MAXLEN).to(model.device)
        o = model.generate(**b, max_length=MAXLEN, num_beams=4)
        out += tok.batch_decode(o, skip_special_tokens=True)
    return out

src = [e["src"] for e in evalr]
tgt = [e["tgt"] for e in evalr]
pred = gen(src)
def norm(s): return re.sub(r"\s+", " ", s.strip())
res = {
    "n": len(evalr),
    "cer_raw_vs_gold": jiwer.cer(tgt, src),
    "cer_model_vs_gold": jiwer.cer(tgt, pred),
    "wer_raw_vs_gold": jiwer.wer(tgt, src),
    "wer_model_vs_gold": jiwer.wer(tgt, pred),
    "exact_raw": float(np.mean([norm(a) == norm(b) for a, b in zip(src, tgt)])),
    "exact_model": float(np.mean([norm(a) == norm(b) for a, b in zip(pred, tgt)])),
}
# reorder-specific subset
ro = [i for i, e in enumerate(evalr) if re.search(r"(?i)\b(si this|sih os|sih si|esihi|tesihi|sihl)\b", e["src"])]
if ro:
    res["reorder_n"] = len(ro)
    res["reorder_exact_raw"] = float(np.mean([norm(src[i]) == norm(tgt[i]) for i in ro]))
    res["reorder_exact_model"] = float(np.mean([norm(pred[i]) == norm(tgt[i]) for i in ro]))
print("=== EVAL (real held-out) ===", flush=True)
print(json.dumps(res, indent=2), flush=True)
json.dump(res, open("/kaggle/working/eval_results.json", "w"), indent=2)
for a, b, c in list(zip(src, pred, tgt))[:25]:
    print(f"  RAW : {a[:100]}\n  PRED: {b[:100]}\n  GOLD: {c[:100]}\n", flush=True)

tr.save_model("/kaggle/working/byt5-nm-corrector")
tok.save_pretrained("/kaggle/working/byt5-nm-corrector")
print("saved /kaggle/working/byt5-nm-corrector", flush=True)
