#!/usr/bin/env python3
"""Assemble the ByT5 training corpus: every real raw->corrected pair we have,
plus synthetic pairs made by replaying REAL observed corruptions onto clean
lines (not random noise). Output: byt5_data/{train,val}.jsonl  {"src","tgt"}.

  python build_corpus.py
"""
import os, re, sys, json, glob, random, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import garble_helpers as G

ROOT = "/home/azureuser/manhwa-recap-studio"
JOB = "cmtug9seh000axiznhpek3nax"
OUT = f"{HERE}/byt5_data"
os.makedirs(OUT, exist_ok=True)
random.seed(13)

NAMES = {re.sub(r"[^a-z]", "", n.lower())
         for n in json.load(open(f"{ROOT}/pipeline/ocr-corrections/nano-machine.json")).get("names", [])}


def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())


def clean_ok(s):
    """target must look like real dialogue, not garble."""
    ws = re.findall(r"[A-Za-z][A-Za-z']*", s)
    if len(ws) < 2:
        return False
    good = sum(1 for w in ws if G.is_known_wordform(w) or w.lower() in G._REAL_SHORT_WORDS
              or re.sub(r"[^a-z]", "", w.lower()) in NAMES or G.is_scream(w))
    return good / len(ws) >= 0.9 and not G.looks_like_ocr_garble(s, NAMES)


# ---- 1. real pairs -----------------------------------------------------------
real = []          # (src, tgt, source)

# the merged applied corrections (grok1 + grok2 + mech), from apply_final.py pairs
cp = f"{HERE}/corrected_pairs.jsonl"
if os.path.exists(cp):
    for ln in open(cp):
        e = json.loads(ln)
        a, b = norm(e["src"]), norm(e["tgt"])
        if a and b and a != b and clean_ok(b):
            real.append((a, b, e.get("kind", "corr")))

# panel-vision pairs (Grok read the actual panel) — HIGH value real garble->truth
pv = f"{HERE}/panel_vision_returns"
if os.path.isdir(pv):
    man = {r["id"]: r for r in json.load(open(f"{HERE}/panel_export/manifest_panels.json"))["rows"]}
    for f in glob.glob(f"{pv}/*.txt"):
        for ln in open(f):
            m = re.match(r"\s*(NM-\d{3}-\d{3})\s*\|\s*(.*)$", ln.rstrip())
            if not m or m.group(1) not in man:
                continue
            a, b = norm(man[m.group(1)]["raw"]), norm(m.group(2))
            if b.upper() == "DROP" or not b or a == b:
                continue
            if clean_ok(b):
                real.append((a, b, "panel"))

# exemplars.jsonl (whole-line, keep only clean targets)
for p in glob.glob(f"{ROOT}/data/ocr-corrections/*.exemplars.jsonl"):
    for l in open(p):
        try:
            e = json.loads(l)
        except Exception:
            continue
        a, b = norm(e.get("from", "")), norm(e.get("to", ""))
        if a and b and a != b and float(e.get("weight", 0)) >= 1.0 and clean_ok(b):
            real.append((a, b, "exemplar"))

# credit-strip pairs straight off the ORIGINAL narration (pre-fix) — teaches
# the model to drop scanlation credit / front-matter / watermarks
import audit_garble as _AG
_orig = {}
for nf in glob.glob(f"{ROOT}/data/jobs/{JOB}/dataset/chapter_*/narration.json"):
    for bk in sorted(glob.glob(nf + ".bak-*")):        # earliest backup = pristine
        for it in json.load(open(bk)):
            t = norm(it.get("text") or "")
            if not t:
                continue
            if _AG.is_credit_line(t):
                real.append((t, "", "credit"))
            else:
                mk = _AG._TRAIL_WM.search(t)
                if mk and mk.start() > 0:
                    head = t[:mk.start()].strip(" .,-·/|")
                    if len(re.findall(r"[A-Za-z][A-Za-z']{2,}", head)) >= 3 and clean_ok(head):
                        real.append((t, head, "credit"))
        break

seen = set()
real_u = []
for a, b, s in real:
    if (a, b) in seen:
        continue
    seen.add((a, b))
    real_u.append((a, b, s))
bysrc = collections.Counter(s for _, _, s in real_u)
print(f"real pairs: {len(real_u)}  {dict(bysrc)}")

# ---- 2. mine corruption ops from the real pairs ----------------------------
# token-level: clean_tok -> garbled_tok  (reverse of a correction)
tok_corrupt = collections.defaultdict(list)
phrase_corrupt = collections.defaultdict(list)
for a, b, _ in real_u:
    aw, bw = a.split(), b.split()
    if len(aw) == len(bw):
        for x, y in zip(aw, bw):
            cx = re.sub(r"[^A-Za-z']", "", x)
            cy = re.sub(r"[^A-Za-z']", "", y)
            if cx and cy and cx.lower() != cy.lower() and len(cy) >= 2:
                tok_corrupt[cy.upper()].append(cx.upper())     # clean -> garbled
    # phrase: short spans that differ
    sm = __import__("difflib").SequenceMatcher(None, [w.lower() for w in bw], [w.lower() for w in aw])
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != "equal" and 1 <= (i2 - i1) <= 4 and 1 <= (j2 - j1) <= 5:
            phrase_corrupt[" ".join(bw[i1:i2]).upper()].append(" ".join(aw[j1:j2]).upper())

# generic char-swaps observed
charswap = collections.Counter()
for cy, cxs in tok_corrupt.items():
    for cx in cxs:
        if len(cx) == len(cy) and 3 <= len(cx) <= 12:
            same = sum(1 for p, q in zip(cx, cy) if p == q)
            if same >= 0.6 * len(cx):
                for p, q in zip(cx, cy):
                    if p != q:
                        charswap[(q, p)] += 1     # right_char -> wrong_char
CS = {k: v for k, v in charswap.items() if v >= 2}
print(f"mined: {len(tok_corrupt)} token-corruptions, {len(phrase_corrupt)} phrase, {len(CS)} char-swaps")

# ---- 3. clean pool from this job's narration (never-flagged lines) ----------
clean_pool = []
sus_ids = {s["id"] for s in json.load(open(f"{HERE}/audit_suspects.json"))}
for r in json.load(open(f"{HERE}/postfix_narration.json")):
    if r["id"] in sus_ids or not r["fixed"]:
        continue
    t = norm(r["fixed"])
    if 4 <= len(t.split()) <= 45 and clean_ok(t):
        clean_pool.append(t)
random.shuffle(clean_pool)
print(f"clean pool: {len(clean_pool)}")


def corrupt(line, k):
    """apply k real observed corruptions to a clean line."""
    ws = line.split()
    ops = 0
    tries = 0
    while ops < k and tries < 30:
        tries += 1
        mode = random.random()
        if mode < 0.45 and tok_corrupt:      # token substitution
            idxs = [i for i, w in enumerate(ws)
                    if re.sub(r"[^A-Za-z']", "", w).upper() in tok_corrupt]
            if idxs:
                i = random.choice(idxs)
                key = re.sub(r"[^A-Za-z']", "", ws[i]).upper()
                bad = random.choice(tok_corrupt[key])
                ws[i] = re.sub(r"[A-Za-z']+", lambda m: bad, ws[i], count=1)
                ops += 1
        elif mode < 0.6 and CS:              # char swap in a random word
            i = random.randrange(len(ws))
            w = ws[i]
            cand = [(p, a, b) for p, ch in enumerate(w) for (a, b) in CS if ch == a]
            if cand:
                p, a, b = random.choice(cand)
                ws[i] = w[:p] + b + w[p+1:]
                ops += 1
        elif mode < 0.78 and len(ws) > 3:    # glue two adjacent words
            i = random.randrange(len(ws) - 1)
            ws[i] = ws[i] + ws[i+1]
            del ws[i+1]
            ops += 1
        elif mode < 0.9 and len(ws) > 4:     # swap two adjacent words (reorder)
            i = random.randrange(len(ws) - 1)
            ws[i], ws[i+1] = ws[i+1], ws[i]
            ops += 1
        else:                                # drop a space inside a word / split
            i = random.randrange(len(ws))
            if len(ws[i]) > 6:
                p = random.randint(2, len(ws[i]) - 2)
                ws[i] = ws[i][:p] + " " + ws[i][p:]
                ops += 1
    return " ".join(ws)


# ---- 4. build synthetic + assemble ----------------------------------------
SYN_N = 6000
syn = []
for t in clean_pool[:SYN_N]:
    k = random.choices([1, 2, 3], weights=[5, 3, 2])[0]
    c = corrupt(t, k)
    if c != t:
        syn.append((c, t, "syn"))
# also identity pairs so the model learns to leave clean text alone
ident = [(t, t, "id") for t in clean_pool[SYN_N:SYN_N + 2000]]

allp = real_u + syn + ident
random.shuffle(allp)
n_val = 600
val = allp[:n_val]
train = allp[n_val:]

def dump(path, rows):
    with open(path, "w") as f:
        for a, b, s in rows:
            f.write(json.dumps({"src": a, "tgt": b, "src_kind": s}) + "\n")

dump(f"{OUT}/train.jsonl", train)
dump(f"{OUT}/val.jsonl", val)
# a held-out REAL-only eval set (no synthetic) for honest metrics
real_eval = [p for p in real_u if p not in set(train) and p not in set(val)][:400] or real_u[:400]
dump(f"{OUT}/eval_real.jsonl", real_eval)

print(f"\ntrain: {len(train)}  val: {len(val)}  eval_real: {len(real_eval)}")
print(f"train mix: {collections.Counter(s for _,_,s in train)}")
