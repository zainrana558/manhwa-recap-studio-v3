#!/usr/bin/env python3
"""Write the final corrected narration into the job's narration.json (backup
first), merging: postfix_narration.json (raw -> grok1 -> guards -> deglue ->
credit-drop) + Grok round-2 returns (if present) + any manual overrides.
Then emit the full raw->corrected pair set for dict-learning + ByT5 training.

  python apply_final.py check      # dry run, show counts + samples
  python apply_final.py write      # back up + write narration.json
  python apply_final.py pairs      # write corrected_pairs.jsonl (all sources)
"""
import os, re, sys, json, glob, time, shutil, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import garble_helpers as G
import audit_garble as A   # reuse is_credit_line / deglue / NAMES

ROOT = "/home/azureuser/manhwa-recap-studio"
JOB = "cmtug9seh000axiznhpek3nax"
PF = f"{HERE}/postfix_narration.json"
R2_DIR = f"{HERE}/round2_returns"
R2_MAN = f"{HERE}/residual_export/manifest_r2.json"

def norm(s):
    return re.sub(r"\s+", " ", (s or "").strip())

# ---- load base (postfix) --------------------------------------------------
base = {r["id"]: r for r in json.load(open(PF))}     # id -> {raw, fixed, ...}

# ---- panel-vision (Grok read the actual panel) — high trust ---------------
pv = {}
_pvf = f"{R2_DIR}/PANEL_VISION.txt"
_pvman = f"{HERE}/panel_export/manifest_panels.json"
if os.path.exists(_pvf) and os.path.exists(_pvman):
    pvman = {r["id"]: r for r in json.load(open(_pvman))["rows"]}
    for ln in open(_pvf):
        m = re.match(r"\s*(NM-\d{3}-\d{3})\s*\|\s*(.*)$", ln.rstrip())
        if not m or m.group(1) not in pvman:
            continue
        v = m.group(2).strip()
        if v.upper() == "DROP":
            pv[m.group(1)] = ""
        elif v and len(v) < 900:
            pv[m.group(1)] = norm(v)

# ---- manual overrides (Claude fixed by eye) ------------------------------
mo = {}
if os.path.exists(f"{HERE}/manual_overrides.json"):
    mo = {k: norm(v) for k, v in json.load(open(f"{HERE}/manual_overrides.json")).items()}

# ---- merge round-2 + round-3 corrections -------------------------------
r2 = {}
r2applied = 0
_R3_DIR = f"{HERE}/round3_returns"
_R4_DIR = f"{HERE}/round4_returns"
_R3_MAN = f"{HERE}/round3_export/manifest_r3.json"
_R4_MAN = f"{HERE}/round4_export/manifest_r4.json"
if os.path.isdir(R2_DIR):
    man = {r["id"]: r for r in json.load(open(R2_MAN))["rows"]}
    for _mp in (_R3_MAN, _R4_MAN):
        if os.path.exists(_mp):
            for r in json.load(open(_mp))["rows"]:
                man.setdefault(r["id"], r)
    txt = ""
    for f in sorted(glob.glob(f"{R2_DIR}/FIXED*.txt")) + sorted(glob.glob(f"{_R3_DIR}/FIXED*.txt")) + sorted(glob.glob(f"{_R4_DIR}/FIXED*.txt")) + sorted(glob.glob(f"{R2_DIR}/*.out")):
        txt += open(f).read() + "\n"
    for ln in txt.splitlines():
        m = re.match(r"\s*(NM-\d{3}-\d{3})\s*(?:\(B\)\s*)?\|\s*(.*)$", ln)
        if not m:
            continue
        mid, val = m.group(1), m.group(2).strip()
        if mid not in man:
            continue
        raw = norm(man[mid]["raw"])
        if val.upper() in ("DROP", "[DROP]", "DROPPED"):
            new = ""
        else:
            new = norm(val)
            if not new or new.upper() == raw.upper():
                continue
        # guard: keep the clean words, no hallucinated tokens, not more garbled.
        # strip credit tails from raw first (Grok correctly drops them).
        raw_nc = A._TRAIL_WM.split(raw)[0].strip() if A._TRAIL_WM.search(raw) else raw
        raw_nc = A._CREDIT_STRONG.split(raw_nc)[0].strip() if A._CREDIT_STRONG.search(raw_nc) else raw_nc
        rc = {w.lower().strip("'") for w in re.findall(r"[A-Za-z][A-Za-z']{2,}", raw_nc)
              if (G.is_known_wordform(w) or re.sub(r"[^a-z]", "", w.lower()) in A.NAMES)
              and len(re.sub(r"[^a-z]", "", w.lower())) >= 3}
        cc = {w.lower().strip("'") for w in re.findall(r"[A-Za-z][A-Za-z']{2,}", new)}
        lost = [w for w in rc if w not in cc and not any(w in x or x in w for x in cc)]
        if new and len(lost) > max(2, len(rc) // 3):
            continue
        raw_low = {x.lower() for x in re.findall(r"[A-Za-z][A-Za-z']*", raw)}
        newtoks = [w for w in re.findall(r"[A-Za-z][A-Za-z']{3,}", new) if w.lower() not in raw_low]
        halluc = []
        for w in newtoks:
            wl = re.sub(r"[^a-z]", "", w.lower())
            if G.is_known_wordform(w) or wl in A.NAMES or G.is_scream(w):
                continue
            # line-break fragment ("TOMOR-", "INSTRUC-") — joins a neighbour?
            if re.search(re.escape(w) + r"\s*-|\-\s*" + re.escape(w), new):
                continue
            halluc.append(w)
        if halluc:
            continue
        r2[mid] = new
        r2applied += 1

# ---- assemble final per-id text ---------------------------------------
# priority: manual override > panel-vision > round-2 > postfix
final = {}
for mid, r in base.items():
    if mid in mo:
        final[mid] = mo[mid]
    elif mid in pv:
        final[mid] = pv[mid]
    elif mid in r2:
        final[mid] = r2[mid]
    else:
        final[mid] = r["fixed"]
print(f"panel-vision applied: {len(pv)}   manual overrides: {len(mo)}")

changed = sum(1 for mid, r in base.items() if norm(final[mid]) != norm(r["raw"]))
dropped = sum(1 for mid, r in base.items() if r["raw"] and not final[mid])
print(f"round-2 file(s): {'yes' if r2 else 'NO (not dropped in yet)'}  applied {r2applied}")
print(f"final: {len(final)} lines, {changed} changed, {dropped} dropped")

mode = sys.argv[1] if len(sys.argv) > 1 else "check"

if mode == "check":
    ids = [mid for mid in base if norm(final[mid]) != norm(base[mid]["raw"])]
    import random
    random.seed(7)
    for mid in random.sample(ids, min(20, len(ids))):
        print(f"- {base[mid]['raw'][:95]}\n+ {final[mid][:95]}\n")

elif mode == "write":
    ts = time.strftime("%Y%m%d-%H%M%S")
    nchg = 0
    for nf in sorted(glob.glob(f"{ROOT}/data/jobs/{JOB}/dataset/chapter_*/narration.json")):
        chn = int(re.search(r"(\d+)", nf.split("/")[-2]).group(1))
        d = json.load(open(nf))
        touched = False
        for idx, it in enumerate(d):
            raw = (it.get("text") or "").strip()
            if not raw:
                continue
            mid = "NM-%03d-%03d" % (chn, idx)
            if mid in final and norm(final[mid]) != norm(raw):
                it["text"] = final[mid]
                nchg += 1
                touched = True
        if touched:
            shutil.copy2(nf, f"{nf}.bak-final-{ts}")
            json.dump(d, open(nf, "w"), ensure_ascii=False, indent=1)
    print(f"WROTE {nchg} line changes. backups: *.bak-final-{ts}")

elif mode == "pairs":
    out = []
    for mid, r in base.items():
        a, b = norm(r["raw"]), norm(final[mid])
        if not (a and a != b):
            continue
        if mid in mo:
            src = "manual"
        elif mid in pv:
            src = "panel"
        elif mid in r2:
            src = "grok2"
        elif re.sub(r"[^a-z]", "", a.lower()) == re.sub(r"[^a-z]", "", b.lower()):
            src = "mech"
        else:
            src = "grok1"
        out.append({"src": a, "tgt": b, "kind": src, "id": mid})
    with open(f"{HERE}/corrected_pairs.jsonl", "w") as f:
        for o in out:
            f.write(json.dumps(o) + "\n")
    print(f"wrote {len(out)} pairs -> corrected_pairs.jsonl  ({collections.Counter(o['kind'] for o in out)})")
