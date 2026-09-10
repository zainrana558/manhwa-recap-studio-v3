#!/usr/bin/env python3
"""Exhaustive residual-garble audit of the POST-FIX narration (raw OCR with the
Grok corrections + guards + mech applied). Text-only layers; emits a suspect
list for a GOT-OCR2 pass.

  python audit_garble.py build     # build postfix_narration.json
  python audit_garble.py scan      # strict scan -> audit_suspects.json
"""
import os, re, sys, json, glob, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import garble_helpers as G

ROOT = "/home/azureuser/manhwa-recap-studio"
JOB = "cmtug9seh000axiznhpek3nax"
PLAN = f"{HERE}/apply_plan.json"
MAN = f"{HERE}/garble_export/manifest.json"
PF = f"{HERE}/postfix_narration.json"

NAMES = {re.sub(r"[^a-z]", "", n.lower())
         for n in json.load(open(f"{ROOT}/pipeline/ocr-corrections/nano-machine.json")).get("names", [])}


def wtoks(s):
    return re.findall(r"[A-Za-z][A-Za-z']*", s or "")


try:
    import wordninja as _WN
except Exception:
    _WN = None

_FUNC = {"a", "an", "the", "to", "of", "in", "on", "is", "it", "as", "at", "i",
         "me", "my", "we", "he", "she", "and", "or", "so", "be", "by", "up",
         "mode", "up", "out", "no", "us", "you", "your", "not", "but"}

def deglue(text):
    """split OCR-glued word runs the mech pass missed: 'CORRECTEDAN'->'CORRECTED AN',
    'WERERESURRECTED', 'LISTENINGTO', 'SELF-HEALINGMODE', 'SomehowI'. Only when
    wordninja splits into pieces that are ALL real words / real short words and
    reconstruct the token exactly, and the glued form itself is not a word."""
    if _WN is None:
        return text
    def _sp(m):
        w = m.group(0)
        if G.is_known_wordform(w) or G.zipf(w) >= 2.2:
            return w
        core = re.sub(r"[^A-Za-z]", "", w)
        p = _WN.split(core.lower())
        if not (2 <= len(p) <= 4) or sum(len(x) for x in p) != len(core):
            return w
        ok = all((len(x) >= 3 and G.is_common_word(x)) or x in _FUNC for x in p)
        if not ok:
            return w
        # at least one substantial (>=4) piece so we're not shredding a name
        if not any(len(x) >= 4 and G.zipf(x) >= 3.0 for x in p):
            return w
        out, i = [], 0
        for x in p:
            out.append(w[i:i + len(x)]); i += len(x)
        return " ".join(out)
    return re.sub(r"[A-Za-z]{8,}", _sp, text)


def build():
    plan = json.load(open(PLAN))
    byid = {}
    for v in plan["staged"].values():
        for idx, raw, new, i in v:
            byid[i] = new
    out = []
    for nf in sorted(glob.glob(f"{ROOT}/data/jobs/{JOB}/dataset/chapter_*/narration.json")):
        ch = nf.split("/")[-2]
        chn = int(re.search(r"(\d+)", ch).group(1))
        d = json.load(open(nf))
        for idx, it in enumerate(d):
            raw = (it.get("text") or "").strip()
            if not raw:
                continue
            mid = "NM-%03d-%03d" % (chn, idx)
            fixed = byid.get(mid, raw)
            # RESCUE: dialogue that trails a watermark ("...MONSTER? READ AT FOR
            # THE FASTEST RELEASES") — if the first pass dropped it, recover the
            # real-dialogue head from the RAW.
            mk = _TRAIL_WM.search(raw)
            if mk and mk.start() > 0:
                head = raw[:mk.start()].strip(" .,-·/|")
                if len(re.findall(r"[A-Za-z][A-Za-z']{2,}", head)) >= 3 and not is_credit_line(head):
                    fixed = head
            if is_credit_line(fixed):
                fixed = ""
            else:
                fixed = deglue(fixed)
            out.append({"id": mid, "ch": ch, "idx": idx, "raw": raw, "fixed": fixed})
    json.dump(out, open(PF, "w"))
    chg = sum(1 for r in out if r["raw"] != r["fixed"])
    drp = sum(1 for r in out if r["raw"] and not r["fixed"])
    print(f"postfix_narration.json: {len(out)} lines, {chg} changed, {drp} dropped")


# ---- strict scanners --------------------------------------------------------
_SIH = re.compile(r'(?i)\b(si\s+this|si\s+sih|sih\s+si|sih\s+os|sih\s+se|si\s+sihe|'
                  r'insi\b|e?sihe?\b|tesihi\b|\bsih\b|\bsihl\b|nnihion\b|xnihi\b|'
                  r'\bos\s+(the|to|this|it|what|how|is)\b|\bsi\s+(the|is|it|this)\b)')
_CREDIT = re.compile(r'(?i)asura|discord\.?gg|red ?ice stud|koreantranslat|recruiting now|'
                     r'zo?doc?c|\bstudi[co]\b|published by|novel ?chapters? ?:|/asuran|'
                     r'for the fastest')

# strong whole-line credit / staff-card / aggregator-watermark signatures
_CREDIT_STRONG = re.compile(
    r'(?i)(?:asura ?sca|sura ?scans|reaper ?scans?|red ?ice stud|han ?joong|wu?eol ?ya|'
    r'guem ?gang ?bul ?gae|discord ?\.? ?gg|/ ?asuran|korean ?translators?|recruiting now|'
    r'\b[a-z]?[a-z]?docc?\b|\bdoce\b|pedoce|by ?river(?:se|s co)|reverse ?co|'
    r'read (?:it |at )?.{0,25}fastest|novel ?chapters? ?:|artist[ao]?\b.{0,30}\bstud|'
    r'\bano ?machine\b|\bang ?machine\b|nano ?machine (?:oaks|dotori|kiro|kishi|brahimm|scarpet)|'
    r'\bbrahimm\b|\bscarpet\b|\bdotori\b|\bzodocc\b)')

_STAFF_TOKENS = {"asurascans", "asuran", "brahimm", "scarpet", "guem", "dotori",
                 "kiro", "kishi", "regis", "rush", "zodocc", "rodocc", "podocc",
                 "zedocc", "pedoce", "redocc", "docc", "reaperscans", "reapersca",
                 "byriverse", "wueol", "joemama", "euwen", "oaks", "lance", "peri"}

_DOCC = re.compile(r'(?i)\b[a-z]{1,3}doc[ce]s?\b|\bdocc\b|pedoce|byriverse')
_TRAIL_WM = re.compile(r'(?i)\bread\s+(?:it\s+)?(?:at\s+)?for\s+the\s+(?:fastest|latest)\b|'
                       r'\bread\s+(?:it\s+)?at\b.{0,40}\bfastest\b|\bnovel ?chapters? ?:|'
                       r'\bano ?machine\b|\band ?machine\b|\bred ?ice stud|we\'?re recruiting|'
                       r'sitemiz ?adresinden')

def is_credit_line(t):
    # the "read at <X>DOCC for the fastest releases" burned-in watermark — the
    # OCR mangles the site token every which way (ZODOCC/RODOCC/PODOCC/PEDOCE/…)
    if _DOCC.search(t):
        ws = [w for w in wtoks(t) if len(re.sub(r"[^a-z]", "", w.lower())) >= 3]
        real = sum(1 for w in ws if G.is_known_wordform(w))
        if real < 5:
            return True
    if _CREDIT_STRONG.search(t):
        # a real dialogue line that merely trails 'Red ice Studio' -> not a full
        # drop; but if >=40% of its words are staff tokens or it's short, drop.
        ws = [re.sub(r"[^a-z]", "", w.lower()) for w in wtoks(t)]
        ws = [w for w in ws if w]
        if not ws:
            return True
        staff = sum(1 for w in ws if w in _STAFF_TOKENS or w in {"red", "ice", "studio",
                    "studi", "author", "artist", "artista", "machine", "nano", "ano",
                    "chapter", "episode", "gang", "bul", "gae", "joong", "han", "ya"})
        real = sum(1 for w in ws if (G.is_known_wordform(w) and w not in _STAFF_TOKENS
                   and w not in {"red", "ice", "author", "artist", "machine", "nano",
                   "chapter", "episode", "gang", "han", "ya"}))
        return real < 4 or staff >= 0.4 * len(ws)
    return False

NAMES |= set("""jaesim bokmajong mukeum seob cheonma keum shaolinquan waijia
neijia shaolin wudang emei kongtong huashan qingcheng cheongseong
jinhwa woojin hyunmu hwaryeon myeonghwa gwak taegyum
chungmyeong munju yeongung dangju baekri sima namgung
yeonwoo jaseoh geumganggwae hyeoldojin
changbai hwangbo namgun yoojong wonmyung jinchang tonghe museong kyum
taeyoon taekgyeom adularia imoogi hengshan cheonghwa menggwang miyang muwol
tongxu cheolyong yujong cheongwoon danbi sorim jinling yeonhwa cheonwoon
hwamyeong hoejang unmun geumin jaseo yongcheon mujin gatelinium
jongsum jonghwan yeowoon woohyuk soyoung""".split())

_VOWELS = set("aeiou")

def token_bad(t, line=""):
    """a token that is NOT a real word / name / scream / short word, and is NOT
    a line-break hyphenation fragment ('IN- TERESTING')."""
    core = re.sub(r"[^a-z]", "", t.lower())
    if len(core) < 3:
        return False
    if core in NAMES or t.lower().strip("'") in G._REAL_SHORT_WORDS:
        return False
    if G.is_known_wordform(t) or G.is_scream(t) or G.is_probable_sfx(t):
        return False
    parts = [re.sub(r"[^a-z]", "", p) for p in re.split(r"['\-]", t.lower())]
    parts = [p for p in parts if p]
    if parts and all(len(p) <= 2 or G.is_known_wordform(p) or p in NAMES for p in parts):
        return False
    # line-break fragment: sits right next to a hyphen in the line, and the
    # fragment joined to its neighbour across the hyphen is a real word
    e = re.escape(t.strip(".,!?"))
    m = re.search(r"([A-Za-z]+)-\s+" + e + r"\b", line) or re.search(r"\b" + e + r"\s*-\s+([A-Za-z]+)", line)
    if m:
        joined = re.sub(r"[^a-z]", "", (m.group(1) + core).lower())
        if G.is_known_wordform(joined) or G.zipf(joined) >= 2.0:
            return False
        # even if the join isn't a dict word, a hyphen-adjacent fragment of a
        # plausible length is almost always a line break, not garble
        return False
    return True


def homograph_suspect(line):
    """Deliberately disabled — the 1-edit heuristic produced almost entirely
    false positives (CLAN'S->class, NEI->new, TSK->ask). Wrong-real-word errors
    need the pixel (GOT-OCR2), not a frequency guess."""
    return None


def scan():
    rows = json.load(open(PF))
    suspects = []
    counts = collections.Counter()
    for r in rows:
        t = r["fixed"]
        if not t:
            continue
        why = []
        if _CREDIT.search(t):
            why.append("credit-residue")
        if _SIH.search(t):
            why.append("sih-residue")
        bad = [w for w in wtoks(t) if token_bad(w, t)]
        if bad:
            why.append("nonword:" + ",".join(bad[:5]))
        if G.looks_like_ocr_garble(t, NAMES):
            why.append("garble-heuristic")
        hg = homograph_suspect(t)
        if hg:
            why.append("homograph:" + hg)
        # run of 2+ consecutive bad tokens
        run = mx = 0
        for w in wtoks(t):
            run = run + 1 if token_bad(w, t) else 0
            mx = max(mx, run)
        if mx >= 2:
            why.append("run%d" % mx)
        if why:
            suspects.append({**r, "why": why})
            for w in why:
                counts[w.split(":")[0]] += 1
    json.dump(suspects, open(f"{HERE}/audit_suspects.json", "w"), indent=0)
    print(f"scanned {len(rows)} post-fix lines")
    print(f"residual suspects: {len(suspects)}")
    print(f"by reason: {dict(counts)}")
    print("\n--- 40 samples ---")
    for s in suspects[:40]:
        print(f"[{s['id']}] {'|'.join(s['why'])[:60]}")
        print(f"   {s['fixed'][:120]}")


def round3():
    """scan the LIVE corrected narration.json for any residual garble; emit
    round3_residual.json for a Grok round-3 pass."""
    rows = []
    cnt = collections.Counter()
    for nf in sorted(glob.glob(f"{ROOT}/data/jobs/{JOB}/dataset/chapter_*/narration.json")):
        ch = nf.split("/")[-2]
        chn = int(re.search(r"(\d+)", ch).group(1))
        lines = [(it.get("text") or "").strip() for it in json.load(open(nf))]
        for i, t in enumerate(lines):
            if not t:
                continue
            bad = [w for w in wtoks(t) if token_bad(w, t)]
            sih = bool(_SIH.search(t))
            cred = is_credit_line(t)
            run = mx = 0
            for w in wtoks(t):
                run = run + 1 if token_bad(w, t) else 0
                mx = max(mx, run)
            if bad or sih or cred or mx >= 2:
                prev = next((lines[j] for j in range(i-1, -1, -1) if lines[j]), "")
                nxt = next((lines[j] for j in range(i+1, len(lines)) if lines[j]), "")
                mid = "NM-%03d-%03d" % (chn, i)
                why = ("sih " if sih else "") + ("credit " if cred else "") + \
                      ("run%d " % mx if mx >= 2 else "") + " ".join(bad[:4])
                rows.append({"id": mid, "ch": ch, "idx": i, "raw": t, "bad": bad,
                             "why": why, "prev": prev[:160], "next": nxt[:160]})
                for w in bad:
                    cnt[w.upper()] += 1
    json.dump(rows, open(f"{HERE}/round3_residual.json", "w"), indent=0)
    print(f"ROUND-3 residual suspects: {len(rows)}")
    print("top tokens:", cnt.most_common(35))
    from collections import Counter as C
    print("by reason:", dict(C(r["why"].strip().split()[0] if r["why"].strip() else "tok" for r in rows)))


if __name__ == "__main__":
    {"build": build, "scan": scan, "round3": round3}[sys.argv[1]]()
