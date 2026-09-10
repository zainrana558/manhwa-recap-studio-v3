"""Standalone copies of the paddleocr-service garble/word helpers, so scripts
can run WITHOUT importing main.py (which loads ~600MB of OCR models and, during
a render, pushes the box into swap). Kept deliberately in sync with
mini-services/paddleocr-service/main.py."""
import re

try:
    import wordninja
    _WORDCOST = wordninja.DEFAULT_LANGUAGE_MODEL._wordcost
except Exception:
    _WORDCOST = {}

try:
    from wordfreq import zipf_frequency as _zipf
except Exception:
    def _zipf(w, lang="en"):
        return 3.0 if re.sub(r"[^a-z]", "", w.lower()) in _WORDCOST else 0.0

def zipf(w):
    return _zipf(re.sub(r"[^a-z']", "", w.lower()), "en")

_EXTRA_WORDS = set()
for _b in ("/home/azureuser/manhwa-recap-studio/pipeline/ocr-corrections/_words.txt",
           "/home/azureuser/manhwa-recap-studio/data/ocr-corrections/_words.txt"):
    try:
        for _ln in open(_b, encoding="utf-8"):
            _ln = _ln.split("#", 1)[0].strip().lower()
            if _ln and re.fullmatch(r"[a-z']+", _ln):
                _EXTRA_WORDS.add(_ln)
    except OSError:
        pass

_SPLIT_FUNCTION_WORDS = {
    "a", "i", "am", "an", "as", "at", "be", "by", "do", "go", "he", "hi", "if",
    "in", "is", "it", "me", "my", "no", "of", "oh", "ok", "on", "or", "so", "to",
    "up", "us", "we", "ye", "ah", "ha", "um", "ow", "eh", "yo", "aw",
}
_REAL_SHORT_WORDS = set(_SPLIT_FUNCTION_WORDS) | {"ox", "mr", "ms", "dr"}
_VOWEL_RE = re.compile(r"[aeiouyAEIOUY]")

_CREDIT_LINE_RE = re.compile(
    r"(?i)(?:asura ?scan?s?|asura ?scams?|asura\.?gg|discord\.?gg|/asuran\b|"
    r"\bzo?doc?c\b|\brodocc\b|\bpedoce\b|korean ?translators?|recruiting now|"
    r"for the fastest releases?|read (?:it |them )?at for the|"
    r"re?[dl] ?ice s[tu][tu][dl][il][oc]|"
    r"\bartista\b|han ?joong ?w[u]?eol ?ya|w[u]?eol ?ya\b|guem ?gang ?bul ?gae|"
    r"\bartist ?a?\b.{0,40}\bstudi[oc]\b|author ?[-:·]? ?han ?joong|"
    r"novel ?chapters? ?:|published by river|read at\b.{0,20}fastest)")
_CREDIT_STAFF_RE = re.compile(
    r"(?i)\b(?:nano ?machine|oaks|scarpet|kiro|dotori|brahimm|regis|kishi|rush|"
    r"lance|peri|victor|joemama|euwen|dan|mado|med|reg is)\b")


_CLITIC = {"s", "re", "ve", "ll", "d", "m", "t", "am", "o", "clock", "all", "cause", "em", "n"}

def is_known_wordform(w):
    raw = w.lower().strip()
    if "'" in raw or "’" in raw:
        parts = [re.sub(r"[^a-z]", "", p) for p in re.split(r"['’]", raw)]
        parts = [p for p in parts if p]
        if parts and all(len(p) <= 1 or p in _CLITIC or p in _SPLIT_FUNCTION_WORDS
                         or _zipf(p, "en") >= 2.0 or p in _WORDCOST for p in parts):
            return True
    wl = re.sub(r"[^a-z]", "", raw)
    if len(wl) < 2:
        return True
    if wl in _SPLIT_FUNCTION_WORDS or wl in _EXTRA_WORDS:
        return True
    # wordfreq is the primary signal; wordninja's polluted list only as backup
    if _zipf(w.lower().strip("'"), "en") >= 1.6:
        return True
    return wl in _WORDCOST and _zipf(wl, "en") >= 0.5


def is_real_split_word(w):
    """strict: a token good enough to be one piece of a word-split."""
    core = re.sub(r"[^a-z]", "", w.lower())
    if core in _EXTRA_WORDS or core in _SPLIT_FUNCTION_WORDS:
        return True
    return len(core) >= 3 and _zipf(core, "en") >= 3.3


def is_common_word(w):
    wl = re.sub(r"[^a-z]", "", w.lower())
    if wl in _EXTRA_WORDS or wl in _SPLIT_FUNCTION_WORDS:
        return True
    return len(wl) >= 2 and _zipf(wl, "en") >= 2.8


def is_probable_sfx(tok):
    core = re.sub(r"[^A-Za-z]", "", tok)
    if len(core) < 2:
        return False
    if core.lower() in {
        "haha", "hahaha", "hehe", "hoho", "heh", "hah", "huh", "hmph", "hmm",
        "mmm", "tch", "tsk", "pfft", "psst", "shh", "aha", "aah", "ahh", "ooh",
        "ohh", "eek", "whew", "phew", "uwah", "waah", "wah", "kya", "kyaa",
        "nng", "hnng", "urgh", "blegh", "ack", "gack", "ugh", "gah", "argh",
        "grr", "grrr", "gasp", "pant", "huff", "sigh", "groan", "moan", "gulp",
    }:
        return True
    if re.search(r"(.)\1{2,}", core):
        return True
    if len(core) >= 4 and re.fullmatch(r"(?:ha|he|hi|ho|hu|ja|ka|ke|na|la|da|ba|wa)+", core.lower()):
        return True
    return False


def token_looks_mangled(tok):
    if is_known_wordform(tok):
        return False
    for part in re.split(r"['’]", tok):
        low = re.sub(r"[^a-z]", "", part.lower())
        if len(low) < 3 or is_known_wordform(part):
            continue
        if part.islower() and re.search(r"(.)\1\1", low):
            return True
        if len(low) >= 4 and not _VOWEL_RE.search(low):
            return True
        cons = sum(c not in "aeiouy" for c in low)
        if len(low) >= 5 and cons / len(low) >= 0.78:
            return True
        if re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", low):
            return True
    return False


_KSFX = re.compile(r"(?i)^(?:k+|g+|h+|b+|d+|t+|p+|s+|w+|n+|m+|r+|ackk?|kkva|kuk|"
                   r"keu\w*|heu\w*|euk\w*|kua\w*|kva\w*|jeh?\w*|hn+h?|un?f?h|ihf|"
                   r"uh+|mph+|ngh+|argh+|urgh+|blegh+|kugh+|kwe?gh+|hnng+)$")

def is_scream(w):
    t = re.sub(r"[^A-Za-z]", "", w)
    if len(t) < 3:
        return False
    if re.search(r"(.)\1\1", t) or re.search(r"[AEIOUaeiou]{3,}", t):
        return True
    if re.fullmatch(r"(?i)[aeiouyhkrgnw]+", t):
        return True
    if len(t) <= 7 and _KSFX.match(t) and re.sub(r"[^a-z]", "", t.lower()) not in _WORDCOST:
        return True
    return is_probable_sfx(w)


def looks_like_ocr_garble(text, names=frozenset()):
    if not text:
        return False

    def known(t):
        return is_known_wordform(t) or re.sub(r"[^a-z]", "", t.lower()) in names

    toks = re.findall(r"[A-Za-z][A-Za-z'’]*", text)
    real = [t for t in toks if len(t) >= 2]
    if len(real) < 3:
        return False
    non_dict = [t for t in real
                if not known(t) and len(re.sub(r"[^a-z]", "", t.lower())) >= 3]
    dict_hits = len(real) - len(non_dict)
    if dict_hits >= 1 and any(token_looks_mangled(t) for t in non_dict):
        return True
    if dict_hits >= 3:
        run = 0
        for t in real:
            is_nd = (not known(t)
                     and len(re.sub(r"[^a-z]", "", t.lower())) >= 3)
            run = run + 1 if is_nd else 0
            if run >= 2:
                return True
    if dict_hits / len(real) < 0.55:
        return False
    caps = [t for t in real if t.isupper()]
    allcaps = len(caps) >= max(2, len(real) - 1)
    for t in non_dict:
        low = re.sub(r"[^a-z]", "", t.lower())
        if allcaps and not t.isupper() and len(low) <= 4:
            return True
    return False


def line_has_real_dialogue(text):
    s = _CREDIT_STAFF_RE.sub(" ", _CREDIT_LINE_RE.sub(" ", text))
    words = re.findall(r"[A-Za-z][A-Za-z']{2,}", s)
    real = [w for w in words if is_known_wordform(w) or w.lower() in _REAL_SHORT_WORDS]
    return len(real) >= 4
