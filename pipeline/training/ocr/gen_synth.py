#!/usr/bin/env python3
"""Synthetic scanlation-bubble OCR data: render dialogue in comic fonts into
speech-bubble / caption shapes over real manhwa backdrops, then degrade to
match the real OCR-input distribution. Perfect labels.

  python gen_synth.py 12000        # N pairs -> synth/img/*.jpg + synth/labels.jsonl
"""
import os, re, sys, json, glob, random, math, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/azureuser/manhwa-recap-studio"
JOB = "cmtug9seh000axiznhpek3nax"
OUT = f"{HERE}/synth"
os.makedirs(f"{OUT}/img", exist_ok=True)
random.seed(7)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12000

FONTS = glob.glob(f"{ROOT}/assets/fonts/*.ttf")
FONTS = [f for f in FONTS if os.path.getsize(f) > 5000]
print("fonts:", [os.path.basename(f) for f in FONTS])

# text corpus: the corrected narration (real dialogue distribution + vocab)
CORPUS = []
for nf in glob.glob(f"{ROOT}/data/jobs/{JOB}/dataset/chapter_*/narration.json"):
    for it in json.load(open(nf)):
        t = re.sub(r"\s+", " ", (it.get("text") or "").strip())
        # keep lines that are clean-ish and a sensible length for one bubble
        if 3 <= len(t.split()) <= 22 and re.match(r"^[A-Za-z0-9 '.,!?\"\-–—…()/:;%*]+$", t):
            CORPUS.append(t)
CORPUS = list(dict.fromkeys(CORPUS))
random.shuffle(CORPUS)
print(f"corpus lines: {len(CORPUS)}")

# real backdrops (composed frames) — crop a random region for behind the bubble
BGS = glob.glob(f"{ROOT}/data/jobs/{JOB}/work/temp_slices/chap_0[0-3]*/frame_000[0-2]*.jpg")[:400]


def wrap(draw, text, font, maxw):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= maxw or not cur:
            cur = t
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def render_one(text):
    caps = random.random() < 0.75
    disp = text.upper() if caps else text
    font = ImageFont.truetype(random.choice(FONTS), random.randint(30, 58))
    Wtxt = random.randint(300, 520)          # max text column width
    shape = random.random()
    # ellipse needs the text column much narrower than the bubble
    ell = shape < 0.55
    tmp = Image.new("L", (10, 10)); d0 = ImageDraw.Draw(tmp)
    lines = wrap(d0, disp, font, Wtxt)
    txtw = max(int(d0.textlength(ln, font=font)) for ln in lines)
    lh = font.size + random.randint(4, 12)
    Htxt = lh * len(lines)
    if ell:
        W = int(txtw * random.uniform(1.5, 1.9)); H = int(Htxt * random.uniform(1.7, 2.2))
    else:
        W = txtw + random.randint(40, 90); H = Htxt + random.randint(30, 70)
    pad = 30
    CW, CH = W + 2*pad, H + 2*pad
    if BGS and random.random() < 0.7:
        bg = Image.open(random.choice(BGS)).convert("RGB")
        x = random.randint(0, max(0, bg.width - CW)); y = random.randint(0, max(0, bg.height - CH))
        canvas = bg.crop((x, y, x + CW, y + CH)).resize((CW, CH))
        canvas = ImageEnhance.Brightness(canvas).enhance(random.uniform(0.6, 1.1))
    else:
        g = random.randint(150, 240)
        canvas = Image.new("RGB", (CW, CH), (g, g, g))
    dr = ImageDraw.Draw(canvas)
    box = [pad, pad, pad + W, pad + H]
    fill = (255, 255, 255) if random.random() < 0.85 else (random.randint(230, 255),) * 3
    outline = (0, 0, 0)
    ow = random.randint(2, 5)
    if ell:
        dr.ellipse(box, fill=fill, outline=outline, width=ow)
    elif shape < 0.85:                    # rounded rect (caption box)
        dr.rounded_rectangle(box, radius=random.randint(0, 22), fill=fill, outline=outline, width=ow)
    else:                                 # jagged/spiky bubble (shout)
        pts = []
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        rx, ry = (box[2]-box[0])/2+8, (box[3]-box[1])/2+6
        for a in range(0, 360, 18):
            rr = random.uniform(0.82, 1.1)
            pts.append((cx + rx*rr*math.cos(math.radians(a)), cy + ry*rr*math.sin(math.radians(a))))
        dr.polygon(pts, fill=fill, outline=outline)
    # text — vertically centred
    tcol = (random.randint(0, 40),) * 3
    ty = pad + (H - Htxt) / 2
    for ln in lines:
        tw = dr.textlength(ln, font=font)
        tx = pad + (W - tw) / 2 + random.randint(-4, 4)
        if random.random() < 0.3:        # faux-bold stroke
            dr.text((tx, ty), ln, font=font, fill=tcol, stroke_width=1, stroke_fill=tcol)
        else:
            dr.text((tx, ty), ln, font=font, fill=tcol)
        ty += lh
    # rotate slightly
    if random.random() < 0.5:
        canvas = canvas.rotate(random.uniform(-3.5, 3.5), expand=True,
                               fillcolor=(200, 200, 200), resample=Image.BICUBIC)
    # ---- degrade to match real OCR input ----
    im = canvas
    s = random.uniform(0.45, 1.0)                    # resolution loss
    im = im.resize((max(8, int(im.width*s)), max(8, int(im.height*s))), Image.BILINEAR)
    if random.random() < 0.8:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.3)))
    im = ImageEnhance.Contrast(im).enhance(random.uniform(0.8, 1.25))
    if random.random() < 0.4:
        a = np.asarray(im).astype(np.int16)
        a += np.random.randint(-12, 12, a.shape, dtype=np.int16)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=random.randint(38, 80))
    im = Image.open(buf)
    # normalise long side ~1100 like the pipeline
    m = max(im.size)
    if m != 1100:
        r = 1100 / m
        im = im.resize((max(1, int(im.width*r)), max(1, int(im.height*r))), Image.BICUBIC)
    return im.convert("RGB"), disp


rows = []
i = 0
while i < N:
    txt = random.choice(CORPUS)
    try:
        img, label = render_one(txt)
    except Exception as e:
        continue
    fn = f"{i:06d}.jpg"
    img.save(f"{OUT}/img/{fn}", quality=88)
    rows.append({"image": f"img/{fn}", "text": label})
    i += 1
    if i % 1000 == 0:
        print(f"  {i}/{N}")
with open(f"{OUT}/labels.jsonl", "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"done: {len(rows)} synthetic pairs -> {OUT}/")
