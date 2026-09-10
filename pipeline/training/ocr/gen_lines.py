#!/usr/bin/env python3
"""Single-TEXT-LINE synthetic crops for PP-OCRv5 rec fine-tuning.
PaddleOCR rec format: <img_path>\t<label> in a train/val list file.
Renders ONE line of dialogue in a comic font, on a light/bubble background,
degraded to match real rec-crop input (short, ~48px tall after resize).

  python gen_lines.py 40000
"""
import os, re, sys, json, glob, random, io
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/home/azureuser/manhwa-recap-studio"
JOB = "cmtug9seh000axiznhpek3nax"
OUT = f"{HERE}/rec_synth"
os.makedirs(f"{OUT}/img", exist_ok=True)
random.seed(11)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40000

FONTS = [f for f in glob.glob(f"{ROOT}/assets/fonts/*.ttf") if os.path.getsize(f) > 5000]

# phrase corpus: split corrected narration into short line-sized fragments
PHRASES = []
for nf in glob.glob(f"{ROOT}/data/jobs/{JOB}/dataset/chapter_*/narration.json"):
    for it in json.load(open(nf)):
        t = re.sub(r"\s+", " ", (it.get("text") or "").strip())
        if not re.match(r"^[A-Za-z0-9 '.,!?\"\-–—…()/:;%*]+$", t):
            continue
        # break long lines into 2-7 word chunks (a "line" in a bubble)
        w = t.split()
        j = 0
        while j < len(w):
            k = random.randint(2, 7)
            frag = " ".join(w[j:j+k]).strip(" ,")
            if 1 <= len(frag) <= 42 and any(c.isalpha() for c in frag):
                PHRASES.append(frag)
            j += k
PHRASES = list(dict.fromkeys(PHRASES))
random.shuffle(PHRASES)
print(f"line phrases: {len(PHRASES)}   fonts: {len(FONTS)}")

BG_TILES = []
for p in glob.glob(f"{ROOT}/data/jobs/{JOB}/work/temp_slices/chap_00[1-9]/frame_000[0-1]*.jpg")[:120]:
    try:
        BG_TILES.append(Image.open(p).convert("RGB"))
    except Exception:
        pass


def one(txt):
    caps = random.random() < 0.8
    disp = txt.upper() if caps else txt
    fs = random.randint(30, 52)
    font = ImageFont.truetype(random.choice(FONTS), fs)
    tmp = ImageDraw.Draw(Image.new("L", (4, 4)))
    tw = int(tmp.textlength(disp, font=font))
    W, H = tw + random.randint(24, 60), fs + random.randint(16, 34)
    # background
    if BG_TILES and random.random() < 0.35:
        b = random.choice(BG_TILES)
        x = random.randint(0, max(0, b.width - W)); y = random.randint(0, max(0, b.height - H))
        im = b.crop((x, y, x + W, y + H)).resize((W, H))
        im = ImageEnhance.Brightness(im).enhance(random.uniform(1.1, 1.6))
        im = Image.blend(im, Image.new("RGB", (W, H), (255, 255, 255)), random.uniform(0.45, 0.8))
    else:
        g = random.randint(225, 255)
        im = Image.new("RGB", (W, H), (g, g, g))
    d = ImageDraw.Draw(im)
    tcol = (random.randint(0, 45),) * 3
    ty = (H - fs) / 2 - random.randint(0, 4)
    tx = (W - tw) / 2 + random.randint(-3, 3)
    if random.random() < 0.25:
        d.text((tx, ty), disp, font=font, fill=tcol, stroke_width=1, stroke_fill=tcol)
    else:
        d.text((tx, ty), disp, font=font, fill=tcol)
    if random.random() < 0.4:
        im = im.rotate(random.uniform(-2.5, 2.5), expand=True, fillcolor=(250,) * 3, resample=Image.BICUBIC)
    # degrade
    s = random.uniform(0.5, 1.0)
    im = im.resize((max(6, int(im.width*s)), max(6, int(im.height*s))), Image.BILINEAR)
    if random.random() < 0.75:
        im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.1)))
    im = ImageEnhance.Contrast(im).enhance(random.uniform(0.85, 1.2))
    if random.random() < 0.35:
        a = np.asarray(im).astype(np.int16) + np.random.randint(-10, 10, (im.height, im.width, 3), np.int16)
        im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=random.randint(45, 85))
    im = Image.open(buf)
    # PP-OCR rec wants ~48px tall
    r = 48 / im.height
    im = im.resize((max(8, int(im.width*r)), 48), Image.BILINEAR)
    return im.convert("RGB"), disp


rows = []
i = 0
while i < N:
    try:
        img, lab = one(random.choice(PHRASES))
    except Exception:
        continue
    fn = f"{i:07d}.jpg"
    img.save(f"{OUT}/img/{fn}", quality=90)
    rows.append(f"img/{fn}\t{lab}")
    i += 1
    if i % 5000 == 0:
        print(f"  {i}/{N}")
random.shuffle(rows)
nv = 2000
open(f"{OUT}/train_list.txt", "w").write("\n".join(rows[nv:]))
open(f"{OUT}/val_list.txt", "w").write("\n".join(rows[:nv]))
print(f"done: {len(rows)} line crops ({len(rows)-nv} train / {nv} val) -> {OUT}/")
