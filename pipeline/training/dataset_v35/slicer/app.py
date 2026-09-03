#!/usr/bin/env python3
"""Webtoon strip slicer — browser version of the Tkinter tool.

Left-click on the strip = add a cut line.  Drag a line = move it.  Right-click a
line = delete it.  Every change auto-saves and rebuilds that chapter's panel zip
on the server (no export button).  Prev / Next walk the chapters.

    .venv/bin/python pipeline/training/dataset_v35/slicer/app.py [port]

Served behind Caddy at  /slicer/  (see the Caddyfile block added by setup).
"""
import csv, io, json, os, re, sys, threading, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STRIPS = os.path.abspath(os.path.join(HERE, "..", "strips"))
OUTDIR = os.path.join(HERE, "output")
CUTDIR = os.path.join(HERE, "cuts")
os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(CUTDIR, exist_ok=True)
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
_lock = threading.Lock()
_seg_cache = {}


def chapters():
    return sorted(d for d in os.listdir(STRIPS) if os.path.isdir(os.path.join(STRIPS, d)))


def seg_meta(ch):
    """[(name, y_offset, h)], total_height, width — from the segment files."""
    d = os.path.join(STRIPS, ch)
    segs = sorted(f for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    out, y, w = [], 0, 800
    for s in segs:
        im = _seg(ch, s)
        h, w = im.shape[:2]
        out.append((s, y, h))
        y += h
    return out, y, w


def _seg(ch, name):
    k = (ch, name)
    if k not in _seg_cache:
        if len(_seg_cache) > 40:
            _seg_cache.clear()
        _seg_cache[k] = cv2.imread(os.path.join(STRIPS, ch, name))
    return _seg_cache[k]


def cuts_path(ch):
    return os.path.join(CUTDIR, ch + ".json")


def load_state(ch):
    """-> (cuts, skips).  skips = y markers; a slice containing one is dropped."""
    p = cuts_path(ch)
    if os.path.exists(p):
        try:
            d = json.load(open(p))
            if isinstance(d, list):                       # old format
                return sorted(int(v) for v in d), []
            return (sorted(int(v) for v in d.get("cuts", [])),
                    sorted(int(v) for v in d.get("skips", [])))
        except Exception:
            return [], []
    return [], []


def is_blank(crop):
    """a gutter / empty band -> auto-drop on export.

    catches white webtoon gutters and thin light manga gutters (mostly white,
    maybe a border line) without dropping real minimal panels (a mostly-black
    impact panel keeps a lot of dark pixels; a near-empty panel with a bubble
    has text pixels).
    """
    if crop is None or crop.size == 0:
        return True
    h = crop.shape[0]
    if h < 12:
        return True
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    content = float((g < 232).mean())          # fraction of non-white pixels
    if content < 0.045:                        # basically an empty white band
        return True
    if h < 320 and content < 0.14:             # thin + sparse -> gutter w/ a line
        return True
    return False


def strip_rows(ch, y0, y1):
    """pixels [y0:y1] of the full strip, spanning segments as needed."""
    meta, H, W = seg_meta(ch)
    y0, y1 = max(0, y0), min(H, y1)
    parts = []
    for name, oy, h in meta:
        a, b = oy, oy + h
        if b <= y0 or a >= y1:
            continue
        im = _seg(ch, name)
        parts.append(im[max(0, y0 - a):min(h, y1 - a)])
    return np.vstack(parts) if parts else None


def rebuild_zip(ch):
    cuts, skips = load_state(ch)
    meta, H, W = seg_meta(ch)
    bounds, last = [], 0
    for c in cuts:
        if c > last + 4:
            bounds.append((last, c))
            last = c
    if last < H - 4:
        bounds.append((last, H))
    zp = os.path.join(OUTDIR, ch + ".zip")
    kept = 0
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for a, b in bounds:
            if any(a < s < b for s in skips):               # user marked it a gap
                continue
            crop = strip_rows(ch, a, b)
            if crop is None or crop.shape[0] < 2 or is_blank(crop):   # auto-drop gutters
                continue
            kept += 1
            if crop.shape[0] <= 65000:
                ok, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                z.writestr(f"{ch}__p{kept:03d}.jpg", buf.tobytes())
            else:
                ok, buf = cv2.imencode(".png", crop)
                z.writestr(f"{ch}__p{kept:03d}.png", buf.tobytes())
    return kept


def zip_all():
    """one flat archive of every sliced panel, ready to drop into a dataset."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(OUTDIR)):
            if not f.endswith(".zip"):
                continue
            with zipfile.ZipFile(os.path.join(OUTDIR, f)) as src:
                for n in src.namelist():
                    z.writestr(n, src.read(n))
    buf.seek(0)
    return buf.read()


HTML = r"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Webtoon Slicer</title><style>
*{box-sizing:border-box}body{margin:0;font:14px/1.4 system-ui,sans-serif;background:#1e1e1e;color:#ddd}
#bar{position:sticky;top:0;z-index:10;background:#2b2b2b;padding:8px 12px;display:flex;gap:8px;align-items:center;flex-wrap:wrap;border-bottom:1px solid #444}
button,select{background:#3a3a3a;color:#eee;border:1px solid #555;border-radius:5px;padding:6px 10px;font-size:13px;cursor:pointer}
button:hover{background:#484848}button:disabled{opacity:.4;cursor:default}
#info{font-weight:600;margin:0 6px}#saved{color:#5fd08a;min-width:120px}
#hint{color:#999;padding:4px 12px;background:#252525;font-size:12px}
#wrap{position:relative;margin:14px auto;width:max-content;cursor:crosshair}
#wrap img{display:block;width:var(--w)}
.line{position:absolute;left:0;width:100%;height:0;border-top:2px solid #ff0055;cursor:ns-resize;z-index:3}
.line span{position:absolute;left:4px;top:-18px;background:#ff0055;color:#fff;font-size:11px;padding:0 4px;border-radius:3px}
.line:hover{border-top-color:#ff5599}
.skip{position:absolute;left:0;width:100%;z-index:2;cursor:pointer;
 background:repeating-linear-gradient(45deg,rgba(120,120,120,.55) 0 8px,rgba(60,60,60,.55) 8px 16px);
 border-top:1px dashed #999;border-bottom:1px dashed #999}
.skip span{position:absolute;left:6px;top:4px;background:#555;color:#fff;font-size:11px;padding:1px 5px;border-radius:3px}
</style></head><body>
<div id=bar>
 <button id=prev>&#9664; Prev</button>
 <select id=sel></select>
 <button id=next>Next &#9654;</button>
 <span id=info></span>
 <button id=clear>Clear</button>
 <span id=saved></span>
 <a id=dlch style="margin-left:auto"><button>Download this chapter</button></a>
 <a id=dl href="api/zip_all"><button>Download ALL panels</button></a>
</div>
<div id=hint>click = add cut &nbsp;·&nbsp; drag a line = move &nbsp;·&nbsp; right-click a line = delete &nbsp;·&nbsp; <b>Shift-click a band = mark it a gap</b> (excluded) &nbsp;·&nbsp; blank gutters are auto-dropped &nbsp;·&nbsp; auto-saves</div>
<div id=wrap></div>
<script>
const B="";let chs=[],ci=0,meta=null,cuts=[],skips=[],scale=1,drag=null;
const wrap=document.getElementById('wrap'),sel=document.getElementById('sel'),
info=document.getElementById('info'),saved=document.getElementById('saved');
async function j(u,o){const r=await fetch(B+u,o);return r.json();}
async function init(){chs=await j('api/chapters');sel.innerHTML=chs.map((c,i)=>`<option value=${i}>${c.id}</option>`).join('');load(0);}
async function load(i){ci=Math.max(0,Math.min(chs.length-1,i));sel.value=ci;
 meta=await j('api/strip/'+chs[ci].id);cuts=meta.cuts.slice();skips=(meta.skips||[]).slice();
 const W=Math.min(720,meta.width);scale=W/meta.width;
 wrap.style.setProperty('--w',W+'px');
 wrap.innerHTML=meta.segments.map(s=>`<img src="seg/${chs[ci].id}/${s.name}" loading=lazy>`).join('');
 wrap.style.height=(meta.height*scale)+'px';
 document.getElementById('prev').disabled=ci==0;
 document.getElementById('next').disabled=ci==chs.length-1;
 document.getElementById('dlch').href='api/zip/'+chs[ci].id;
 info.textContent=`${ci+1}/${chs.length}  ·  ${chs[ci].id}  ·  ${meta.height}px`;
 render();}
let suppressClick=false;
function bounds(){const b=[0,...cuts.slice().sort((a,b)=>a-b),meta.height];
 const o=[];for(let k=0;k<b.length-1;k++)o.push([b[k],b[k+1]]);return o;}
function render(){
 wrap.querySelectorAll('.line,.skip').forEach(e=>e.remove());
 cuts.sort((a,b)=>a-b);skips.sort((a,b)=>a-b);
 bounds().forEach(([a,c])=>{ if(!skips.some(s=>s>a&&s<c))return;
  const d=document.createElement('div');d.className='skip';
  d.style.top=(a*scale)+'px';d.style.height=((c-a)*scale)+'px';
  d.innerHTML='<span>GAP — not exported (shift-click to keep)</span>';
  d.onclick=e=>{e.stopPropagation();skips=skips.filter(s=>!(s>a&&s<c));render();save();};
  wrap.appendChild(d);});
 cuts.forEach((y,k)=>{const d=document.createElement('div');d.className='line';
  d.style.top=(y*scale)+'px';d.innerHTML=`<span>#${k+1} · ${y}px</span>`;
  d.onmousedown=e=>{e.preventDefault();e.stopPropagation();drag=k;};
  d.onclick=e=>e.stopPropagation();
  d.oncontextmenu=e=>{e.preventDefault();e.stopPropagation();cuts.splice(k,1);render();save();};
  wrap.appendChild(d);});
 const skipped=bounds().filter(([a,c])=>skips.some(s=>s>a&&s<c)).length;
 saved.textContent=`~${cuts.length+1-skipped} slices`;}
wrap.addEventListener('click',e=>{
 if(drag!==null||suppressClick){suppressClick=false;return;}
 const r=wrap.getBoundingClientRect();
 const y=Math.round((e.clientY-r.top)/scale);
 if(y<=2||y>=meta.height-2)return;
 if(e.shiftKey){                                   // mark the band around y as a gap
  const bd=bounds().find(([a,c])=>y>a&&y<c);if(!bd)return;
  if(skips.some(s=>s>bd[0]&&s<bd[1]))skips=skips.filter(s=>!(s>bd[0]&&s<bd[1]));
  else skips.push(Math.round((bd[0]+bd[1])/2));
  render();save();return;}
 cuts.push(y);render();save();});
window.addEventListener('mousemove',e=>{if(drag===null)return;
 const r=wrap.getBoundingClientRect();let y=Math.round((e.clientY-r.top)/scale);
 y=Math.max(2,Math.min(meta.height-2,y));cuts[drag]=y;render();});
window.addEventListener('mouseup',()=>{if(drag!==null){drag=null;suppressClick=true;
 cuts.sort((a,b)=>a-b);render();save();}});
let t=null;function save(){clearTimeout(t);
 t=setTimeout(async()=>{saved.textContent='saving…';
  const r=await j('api/save/'+chs[ci].id,{method:'POST',
  headers:{'content-type':'application/json'},body:JSON.stringify({cuts,skips})});
  saved.textContent=`✓ ${r.panels} panels`;},350);}
document.getElementById('prev').onclick=()=>load(ci-1);
document.getElementById('next').onclick=()=>load(ci+1);
sel.onchange=()=>load(+sel.value);
document.getElementById('clear').onclick=()=>{cuts=[];skips=[];render();save();};
init();
</script></body></html>"""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _path(self):
        return re.sub(r"^/slicer", "", self.path.split("?")[0]).lstrip("/")

    def do_GET(self):
        p = self._path()
        try:
            if p in ("", "index.html"):
                return self._send(200, HTML, "text/html; charset=utf-8")
            if p == "api/chapters":
                out = []
                for c in chapters():
                    _, hh, _w = seg_meta(c)
                    cu, _sk = load_state(c)
                    out.append({"id": c, "height": hh, "cuts": len(cu)})
                return self._send(200, json.dumps(out))
            if p.startswith("api/strip/"):
                ch = p[len("api/strip/"):]
                meta, hh, w = seg_meta(ch)
                cu, sk = load_state(ch)
                return self._send(200, json.dumps({
                    "width": w, "height": hh, "cuts": cu, "skips": sk,
                    "segments": [{"name": n, "y": y, "h": h} for n, y, h in meta]}))
            if p.startswith("seg/"):
                _, ch, name = p.split("/", 2)
                fp = os.path.join(STRIPS, ch, os.path.basename(name))
                if not os.path.isfile(fp):
                    return self._send(404, "no")
                ct = "image/png" if fp.endswith(".png") else "image/jpeg"
                return self._send(200, open(fp, "rb").read(), ct)
            if p.startswith("api/zip/"):
                ch = p[len("api/zip/"):]
                with _lock:
                    rebuild_zip(ch)
                fp = os.path.join(OUTDIR, ch + ".zip")
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", f'attachment; filename="{ch}.zip"')
                b = open(fp, "rb").read()
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                return self.wfile.write(b)
            if p == "api/zip_all":
                with _lock:
                    for c in chapters():
                        rebuild_zip(c)
                    b = zip_all()
                self.send_response(200)
                self.send_header("Content-Type", "application/zip")
                self.send_header("Content-Disposition", 'attachment; filename="all_sliced_panels.zip"')
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                return self.wfile.write(b)
            return self._send(404, "no")
        except Exception as e:
            return self._send(500, json.dumps({"error": str(e)}))

    def do_HEAD(self):
        self.do_GET()

    def do_POST(self):
        p = self._path()
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if p.startswith("api/save/"):
            ch = p[len("api/save/"):]
            cuts = sorted({int(v) for v in body.get("cuts", []) if 0 < int(v)})
            skips = sorted({int(v) for v in body.get("skips", []) if 0 < int(v)})
            with _lock:
                json.dump({"cuts": cuts, "skips": skips}, open(cuts_path(ch), "w"))
                panels = rebuild_zip(ch)
            return self._send(200, json.dumps({"panels": panels}))
        return self._send(404, "no")


if __name__ == "__main__":
    print(f"slicer on :{PORT}  strips={STRIPS}  out={OUTDIR}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
