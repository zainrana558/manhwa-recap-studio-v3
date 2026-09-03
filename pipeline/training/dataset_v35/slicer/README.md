# Webtoon strip slicer

Browser tool to hand-slice the stitched chapter strips into panels for the
dataset. Same UX as the Tkinter script, runs headless on the box.

- **URL:** http://80.225.248.230/slicer/   (Caddy routes `/slicer/*` → `localhost:8899`)
- **service:** `webtoon-slicer.service` (systemd, `Restart=always`, enabled)
- reads strips from `../strips/<series>__<chapter>/strip_NN.jpg`
- left-click = add cut · drag = move · right-click a line = delete · Prev/Next = chapters
- every change auto-writes `output/<chapter>.zip` (panels as `<chapter>__pNNN.jpg`, q95)
- "Download this chapter" / "Download ALL panels" (one flat zip) in the top bar
- cut positions persist in `cuts/<chapter>.json`

Run manually: `.venv/bin/python pipeline/training/dataset_v35/slicer/app.py 8899`

Caddyfile block added under `:80` (backup at `/etc/caddy/Caddyfile.bak`):
```
@slicer { path /slicer /slicer/* }
handle @slicer { reverse_proxy localhost:8899 }
```
