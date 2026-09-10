#!/usr/bin/env python3
"""cf_fetch — fetch a URL through curl_cffi's browser-TLS impersonation.

Cloudflare-fronted manga aggregators (toonily, comick, weebcentral) 403 a
plain `fetch()` from a datacenter IP on TLS/JA3 fingerprint alone. curl_cffi
replays a real Chrome handshake (HTTP/2 settings, cipher order, GREASE, …) and
gets a 200 — no headless browser, ~0 RAM. Verified 2026-09: `impersonate="chrome"`
(latest) clears weebcentral / comick.io / toonily.me; the older pinned
`chrome124` does NOT (Cloudflare flags stale versions).

Usage:  python cf_fetch.py '<json request on stdin or argv[1]>'
Request:  {"url": "...", "headers": {...}, "binary": false, "timeout": 25,
           "impersonate": "chrome"}
Response (stdout, one JSON line):
  {"ok": true, "status": 200, "url": "<final>", "headers": {...},
   "body": "<text>"}            # or "body_b64": "<base64>" when binary
  {"ok": false, "error": "..."}
Exit 0 on a completed HTTP round-trip (even a 4xx/5xx), 1 on a transport error.
"""
import base64
import json
import sys


def main() -> int:
    raw = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    try:
        req = json.loads(raw)
        url = req["url"]
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"bad request: {e}"}))
        return 1

    try:
        from curl_cffi import requests as creq
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"curl_cffi not installed: {e}"}))
        return 1

    headers = req.get("headers") or {}
    imp = req.get("impersonate") or "chrome"
    timeout = float(req.get("timeout") or 25)
    binary = bool(req.get("binary"))

    try:
        r = creq.get(url, impersonate=imp, headers=headers, timeout=timeout,
                     allow_redirects=True)
    except Exception as e:
        print(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return 1

    out = {
        "ok": True,
        "status": r.status_code,
        "url": str(r.url),
        "headers": {k.lower(): v for k, v in dict(r.headers).items()},
    }
    if binary:
        out["body_b64"] = base64.b64encode(r.content).decode("ascii")
    else:
        # curl_cffi decodes text by declared charset; fall back to utf-8
        try:
            out["body"] = r.text
        except Exception:
            out["body"] = r.content.decode("utf-8", "replace")
    sys.stdout.write(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
