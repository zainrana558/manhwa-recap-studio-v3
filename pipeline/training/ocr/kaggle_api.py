#!/usr/bin/env python3
"""Minimal Kaggle REST client driven by a KGAT_ bearer token (the kaggle CLI
1.7.4.5 can't use these; kagglehub can only do model/dataset artifacts, not
kernel runs). Covers: dataset create/version, kernel push/status/output.

  python kaggle_api.py whoami
  python kaggle_api.py dataset-new  <dir> <slug> "<title>"
  python kaggle_api.py dataset-version <dir> <slug> "<notes>"
  python kaggle_api.py kernel-push  <dir>            # dir has kernel-metadata.json
  python kaggle_api.py kernel-status <owner/slug>
  python kaggle_api.py kernel-output <owner/slug> <out_dir>
"""
import os, sys, json, time, base64, io, zipfile, urllib.request, urllib.error

TOKEN = os.environ.get("KAGGLE_API_TOKEN") or open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
BASE = "https://www.kaggle.com/api/v1"
H = {"Authorization": f"Bearer {TOKEN}"}


def _req(method, path, body=None, headers=None, raw=False):
    url = path if path.startswith("http") else BASE + path
    data = None
    hh = dict(H)
    if headers:
        hh.update(headers)
    if body is not None and not raw:
        data = json.dumps(body).encode()
        hh["Content-Type"] = "application/json"
    elif raw:
        data = body
    r = urllib.request.Request(url, data=data, headers=hh, method=method)
    try:
        with urllib.request.urlopen(r, timeout=120) as resp:
            b = resp.read()
            return resp.status, b
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def whoami():
    import kagglehub
    print(kagglehub.whoami())


# ---- datasets ------------------------------------------------------------
def _upload_file(fp, name):
    """create-upload-token flow: POST /datasets/upload/file/<contentLength>/<name>"""
    blob = open(fp, "rb").read()
    body = {"fileName": name, "contentLength": len(blob),
            "lastModifiedEpochSeconds": int(os.path.getmtime(fp))}
    st, b = _req("POST", f"/datasets/upload/file/{len(blob)}/{name}", body)
    j = json.loads(b)
    token = j["token"]
    put_url = j.get("createUrl") or j.get("createUrlNullable")
    if put_url and put_url.startswith("/"):
        put_url = "https://www.kaggle.com/api/v1" + put_url
    r = urllib.request.Request(put_url, data=blob, method="PUT",
                               headers={"Content-Type": "application/octet-stream",
                                        "x-goog-content-length-range": f"{len(blob)},{len(blob)}"})
    with urllib.request.urlopen(r, timeout=900) as resp:
        assert resp.status in (200, 201), resp.status
    return token


def dataset_new(dirp, slug, title):
    owner = __import__("kagglehub").whoami()["username"]
    files = [f for f in os.listdir(dirp) if os.path.isfile(os.path.join(dirp, f))]
    toks = []
    for f in files:
        t = _upload_file(os.path.join(dirp, f), f)
        toks.append({"token": t})
        print("  uploaded", f)
    body = {"title": title, "slug": slug, "ownerSlug": owner,
            "licenseName": "CC0-1.0", "isPrivate": True,
            "files": toks, "convertToCsv": False}
    st, b = _req("POST", "/datasets/create/new", body)
    print(st, b.decode()[:400])


def dataset_version(dirp, slug, notes):
    owner = __import__("kagglehub").whoami()["username"]
    files = [f for f in os.listdir(dirp) if os.path.isfile(os.path.join(dirp, f))]
    toks = []
    for f in files:
        t = _upload_file(os.path.join(dirp, f), f)
        toks.append({"token": t})
        print("  uploaded", f)
    body = {"versionNotes": notes, "files": toks, "convertToCsv": False,
            "deleteOldVersions": False}
    st, b = _req("POST", f"/datasets/create/version/{owner}/{slug}", body)
    print(st, b.decode()[:400])


# ---- kernels -----------------------------------------------------------
def kernel_push(dirp):
    meta = json.load(open(os.path.join(dirp, "kernel-metadata.json")))
    src_name = meta["code_file"]
    src = open(os.path.join(dirp, src_name)).read()
    body = {
        "slug": meta["id"],
        "newTitle": meta.get("title", meta["id"].split("/")[-1]),
        "language": meta.get("language", "python"),
        "kernelType": meta.get("kernel_type", "script"),
        "isPrivate": meta.get("is_private", True),
        "enableGpu": meta.get("enable_gpu", False),
        "enableTpu": meta.get("enable_tpu", False),
        "enableInternet": meta.get("enable_internet", True),
        "datasetDataSources": meta.get("dataset_sources", []),
        "kernelDataSources": meta.get("kernel_sources", []),
        "modelDataSources": meta.get("model_sources", []),
        "competitionDataSources": meta.get("competition_sources", []),
        "text": src,
    }
    st, b = _req("POST", "/kernels/push", body)
    print(st, b.decode()[:500])


def kernel_status(ref):
    owner, slug = ref.split("/")
    st, b = _req("GET", f"/kernels/status?userName={owner}&kernelSlug={slug}")
    print(st, b.decode()[:400])


def kernel_output(ref, out_dir):
    owner, slug = ref.split("/")
    st, b = _req("GET", f"/kernels/output?userName={owner}&kernelSlug={slug}")
    j = json.loads(b)
    os.makedirs(out_dir, exist_ok=True)
    for f in j.get("files", []):
        u = f["url"]
        code, data = _req("GET", u)
        open(os.path.join(out_dir, f["fileName"]), "wb").write(data)
        print("  got", f["fileName"], len(data), "bytes")
    if j.get("log"):
        code, data = _req("GET", j["log"])
        open(os.path.join(out_dir, "run.log"), "wb").write(data)
        print("  got run.log")


if __name__ == "__main__":
    c = sys.argv[1]
    a = sys.argv[2:]
    {"whoami": whoami, "dataset-new": dataset_new, "dataset-version": dataset_version,
     "kernel-push": kernel_push, "kernel-status": kernel_status,
     "kernel-output": kernel_output}[c](*a)
