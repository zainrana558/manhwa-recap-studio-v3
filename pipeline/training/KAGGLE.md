# Train the webtoon panel detector on Kaggle (free GPU, ~1 h)

Kaggle gives **30 GPU-h / week free** (P100 or T4×2), no credit card. You spend
~10 min clicking; the training runs on their GPU.

---

## 1 · Account (once)

1. Sign up at <https://www.kaggle.com> — Google or email, **no payment method**.
2. Top-right avatar → **Settings** → scroll to **Phone Verification** → verify.
   *(This is what unlocks GPU. One SMS.)*

## 2 · Upload the dataset

I'll send you **`webtoon-yolo.zip`** (~1.5 GB — images + YOLO labels + `data.yaml`).

1. Left sidebar → **Datasets** → **+ New Dataset**.
2. Drag `webtoon-yolo.zip` in. Title: **`webtoon-yolo`**. → **Create**.
   Kaggle unzips it automatically. Wait for "Dataset created".

## 3 · Create the notebook

1. Left sidebar → **Code** → **+ New Notebook**.
2. Right panel **Input** → **+ Add Input** → search `webtoon-yolo` (your dataset) → **Add**.
3. Right panel **Session options**:
   - **Accelerator** → **GPU T4 x2**  (or **GPU P100**)
   - **Persistence** → *Variables and Files*  (optional, lets you resume)
4. Delete the default cell. Paste the three cells below.
5. Top menu → **Run All**.

### Cell 1 — locate the data
```python
!pip -q install "ultralytics>=8.3" onnx onnxslim
import glob, yaml, os
yml = glob.glob('/kaggle/input/**/data.yaml', recursive=True)[0]
d = yaml.safe_load(open(yml))
d['path'] = os.path.dirname(yml)            # point it at the Kaggle mount
yaml.safe_dump(d, open('/kaggle/working/data.yaml', 'w'))
print(d)
print('train imgs:', len(glob.glob(d['path'] + '/images/train/*')))
```

### Cell 2 — train  (~30–60 min on T4×2)
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')          # fine-tune, not from scratch
model.train(
    data='/kaggle/working/data.yaml',
    epochs=120, imgsz=1024, batch=16, device=[0, 1],   # both T4s; use device=0 for P100
    patience=25,
    fliplr=0.0, degrees=0.0, shear=0.0, perspective=0.0, mosaic=0.4,
    hsv_h=0.0, hsv_s=0.3, hsv_v=0.3,
    project='/kaggle/working/runs', name='webtoon',
)
```

### Cell 3 — validate + export ONNX
```python
from ultralytics import YOLO, settings
m = YOLO('/kaggle/working/runs/webtoon/weights/best.pt')
print('mAP50-95:', m.val(data='/kaggle/working/data.yaml', imgsz=1024).box.map)
onnx = m.export(format='onnx', imgsz=1024, opset=13, simplify=True, nms=True)
import shutil; shutil.copy(onnx, '/kaggle/working/webtoon_panel_yolo.onnx')
print('DONE ->', '/kaggle/working/webtoon_panel_yolo.onnx')
```

## 4 · Send me the model

When Cell 3 prints `DONE`:
- Right panel → **Output** → find `webtoon_panel_yolo.onnx` (~12 MB) → **⬇ download**.
- Send me that file.

I drop it into `pipeline/models/manga-panel-yolo/manga_panel_detector_fp32_1024.onnx`,
run `pytest tests/test_image_slicing.py`, and re-render Solo Leveling so you can
compare against the current slicer.

---

### If something breaks

| Symptom | Fix |
|---|---|
| "No accelerator" / training on CPU | Session options → Accelerator → GPU. Re-run. |
| Cell 1 `IndexError` | The dataset didn't mount — re-add it in the Input panel. |
| Session dies at ~9 h | Free GPU limit. `best.pt` is still saved at the last epoch; `patience=25` means it converged long before. Just run Cell 3. |
| `device=[0,1]` error | You picked P100 (single GPU) — change to `device=0`. |

Keep `imgsz=1024` — webtoon pages are tall and 640 drops small panels.
