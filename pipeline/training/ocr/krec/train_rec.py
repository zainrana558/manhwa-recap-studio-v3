# PP-OCRv5 English recognition fine-tune (Kaggle GPU) for manhwa/webtoon scanlation text.
# Data: attached dataset  ppocr_rec_data/  with:
#   img/*.jpg
#   train_list.txt / val_list.txt   ("<relpath>\t<label>" per line, paths relative to data root)
#   real_eval.txt                   (held-out REAL Qwen-labelled crops, same format; optional)
# Output: /kaggle/working/rec_ppocrv5_ft/  (inference model + en_PP-OCRv5_rec_mobile_ft.onnx)
import os, sys, subprocess, glob, json, time, shutil, re

def sh(cmd, check=True):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if check and r.returncode:
        raise SystemExit(f"cmd failed ({r.returncode}): {cmd}")

# ---- 1. deps -------------------------------------------------------------
# PaddleOCR 3.x ships the PP-OCRv5 rec configs. paddlepaddle-gpu matched to
# Kaggle's CUDA 12.x. (Paddle has its own CUDA runtime — the torch/sm_60
# issue that bit the ByT5 kernel does not apply here.)
sh("pip -q install 'paddlepaddle-gpu==3.0.0' -i https://www.paddlepaddle.org.cn/packages/stable/cu123/ || "
   "pip -q install paddlepaddle-gpu==3.0.0", check=False)
sh("pip -q install paddleocr==3.1.0 paddle2onnx==2.0.1 onnx onnxruntime jiwer", check=False)

import paddle
print("paddle:", paddle.__version__, "gpu:", paddle.device.is_compiled_with_cuda(),
      "dev:", paddle.device.get_device(), flush=True)

# ---- 2. get PaddleOCR repo (training entrypoints) + PP-OCRv5 rec config/pretrained
sh("git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git /kaggle/working/PaddleOCR", check=False)
OCRDIR = "/kaggle/working/PaddleOCR"
os.chdir(OCRDIR)

# PP-OCRv5 mobile rec: config + pretrained student weights + char dict
CFG = "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml"
assert os.path.exists(CFG), sorted(glob.glob("configs/rec/PP-OCRv5/*"))
sh("mkdir -p pretrain && cd pretrain && "
   "wget -q https://paddleocr.bj.bcebos.com/PP-OCRv5/english/en_PP-OCRv5_mobile_rec_pretrained.pdparams || "
   "wget -q https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/en_PP-OCRv5_mobile_rec_pretrained.pdparams",
   check=False)
PRE = glob.glob("pretrain/*.pdparams")
print("pretrained:", PRE, flush=True)

# ---- 3. data ----------------------------------------------------------
DATA = None
for root, _, files in os.walk("/kaggle/input"):
    if "train_list.txt" in files:
        DATA = root
assert DATA, "attach ppocr_rec_data"
print("data:", DATA, flush=True)

def abspath_list(src, dst):
    n = 0
    with open(src) as f, open(dst, "w") as o:
        for ln in f:
            ln = ln.rstrip("\n")
            if "\t" not in ln:
                continue
            rel, lab = ln.split("\t", 1)
            ap = rel if os.path.isabs(rel) else os.path.join(DATA, rel)
            if os.path.exists(ap) and lab.strip():
                o.write(f"{ap}\t{lab}\n"); n += 1
    return n

os.makedirs("/kaggle/working/lists", exist_ok=True)
ntr = abspath_list(f"{DATA}/train_list.txt", "/kaggle/working/lists/train.txt")
nva = abspath_list(f"{DATA}/val_list.txt", "/kaggle/working/lists/val.txt")
print(f"train {ntr}  val {nva}", flush=True)

# char dict that ships with the config
DICT = None
for c in ["ppocr/utils/dict/en_dict.txt", "ppocr/utils/en_dict.txt"]:
    if os.path.exists(c):
        DICT = c
print("dict:", DICT, flush=True)

# ---- 4. patch config: fine-tune schedule, our data, our dict ----------
import yaml
cfg = yaml.safe_load(open(CFG))
cfg["Global"]["epoch_num"] = 40
cfg["Global"]["eval_batch_step"] = [0, 500]
cfg["Global"]["print_batch_step"] = 50
cfg["Global"]["save_epoch_step"] = 5
cfg["Global"]["save_model_dir"] = "/kaggle/working/rec_ppocrv5_ft"
cfg["Global"]["character_dict_path"] = DICT
cfg["Global"]["use_space_char"] = True
cfg["Global"]["pretrained_model"] = PRE[0].replace(".pdparams", "") if PRE else None
cfg["Global"]["save_inference_dir"] = "/kaggle/working/rec_ppocrv5_ft/inference"
if "Optimizer" in cfg:
    # low LR for fine-tune
    lr = cfg["Optimizer"].get("lr", {})
    if isinstance(lr, dict):
        lr["learning_rate"] = 0.0004
        if "warmup_epoch" in lr:
            lr["warmup_epoch"] = 2
cfg["Train"]["dataset"]["data_dir"] = "/"
cfg["Train"]["dataset"]["label_file_list"] = ["/kaggle/working/lists/train.txt"]
cfg["Train"]["loader"]["batch_size_per_card"] = 128
cfg["Eval"]["dataset"]["data_dir"] = "/"
cfg["Eval"]["dataset"]["label_file_list"] = ["/kaggle/working/lists/val.txt"]
cfg["Eval"]["loader"]["batch_size_per_card"] = 128
open("nm_rec.yml", "w").write(yaml.safe_dump(cfg))
print(open("nm_rec.yml").read()[:2000], flush=True)

# ---- 5. train -------------------------------------------------------
t0 = time.time()
sh(f"{sys.executable} tools/train.py -c nm_rec.yml")
print(f"train wall {(time.time()-t0)/60:.1f} min", flush=True)

# ---- 6. export inference model + onnx ------------------------------
BEST = "/kaggle/working/rec_ppocrv5_ft/best_accuracy"
sh(f"{sys.executable} tools/export_model.py -c nm_rec.yml -o Global.pretrained_model={BEST} "
   f"Global.save_inference_dir=/kaggle/working/rec_ppocrv5_ft/inference", check=False)
sh("paddle2onnx --model_dir /kaggle/working/rec_ppocrv5_ft/inference "
   "--model_filename inference.json --params_filename inference.pdiparams "
   "--save_file /kaggle/working/en_PP-OCRv5_rec_mobile_ft.onnx --opset_version 14 "
   "--enable_onnx_checker True || "
   "paddle2onnx --model_dir /kaggle/working/rec_ppocrv5_ft/inference "
   "--model_filename inference.pdmodel --params_filename inference.pdiparams "
   "--save_file /kaggle/working/en_PP-OCRv5_rec_mobile_ft.onnx --opset_version 14", check=False)
print("onnx:", glob.glob("/kaggle/working/*.onnx"), flush=True)

# ---- 7. honest eval on held-out REAL crops (base vs fine-tuned) ----
real = f"{DATA}/real_eval.txt"
if os.path.exists(real):
    sh(f"{sys.executable} tools/eval.py -c nm_rec.yml -o Global.checkpoints={BEST} "
       f"Eval.dataset.label_file_list=['{real}'] Eval.dataset.data_dir=/ "
       f"2>&1 | tee /kaggle/working/real_eval_ft.txt", check=False)
print("DONE", flush=True)
