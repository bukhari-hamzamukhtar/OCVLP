"""Kaggle GPU job: collect VLM answers for one job file.

Kaggle mounts the two private datasets read-only under /kaggle/input:
  <code dataset>   ocvlp/, vlp/, jobs/<job>.json
  <images dataset> data/CLEVR-Hans*/<split>/images/*.png   (only the images the tasks use)
This script links them into one working folder so the relative image paths
used as cache keys ('data/CLEVR-Hans7/...') are identical to the laptop's,
then runs ocvlp.collect with PYTHONHASHSEED=0.
"""

import glob
import os
import subprocess
import sys

JOB = "__JOB__"          # filled in by prepare_kaggle.py
BATCH = "__BATCH__"
HOURS = "__HOURS__"
DTYPE = "__DTYPE__"
STAGES = "__STAGES__"
OC_SCALE = "__OC_SCALE__"

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers==4.51.3", "accelerate", "qwen-vl-utils", "vose", "rtpt"],
               check=True)

hits = glob.glob("/kaggle/input/**/ocvlp/collect.py", recursive=True)
if not hits:
    raise SystemExit("code dataset not found under /kaggle/input")
code = os.path.dirname(os.path.dirname(hits[0]))

# Kaggle may or may not keep the top 'data/' folder of the image dataset,
# so find each CLEVR-Hans folder wherever it is and link it under data/.
hits = glob.glob("/kaggle/input/**/CLEVR-Hans*/*/images/*.png", recursive=True)
if not hits:
    raise SystemExit("image dataset not found under /kaggle/input")
roots = {}
for h in hits:
    i = h.index("/CLEVR-Hans")
    name = h[i + 1:].split("/")[0]
    roots.setdefault(name, h[:i] + "/" + name)

work = "/tmp/proj"  # outside /kaggle/working, so Kaggle does not save copies of the inputs
os.makedirs(work, exist_ok=True)
for name, target in [("ocvlp", os.path.join(code, "ocvlp")),
                     ("vlp", os.path.join(code, "vlp")),
                     ("jobs", os.path.join(code, "jobs")),
                     ("prompts", os.path.join(code, "vlp", "prompts"))]:
    link = os.path.join(work, name)
    if not os.path.exists(link):
        os.symlink(target, link)
os.makedirs(os.path.join(work, "data"), exist_ok=True)
for name, target in roots.items():
    link = os.path.join(work, "data", name)
    if not os.path.exists(link):
        os.symlink(target, link)
print("code:", code, "| data:", roots, flush=True)

# expandable segments: less memory fragmentation over a long job (numerics unchanged)
env = dict(os.environ, PYTHONHASHSEED="0", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
subprocess.run(["nvidia-smi"], check=False)
out = f"/kaggle/working/out/{JOB}"
cmd = [sys.executable, "-m", "ocvlp.collect", "--tasks", f"jobs/{JOB}.json",
       "--out", out, "--batch", BATCH, "--hours", HOURS, "--dtype", DTYPE,
       "--stages", STAGES, "--oc-scale", OC_SCALE]
print("running:", " ".join(cmd), flush=True)
subprocess.run(cmd, cwd=work, env=env, check=False)
