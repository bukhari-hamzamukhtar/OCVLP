"""Build everything the Kaggle GPU runs need, under kaggle/build/:

  code/     ocvlp package, VLP clone (no .git, no docs), jobs/<job>.json
  images/   data/CLEVR-Hans*/<split>/images/*.png for every image any job uses
  kernels/<job>/  run.py + kernel-metadata.json  (GPU T4 x2, internet on, private)

Usage:  PYTHONHASHSEED=0 python kaggle/prepare_kaggle.py --user <kaggle-username>
Then:   kaggle datasets create -p kaggle/build/code   --dir-mode zip
        kaggle datasets create -p kaggle/build/images --dir-mode zip
        kaggle kernels push   -p kaggle/build/kernels/<job>
"""

import argparse
import glob
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ocvlp import data as D  # noqa: E402
from ocvlp.experiment import TASKSETS  # noqa: E402

JOBS = {
    # job name: list of (taskset, only these class ids or None)
    "smoke": [("hans7_dev", [2])],
    "probe2x": [("hans7_dev", [2])],   # same images as smoke, object lists at 2x size
    "a": [("hans3_vlp", None), ("hans7_dev", None)],
    "b": [("hans7_bal", None)],
    # reruns after the model-reload bug (a and b loaded a new model copy per task)
    "a2": [("hans3_vlp", None), ("hans7_dev", None)],
    "b2": [("hans7_bal", None)],
}
HOURS = {"smoke": 1.0, "probe2x": 1.0, "a": 10.5, "b": 10.5, "a2": 3.0, "b2": 7.5}
# instances already finished in these downloaded jobs are left out
SKIP_DONE = {"b2": ["b"]}
# per-job stages and object-centric image scale (default: all stages, scale from --oc-scale)
JOB_OPTS = {"probe2x": {"stages": "oc", "oc_scale": 2}}
CODE_SLUG = "ocvlp-code"
IMAGES_SLUG = "ocvlp-clevr-hans-images"


def finished(jobs):
    """(task_id, seed) of every instance in the manifests of downloaded jobs."""
    done = set()
    for job in jobs:
        for mf in glob.glob(os.path.join(ROOT, "kaggle", "outputs", job, "**", "manifest.jsonl"),
                            recursive=True):
            for line in open(mf, encoding="utf-8"):
                r = json.loads(line)
                done.add((r["task_id"], r["seed"]))
    return done


def job_instances(spec, skip=()):
    out = []
    for taskset, classes in spec:
        dataset, protocol, draws = TASKSETS[taskset]
        for draw, seed in draws:
            for t in D.make_tasks(dataset, protocol, draw):
                if (classes is None or t["cls"] in classes) and (t["task_id"], seed) not in skip:
                    out.append({"task": t, "seed": seed})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--machine", default="NvidiaTeslaT4")
    ap.add_argument("--dtype", default="float16", choices=["float16", "mixed", "float32", "bfloat16"])
    ap.add_argument("--oc-scale", type=int, default=1)
    a = ap.parse_args()

    build = os.path.join(ROOT, "kaggle", "build")
    code_dir = os.path.join(build, "code")
    img_dir = os.path.join(build, "images")
    for d in (code_dir, img_dir):
        os.makedirs(d, exist_ok=True)

    # --- code dataset -------------------------------------------------------
    shutil.rmtree(os.path.join(code_dir, "ocvlp"), ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "ocvlp"), os.path.join(code_dir, "ocvlp"),
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.rmtree(os.path.join(code_dir, "vlp"), ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "vlp"), os.path.join(code_dir, "vlp"),
                    ignore=shutil.ignore_patterns(".git", "docs", "__pycache__", "results"))
    jobs_dir = os.path.join(code_dir, "jobs")
    os.makedirs(jobs_dir, exist_ok=True)
    all_imgs = []
    for job, spec in JOBS.items():
        inst = job_instances(spec, finished(SKIP_DONE.get(job, [])))
        with open(os.path.join(jobs_dir, f"{job}.json"), "w") as f:
            json.dump(inst, f)
        for it in inst:
            all_imgs += D.all_images([it["task"]])
        print(f"job {job}: {len(inst)} instances")
    with open(os.path.join(code_dir, "dataset-metadata.json"), "w") as f:
        json.dump({"title": CODE_SLUG, "id": f"{a.user}/{CODE_SLUG}",
                   "licenses": [{"name": "other"}]}, f, indent=2)

    # --- image dataset (only the images any job uses) -----------------------
    uniq = sorted(set(all_imgs))
    copied = 0
    for rel in uniq:
        src = os.path.join(ROOT, rel)
        dst = os.path.join(img_dir, rel)
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
    size = sum(os.path.getsize(os.path.join(img_dir, r)) for r in uniq) / 1e6
    print(f"images: {len(uniq)} unique ({copied} newly copied), {size:.0f} MB")
    with open(os.path.join(img_dir, "dataset-metadata.json"), "w") as f:
        json.dump({"title": IMAGES_SLUG, "id": f"{a.user}/{IMAGES_SLUG}",
                   "licenses": [{"name": "MIT"}]}, f, indent=2)

    # --- one kernel per job ---------------------------------------------------
    template = open(os.path.join(ROOT, "kaggle", "kernel_run.py"), encoding="utf-8").read()
    for job in JOBS:
        kdir = os.path.join(build, "kernels", job)
        os.makedirs(kdir, exist_ok=True)
        code = (template.replace("__JOB__", job).replace("__BATCH__", str(a.batch))
                .replace("__HOURS__", str(HOURS[job])).replace("__DTYPE__", a.dtype)
                .replace("__STAGES__", JOB_OPTS.get(job, {}).get("stages", "vlp,oc,direct"))
                .replace("__OC_SCALE__", str(JOB_OPTS.get(job, {}).get("oc_scale", a.oc_scale))))
        with open(os.path.join(kdir, "run.py"), "w", encoding="utf-8") as f:
            f.write(code)
        meta = {"id": f"{a.user}/ocvlp-collect-{job}", "title": f"ocvlp-collect-{job}",
                "code_file": "run.py", "language": "python", "kernel_type": "script",
                "is_private": True, "enable_gpu": True, "enable_tpu": False,
                "enable_internet": True, "machine_shape": a.machine,
                "dataset_sources": [f"{a.user}/{CODE_SLUG}", f"{a.user}/{IMAGES_SLUG}"],
                "competition_sources": [], "kernel_sources": [], "model_sources": []}
        with open(os.path.join(kdir, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f, indent=2)
    print("built", build)


if __name__ == "__main__":
    main()
