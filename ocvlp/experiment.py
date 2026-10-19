"""Run the conditions on the laptop (program search only; VLM answers are replayed).

Conditions
  gt_vlp / gt_count / gt_oc   ground-truth scenes, three languages (upper bound; no VLM)
  vlp_original                VLP exactly as released: VLP's prompt, VLP's language
  oc_hidden                   our object prompt, positions hidden, VLP's language
  oc_count                    our object prompt, positions hidden, VLP + counting
  oc_full                     our object prompt with positions, full object-centric language
  direct                      VLP's direct prompting baseline (no program search)
Moving down the list changes one thing at a time:
  vlp_original -> oc_hidden  : the perception prompt
  oc_hidden    -> oc_count   : counting
  oc_count     -> oc_full    : object geometry (the contribution being tested)

Usage (PYTHONHASHSEED must be 0):
  PYTHONHASHSEED=0 python -m ocvlp.experiment --conditions gt_vlp,gt_count,gt_oc \
      --tasksets hans7_dev --out results/dev_gt.jsonl
"""

import argparse
import glob
import json
import os
import time

from PIL import Image

from .vlp_bridge import ROOT
from . import data as D
from .runner import run_task
from .symbols import ObjList, ObjRec, parse_oc_response
from .prompters import AnswerCache, CachedPrompter
from .prompts import oc_prompt, OC_MAX_NEW_TOKENS
from . import direct

TASKSETS = {
    # name: (dataset, protocol, [(draw, seed), ...])
    "hans3_vlp": ("CLEVR-Hans3", "vlp", [(0, 0)]),
    "hans3_bal": ("CLEVR-Hans3", "balanced", [(0, 0), (1, 1), (2, 2)]),
    "hans7_bal": ("CLEVR-Hans7", "balanced", [(0, 0), (1, 1), (2, 2)]),
    "hans7_dev": ("CLEVR-Hans7", "dev", [(0, 0)]),
    # extra development draws, ground truth only (used to choose the prior weight)
    "hans7_dev1": ("CLEVR-Hans7", "dev", [(1, 1)]),
    "hans3_dev": ("CLEVR-Hans3", "dev", [(0, 0), (1, 1)]),
}
GT_CONDITIONS = {"gt_vlp": "vlp", "gt_count": "count", "gt_oc": "oc"}
VLM_CONDITIONS = {"vlp_original": "vlp", "oc_hidden": "vlp", "oc_count": "count", "oc_full": "oc"}
MODEL = "Qwen2.5-VL-7B-Instruct"


def instances(taskset_names):
    out = []
    for name in taskset_names:
        dataset, protocol, draws = TASKSETS[name]
        for draw, seed in draws:
            for t in D.make_tasks(dataset, protocol, draw):
                out.append((name, t, seed))
    return out


_SIZES = {}


def image_size(rel):
    if rel not in _SIZES:
        with Image.open(D.abs_path(rel)) as im:
            _SIZES[rel] = im.size
    return _SIZES[rel]


def load_vlm(cache_dirs):
    files = []
    manifest = {}
    for d in cache_dirs:
        files += glob.glob(os.path.join(d, "**", "cache.jsonl"), recursive=True)
        for mf in glob.glob(os.path.join(d, "**", "manifest.jsonl"), recursive=True):
            for line in open(mf, encoding="utf-8"):
                r = json.loads(line)
                manifest[(r["task_id"], r["seed"])] = r
    return AnswerCache(files), manifest


def vlm_get_objects(condition, prompter, variables, oc_scale=1):
    """get_objects(path) -> ObjList for one VLM condition."""
    if condition == "vlp_original":
        from method.DSL.dsl_with_img_repr import get_dsl
        sem, _ = get_dsl(MODEL, "CLEVR-Hans3-unconfounded", prompter, variables)
        memo = {}

        def get(path):
            if path not in memo:
                raw = sem["get_objects"](path)  # VLP's own prompt + parser
                memo[path] = ObjList(ObjRec([str(x) for x in e]) for e in raw
                                     if isinstance(e, list) and len(e) > 0)
            return memo[path]
        return get

    prompt = oc_prompt(variables["objects"], variables["properties"])
    memo = {}

    def get(path):
        if path not in memo:
            text = prompter.prompt_with_images(prompt_text=prompt, paths=[path],
                                               max_new_tokens=OC_MAX_NEW_TOKENS,
                                               image_scale=oc_scale)
            w, h = image_size(path)
            memo[path], _ = parse_oc_response(text, w, h, oc_scale)
        return memo[path]
    return get


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", required=True)
    ap.add_argument("--tasksets", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", default="kaggle/outputs", help="comma-separated folders")
    ap.add_argument("--timeout", type=int, default=10)
    ap.add_argument("--oc-scale", type=int, default=1,
                    help="image scale used for the object-centric answers")
    ap.add_argument("--ext-weight", type=float, default=0.1,
                    help="prior weight of the added functions; 0.1 was chosen on the dev "
                         "tasks only (results/tune), before any evaluation run")
    a = ap.parse_args(argv)

    conditions = a.conditions.split(",")
    out_path = ROOT / a.out
    os.makedirs(out_path.parent, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["condition"], r["task_id"], r["seed"]))

    need_vlm = any(c in VLM_CONDITIONS or c == "direct" for c in conditions)
    cache, manifest = load_vlm([str(ROOT / c) for c in a.cache.split(",")]) if need_vlm else (None, {})

    todo = instances(a.tasksets.split(","))
    t_start = time.time()
    with open(out_path, "a", encoding="utf-8") as fh:
        for cond in conditions:
            for taskset, task, seed in todo:
                if (cond, task["task_id"], seed) in done:
                    continue
                rec = {"condition": cond, "taskset": taskset, "task_id": task["task_id"],
                       "cls": task["cls"], "dataset": task["dataset"], "seed": seed}
                try:
                    if cond in GT_CONDITIONS:
                        res = run_task(task, GT_CONDITIONS[cond], D.oracle_objects,
                                       timeout=a.timeout, ext_weight=a.ext_weight)
                    else:
                        info = manifest.get((task["task_id"], seed))
                        if info is None:
                            print(f"skip {cond} {task['task_id']}: not collected yet")
                            continue
                        prompter = CachedPrompter(MODEL, seed, cache, backend="replay")
                        if cond == "direct":
                            res = direct.evaluate(prompter, task)
                        else:
                            get = vlm_get_objects(cond, prompter, info["variables"], a.oc_scale)
                            res = run_task(task, VLM_CONDITIONS[cond], get, timeout=a.timeout,
                                           ext_weight=a.ext_weight)
                except KeyError as e:
                    print(f"skip {cond} {task['task_id']}: {e}")
                    continue
                rec.update(res)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                t_split = f" test-split {rec['test_acc_t']:.3f}" if "test_acc_t" in rec else ""
                print(f"{cond:13s} {task['task_id']:32s} seed {seed}  val {rec['test_acc']:.3f}"
                      f"{t_split}  {rec.get('program') or rec.get('rule', '')}"[:240], flush=True)
    print(f"finished in {time.time() - t_start:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
