"""End-to-end dry run of the Kaggle collection + laptop replay, with a fake VLM.

Checks that
  * collect.py records VLP's exact prompts and fills the cache,
  * experiment.py finds every answer again (same keys) for every condition,
  * the direct baseline and all program-search conditions produce results.
No GPU, no images, no network.

Run:  PYTHONHASHSEED=0 python tests/test_collect_mock.py
"""

import json
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
if not os.path.exists(os.path.join(ROOT, "prompts")):
    shutil.copytree(os.path.join(ROOT, "vlp", "prompts"), os.path.join(ROOT, "prompts"))

from ocvlp import collect, experiment, prompters  # noqa: E402
from ocvlp import data as D  # noqa: E402


def fake_answer(prompt, paths):
    p = prompt.lower()
    if "objects = [...]" in prompt or "identify notable objects" in p:
        return "```python\nobjects = ['cube', 'sphere', 'cylinder']\n```"
    if "property discovery" in p:
        return "```python\nproperties = ['red', 'cyan', 'large', 'small', 'metal']\n```"
    if "list every object" in p:
        # three objects; one cyan in front of two red
        return json.dumps([
            {"label": "cube", "properties": ["small", "cyan"], "bbox_2d": [200, 250, 240, 290]},
            {"label": "sphere", "properties": ["large", "red"], "bbox_2d": [100, 60, 160, 120]},
            {"label": "cylinder", "properties": ["red"], "bbox_2d": [300, 80, 350, 140]}])
    if "identify objects and their properties" in p:
        return "```python\nobjects = [['cube', 'small', 'cyan'], ['sphere', 'large', 'red']]\n```"
    if "underlying concept" in p:
        return "```python\nanswer = {\n'rule': 'a cyan object in front of red objects',\n}\n```"
    if "determine if the image follows the rule" in p:
        return "Yes"
    return "unknown prompt"


def fake_generate(self, prompt_text, paths, max_new_tokens, image_scale=1):
    return fake_answer(prompt_text, paths), 10, 0.01


def fake_batch(prompter, reqs, max_new_tokens, scale=1):
    return [fake_answer(p, ps) for p, ps in reqs], [10] * len(reqs)


def main():
    prompters.CachedPrompter._generate = fake_generate
    collect.hf_batch = fake_batch
    experiment.image_size = lambda rel: (480, 320)

    tmp = tempfile.mkdtemp()
    task = D.make_tasks("CLEVR-Hans7", "dev", 0)[2]
    jobs = os.path.join(tmp, "jobs.json")
    json.dump([{"task": task, "seed": 0}], open(jobs, "w"))
    out = os.path.join(tmp, "out")
    collect.main(["--tasks", jobs, "--out", out, "--batch", "4", "--hours", "1"])

    lines = open(os.path.join(out, "cache.jsonl")).read().splitlines()
    n_img = len(D.all_images([task]))
    n_q = n_img - len(task["pos"]) - len(task["neg"])
    expected = 2 + n_img + n_img + 1 + n_q
    print("cached answers:", len(lines), f"(expected 2 + {n_img} + {n_img} + 1 + {n_q} = {expected})")
    assert len(lines) == expected

    res = os.path.join(tmp, "res.jsonl")
    # run replay on exactly this one task by monkeypatching the task list
    experiment.instances = lambda names: [("hans7_dev", task, 0)]
    experiment.ROOT = type(experiment.ROOT)(tmp)
    experiment.main(["--conditions", "vlp_original,oc_hidden,oc_count,oc_full,direct",
                     "--tasksets", "hans7_dev", "--out", "res.jsonl", "--cache", "out",
                     "--timeout", "2"])
    rows = [json.loads(l) for l in open(res)]
    for r in rows:
        print(f"  {r['condition']:13s} test {r['test_acc']:.2f}  {r.get('program') or r.get('rule')}")
    assert len(rows) == 5, "every condition must replay without a cache miss"
    print("OK")


if __name__ == "__main__":
    main()
