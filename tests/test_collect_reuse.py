"""Regression test for the Kaggle job b failure (2026-09-26).

Job b loaded a new copy of the model for every task; the old copy still held
GPU memory, so task 2 ran partly on the CPU and task 3 ran out of GPU memory.
This test runs the collection over several tasks with a fake model and checks
  * the model is loaded exactly once for the whole job,
  * a batch that runs out of GPU memory is retried in halves, and no answer is lost,
  * every condition then replays on the laptop without a missing answer.
No GPU, no images, no network.

Run:  PYTHONHASHSEED=0 python tests/test_collect_reuse.py
"""

import json
import os
import sys
import tempfile
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
os.chdir(ROOT)

from test_collect_mock import fake_answer  # noqa: E402
from ocvlp import collect, experiment, prompters  # noqa: E402
from ocvlp import data as D  # noqa: E402

LOADS = []
OOM_SPLITS = []


def fake_load_model(model_id, dtype_name):
    LOADS.append(model_id)
    return SimpleNamespace(device="cpu"), SimpleNamespace(tokenizer=SimpleNamespace(padding_side="left"))


def fake_generate(self, prompt_text, paths, max_new_tokens, image_scale=1):
    if not self.loaded:
        self._load()
    return fake_answer(prompt_text, paths), 10, 0.01


def fake_batch(prompter, reqs, max_new_tokens, scale=1):
    if not prompter.loaded:
        prompter._load()
    if len(reqs) > 2:  # pretend the GPU only fits two requests at once
        OOM_SPLITS.append(len(reqs))
        raise RuntimeError("CUDA out of memory. Tried to allocate 1.51 GiB.")
    return [fake_answer(p, ps) for p, ps in reqs], [10] * len(reqs)


def main():
    prompters._load_model = fake_load_model
    prompters._MODELS.clear()
    prompters.CachedPrompter._generate = fake_generate
    collect.hf_batch = fake_batch
    experiment.image_size = lambda rel: (480, 320)

    tmp = tempfile.mkdtemp()
    tasks = [D.make_tasks("CLEVR-Hans7", "dev", 0)[k] for k in (2, 3, 4)]
    jobs = os.path.join(tmp, "jobs.json")
    json.dump([{"task": t, "seed": 0} for t in tasks], open(jobs, "w"))
    out = os.path.join(tmp, "out")
    collect.main(["--tasks", jobs, "--out", out, "--batch", "4", "--hours", "1"])

    manifest = open(os.path.join(out, "manifest.jsonl")).read().splitlines()
    print(f"model loads: {len(LOADS)} for {len(tasks)} tasks; batches split after "
          f"running out of memory: {len(OOM_SPLITS)}; tasks finished: {len(manifest)}")
    assert len(LOADS) == 1, "the model must be loaded once per job"
    assert OOM_SPLITS, "the out-of-memory path was not exercised"
    assert len(manifest) == len(tasks)

    experiment.instances = lambda names: [("hans7_dev", t, 0) for t in tasks]
    experiment.ROOT = type(experiment.ROOT)(tmp)
    experiment.main(["--conditions", "vlp_original,oc_hidden,oc_count,oc_full,direct",
                     "--tasksets", "hans7_dev", "--out", "res.jsonl", "--cache", "out",
                     "--timeout", "1"])
    rows = [json.loads(l) for l in open(os.path.join(tmp, "res.jsonl"))]
    assert len(rows) == 5 * len(tasks), "every condition must replay without a missing answer"
    print("OK")


if __name__ == "__main__":
    main()
