"""Collect every VLM answer the experiments need (runs on the Kaggle GPU).

For each task instance (task, seed):
  1. VLP variable discovery (VLP's own discover_properties.variable_discovery,
     n_objects=10, n_properties=10, n_actions=0, all 20 support images).
  2. VLP's own object extraction for every support and query image. The exact
     prompts are captured by running VLP's get_objects against a recorder, then
     generated in batches, so VLP's prompt text is never copied by hand.
  3. Object-centric extraction (ocvlp.prompts) for the same images, optionally
     with the image fed at a larger scale (--oc-scale; VLP's calls keep scale 1).
  4. Direct prompting baseline: one rule call on the support images, then one
     yes/no call per query image.
Answers go to an append-only JSONL cache; discovered vocabularies and timings
go to a manifest. The laptop replays both.

Qwen2.5-VL's default generation config has temperature 1e-6, so VLP's
"sampling" is effectively greedy; batching does not change which answer wins.
"""

import argparse
import json
import os
import time
from types import SimpleNamespace

from .prompters import AnswerCache, CachedPrompter, _key, image_entry
from .prompts import oc_prompt, OC_MAX_NEW_TOKENS
from . import direct

ALL_STAGES = ("vlp", "oc", "direct")


class Recorder:
    """Stands in for a prompter: remembers requests, answers with ''."""

    def __init__(self):
        self.requests = []

    def prompt_with_images(self, prompt_text, paths, max_new_tokens=128, **kw):
        self.requests.append((prompt_text, list(paths), max_new_tokens, 1))
        return ""

    def get_produced_tokens(self):
        return 0

    def reset_produced_tokens(self):
        pass


def batch_generate(prompter, requests, batch_size, log):
    """requests: list of (prompt, paths, max_new_tokens, image_scale). Skips cached ones."""
    todo = []
    seen = set()
    for prompt, paths, mnt, scale in requests:
        k = _key(prompter.model_name, prompter.seed, prompter.sampling, prompt, paths, scale)
        if prompter.cache.get(k) is None and k not in seen:
            seen.add(k)
            todo.append((prompt, tuple(paths), mnt, scale))
    if not todo:
        return
    # group by (max_new_tokens, scale) so a batch never waits on a longer limit
    groups = {}
    for r in todo:
        groups.setdefault((r[2], r[3]), []).append(r)
    for (mnt, scale), items in groups.items():
        for i in range(0, len(items), batch_size):
            chunk = items[i:i + batch_size]
            t0 = time.time()
            texts, ntoks = _generate_chunk(prompter, chunk, mnt, scale, log)
            secs = time.time() - t0
            for (p, ps, _, _), text, nt in zip(chunk, texts, ntoks):
                k = _key(prompter.model_name, prompter.seed, prompter.sampling, p, list(ps), scale)
                prompter.cache.put(k, {"key": k, "model": prompter.model_name,
                                       "seed": prompter.seed, "sampling": prompter.sampling,
                                       "paths": list(ps), "prompt": p, "response": text,
                                       "new_tokens": nt, "seconds": round(secs / len(chunk), 2),
                                       "max_new_tokens": mnt, "batch": len(chunk),
                                       "image_scale": scale})
            log(f"    batch {len(chunk)} x {mnt} tok (scale {scale}): {secs:.1f}s")


def _generate_chunk(prompter, chunk, mnt, scale, log):
    """hf_batch on one chunk; if the GPU runs out of memory, retry it in two halves."""
    try:
        return hf_batch(prompter, [(p, list(ps)) for p, ps, _, _ in chunk], mnt, scale)
    except Exception as e:  # torch.OutOfMemoryError, matched by text so torch stays optional
        if "out of memory" not in str(e).lower() or len(chunk) == 1:
            raise
        free_gpu()
        log(f"    out of GPU memory at batch {len(chunk)}, retrying in two halves")
        h = len(chunk) // 2
        t1, n1 = _generate_chunk(prompter, chunk[:h], mnt, scale, log)
        t2, n2 = _generate_chunk(prompter, chunk[h:], mnt, scale, log)
        return t1 + t2, n1 + n2


def free_gpu():
    import gc
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_memory():
    """GPU memory held by PyTorch, e.g. 'gpu0 8.9G gpu1 7.7G' ('' without a GPU)."""
    try:
        import torch
    except ImportError:
        return ""
    if not torch.cuda.is_available():
        return ""
    return " ".join(f"gpu{i} {torch.cuda.memory_allocated(i) / 2 ** 30:.1f}G"
                    for i in range(torch.cuda.device_count()))


def hf_batch(prompter, reqs, max_new_tokens, scale=1):
    """Batched version of CachedPrompter._generate (same messages, same settings)."""
    import torch
    from qwen_vl_utils import process_vision_info
    if not prompter.loaded:
        prompter._load()  # shared model; the processor pads on the left
    all_msgs, texts = [], []
    for prompt, paths in reqs:
        content = [{"type": "text", "text": prompt}]
        if len(paths) > 1:
            content += [{"type": "image", "image": p, "max_pixels": 224 * 224} for p in paths]
        else:
            content += [image_entry(p, scale) for p in paths]
        msgs = [{"role": "user", "content": content}]
        all_msgs.append(msgs)
        texts.append(prompter.processor.apply_chat_template(msgs, tokenize=False,
                                                            add_generation_prompt=True))
    image_inputs, video_inputs = process_vision_info(all_msgs)
    inputs = prompter.processor(text=texts, images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(prompter.model.device)
    torch.manual_seed(prompter.seed)
    torch.cuda.manual_seed_all(prompter.seed)
    with torch.inference_mode():
        gen = prompter.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=prompter.sampling)
    trimmed = [o[inputs.input_ids.shape[1]:] for o in gen]
    outs = prompter.processor.batch_decode(trimmed, skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)
    pad = prompter.processor.tokenizer.pad_token_id
    ntoks = [int((t != pad).sum()) for t in trimmed]
    prompter.produced_tokens += sum(ntoks)
    return outs, ntoks


def run_instance(task, seed, model, cache, batch_size, log, dtype="float16",
                 stages=ALL_STAGES, oc_scale=1):
    from discover_properties import variable_discovery
    from method.DSL.dsl_with_img_repr import get_dsl

    t0 = time.time()
    prompter = CachedPrompter(model, seed, cache, backend="hf", sampling=True, dtype=dtype)
    args = SimpleNamespace(n_objects=10, n_properties=10, n_actions=0)
    support = task["pos"] + task["neg"]
    queries = task["pos_q"] + task["neg_q"] + task.get("pos_t", []) + task.get("neg_t", [])
    secs = {}

    # 1. discovery (VLP code, sequential, 2 calls; cached answers are reused)
    variables = variable_discovery(prompter, support, args)
    secs["discovery"] = round(time.time() - t0, 1)

    # 2. VLP's object extraction, captured then batched
    if "vlp" in stages:
        t = time.time()
        rec = Recorder()
        semantics, _ = get_dsl(model, "CLEVR-Hans3-unconfounded", rec, variables)
        for img in support + queries:
            semantics["get_objects"](img)
        batch_generate(prompter, rec.requests, batch_size, log)
        secs["vlp_extract"] = round(time.time() - t, 1)

    # 3. object-centric extraction
    if "oc" in stages:
        t = time.time()
        p = oc_prompt(variables["objects"], variables["properties"])
        batch_generate(prompter, [(p, [img], OC_MAX_NEW_TOKENS, oc_scale)
                                  for img in support + queries], batch_size, log)
        secs[f"oc_extract_x{oc_scale}"] = round(time.time() - t, 1)

    # 4. direct baseline: rule (sequential, long), then batched yes/no
    rule = None
    if "direct" in stages:
        t = time.time()
        raw, rule = direct.get_rule(prompter, task)
        tp = direct.test_prompt(rule)
        batch_generate(prompter, [(tp, [img], 128, 1) for img in queries], batch_size, log)
        secs["direct"] = round(time.time() - t, 1)

    return {"task_id": task["task_id"], "seed": seed, "model": model,
            "variables": variables, "direct_rule": rule, "stages": list(stages),
            "oc_scale": oc_scale, "seconds": secs,
            "produced_tokens": prompter.produced_tokens}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True, help="JSON list of {task, seed}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--hours", type=float, default=10.5, help="stop starting new instances after this")
    ap.add_argument("--dtype", default="float16", choices=["float16", "mixed", "float32", "bfloat16"])
    ap.add_argument("--stages", default="vlp,oc,direct")
    ap.add_argument("--oc-scale", type=int, default=1)
    a = ap.parse_args(argv)
    stages = tuple(s for s in a.stages.split(",") if s)

    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, "log.txt"), "a", encoding="utf-8")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    jobs = json.load(open(a.tasks, encoding="utf-8"))
    cache = AnswerCache([os.path.join(a.out, "cache.jsonl")],
                        write_file=os.path.join(a.out, "cache.jsonl"))
    manifest_path = os.path.join(a.out, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest_path):
        for line in open(manifest_path, encoding="utf-8"):
            r = json.loads(line)
            done.add((r["task_id"], r["seed"]))

    start = time.time()
    for n, job in enumerate(jobs):
        task, seed = job["task"], job["seed"]
        if (task["task_id"], seed) in done:
            continue
        if (time.time() - start) / 3600 > a.hours:
            log("time budget reached, stopping cleanly")
            break
        log(f"instance {n + 1}/{len(jobs)}: {task['task_id']} seed {seed}")
        try:
            info = run_instance(task, seed, a.model, cache, a.batch, log, a.dtype,
                                stages, a.oc_scale)
            with open(manifest_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(info) + "\n")
            free_gpu()
            log(f"  done in {sum(info['seconds'].values()):.0f}s {info['seconds']} {gpu_memory()}")
        except Exception as e:  # keep going; the failure is logged and retried next run
            import traceback
            log(f"  FAILED: {e}\n{traceback.format_exc()}")
            free_gpu()
    log("finished")


if __name__ == "__main__":
    main()
