"""VLM access with an append-only answer cache.

Every call is stored as one JSON line keyed by (model, seed, sampling, prompt,
image paths). The GPU run on Kaggle fills the cache; the laptop replays it, so
all program searches run locally on identical answers and no answer is ever
requested twice.

The interface matches VLP's prompters (prompt_with_images, prompt_with_text,
get_produced_tokens, reset_produced_tokens, remove_from_gpu), so VLP's own
discovery code and DSL functions can use it unchanged.

HF backend: a faithful copy of VLP's models/qwen/main.py (Qwen2_5Prompter):
text first, then images; 224*224 max pixels per image when several images are
sent; torch.manual_seed(seed) before every generation; do_sample = sampling.
"""

import hashlib
import json
import os
import threading
import time


def _key(model, seed, sampling, prompt, paths, scale=1):
    h = hashlib.sha1()
    parts = [model, seed, bool(sampling), prompt, list(paths)]
    if scale != 1:  # scale-1 keys stay exactly as before
        parts.append({"image_scale": scale})
    h.update(json.dumps(parts, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def image_entry(path, scale=1):
    """Chat-message entry for one image. scale=1 is VLP's default (native size,
    rounded to multiples of 28); scale=s feeds the image s times larger."""
    if scale == 1:
        return {"type": "image", "image": path}
    from PIL import Image
    from .symbols import smart_resize
    with Image.open(path) as im:
        w, h = im.size
    rh, rw = smart_resize(h, w)
    return {"type": "image", "image": path,
            "resized_height": int(rh * scale), "resized_width": int(rw * scale)}


class AnswerCache:
    def __init__(self, files, write_file=None):
        self.table = {}
        self.write_file = write_file
        self.lock = threading.Lock()
        for f in files:
            if os.path.exists(f):
                with open(f, encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            rec = json.loads(line)
                            self.table[rec["key"]] = rec
        if write_file:
            os.makedirs(os.path.dirname(os.path.abspath(write_file)), exist_ok=True)

    def get(self, key):
        rec = self.table.get(key)
        return None if rec is None else rec["response"]

    def put(self, key, rec):
        with self.lock:
            self.table[key] = rec
            if self.write_file:
                with open(self.write_file, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _load_model(model_id, dtype_name):
    """Load Qwen2.5-VL and its processor (the only heavy step; GPU runs only)."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    # 'mixed' = language model in float16, vision encoder in float32
    # (T4 GPUs have no bfloat16; pure float16 can overflow in the vision tower).
    dtype = {"float16": torch.float16, "mixed": torch.float16, "float32": torch.float32,
             "bfloat16": torch.bfloat16, "auto": "auto"}[dtype_name]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=dtype, device_map="auto").eval()
    if dtype_name == "mixed":
        model.visual.to(torch.float32)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"  # needed by batched generation (collect.hf_batch)
    return model, processor


# One loaded model per process, shared by every CachedPrompter. Kaggle job b
# (2026-09-26) made a new prompter per task and loaded a new copy each time;
# the old copy still held GPU memory, so the second copy was partly placed on
# the CPU (about 25 times slower) and the third ran out of GPU memory.
_MODELS = {}


class CachedPrompter:
    """backend='replay': answers must already be cached (laptop).
    backend='hf': Qwen2.5-VL through transformers on GPU (Kaggle)."""

    MODEL_IDS = {
        "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
        "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    }

    def __init__(self, model, seed, cache, backend="replay", sampling=True, dtype="float16"):
        self.model_name = model
        self.seed = seed
        self.cache = cache
        self.backend = backend
        self.sampling = sampling
        self.dtype = dtype
        self.loaded = False
        self.produced_tokens = 0
        self.calls = 0
        self.gpu_seconds = 0.0
        self.misses = []

    # -- VLP interface -----------------------------------------------------
    def get_produced_tokens(self):
        return self.produced_tokens

    def reset_produced_tokens(self):
        self.produced_tokens = 0

    def remove_from_gpu(self):
        if self.loaded:
            import gc
            import torch
            _MODELS.pop((self.model_name, self.dtype), None)
            del self.model, self.processor
            gc.collect()
            torch.cuda.empty_cache()
            self.loaded = False

    def prompt_with_text(self, prompt_text, paths=None, url=False, use_memory=True,
                         max_new_tokens=5000, overwrite_memory=False, **kw):
        return self.prompt_with_images(prompt_text, [], max_new_tokens=max_new_tokens)

    def prompt_with_images(self, prompt_text, paths, url=False, use_memory=True,
                           max_new_tokens=128, overwrite_memory=False, image_scale=1, **kw):
        paths = list(paths)
        key = _key(self.model_name, self.seed, self.sampling, prompt_text, paths, image_scale)
        hit = self.cache.get(key)
        if hit is not None:
            return hit
        if self.backend == "replay":
            self.misses.append((prompt_text[:80], paths[:2]))
            raise KeyError(f"answer not cached: {paths[:1]} {prompt_text[:60]!r}")
        text, n_tokens, secs = self._generate(prompt_text, paths, max_new_tokens, image_scale)
        self.cache.put(key, {"key": key, "model": self.model_name, "seed": self.seed,
                             "sampling": self.sampling, "paths": paths,
                             "prompt": prompt_text, "response": text,
                             "new_tokens": n_tokens, "seconds": round(secs, 2),
                             "max_new_tokens": max_new_tokens, "image_scale": image_scale})
        return text

    # -- HF generation (copy of VLP's Qwen2_5Prompter) ---------------------
    def _load(self):
        key = (self.model_name, self.dtype)
        if key not in _MODELS:
            _MODELS[key] = _load_model(self.MODEL_IDS[self.model_name], self.dtype)
        self.model, self.processor = _MODELS[key]
        self.loaded = True

    def _generate(self, prompt_text, paths, max_new_tokens, image_scale=1):
        import torch
        from qwen_vl_utils import process_vision_info
        if not self.loaded:
            self._load()
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        content = [{"type": "text", "text": prompt_text}]
        if len(paths) > 1:
            content += [{"type": "image", "image": p, "max_pixels": 224 * 224} for p in paths]
        else:
            content += [image_entry(p, image_scale) for p in paths]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)
        t0 = time.time()
        with torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                      do_sample=self.sampling)
        secs = time.time() - t0
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        out = self.processor.batch_decode(trimmed, skip_special_tokens=True,
                                          clean_up_tokenization_spaces=False)
        n_new = int(trimmed[0].shape[0]) if trimmed else 0
        self.produced_tokens += n_new
        self.calls += 1
        self.gpu_seconds += secs
        return (out[0] if out else ""), n_new, secs
