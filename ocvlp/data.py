"""CLEVR-Hans tasks and ground-truth scenes.

Folder layout expected (from the official zips, unzipped into data/):
    data/CLEVR-Hans3/{train,val,test}/images/CLEVR_Hans_classid_{c}_{id:06d}.png
    data/CLEVR-Hans3/{train,val,test}/*scenes*.json
    data/CLEVR-Hans7/...            (same)

A task is one class versus the rest, as in VLP: positives are images of the
class, negatives are images of the other classes. Image paths are kept
relative to the project root (e.g. 'data/CLEVR-Hans7/train/images/...png'),
because VLM answers are cached under the exact path string and the same
strings are used on Kaggle and on the laptop.

Protocols
  vlp       VLP's released loader (utils/dataset_utils.py, 'CLEVR-Hans3-unconfounded'),
            reproduced exactly: fixed first images, support from train, query from val
            (10 + 10). We add a second query set from the test split (10 + 10).
  balanced  our protocol: random images with a fixed seed per draw; negatives spread
            evenly over all other classes; support (10 + 10) from train ids < 1500,
            queries (20 + 20) from val and a second query set (20 + 20) from test.
  dev       like balanced, but support AND queries come from train ids >= 1500 and
            there is no test query set. Every design decision is made here.

Why two query sets: CLEVR-Hans plants shortcuts in train and val (in class
"cyan object in front of two red objects" the cyan object is always a small
cube there). val queries follow VLP and reward a shortcut; test queries (the
dataset's unconfounded split) only reward the true rule.
"""

import json
import os
import random

from .vlp_bridge import ROOT
from .symbols import ObjRec, ObjList

N_CLASSES = {"CLEVR-Hans3": 3, "CLEVR-Hans7": 7}
IMAGES_PER_CLASS = {"train": 3000, "val": 750, "test": 750}
N_SUPPORT = 10       # per side; VLP's max_imgs for CLEVR-Hans3
N_QUERY = 10         # per side; VLP's N_TEST_IMAGES_CLEVR (vlp protocol)
N_QUERY_BAL = 20     # per side; balanced and dev protocols (more power per task)
DEV_SPLIT_ID = 1500  # train ids below this may be used for evaluation support sets

# Class rules as written in the CLEVR-Hans README (0-indexed class ids).
CLASS_RULES = {
    "CLEVR-Hans3": [
        "large cube and large cylinder",
        "small metal cube and small sphere",
        "large blue sphere and small yellow sphere",
    ],
    "CLEVR-Hans7": [
        "large cube and large cylinder",
        "small metal cube and small sphere",
        "small cyan object in front of two red objects",
        "at least five small objects, one green, one brown, one purple",
        "three spheres in the left half (10%: plus three metal cylinders in the right half)",
        "three metal cylinders in the right half",
        "large blue sphere and small yellow sphere",
    ],
}
SPATIAL_CLASSES = {"CLEVR-Hans3": [], "CLEVR-Hans7": [2, 4, 5]}


def img_path(dataset, split, cls, idx):
    return f"data/{dataset}/{split}/images/CLEVR_Hans_classid_{cls}_{idx:06d}.png"


def abs_path(rel):
    return str(ROOT / rel)


# ---------------------------------------------------------------------------
# Task construction
# ---------------------------------------------------------------------------
def _vlp_tasks(dataset, max_imgs=N_SUPPORT):
    """Exact copy of VLP's CLEVR-Hans3 loader logic, generalised to n classes."""
    n = N_CLASSES[dataset]
    tasks = []
    for cls in range(n):
        pos = [img_path(dataset, "train", cls, i) for i in range(50)]
        pos_q = [img_path(dataset, "val", cls, i) for i in range(50)]
        per_other = (max_imgs + 1) // 2
        neg, neg_q = [], []
        for other in range(n):
            if other == cls:
                continue
            neg += [img_path(dataset, "train", other, i) for i in range(per_other)]
            neg_q += [img_path(dataset, "val", other, i) for i in range(N_QUERY)]
        if len(pos) > max_imgs:
            pos, neg = pos[:max_imgs], neg[:max_imgs]
            pos_q, neg_q = pos_q[:N_QUERY], neg_q[:N_QUERY]
        # our addition: a test-split query set, negatives spread over the other classes
        others = [o for o in range(n) if o != cls]
        pos_t = [img_path(dataset, "test", cls, i) for i in range(N_QUERY)]
        neg_t = []
        for o, k in _spread(N_QUERY, others).items():
            neg_t += [img_path(dataset, "test", o, i) for i in range(k)]
        tasks.append({"task_id": f"{dataset}:vlp:0:class{cls}", "dataset": dataset,
                      "protocol": "vlp", "draw": 0, "cls": cls,
                      "pos": pos, "neg": neg, "pos_q": pos_q, "neg_q": neg_q,
                      "pos_t": pos_t, "neg_t": neg_t})
    return tasks


def _spread(k, others):
    """How many negatives to take from each other class: as even as possible."""
    counts = {o: 0 for o in others}
    for i in range(k):
        counts[others[i % len(others)]] += 1
    return counts


def _balanced_tasks(dataset, draw, protocol):
    n = N_CLASSES[dataset]
    tasks = []
    for cls in range(n):
        rng = random.Random(f"{dataset}|{protocol}|{draw}|{cls}")
        others = [o for o in range(n) if o != cls]
        rng.shuffle(others)
        if protocol == "balanced":
            support = ("train", list(range(0, DEV_SPLIT_ID)))
            query = ("val", list(range(IMAGES_PER_CLASS["val"])))
            test = ("test", list(range(IMAGES_PER_CLASS["test"])))
        else:  # dev: two disjoint halves of train ids >= DEV_SPLIT_ID
            pool = list(range(DEV_SPLIT_ID, IMAGES_PER_CLASS["train"]))
            rng.shuffle(pool)
            half = len(pool) // 2
            support, query, test = ("train", pool[:half]), ("train", pool[half:]), None

        def draw_set(split_ids, k):
            split, ids = split_ids
            pos = [img_path(dataset, split, cls, i) for i in rng.sample(ids, k)]
            neg = []
            for o, kk in _spread(k, others).items():
                neg += [img_path(dataset, split, o, i) for i in rng.sample(ids, kk)]
            return pos, neg

        pos, neg = draw_set(support, N_SUPPORT)
        pos_q, neg_q = draw_set(query, N_QUERY_BAL)
        task = {"task_id": f"{dataset}:{protocol}:{draw}:class{cls}", "dataset": dataset,
                "protocol": protocol, "draw": draw, "cls": cls,
                "pos": pos, "neg": neg, "pos_q": pos_q, "neg_q": neg_q}
        if test is not None:
            task["pos_t"], task["neg_t"] = draw_set(test, N_QUERY_BAL)
        tasks.append(task)
    return tasks


def make_tasks(dataset, protocol, draw=0):
    if protocol == "vlp":
        return _vlp_tasks(dataset)
    if protocol in ("balanced", "dev"):
        return _balanced_tasks(dataset, draw, protocol)
    raise ValueError(protocol)


def all_images(tasks):
    seen = []
    s = set()
    for t in tasks:
        for p in (t["pos"] + t["neg"] + t["pos_q"] + t["neg_q"]
                  + t.get("pos_t", []) + t.get("neg_t", [])):
            if p not in s:
                s.add(p)
                seen.append(p)
    return seen


# ---------------------------------------------------------------------------
# Ground-truth scenes (used for the oracle condition and for checking the VLM)
# ---------------------------------------------------------------------------
_SCENES = {}


def load_scenes(dataset, split):
    """Map relative image path -> scene dict (objects with shape, color, size,
    material, 3d_coords, pixel_coords)."""
    key = (dataset, split)
    if key in _SCENES:
        return _SCENES[key]
    path = ROOT / "data" / dataset / split / f"CLEVR_HANS_scenes_{split}.json"
    with open(path) as f:
        scenes = json.load(f)["scenes"]
    table = {}
    for sc in scenes:
        table[f"data/{dataset}/{split}/images/{sc['image_filename']}"] = sc
    _SCENES[key] = table
    return table


def scene_for(rel_path):
    parts = rel_path.split("/")
    return load_scenes(parts[1], parts[2])[rel_path]


IMG_W, IMG_H = 480, 320  # CLEVR-Hans render size (checked on the files: 480 x 320)


def oracle_objects(rel_path):
    """Ground-truth representation in VLP's format: [shape, size, color, material]
    plus the true bounding box. The position is the box centre, the same kind of
    position the VLM's boxes give, so the oracle and the VLM are compared like
    for like."""
    sc = scene_for(rel_path)
    out = ObjList()
    for o in sc["objects"]:
        x, y, w, h = o["bbox"]
        box = [x / IMG_W, y / IMG_H, (x + w) / IMG_W, (y + h) / IMG_H]
        out.append(ObjRec([o["shape"], o["size"], o["color"], o["material"]],
                          pos=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), box=box))
    return out
