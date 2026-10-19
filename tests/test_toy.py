"""Synthetic check of the whole search path, no dataset or VLM needed.

Scenes are random CLEVR-like object lists. The hidden rule is spatial:
"at least three spheres in the left half of the image". VLP's language has no
way to say "left half", so it should stay near chance on the query images;
the object-centric language should find the rule.

Run:  PYTHONHASHSEED=0 python tests/test_toy.py
"""

import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocvlp.symbols import ObjRec, ObjList  # noqa: E402
from ocvlp.runner import run_task  # noqa: E402
from ocvlp.dsl import in_region  # noqa: E402

SHAPES = ["cube", "sphere", "cylinder"]
COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]


def random_scene(rng, n=None):
    n = n or rng.randint(3, 8)
    return ObjList(
        ObjRec([rng.choice(SHAPES), rng.choice(["large", "small"]), rng.choice(COLORS),
                rng.choice(["rubber", "metal"])], pos=(rng.random(), rng.random()))
        for _ in range(n))


def rule(scene):
    return sum(1 for r in scene if r[0] == "sphere" and in_region(r, "left_half")) >= 3


def make_scene(rng, positive):
    while True:
        s = random_scene(rng)
        if positive:
            # plant three left spheres
            for k in range(3):
                s[k] = ObjRec(["sphere", rng.choice(["large", "small"]), rng.choice(COLORS),
                               rng.choice(["rubber", "metal"])],
                              pos=(rng.random() * 0.45, rng.random()))
        if rule(s) == positive:
            return s


def main():
    rng = random.Random(0)
    store = {}

    def add(name, scene):
        store[name] = scene
        return name

    task = {"task_id": "toy_three_left_spheres",
            "pos": [add(f"p{i}.png", make_scene(rng, True)) for i in range(10)],
            "neg": [add(f"n{i}.png", make_scene(rng, False)) for i in range(10)],
            "pos_q": [add(f"pq{i}.png", make_scene(rng, True)) for i in range(50)],
            "neg_q": [add(f"nq{i}.png", make_scene(rng, False)) for i in range(50)]}

    for kind in ("vlp", "count", "oc"):
        res = run_task(task, kind, store.__getitem__, timeout=10)
        print(f"{kind:5s} train {res['train_acc']:.2f}  query {res['test_acc']:.2f}  "
              f"programs {res['nb_programs']:>7}  {res['program']}")


if __name__ == "__main__":
    main()
