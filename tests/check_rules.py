"""Can our language express the true CLEVR-Hans7 class rules, with our
image-space geometry? Checked on ground-truth scenes of the DEV pool
(train ids >= 1500), never on val.

For each class the README rule is written with our DSL helpers and evaluated
on every dev image: 'hit' = share of the class's images where the rule holds,
'false' = share of other classes' images where it holds. A faithful language
and geometry give hit ~100% and false ~0%.

Run:  PYTHONHASHSEED=0 python tests/check_rules.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocvlp import data as D  # noqa: E402
from ocvlp.dsl import in_region, count_related_value  # noqa: E402


def has(objs, shape=None, **attrs):
    for r in objs:
        if shape and r[0] != shape:
            continue
        if all(v in r[1:] for v in attrs.values()):
            return True
    return False


def count(objs, shape=None, props=(), region=None):
    return sum(1 for r in objs if (shape is None or r[0] == shape)
               and all(p in r[1:] for p in props)
               and (region is None or in_region(r, region)))


RULES = {
    0: lambda o: has(o, "cube", s="large") and has(o, "cylinder", s="large"),
    1: lambda o: has(o, "cube", s="small", m="metal") and has(o, "sphere", s="small"),
    2: lambda o: count_related_value(o, "cyan", "in_front_of", "red") >= 2,
    3: lambda o: count(o, props=["small"]) >= 5 and has(o, a="small", c="green")
    and has(o, a="small", c="brown") and has(o, a="small", c="purple"),
    4: lambda o: count(o, "sphere", region="left_half") >= 3,
    5: lambda o: count(o, "cylinder", ["metal"], "right_half") >= 3,
    6: lambda o: has(o, "sphere", s="large", c="blue") and has(o, "sphere", s="small", c="yellow"),
}


def main():
    dataset = "CLEVR-Hans7"
    ids = range(D.DEV_SPLIT_ID, D.IMAGES_PER_CLASS["train"])
    images = {c: [D.img_path(dataset, "train", c, i) for i in ids] for c in range(7)}
    print(f"{'class':>5} {'rule':60s} {'hit':>6} {'false':>6}")
    for c, rule in RULES.items():
        hit = sum(rule(D.oracle_objects(p)) for p in images[c]) / len(images[c])
        others = [p for k in range(7) if k != c for p in images[k]]
        false = sum(rule(D.oracle_objects(p)) for p in others) / len(others)
        print(f"{c + 1:>5} {D.CLASS_RULES[dataset][c][:60]:60s} {100 * hit:6.1f} {100 * false:6.1f}")


if __name__ == "__main__":
    main()
