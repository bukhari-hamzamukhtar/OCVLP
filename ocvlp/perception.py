"""How good is the VLM's perception? Compare cached answers with ground truth.

For every image with an object-centric answer (at the chosen image scale):
  * object count error (VLM objects minus true objects)
  * attribute-matched objects: a VLM object and a true object are paired when
    their (shape, colour) is unique in both lists; positions play no part in
    the pairing, so position errors are measured without bias
  * for paired objects: correlation and linear fit of VLM vs true centre
    coordinates, agreement on "left half" / "front half", and, for every two
    paired objects in the same image, agreement on which is further left and
    which is further in front
For VLP's own answers (no boxes) only the object count is compared.

  PYTHONHASHSEED=0 python -m ocvlp.perception --cache kaggle/outputs/smoke --scale 1
"""

import argparse
import ast
import itertools
import statistics as st
from collections import defaultdict

from PIL import Image

from . import data as D
from .experiment import load_vlm
from .symbols import parse_oc_response

COLORS = {"gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"}


def _corr(a, b):
    ma, mb = st.mean(a), st.mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = (sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)) ** 0.5
    return num / den if den else float("nan")


def _fit(a, b):
    ma, mb = st.mean(a), st.mean(b)
    s = sum((x - ma) * (y - mb) for x, y in zip(a, b)) / sum((x - ma) ** 2 for x in a)
    return s, mb - s * ma


def attribute_pairs(vlm, gt):
    """Pair objects whose (shape, colour) is unique in both lists."""
    gk = defaultdict(list)
    for g in gt:
        gk[(g[0], g[2])].append(g)
    vk = defaultdict(list)
    for v in vlm:
        cols = [p for p in v[1:] if p in COLORS]
        if len(cols) == 1 and v.pos is not None:
            vk[(v[0], cols[0])].append(v)
    return [(vk[k][0], gk[k][0]) for k in gk if len(gk[k]) == 1 and len(vk.get(k, [])) == 1]


def measure(cache, scale=1):
    s = defaultdict(list)
    per_image_pairs = []
    sizes = {}
    for rec in cache.table.values():
        prompt, paths = rec["prompt"], rec["paths"]
        if len(paths) != 1 or "/images/" not in paths[0]:
            continue
        path = paths[0]
        try:
            gt = D.oracle_objects(path)
        except Exception:
            continue
        if "List every object" in prompt and rec.get("image_scale", 1) == scale:
            if path not in sizes:
                with Image.open(D.abs_path(path)) as im:
                    sizes[path] = im.size
            w, h = sizes[path]
            objs, bad = parse_oc_response(rec["response"], w, h, scale)
            s["images"].append(1)
            s["unparsed_or_empty"].append(1 if len(objs) == 0 else 0)
            s["count_error"].append(len(objs) - len(gt))
            s["abs_count_error"].append(abs(len(objs) - len(gt)))
            pairs = attribute_pairs(objs, gt)
            per_image_pairs.append(pairs)
            for v, g in pairs:
                s["shape_ok"].append(1)  # shape is part of the pairing key
                props = set(v[1:])
                s["size_named_ok"].append(1 if g[1] in props else 0)
                s["material_named_ok"].append(1 if g[3] in props else 0)
        elif "Identify objects and their properties" in prompt:
            try:
                body = rec["response"].replace("\n", "").split("objects =")[-1].split("```")[0]
                n = len([e for e in ast.literal_eval(body) if isinstance(e, list) and e])
                s["vlp_abs_count_error"].append(abs(n - len(gt)))
                s["vlp_unparsed_share"].append(0)
            except Exception:
                s["vlp_unparsed_share"].append(1)

    out = {k: (sum(v) / len(v) if k != "images" else len(v)) for k, v in s.items()}
    flat = [p for ps in per_image_pairs for p in ps]
    if flat:
        gx = [g.pos[0] for v, g in flat]
        gy = [g.pos[1] for v, g in flat]
        vx = [v.pos[0] for v, g in flat]
        vy = [v.pos[1] for v, g in flat]
        out["paired_objects"] = len(flat)
        out["x_corr"] = _corr(gx, vx)
        out["y_corr"] = _corr(gy, vy)
        out["x_fit_slope"], out["x_fit_offset"] = _fit(gx, vx)
        out["y_fit_slope"], out["y_fit_offset"] = _fit(gy, vy)
        out["left_half_agree"] = st.mean([(g.pos[0] < 0.5) == (v.pos[0] < 0.5) for v, g in flat])
        out["front_half_agree"] = st.mean([(g.pos[1] >= 0.5) == (v.pos[1] >= 0.5) for v, g in flat])
        lr = fb = n = 0
        for ps in per_image_pairs:
            for (v1, g1), (v2, g2) in itertools.combinations(ps, 2):
                n += 1
                lr += (g1.pos[0] < g2.pos[0]) == (v1.pos[0] < v2.pos[0])
                fb += (g1.pos[1] < g2.pos[1]) == (v1.pos[1] < v2.pos[1])
        if n:
            out["left_right_order_agree"] = lr / n
            out["front_behind_order_agree"] = fb / n
            out["object_pairs"] = n
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="kaggle/outputs")
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cache, _ = load_vlm([str(D.ROOT / c) for c in a.cache.split(",")])
    res = measure(cache, a.scale)
    lines = [f"| measure (object-centric answers, scale {a.scale}) | value |", "|---|---:|"]
    for k, v in res.items():
        lines.append(f"| {k} | {v:.3f} |" if isinstance(v, float) else f"| {k} | {v} |")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
