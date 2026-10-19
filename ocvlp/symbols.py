"""Symbolic image representations shared by every condition.

VLP represents an image as a list of object entries, each a list of strings:
    [['cube', 'large', 'gray'], ['sphere', 'small', 'red'], ...]
The name comes first, the properties after it. All of VLP's DSL functions,
its vocabulary collection and its PCFG weighting read exactly this format.

Our extension keeps that format untouched and attaches the object's position
as an attribute. `ObjRec` is a list of strings, so every VLP function sees the
same entry it always saw; our spatial functions additionally read `rec.pos`.
Because the position never appears inside the list, it can never leak into
the vocabulary as a fake "property".
"""

import ast
import json
import math
import re


class ObjRec(list):
    """One object: list of strings [name, prop, ...] plus .pos and .box.

    pos = (x, y): box centre in normalised image coordinates, x to the right,
    y downwards, both in [0, 1]. In CLEVR's camera view a larger y means the
    object is closer to the camera, i.e. further "in front".
    """

    __slots__ = ("pos", "box")

    def __init__(self, items, pos=None, box=None):
        super().__init__(items)
        self.pos = pos
        self.box = box


class ObjList(list):
    """The representation of one image: a list of ObjRec."""


def strip_positions(obj_list):
    """Same entries, positions removed (used by the 'positions hidden' conditions)."""
    return ObjList(ObjRec(list(r)) for r in obj_list)


def to_jsonable(obj_list):
    return [{"entry": list(r), "pos": r.pos, "box": r.box} for r in obj_list]


def from_jsonable(items):
    return ObjList(ObjRec(it["entry"], tuple(it["pos"]) if it["pos"] else None,
                          it.get("box")) for it in items)


# ---------------------------------------------------------------------------
# Qwen2.5-VL returns boxes in pixel coordinates of the image *after* its
# processor resized it to multiples of 28. This is qwen_vl_utils.smart_resize
# with the library defaults, copied here so the laptop needs no transformers.
# ---------------------------------------------------------------------------
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28


def _round_by(n, f):
    return round(n / f) * f


def _ceil_by(n, f):
    return math.ceil(n / f) * f


def _floor_by(n, f):
    return math.floor(n / f) * f


def smart_resize(height, width, factor=IMAGE_FACTOR, min_pixels=MIN_PIXELS,
                 max_pixels=MAX_PIXELS):
    h_bar = max(factor, _round_by(height, factor))
    w_bar = max(factor, _round_by(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by(height / beta, factor)
        w_bar = _floor_by(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by(height * beta, factor)
        w_bar = _ceil_by(width * beta, factor)
    return h_bar, w_bar


# ---------------------------------------------------------------------------
# Parsing the object-centric VLM answer
# ---------------------------------------------------------------------------
def _clean(s):
    return str(s).strip().lower()


def _extract_json_list(text):
    """Return the first JSON-like list in the model's answer, tolerating code
    fences, trailing commas and a cut-off final element."""
    t = text.strip()
    fence = re.search(r"```(?:json|python)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1)
    start = t.find("[")
    if start < 0:
        return []
    t = t[start:]
    for candidate in (t, t[: t.rfind("]") + 1] if "]" in t else t):
        cand = re.sub(r",\s*([\]}])", r"\1", candidate)
        for loader in (json.loads, ast.literal_eval):
            try:
                val = loader(cand)
                if isinstance(val, list):
                    return val
            except Exception:
                pass
    # Cut-off answer: keep every complete {...} element.
    items = []
    for m in re.finditer(r"\{[^{}]*\}", t):
        frag = re.sub(r",\s*}", "}", m.group(0))
        for loader in (json.loads, ast.literal_eval):
            try:
                items.append(loader(frag))
                break
            except Exception:
                pass
    return items


def parse_oc_response(text, img_w, img_h, scale=1):
    """Parse the object-centric answer into an ObjList.

    Expected elements: {"label": ..., "properties": [...], "bbox_2d": [x1,y1,x2,y2]}
    with pixel coordinates in the model's resized frame. Returns
    (ObjList, n_malformed) where n_malformed counts dropped elements.
    """
    rh, rw = smart_resize(img_h, img_w)
    rh, rw = rh * scale, rw * scale  # the frame the model saw
    out = ObjList()
    bad = 0
    for it in _extract_json_list(text):
        if not isinstance(it, dict):
            bad += 1
            continue
        label = it.get("label") or it.get("object") or it.get("name")
        if not label:
            bad += 1
            continue
        props = it.get("properties") or it.get("props") or []
        if isinstance(props, str):
            props = [props]
        entry = [_clean(label)] + [_clean(p) for p in props if str(p).strip()]
        box = it.get("bbox_2d") or it.get("bbox") or it.get("box")
        pos = norm_box = None
        try:
            x1, y1, x2, y2 = [float(v) for v in box]
            # No clipping: the model's boxes can overshoot the frame, and the
            # overshoot still carries which object is further left / in front.
            norm_box = [x1 / rw, y1 / rh, x2 / rw, y2 / rh]
            pos = ((norm_box[0] + norm_box[2]) / 2, (norm_box[1] + norm_box[3]) / 2)
        except Exception:
            bad += 1
        out.append(ObjRec(entry, pos, norm_box))
    return out, bad
