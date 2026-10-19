"""The three rule languages compared in this project.

  vlp    VLP's own CLEVR-Hans3 language, imported unchanged from VLP's code:
         exists_object, exists_object_with_property(ies), exists_property(ies),
         and / or / xor / not.
  count  vlp + counting: count_object, count_property,
         count_object_with_property, gt?, eq?, integers 0..6.
  oc     count + object geometry (this project): region counts
         (left/right/front/back half of the image) and relations between
         objects (left of, right of, in front of, behind).

Only the language changes between conditions; the program search, the PCFG
weighting, the time budget and the images are identical.
"""

from .vlp_bridge import Arrow, List, PrimitiveType, IMG, OBJECT, PROPERTY, INT, BOOL, STRING
from .symbols import strip_positions

LLS = List(List(STRING))           # VLP's image representation type
REGION = PrimitiveType("region")
RELATION = PrimitiveType("relation")

REGIONS = ["left_half", "right_half", "front_half", "back_half"]
RELATIONS = ["left_of", "right_of", "in_front_of", "behind"]
INTS = [str(i) for i in range(7)]  # same integer constants as VLP's cocologic DSL

# Functions this project adds on top of VLP's language. Their prior weight can
# be scaled down (runner.run_task(ext_weight=...)) so the search still tries
# VLP's own rules first; integer / region / relation constants are only
# reachable through these functions, so scaling the functions is enough.
COUNT_FUNCS = ["count_object", "count_property", "count_object_with_property", "gt?", "eq?"]
OC_FUNCS = ["count_object_in_region", "count_property_in_region",
            "count_object_with_property_in_region", "count_related", "exists_related"]
EXTENSION_FUNCS = set(COUNT_FUNCS + OC_FUNCS)


# ---------------------------------------------------------------------------
# Geometry (normalised image coordinates: x right, y down; larger y = closer
# to the camera = "in front" in CLEVR's view)
# ---------------------------------------------------------------------------
def in_region(rec, region):
    if rec.pos is None:
        return False
    x, y = rec.pos
    if region == "left_half":
        return x < 0.5
    if region == "right_half":
        return x >= 0.5
    if region == "front_half":
        return y >= 0.5
    if region == "back_half":
        return y < 0.5
    raise ValueError(region)


def related(a, rel, b, margin=0.0):
    if a.pos is None or b.pos is None:
        return False
    (ax, ay), (bx, by) = a.pos, b.pos
    if rel == "left_of":
        return ax < bx - margin
    if rel == "right_of":
        return ax > bx + margin
    if rel == "in_front_of":
        return ay > by + margin
    if rel == "behind":
        return ay < by - margin
    raise ValueError(rel)


def _has_prop(rec, p):
    return p in rec[1:]


def _is(rec, o):
    return len(rec) > 0 and rec[0] == o


# ---------------------------------------------------------------------------
# Counting primitives (curried, VLP style)
# ---------------------------------------------------------------------------
def _count_object(objs):
    return lambda o: sum(1 for r in objs if _is(r, o))


def _count_property(objs):
    return lambda p: sum(1 for r in objs if _has_prop(r, p))


def _count_object_with_property(objs):
    return lambda o: lambda p: sum(1 for r in objs if _is(r, o) and _has_prop(r, p))


# ---------------------------------------------------------------------------
# Region primitives
# ---------------------------------------------------------------------------
def _count_object_in_region(objs):
    return lambda o: lambda reg: sum(1 for r in objs if _is(r, o) and in_region(r, reg))


def _count_property_in_region(objs):
    return lambda p: lambda reg: sum(1 for r in objs if _has_prop(r, p) and in_region(r, reg))


def _count_object_with_property_in_region(objs):
    return lambda o: lambda p: lambda reg: sum(
        1 for r in objs if _is(r, o) and _has_prop(r, p) and in_region(r, reg))


# ---------------------------------------------------------------------------
# Relation primitives
# ---------------------------------------------------------------------------
def count_related_value(objs, p1, rel, p2):
    """Largest number of p2-objects that a single p1-object stands in `rel` to.
    Example: count_related(cyan, in_front_of, red) = 2 when some cyan object is
    in front of two red objects."""
    best = 0
    for a in objs:
        if not _has_prop(a, p1):
            continue
        n = sum(1 for b in objs if b is not a and _has_prop(b, p2) and related(a, rel, b))
        best = max(best, n)
    return best


def _count_related(objs):
    return lambda p1: lambda rel: lambda p2: count_related_value(objs, p1, rel, p2)


def _exists_related(objs):
    return lambda p1: lambda rel: lambda p2: count_related_value(objs, p1, rel, p2) > 0


def _gt(x):
    return lambda y: x > y


def _eq(x):
    return lambda y: x == y


# ---------------------------------------------------------------------------
# Build a (semantics, primitive_types) pair for one language
# ---------------------------------------------------------------------------
def vlp_base_semantics(model_name, prompter, variables):
    """VLP's CLEVR-Hans3 language exactly as released (includes VLP's own
    get_objects, which prompts the VLM with VLP's object prompt)."""
    from method.DSL.dsl_with_img_repr import get_dsl
    return get_dsl(model_name, "CLEVR-Hans3-unconfounded", prompter, variables)


def build_language(kind, get_objects, variables, model_name="none", prompter=None):
    """kind in {'vlp', 'count', 'oc'}; get_objects(img_path) -> ObjList."""
    semantics, types = vlp_base_semantics(model_name, prompter, variables)

    if kind in ("vlp", "count"):
        semantics["get_objects"] = lambda img: strip_positions(get_objects(img))
    else:
        semantics["get_objects"] = get_objects
    types["get_objects"] = Arrow(IMG, LLS)

    if kind in ("count", "oc"):
        semantics.update({
            "count_object": _count_object,
            "count_property": _count_property,
            "count_object_with_property": _count_object_with_property,
            "gt?": _gt,
            "eq?": _eq,
        })
        types.update({
            "count_object": Arrow(LLS, Arrow(OBJECT, INT)),
            "count_property": Arrow(LLS, Arrow(PROPERTY, INT)),
            "count_object_with_property": Arrow(LLS, Arrow(OBJECT, Arrow(PROPERTY, INT))),
            "gt?": Arrow(INT, Arrow(INT, BOOL)),
            "eq?": Arrow(INT, Arrow(INT, BOOL)),
        })
        for i in INTS:
            semantics[i] = int(i)
            types[i] = INT

    if kind == "oc":
        semantics.update({
            "count_object_in_region": _count_object_in_region,
            "count_property_in_region": _count_property_in_region,
            "count_object_with_property_in_region": _count_object_with_property_in_region,
            "count_related": _count_related,
            "exists_related": _exists_related,
        })
        types.update({
            "count_object_in_region": Arrow(LLS, Arrow(OBJECT, Arrow(REGION, INT))),
            "count_property_in_region": Arrow(LLS, Arrow(PROPERTY, Arrow(REGION, INT))),
            "count_object_with_property_in_region": Arrow(
                LLS, Arrow(OBJECT, Arrow(PROPERTY, Arrow(REGION, INT)))),
            "count_related": Arrow(LLS, Arrow(PROPERTY, Arrow(RELATION, Arrow(PROPERTY, INT)))),
            "exists_related": Arrow(LLS, Arrow(PROPERTY, Arrow(RELATION, Arrow(PROPERTY, BOOL)))),
        })
        for r in REGIONS:
            semantics[r] = r
            types[r] = REGION
        for r in RELATIONS:
            semantics[r] = r
            types[r] = RELATION

    if kind not in ("vlp", "count", "oc"):
        raise ValueError(kind)
    return semantics, types
