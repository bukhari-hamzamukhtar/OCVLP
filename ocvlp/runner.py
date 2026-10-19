"""Run VLP's program search on one task with a chosen rule language.

This mirrors the loop body of VLP's main.py step by step:
  1. examples = support images (positives first, then negatives)
  2. image representations of the support images (VLP: pre_collect_img_representations)
  3. vocabulary = every object name / property that appears in them
  4. DSL -> CFG (max depth 6) -> PCFG weighted with VLP's 'naive_weighted' rule
  5. heap search with a 10 s budget, keep 10 candidates
  6. best program = highest support accuracy, ties broken by probability
  7. balanced accuracy of the best program on the query images
Only step 4's language differs between our conditions.
"""

import time

from .vlp_bridge import DSL, Arrow, IMG, BOOL, run_algorithm, make_program_checker_with_accuracy
from .dsl import build_language, EXTENSION_FUNCS
from .symbols import strip_positions


def vocabulary_from_representations(reprs):
    """VLP's rule (precollect_vlm_answers.pre_collect_img_representations)."""
    objects, properties = [], []
    for rep in reprs:
        for entry in rep:
            if not isinstance(entry, list) or len(entry) == 0:
                continue
            name = entry[0] if isinstance(entry[0], str) else str(entry[0])
            if name and name not in objects:
                objects.append(name)
            for p in entry[1:]:
                p = p if isinstance(p, str) else str(p)
                if p and p not in properties:
                    properties.append(p)
    return {"objects": objects, "properties": properties, "actions": []}


def add_variables(semantics, types, variables):
    """VLP main.add_variables_to_dsl."""
    from .vlp_bridge import OBJECT, PROPERTY
    for o in variables["objects"]:
        if o not in semantics:
            semantics[o] = o
            types[o] = OBJECT
    for p in variables["properties"]:
        if p not in semantics:
            semantics[p] = p
            types[p] = PROPERTY


def balanced_accuracy(preds_pos, preds_neg):
    pa = sum(1 for p in preds_pos if p is True) / max(1, len(preds_pos))
    na = sum(1 for p in preds_neg if p is False) / max(1, len(preds_neg))
    return (pa + na) / 2


def scale_extension_prior(pcfg, factor):
    """Multiply the weight of every production that uses one of our added
    functions by `factor`, then let VLP's PCFG re-normalise (clean=True).
    factor = 1 leaves VLP's weighting untouched."""
    from method.pcfg import PCFG
    from method.program import BasicPrimitive
    if factor == 1.0:
        return pcfg
    rules = {}
    for S, prods in pcfg.rules.items():
        rules[S] = {}
        for P, (args, w) in prods.items():
            if isinstance(P, BasicPrimitive) and P.primitive in EXTENSION_FUNCS:
                w = w * factor
            rules[S][P] = (args, w)
    return PCFG(start=pcfg.start, rules=rules, max_program_depth=pcfg.max_program_depth,
                clean=True)


def run_task(task, kind, get_objects, timeout=10, max_depth=6, n_candidates=10,
             ext_weight=1.0, verbose=False):
    """task: dict with 'pos', 'neg' (support paths) and 'pos_q', 'neg_q' (query paths).
    get_objects(path) -> ObjList. Returns a result dict."""
    t0 = time.time()
    support = task["pos"] + task["neg"]
    examples = [([p], True) for p in task["pos"]] + [([p], False) for p in task["neg"]]

    reprs = [get_objects(p) for p in support]
    if kind in ("vlp", "count"):
        reprs = [strip_positions(r) for r in reprs]
    variables = vocabulary_from_representations(reprs)

    semantics, types = build_language(kind, get_objects, variables)
    add_variables(semantics, types, variables)
    dsl = DSL(semantics, types)

    cfg = dsl.DSL_to_CFG(Arrow(IMG, BOOL), max_program_depth=max_depth,
                         min_variable_depth=1, upper_bound_type_size=10, n_gram=2)
    img_repr = {"objects": reprs, "properties": reprs, "actions": []}
    pcfg = cfg.CFG_to_PCFG_with_naive_weighted(img_repr, variables)
    if kind in ("count", "oc"):
        pcfg = scale_extension_prior(pcfg, ext_weight)

    checker = make_program_checker_with_accuracy(dsl, examples)
    res = run_algorithm(checker, pcfg, 0, timeout=timeout, n_candidates=n_candidates)
    candidates, search_time, eval_time, nb_programs = res[0], res[1], res[2], res[3]
    if not candidates:
        out = {"task_id": task["task_id"], "kind": kind, "program": None,
               "train_acc": 0.0, "test_acc": 0.5, "nb_programs": nb_programs,
               "vocab": variables, "seconds": time.time() - t0}
        if task.get("pos_t"):
            out["test_acc_t"] = 0.5
        return out

    ranked = sorted(candidates, key=lambda c: (c[0], c[2]), reverse=True)
    train_acc, best, prob = ranked[0]

    def predict(path):
        try:
            out = best.eval_naive(dsl, [path])
        except Exception:
            out = None
        return out if isinstance(out, bool) else None

    preds_pos = [predict(p) for p in task["pos_q"]]
    preds_neg = [predict(p) for p in task["neg_q"]]
    out = {
        "task_id": task["task_id"],
        "kind": kind,
        "ext_weight": ext_weight,
        "program": str(best),
        "probability": prob,
        "train_acc": train_acc,
        "test_acc": balanced_accuracy(preds_pos, preds_neg),
        "preds_pos": preds_pos,
        "preds_neg": preds_neg,
        "candidates": [(round(a, 4), str(p), float(pr)) for a, p, pr in ranked],
        "nb_programs": nb_programs,
        "search_time": round(search_time, 2),
        "eval_time": round(eval_time, 2),
        "vocab": variables,
        "seconds": round(time.time() - t0, 2),
    }
    if task.get("pos_t"):
        pt = [predict(p) for p in task["pos_t"]]
        nt = [predict(p) for p in task["neg_t"]]
        out.update({"test_acc_t": balanced_accuracy(pt, nt), "preds_pos_t": pt, "preds_neg_t": nt})
    if verbose:
        print(f"  [{kind}] {task['task_id']}: train {train_acc:.2f} test {out['test_acc']:.2f} "
              f"programs {nb_programs}  {best}")
    return out
