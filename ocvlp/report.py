"""Tables and paired tests from experiment results.

  python -m ocvlp.report results/*.jsonl --out results/report.md

Accuracy = VLP's balanced accuracy on the query images of a task, averaged
over tasks. Two query sets are reported separately:
  test_acc    queries from val (VLP's protocol; CLEVR-Hans shortcuts still pay)
  test_acc_t  queries from the test split (unconfounded; only true rules pay)
Paired tests match the same task instance (task_id, seed) across two conditions:
  * sign test on per-task accuracy differences (exact binomial, two-sided)
  * McNemar on per-image correctness, pooled over all query images
"""

import argparse
import glob
import json
import math
from collections import defaultdict

SPATIAL = {"CLEVR-Hans7": {2, 4, 5}, "CLEVR-Hans3": set()}
ORDER = ["direct", "vlp_original", "oc_hidden", "oc_count", "oc_full",
         "gt_vlp", "gt_count", "gt_oc"]
PAIRS = [("vlp_original", "oc_hidden"), ("oc_hidden", "oc_count"), ("oc_count", "oc_full"),
         ("vlp_original", "oc_full"), ("direct", "oc_full"),
         ("gt_vlp", "gt_count"), ("gt_count", "gt_oc")]


def load(paths):
    rows = []
    for p in paths:
        for f in glob.glob(p):
            rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
    last = {}
    for r in rows:  # keep the last row per (condition, task, seed)
        last[(r["condition"], r["task_id"], r["seed"])] = r
    return list(last.values())


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    k = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def permutation_p(diffs):
    """Exact two-sided paired sign-flip test on per-task differences.

    Tasks are the independent units (all images of a task share one learned
    rule), so this is the task-level test. Balanced accuracies here are
    multiples of 1/40 or 1/20, so the null distribution of the summed
    differences is built exactly on an integer grid."""
    scale = 120
    ints = [round(d * scale) for d in diffs]
    if any(abs(d * scale - i) > 1e-6 for d, i in zip(diffs, ints)):
        raise ValueError("differences are not on the expected grid")
    ints = [abs(i) for i in ints if i != 0]
    if not ints:
        return 1.0
    dist = {0: 1}
    for v in ints:
        nxt = {}
        for s, n in dist.items():
            nxt[s + v] = nxt.get(s + v, 0) + n
            nxt[s - v] = nxt.get(s - v, 0) + n
        dist = nxt
    obs = abs(round(sum(d * scale for d in diffs)))
    return sum(n for s, n in dist.items() if abs(s) >= obs) / 2 ** len(ints)


def mcnemar(b, c):
    """b = A right / B wrong, c = A wrong / B right. Normal approximation."""
    if b + c == 0:
        return 0.0, 1.0
    z = (c - b) / math.sqrt(b + c)
    return z, math.erfc(abs(z) / math.sqrt(2))


def correct_vector(r, suffix=""):
    pos = [1 if p is True else 0 for p in r.get("preds_pos" + suffix, [])]
    neg = [1 if p is False else 0 for p in r.get("preds_neg" + suffix, [])]
    return pos + neg


def fmt(x):
    return "--" if x is None else f"{100 * x:.1f}"


def mean(v):
    return sum(v) / len(v) if v else None


def tables(rows, metric, suffix, title):
    rows = [r for r in rows if metric in r]
    lines = []
    if not rows:
        return lines
    by = defaultdict(list)
    for r in rows:
        by[(r["taskset"], r["condition"])].append(r)
    tasksets = sorted({r["taskset"] for r in rows})
    conds = [c for c in ORDER if any(r["condition"] == c for r in rows)]

    lines.append(f"## {title}: mean balanced accuracy (%)")
    lines.append("")
    lines.append("| condition | " + " | ".join(tasksets) + " |")
    lines.append("|---|" + "---:|" * len(tasksets))
    for c in conds:
        cells = []
        for ts in tasksets:
            rs = by.get((ts, c), [])
            if not rs:
                cells.append("--")
                continue
            per_draw = defaultdict(list)
            for r in rs:
                per_draw[r["task_id"].split(":")[2] + "/" + str(r["seed"])].append(r[metric])
            dm = [mean(v) for v in per_draw.values()]
            sd = None
            if len(dm) > 1:
                mu = mean(dm)
                sd = (sum((x - mu) ** 2 for x in dm) / (len(dm) - 1)) ** 0.5
            cell = fmt(mean([r[metric] for r in rs]))
            if sd is not None:
                cell += f" ± {100 * sd:.1f}"
            cells.append(cell + f" (n={len(rs)})")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")

    for ts in [t for t in tasksets if t.startswith("hans7")]:
        lines.append("")
        lines.append(f"### {title}, {ts}: per class (%), spatial classes marked *")
        lines.append("")
        classes = sorted({r["cls"] for r in rows if r["taskset"] == ts})
        head = [f"{c + 1}{'*' if c in SPATIAL['CLEVR-Hans7'] else ''}" for c in classes]
        lines.append("| condition | " + " | ".join(head) + " | spatial | non-spatial |")
        lines.append("|---|" + "---:|" * (len(classes) + 2))
        for c in conds:
            rs = by.get((ts, c), [])
            if not rs:
                continue
            cells = [fmt(mean([r[metric] for r in rs if r["cls"] == k])) for k in classes]
            cells.append(fmt(mean([r[metric] for r in rs if r["cls"] in SPATIAL["CLEVR-Hans7"]])))
            cells.append(fmt(mean([r[metric] for r in rs if r["cls"] not in SPATIAL["CLEVR-Hans7"]])))
            lines.append(f"| {c} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(f"### {title}: paired comparisons (same task instances)")
    lines.append("")
    lines.append("Task-level tests (tasks are independent): sign test on better / worse, exact "
                 "permutation test on the differences. McNemar pools images, which share one "
                 "rule per task, so it overstates significance; it is shown for reference.")
    lines.append("")
    lines.append("| taskset | A -> B | tasks | mean diff (pp) | B better / worse | sign p "
                 "| perm p | images b / c | McNemar z | p |")
    lines.append("|---|---|---:|---:|---|---:|---:|---|---:|---:|")
    groups = []
    for ts in tasksets:
        groups.append((ts, ts, None))
        if ts.startswith("hans7"):  # the hypothesis is about the spatial classes
            groups.append((ts, f"{ts} spatial", True))
            groups.append((ts, f"{ts} non-spatial", False))
    for ts, label, spatial in groups:
        for ca, cb in PAIRS:
            A = {(r["task_id"], r["seed"]): r for r in by.get((ts, ca), [])}
            B = {(r["task_id"], r["seed"]): r for r in by.get((ts, cb), [])}
            keys = sorted(k for k in set(A) & set(B)
                          if spatial is None
                          or (A[k]["cls"] in SPATIAL[A[k]["dataset"]]) == spatial)
            if not keys:
                continue
            diffs = [B[k][metric] - A[k][metric] for k in keys]
            up = sum(1 for d in diffs if d > 1e-9)
            down = sum(1 for d in diffs if d < -1e-9)
            p_sign = binom_two_sided(up, up + down)
            p_perm = permutation_p(diffs)
            bb = cc = 0
            for k in keys:
                va, vb = correct_vector(A[k], suffix), correct_vector(B[k], suffix)
                if len(va) != len(vb):
                    continue
                bb += sum(1 for x, y in zip(va, vb) if x == 1 and y == 0)
                cc += sum(1 for x, y in zip(va, vb) if x == 0 and y == 1)
            z, p = mcnemar(bb, cc)
            lines.append(f"| {label} | {ca} -> {cb} | {len(keys)} | {100 * mean(diffs):+.1f} "
                         f"| {up} / {down} | {p_sign:.3f} | {p_perm:.3f} | {bb} / {cc} "
                         f"| {z:+.2f} | {p:.3f} |")
    lines.append("")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = load(a.files)
    lines = tables(rows, "test_acc", "", "Queries from val (VLP's protocol; shortcuts still pay)")
    lines += tables(rows, "test_acc_t", "_t", "Queries from test (unconfounded; only true rules pay)")
    text = "\n".join(lines)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
