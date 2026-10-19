"""Direct prompting baseline, ported line for line from VLP's baseline.py
(eval()), without that file's module-level imports (matplotlib, InternVL, ...).

1. One call with all support images and prompts/baseline_prompt.txt: the VLM
   writes the rule.
2. One yes/no call per query image: "Given the rule '...', determine if the
   image follows the rule or not."
3. Balanced accuracy, with VLP's exact string test ("yes" in answer and "no"
   not in answer).
"""

import ast
import re

from .vlp_bridge import VLP_DIR


def parse_response(response, dict_key="rule"):
    code_block = re.search(r"```python(.*?)```", response, re.DOTALL)
    code = code_block.group(1).strip() if code_block else response.strip()
    dict_match = re.search(r"=\s*(\{.*\})", code, re.DOTALL)
    if not dict_match:
        raise ValueError("No dictionary found after '=' in the response.")
    parsed = ast.literal_eval(dict_match.group(1))
    if dict_key in parsed:
        return parsed[dict_key]
    return parsed


def rule_prompt(task):
    prompt = (VLP_DIR / "prompts" / "baseline_prompt.txt").read_text(encoding="utf-8")
    n = len(task["pos"]) + len(task["neg"])
    prompt = prompt.replace("{n}", str(n))
    prompt = prompt.replace("{m}", str(len(task["pos"])))
    prompt = prompt.replace("{o}", str(len(task["neg"])))
    return prompt


def test_prompt(rule):
    return (f"Given the rule '{rule}', determine if the image follows the rule or not. "
            f"Answer with 'Yes' or 'No', nothing else.")


def get_rule(prompter, task):
    raw = prompter.prompt_with_images(prompt_text=rule_prompt(task),
                                      paths=task["pos"] + task["neg"],
                                      max_new_tokens=5000)
    try:
        rule = parse_response(raw, dict_key="rule")
    except Exception:
        rule = "Error parsing response"
    return raw, rule


def _score(prompter, tp, pos_imgs, neg_imgs):
    pos = [prompter.prompt_with_images(prompt_text=tp, paths=[p]) for p in pos_imgs]
    neg = [prompter.prompt_with_images(prompt_text=tp, paths=[p]) for p in neg_imgs]
    pos_ok = [("yes" in r.lower() and "no" not in r.lower()) for r in pos]
    neg_ok = [("no" in r.lower() and "yes" not in r.lower()) for r in neg]
    acc = (sum(pos_ok) / len(pos) + sum(neg_ok) / len(neg)) / 2
    # predictions: True = "follows the rule" in VLP's scoring
    return acc, pos, neg, pos_ok, [not ok for ok in neg_ok]


def evaluate(prompter, task):
    raw, rule = get_rule(prompter, task)
    tp = test_prompt(rule)
    acc, pos, neg, pp, pn = _score(prompter, tp, task["pos_q"], task["neg_q"])
    out = {"task_id": task["task_id"], "kind": "direct", "rule": rule, "test_acc": acc,
           "pos_answers": pos, "neg_answers": neg, "preds_pos": pp, "preds_neg": pn}
    if task.get("pos_t"):
        acc_t, pos_t, neg_t, ppt, pnt = _score(prompter, tp, task["pos_t"], task["neg_t"])
        out.update({"test_acc_t": acc_t, "pos_answers_t": pos_t, "neg_answers_t": neg_t,
                    "preds_pos_t": ppt, "preds_neg_t": pnt})
    return out
