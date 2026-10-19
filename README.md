# OCVLP: Object-Centric Vision-Language Programs

Syed Hamza Mukhtar Bukhari, Ghulam Ishaq Khan Institute (GIKI), Pakistan.
Paper: [`paper/OCVLP_paper.pdf`](paper/OCVLP_paper.pdf)

This code extends [Vision-Language Programs](https://github.com/ml-research/vision-language-programs)
(VLP; Wüst et al., CVPR 2026) so that rules can depend on where objects are. The same VLM that
lists the objects in an image now also returns a bounding box for each one, and VLP's program
grammar gains counting, region and relation predicates (for example
`exists_related(cyan, in_front_of, red)`). The new functions enter the search with a small prior,
so VLP's own rules are still tried first. Nothing is trained.

## Results
Qwen2.5-VL-7B-Instruct, CLEVR-Hans7, 21 tasks (7 classes, 3 random draws each). Mean balanced
accuracy (%) on query images from the test split, where the dataset's planted shortcuts do not hold.

| condition | all 21 tasks | 9 spatial tasks | 12 other tasks |
|---|---:|---:|---:|
| direct prompting | 53.0 | 53.6 | 52.5 |
| VLP as released | 69.5 | 65.6 | 72.5 |
| our object prompt, VLP's language | 69.2 | 62.5 | 74.2 |
| + counting | 75.8 | 77.5 | 74.6 |
| + positions, regions, relations (ours) | **80.2** | **90.8** | 72.3 |
| ground-truth scenes, VLP's language | 74.9 | 63.9 | 83.1 |
| ground-truth scenes, our language | 85.7 | 89.4 | 82.9 |

Tasks are the independent units, so tests are task level. VLP to ours: +10.7 points, 13 tasks
better and 6 worse, exact permutation p = 0.013. Spatial tasks: +25.3 points, p = 0.008.
On CLEVR-Hans3 with VLP's own protocol, our run of VLP scores 86.7% (the paper reports 86.1%).

Known limitations: positions cost 2.3 points on the non-spatial classes, where two tasks picked a
wrong spatial rule, and VLP's variable discovery with this model almost never proposes size words,
so the size part of classes 1, 2, 4 and 7 is out of reach.

## Setup
VLP's code is not included here. Clone the exact commit this work builds on into `vlp/`:

    git clone https://github.com/ml-research/vision-language-programs vlp
    git -C vlp checkout 6474ff7

Put the CLEVR-Hans3 and CLEVR-Hans7 images in `data/` (download them in a browser from the
[CLEVR-Hans page](https://github.com/ml-research/CLEVR-Hans)). Run everything from the repository
root with `PYTHONHASHSEED=0`.

## Layout
| path | what it is |
|---|---|
| `ocvlp/vlp_bridge.py` | imports VLP's search, grammar and PCFG weighting (stubs `ray`, which heap search never uses) |
| `ocvlp/symbols.py` | object entries in VLP's format plus a position, and the parser for the VLM's boxes |
| `ocvlp/dsl.py` | the three rule languages: `vlp`, `count`, `oc` |
| `ocvlp/runner.py` | one task: VLP's search loop with a chosen language and prior weight |
| `ocvlp/data.py` | CLEVR-Hans tasks (protocols `vlp`, `balanced`, `dev`) and ground-truth scenes |
| `ocvlp/prompts.py` | object-centric prompt: VLP's object prompt plus a box per object |
| `ocvlp/prompters.py` | Qwen2.5-VL prompter with an append-only answer cache |
| `ocvlp/collect.py` | GPU job that asks every VLM question the experiments need, in batches |
| `ocvlp/direct.py` | VLP's direct-prompting baseline |
| `ocvlp/experiment.py` | replays cached answers on a CPU and runs every condition |
| `ocvlp/report.py` | tables and task-level paired tests |
| `ocvlp/perception.py` | accuracy of the VLM's boxes against ground truth |
| `kaggle/` | packaging and the job script for a free Kaggle GPU |
| `tests/` | synthetic spatial-rule test and a mock end-to-end run |

## Conditions
Each condition changes one thing from the one above it.

| condition | perception | rule language |
|---|---|---|
| `direct` | VLM answers yes or no directly | none |
| `vlp_original` | VLP's prompt | VLP |
| `oc_hidden` | our prompt, positions hidden | VLP |
| `oc_count` | our prompt, positions hidden | VLP + counting |
| `oc_full` | our prompt with positions | VLP + counting + regions + relations |
| `gt_vlp`, `gt_count`, `gt_oc` | ground-truth scenes | the three languages (upper bound) |

Protocols: `vlp` reproduces VLP's CLEVR-Hans3 loader exactly; `balanced` uses 3 random draws with
negatives spread over all other classes; `dev` uses train images only, and every design choice was
made there.

## Running
```
python tests/test_toy.py              # synthetic check, needs no data
python tests/test_collect_mock.py     # mock GPU job and replay, needs no data
python -m ocvlp.experiment --conditions gt_vlp,gt_count,gt_oc --tasksets hans7_dev --out results/dev_gt.jsonl
python kaggle/prepare_kaggle.py --user <kaggle-user>   # then push the dataset and kernel to Kaggle
python -m ocvlp.experiment --conditions direct,vlp_original,oc_hidden,oc_count,oc_full \
    --tasksets hans7_bal --cache <kaggle-output-dirs> --oc-scale 2 --out results/vlm_hans7_bal.jsonl
python -m ocvlp.report results/*.jsonl --out results/report.md
```

## Design decisions
* **Prior weight of the added functions (`--ext-weight 0.1`).** At full weight the larger grammar
  can use up the 10 second search budget before it reaches a rule that VLP's language finds. Scaling
  the new functions' prior makes the search try VLP's rules first. The value 0.1 was chosen on dev
  tasks only and kept fixed for every later run.
* **Object lists at double image size (`--oc-scale 2`).** At the native 480x320 size Qwen2.5-VL's
  boxes come back stretched vertically. At twice the size, position correlation with ground truth
  is 0.99 (x) and 0.98 (y), and front to behind order between two objects is right 98% of the time.
  VLP's own calls keep the native size, as released.
* **Two query sets.** CLEVR-Hans's train and val splits contain planted shortcuts and its test split
  does not, so every task is scored on both.
* Qwen2.5-VL's default temperature is 1e-6, so VLP's three sampling seeds give identical answers;
  one seed is run and the image draws are varied instead.
