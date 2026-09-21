# DATASET_AND_SPLITS.md

## TGIF

The whole project runs on **TGIF** (Text-Guided Image Forgery). Only one subset is used
for the main line: **`sd2-sp`**.

### What the subset names mean

| token | meaning |
|---|---|
| `sd2` | the generator — Stable Diffusion 2 inpainting |
| `sp` | **splice**: the edited region is composited into the original, so pixels outside the mask are *unchanged* |
| `fr` | **full reconstruction**: the whole image is passed through the generator, so every pixel changes |

### Why `sd2-sp` is the main experimental domain

Phase 0 verification on 2058 triples measured the median mean absolute difference
*outside* the mask at **0.0030** — i.e. a pixel-exact local splice. That property is
what makes the experiment interpretable:

- the ground-truth mask really does mark the only altered region, so a spatial claim can
  be scored;
- `fr` variants would change every pixel, making "is the cited region the manipulated
  one" meaningless;
- a real/fake pair differs *only* inside the mask, which is what lets the semantic audit
  use the paired real image as a comparison.

### Why CocoGlide was rejected

An earlier attempt used CocoGlide. It was abandoned because its manipulations did not
give the clean local-splice structure the mechanism experiments need. Recorded in
`docs/EXPERIMENTAL_HISTORY.md`; no CocoGlide result is cited anywhere.

## Triple structure and pairing

Each sample is a triple:

```
fake image   the manipulated photograph
real image   the unmodified source, same scene, same size
mask         TGIF ps_mask PNG marking the edited region
```

Pairing is by `coco_id`, and **every fake has exactly one paired real**. All decision
metrics are computed on `184 fake + 184 paired real = 368` images, so recall and FPR come
from a balanced, paired set.

### Mask binarization — a preprocessing decision, not a dataset property

TGIF `ps_mask` PNGs are **RGB with feathered edges**. They are binarized at
**`> 127`** (`configs/exp01.yaml: mask_binarize_threshold`). This is an experiment
choice, stated as such in the config, and it is held fixed across every round. Any
re-analysis must use the same threshold or the grid cells will move.

### Mask type

| type | count in the frozen subset |
|---|---|
| `bbox` | 73 |
| `segm` | 111 |

## Source-level splitting

**Everything is split at the source level, never at the image level.** A `coco_id`
appears in exactly one subset. This matters because a fake and its paired real share a
`coco_id`; splitting by image would put the two halves of one pair on opposite sides.

The bootstrap exploits the same property: in the TruFor rounds each source contributes
exactly one variant, so `n_samples == n_sources == 184` and cluster dependence is
eliminated rather than modelled.

## Frozen subsets

| subset | size | seed | used by |
|---|---|---|---|
| **184-source main subset** | 184 fake + 184 paired real, 184 unique `coco_id` | frozen in `runs/trufor_mllm/trufor_mllm_frozen.json` | every TruFor→MLLM round (T02–F01) and X01 |
| **200-source standalone subset** | 200 fake + 200 paired real | frozen in `runs/trufor_feasibility/trufor_feasibility_frozen.json` | T01 TruFor standalone feasibility only |
| **mirror-eligible subset** | 70 fake / 80 real | frozen in `runs/trufor_abstraction/…_frozen.json` before any mirror result | S01 `E4`, F01 `B4` |
| **semantic audit subset** | 100 fake + 50 real | `20260918`, frozen in `runs/explainability/x3/x2_semantic_audit_frozen.json` | X02, X03 |

### Calibration / validation separation

The ELA stage used a calibration subset for its window readout
(`runs/calibration_tgif/`, `runs/readout_cv/`) and a separate validation subset for
confirmation (`runs/v4_stageA`, `runs/v4_stageB`). The calibration set was never reused
for confirmation.

### Cumulative source usage

**677 distinct `coco_id`** were consumed across the project (243 + 184 + 150 + 100).

An important negative check: the TGIF **training** split's sd2 pool contains 1558
`coco_id` and its intersection with those 677 is **zero**. The three TGIF generators
share the same 2242 `coco_id`, which is why switching generator could *not* have bought
fresh sources.

### Statement required whenever TruFor is described

> The TGIF training split was used only as an untouched evaluation pool for the external
> forensic model; no model was trained or tuned on these images.

## Semantic audit stratification (X02)

The 100 fake were drawn proportionally over **GT area quartile × grid spread**
(quartile cuts `0.0149 / 0.0383 / 0.1089`):

| factor | composition |
|---|---|
| area quartile | Q1 25, Q2 26, Q3 24, Q4 25 |
| grid spread | 1 cell 23, 2–3 cells 48, ≥4 cells 29 |
| mask type | bbox 40, segm 60 |
| categories | 12 distinct |

The 50 real were drawn **only from the 84 sources not used as fake units**, because a
fake unit shows the annotator that source's real image as a comparison photograph;
reusing it as a real unit would mean the annotator had already seen it.

Selection used **no** verdict, explanation-quality or spatial-correctness criterion.

## `docs/data_manifest.csv`

One row per source in the frozen 184 subset, generated from the frozen protocols:

| column | meaning |
|---|---|
| `sample_id` | stable identifier used in every verdicts file |
| `coco_id` | source id; the unit of splitting |
| `category`, `mask_type`, `tamper_ratio` | sample properties |
| `fake_relpath`, `real_relpath`, `mask_relpath` | paths **relative to the TGIF root**, never absolute |
| `split` | which frozen subset |
| `used_in_experiment` | which rounds consumed this source |

Paths are deliberately relative: point `configs/exp01.yaml: datasets.tgif_sp.root` at
your own TGIF copy and the manifest stays valid. The 12 GB of imagery is **not** in this
repository.

### tamper_ratio distribution (184 sources)

| quartile | value |
|---|---|
| q1 | 0.0149 |
| median | 0.0383 |
| q3 | 0.1089 |

The manipulated regions are **small** — a median of under 4% of the image. That fact
drives several findings, notably that the 7B's semantic failures worsen as the region
shrinks.
