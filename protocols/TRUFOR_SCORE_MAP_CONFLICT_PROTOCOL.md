# TRUFOR_SCORE_MAP_CONFLICT_PROTOCOL.md

Frozen protocol for the **score–map conflict causal intervention**.
Frozen **before** any inference. Machine-readable twin:
`runs/trufor_conflict/trufor_score_map_conflict_frozen.json`
(protocol sha1 `f65614514b090bd12eca59a139ee1963e6921021`)

## Question

When TruFor's scalar score and its spatial map disagree, which governs the MLLM's
final verdict? Tests the mechanism proposed after the 2×2 round: that the model
reads a small / visually unimpressive localization map as counter-evidence against a
high scalar score.

## Design principle

**The image-level score is always the target's own score.** Only the accompanying
visualization changes. Every fake contrast therefore isolates a property of the map
while the scalar evidence is held fixed — this is what separates F3 from the earlier
T2 (which swapped score *and* map together).

## Samples

184 sources, one fake + paired real each, inherited unchanged from
`trufor_score_map_ablation_frozen.json` (sha1 `8f65faf52921…`).
`n_samples == n_sources == 184`, so no variant clustering.

## Conditions

| id | label | map shown | score |
|---|---|---|---|
| **F1** own | fake | raw TruFor map | own |
| **F2** blank | fake | all-zero map | own |
| **F3** donor | fake | another source's map | **own (retained)** |
| **F4** shifted | fake | own map, cyclic shift 50 % width | own |
| **F5** amplified | fake | `clip(raw×2, 0, 1)` | own |
| **F6** binary | fake | `raw > 0.5` | own |
| **R1** own | real | raw TruFor map | own |
| **R2** blank | real | all-zero map | own |
| **R3** donor | real | another real's map | own |

**F5 and F6 change only how the map is DISPLAYED.** They are not new TruFor outputs
and must never be reported as improved tool performance. The `>0.5` threshold is a
visual-representation choice, never a performance tuning knob.

## Transform verification (on a real map, 1024×777)

| variant | min | max | mean | frac > 0.5 |
|---|---|---|---|---|
| own | 0.0000 | 0.9976 | 0.02394 | 0.02275 |
| blank | 0.0000 | 0.0000 | 0.00000 | 0.00000 |
| shifted | 0.0000 | 0.9976 | **0.02394** | **0.02275** |
| amplified | 0.0000 | 1.0000 | 0.02988 | 0.02811 |
| binary | 0.0000 | 1.0000 | 0.02275 | 0.02275 |

**`shifted` preserves the value multiset exactly** (verified by sorted-array
equality) and differs spatially from `own`. No renormalisation is applied. This makes
F4 the cleanest correspondence test: identical map statistics, broken alignment —
unlike the donor map, which differs in texture as well as position.

Note the scale of the problem this exposes: the mean map value is 0.024 and only
2.3 % of pixels exceed 0.5. The "map" a 0.999-scoring fake comes with is almost
entirely dark.

## Prompt

All conditions supply both a score and a map image, so the **frozen C11 prompt is
reused verbatim**: sha1 `2f1f3ef0de63100006844f491c9d027efa0a9ef1` (matches the 2×2
freeze, contains no Stage-A summary).

The prompt never mentions blank / shifted / amplified / binary / donor. The model is
told only that a localization map is provided.

Consequence worth noting: **F1 is an exact protocol replicate of the 2×2 C11 cell**,
so it doubles as a replication check rather than a reused number.

## Donor mapping

Label-matched, `donor_coco_id ≠ target_coco_id`, never self, frozen before
inference. 184 assigned, 0 same-coco/self, and **identical to the earlier round's
mapping (184/184)**.

## Metrics and contrasts

Per condition: fake recall, real FPR. Paired against F1 (or R1) with
**effect size → source-level bootstrap 95 % CI → McNemar p** (supplementary).

| contrast | reads as |
|---|---|
| **F2 − F1** blank | is the real map itself suppressing the score, or is any second image enough? |
| **F3 − F1** mismatch | does a wrong-image map hurt further? |
| **F4 − F1** shift | cleanest correspondence test |
| **F5 − F1**, **F6 − F1** saliency | does a more visible map restore acceptance of the high score? |
| **R2 − R1**, **R3 − R1** | do the interventions reintroduce real FPR? |

## Pre-registered mechanisms

- **A — weak-visual-evidence veto:** blank > own, amplified/binary > own;
  shifted/donor need not be worse
- **B — spatial-mismatch sensitivity:** own > donor, own > shifted; blank need not be
  better
- **C — generic second-image interference:** own ≈ donor ≈ shifted ≈ blank, all below
  score-only
- **D — mixed:** several may hold; no single label is forced

## Suppression subgroup (secondary)

Defined from the 2×2 round as `C10 = fake AND C11 = real`.

**Count is 56, not 42.** The earlier report's 42 was the *net* recall difference
(139 − 97); the actual flip count is **56 suppressed with 0 reverse flips**. This
correction is recorded in the frozen JSON.

Suppressed cases have median `tamper_ratio` 0.0274 vs 0.0383 for all sources —
consistent with, though not proof of, the small-region hypothesis.

Planned per-case analysis: tamper_ratio, map active area, max/mean map probability,
score, reason text, and recovery under F2 / F5 / F6. **Primary analysis stays on all
184 sources.**

## Explanation audit

Deterministic keyword rules over saved reasons (no LLM scoring), tracking:
`score_high`, `map_shows_little`, `map_inconsistent`, `region_small`, `map_supports`.
Key question: among fakes judged Real under F1, how often the reason calls the map
weak/insignificant — and whether that falls under F2/F5/F6.

## Budget

| group | inferences |
|---|---|
| fake, 6 conditions | 1104 |
| real, 3 controls | 552 |
| **total** | **1656** (~61 min) |

## Forbidden this round

prompt tuning, threshold optimization, retraining, gate, verifier, LoRA, confidence
map, new detector, new dataset.

## Reference (2×2 round)

| | recall | FPR | J |
|---|---|---|---|
| C00 image only | 0.038 | 0.016 | +0.022 |
| C10 score only | 0.755 | 0.011 | **+0.745** |
| C01 map only | 0.098 | 0.033 | +0.065 |
| C11 score + map | 0.527 | 0.011 | +0.516 |

map effect with score on J: **−0.228**; score×map interaction on J: **−0.272**.
