# PROTOCOL v4 — INDEPENDENT VALIDATION (FROZEN before any inference)

**Frozen on:** Phase V0, before any Qwen inference on the TGIF validation split.
Everything below is fixed. No validation outcome may change eligibility, the cue
representation, ELA parameters, the p99 read-out, B2, the donor rule, tamper bins,
the statistical plan, or the prompts.

## Phenomenon under confirmation

> **Forensic Cue Grounding → Verdict Disconnect** (Perception–Decision Disconnect).
> Qwen2.5-VL-7B can exploit a matched ELA cue during Stage-A spatial localization —
> correct cue beats both no-tool and wrong donor cue — yet at the Stage-B
> authenticity verdict, correct and wrong ELA produce almost identical behaviour.
> The spatial cue-matching information appears not to propagate into the final
> authenticity decision.

## Exploratory background (NOT usable for tuning)

| Measurement | Pilot (discovery) | Primary (independent, Stage A) | Primary (exploratory, Stage B) |
|---|---|---|---|
| Stage-A GT loc: correct | 0.589 | 0.565 | — |
| Stage-A GT loc: no-tool | 0.333 | 0.323 | — |
| Stage-A GT loc: wrong | 0.500 | 0.427 | — |
| Stage-A correct − wrong | +0.089 [+0.006,+0.170] | **+0.137 [+0.017,+0.255]** | — |
| Stage-B fake recall: correct | — | — | 0.427 |
| Stage-B fake recall: no-tool | — | — | 0.137 |
| Stage-B fake recall: wrong | — | — | 0.452 |
| Stage-B correct − wrong | — | — | **−0.024 [−0.134,+0.093]** |
| B2 P(Real \| correct) | — | — | 0.532 [0.343,0.717] (n=47/20 clusters) |

The contrast between Stage-A **+0.137** and Stage-B **−0.024** is the pattern to be
confirmed. These numbers are background only.

## Data: TGIF validation split

A different TGIF split from the one used so far. All prior work used the
**testing** split (2058 triples, 343 coco_ids). Coco_ids touched by any previous
experiment: **243** (tool calibration 69 + pilot 129 + Primary 45 + Secondary 159,
deduplicated), recorded in `runs/used_coco_ids.json`.

Required files (sizes read from the official WebDAV listing):

| file | bytes |
|---|---|
| `masks_validation.tar.gz` | 5,295,371 |
| `orig_validation.tar.gz` | 859,947,874 |
| `sd2-sp_validation.tar.gz` | 2,172,017,290 |

TGIF documents the validation split as **341 authentic source images** with 8184
manipulated images across all sub-datasets; the `sd2-sp` subset is the one used here,
matching the frozen method.

**Mandatory V0 gate, to be executed once the data is present:**
`validation coco_id ∩ used_coco_ids = ∅`. TGIF's train/val/test splits are disjoint
by construction, but this is verified numerically, not assumed. If any overlap
exists, the run stops and is reported.

Also verified before freezing the sample list: directory structure and naming
identical to the testing split
(`{coco_id}_mask_{bbox|segm}.png_ps_mask.png_sd2_{var}.png`,
`{coco_id}_orig.png`, `{coco_id}_mask_{type}.png_ps_mask.png`); triple completeness;
image format; that `>127` mask binarization still applies; and that the ELA pipeline
runs unmodified.

## Frozen method (unchanged from previous phases)

- **Tool:** ELA only. Parameters as in `configs/exp01.yaml` (`jpeg_quality=90`,
  `scale=20`). No TruFor, MMFusion, noise-residual, generic cue, or new read-out.
- **Blind cue:** `B_region_p99` — 99th-percentile threshold on the ELA map, largest
  connected component. No GT access during generation.
- **Eligibility** (GT used only post-hoc, after the blind candidate exists): ELA
  valid; candidate non-degenerate; candidate hits GT; candidate centre inside GT; GT
  occupies ≤ 6 of 9 cells. **No image-level ELA threshold** — that approach was
  already rejected (CV AUROC 0.531–0.581).
- **Grid:** 3×3, `gt_cell_min_overlap = 0.05`.
- **Decoding:** frozen deterministic (`do_sample=false`, `max_new_tokens=320`),
  visual-token budget 256–1024.
- **Donor rule:** sorted pair_ids rotated by half the list; on coco_id collision walk
  forward to the first different coco_id. Frozen before inference; asserted
  `donor != self` and `donor_coco_id != target_coco_id`.

## Hypotheses

- **H1 — Stage-A matched-cue grounding.** correct ELA > no-tool on GT localization.
- **H2 — Stage-A cue matching.** correct ELA > wrong donor ELA on GT localization.
  If H2 fails, the premise for any Stage-B disconnect claim is absent.
- **H3 — Stage-B heatmap effect.** Presence of an ELA visualization changes the
  verdict: correct vs no-tool and wrong vs no-tool on fake recall, with paired-real
  specificity reported alongside.
- **H4 — Stage-B loss of cue matching (primary confirmatory).** The correct-vs-wrong
  advantage present in Stage A is markedly attenuated or absent in Stage B.

**H4 assessment rule, fixed now.** No equivalence margin is invented: there is no
principled basis for one, so H4 is judged on effect sizes and cluster CIs, reported
as an attenuation contrast —

`Δ_A = loc(correct) − loc(wrong)` versus `Δ_B = recall(correct) − recall(wrong)`,

each with its own coco_id-cluster bootstrap CI, plus the paired per-cluster
difference `Δ_A − Δ_B` with a cluster CI. Attenuation is supported if `Δ_A`'s CI
excludes zero while `Δ_B`'s CI contains zero **and** the CI for `Δ_A − Δ_B` excludes
zero. No `p > 0.05` claim of sameness will be made.

## Conditions

Stage A and Stage B each run three fake conditions — `correct_ela`, `no_tool`,
`wrong_donor_ela` — plus paired-real controls.

**Paired-real controls, frozen now (both, decided before seeing results):**
`real_own_ela` (real image + its own ELA) and `real_no_tool` (real image only). The
second is added now, in advance, so specificity can be separated from a general
"a heatmap is present" context prior; it will not be added later on the basis of
outcomes.

Prompt discipline, carried over verbatim: `correct_ela` and `wrong_donor_ela` share
an identical template and identical tool description, so the only difference is the
second image's pixels. Stage B additionally receives that condition's own frozen
Stage-A summary and may not revise it. Forbidden in any prompt: GT, cue bbox,
eligibility facts, cue correctness, B2 membership, localization outcomes,
tamper_ratio, condition names, and any wording urging trust in the tool
("trust the forensic evidence", "the tool is reliable", "prioritize technical
evidence"). A word-boundary blacklist assertion runs before every generation
(verified: 0 false positives on legitimate text, 0 misses on 9 crafted violations).

## Pre-registered behavioral subset

`B2 = mllm_vs_gt AND mllm_vs_correct_cue` on the `correct_ela` condition — the
definition formed on the earlier data, now used as a genuine pre-registered
confirmatory subset. Unmodified.

**Grounding–Verdict Inconsistency** = `final_verdict == "real"` within B2 under
`correct_ela`. Forbidden descriptions: "evidence rejection", "rejection of
definitive evidence", "the model rejected conclusive evidence" — ELA is not an
image-level fake proof. Permitted description: *spatial cue grounding and the final
authenticity decision are inconsistent.*

If B2 is small, insufficient statistical power is reported as such; the definition
is not relaxed.

## Statistics

Primary uncertainty: **coco_id-cluster bootstrap 95% CI**, 10,000 iterations,
seed 20260918. Supplementary: Wilson CIs, McNemar (exact binomial), Cochran's Q —
each labelled with the source-image clustering caveat.

Reporting order everywhere: **1. effect size → 2. coco_id-cluster CI →
3. sample-level p/CI.** No hypothesis is judged on a p-value alone.

## Required headline table

| Stage | correct ELA | no-tool | wrong ELA | correct − no-tool | correct − wrong |
|---|---|---|---|---|---|
| Stage A GT localization | | | | | |
| Stage B fake recall | | | | | |

## Pre-registered interpretation

- **A.** Stage A `correct > wrong`, Stage B `correct ≈ wrong`, and B2 shows clear
  Grounding–Verdict Inconsistency → **supports the disconnect**: the model grounds a
  matched cue spatially, but that matching information does not reliably reach the
  authenticity decision.
- **B.** `correct > wrong` in *both* stages → the exploratory disconnect did **not**
  replicate; the model propagates cue correctness into the decision. Stop the
  disconnect line.
- **C.** Stage A `correct` no longer beats `wrong`/`no-tool` → grounding itself did
  not replicate. Stop this direction entirely.
- **D.** All Stage-B conditions poor *and* paired-real false-positive rate high →
  interpret primarily as a basic authenticity-classification / prompt-prior problem,
  **not** evidence arbitration. Do not force a disconnect reading.

Note in advance, given Primary's paired-real specificity of 0.460: outcome D is a
live possibility and will be checked explicitly against the `real_no_tool` and
`real_own_ela` controls.

## Execution order

- **V0** — audit (this document + `validation_v4_frozen.json`). **Pause for
  confirmation before any inference.**
- **V1** — Stage A, three fake conditions + two real controls. Evaluate H1/H2. If
  Stage-A grounding fails to replicate, stop; do not run Stage B.
- **V2** — Stage B, same conditions. Evaluate H3/H4 and B2 inconsistency. Stop.

Deterministic inference only in this round. A stochastic stability probe is
considered **only** if B2 + correct + final-Real appears in materially meaningful
numbers, and is run separately afterwards.

## Prohibited

Changing ELA parameters, p99, eligibility, B2, the donor rule, tamper bins, or
prompts; adding tools or gates; re-tuning on validation output; defining a new
primary metric after seeing results; substituting the frozen Secondary 521 set for
this independent validation.
