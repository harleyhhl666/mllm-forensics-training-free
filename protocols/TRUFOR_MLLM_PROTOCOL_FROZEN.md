# TRUFOR_MLLM_PROTOCOL_FROZEN.md

Phase T0 freeze for the **TruFor → MLLM evidence integration replication**.
Frozen **before** any MLLM inference. Nothing below may change afterwards.

Machine-readable twin: `runs/trufor_mllm/trufor_mllm_frozen.json`
(protocol sha1 `3a9bb651dcded055f0e82bd6bce7ef2fda09f11b`)

## Question

Does the forensic-context bias / cue-grounding–verdict disconnect seen with ELA
persist when the evidence source is itself highly discriminative (TruFor:
AUROC 0.9845, TPR@5 % FPR 0.955, IoU 0.758)?

This is a **tool-replacement replication**, not a new direction. The ELA
Stage-A/Stage-B structure is reused verbatim; only the evidence source changes.

## 1. Samples

| field | value |
|---|---|
| source pool | V4 validation frozen set |
| sources selected | **184**, one variant per coco_id |
| `n_samples == n_sources` | **True** → no variant clustering |
| **with ELA Stage-A history** | **184/184** → genuine paired ELA-vs-TruFor comparison |
| tamper_ratio | min 0.0032, median 0.0383, max 0.2837 |
| mask types | bbox 73 / segm 111 |
| seed | 20260918 |

Selection used **neither** TruFor scores **nor** prior ELA results — one variant per
coco_id, deterministic under the seed.

Choosing the V4 pool is deliberate: every selected image already has ELA Stage-A
output with the same condition structure (`fake_correct / fake_notool / fake_wrong /
real_own / real_notool`), so §11's ELA comparison will be a true same-image paired
comparison rather than a same-domain approximation.

## 2. Six conditions

| condition | image | evidence package |
|---|---|---|
| `fake_T0_notool` | fake | none |
| `fake_T1_correct` | fake | own TruFor map + own score |
| `fake_T2_wrong` | fake | **donor** map + **donor** score |
| `real_T0_notool` | real | none |
| `real_T1_correct` | real | own TruFor map + own score |
| `real_T2_wrong` | real | **donor** map + **donor** score |

**Evidence package rule (§5):** in T2 the localization map and the image-level score
move **together** as one donor package. A donor map is never paired with the
target's score.

**Donor rule:** label-matched (fake←fake, real←real), `donor_coco_id ≠
target_coco_id`, never self, frozen before inference. Verified: 184 assigned,
**0 same-coco, 0 self**.

T1 and T2 prompts are **byte-identical**; only the pixels and the score value
differ. The model is never told the condition, the donor, or the GT.

## 3. Confidence map: excluded this round

Per §2 of the brief, the pixel-level confidence map is **not** shown to the MLLM.
It is stored in every evidence `.npz` for secondary analysis only.

## 4. Prompts (neutral)

Tool description, verbatim and frozen:

> An automated forensic analysis of the image is provided. The accompanying
> localization map indicates where the tool assigns higher manipulation-related
> response. A whole-image score in [0,1] is also provided, with higher values
> corresponding to stronger manipulation evidence. Treat these outputs as auxiliary
> evidence to be interpreted together with the original image, not as proof.

Automated audit: **0 hits** for `reliable / accurate / state-of-the-art / trusted /
trustworthy / auroc / benchmark / proven / prioritize / should trust / correct /
wrong / donor / ground truth`. The model is **not** told TruFor's AUROC.

Stage A forbids any authenticity verdict and returns
`{evidence_used, predicted_regions, tool_score_interpretation, reason}`.
Stage B returns `{final_verdict, reason}` and receives the image, the current
condition's evidence, and **that condition's own** Stage-A summary.

| prompt | sha1 |
|---|---|
| `STAGE_A_TOOL` | `ce41af3d9efff2d5ba77643ed070621b41c6c506` |
| `STAGE_A_NOTOOL` | `fc89d1cee3f0ddcb14cba5509628474250514d4f` |
| `STAGE_B_TOOL` | `9996b1a9d2bf5eda90bcfe529d647668b5356825` |
| `STAGE_B_NOTOOL` | `4c15a0b7449b20d19a5592195ff2045b6d6228b2` |

## 5. Score presentation

`TruFor image-level manipulation score: 0.873` — raw continuous value, **3
decimals**, no percentage, no high/medium/low label, no threshold, no
discretisation.

## 6. Map rendering (frozen)

`inferno` colormap on a **fixed 0–1 scale** (`vmin=0, vmax=1`), **no min-max
stretch**, so displayed intensity equals the model's probability and is comparable
across images and conditions. Resized to the analysed image only if sizes differ
(bilinear). RGB PNG.

## 7. Score distributions (reported, not used to re-pick donors)

| package | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| fake, own (T1) | 184 | 0.9657 | 0.9966 | 0.9186 | 0.9998 |
| fake, donor (T2) | 184 | 0.9657 | 0.9966 | 0.9186 | 0.9998 |
| real, own (T1) | 184 | 0.2524 | 0.1877 | 0.0665 | 0.5552 |
| real, donor (T2) | 184 | 0.2524 | 0.1877 | 0.0665 | 0.5552 |

Mean difference own−donor: **+0.0000** for both labels.

§14's imbalance concern is satisfied **structurally**: donors are a permutation of
the same label-matched pool, so T1 and T2 score *marginals* are identical by
construction — only the image↔score correspondence is broken. No post-hoc
re-matching was performed or needed.

One consequence worth stating plainly: because fake scores are tightly clustered
high (median 0.997), a donor fake score is numerically almost indistinguishable from
the correct one. So for fake targets, **T1 vs T2 is essentially a test of spatial
map correspondence**, not of score value. That is a property of TruFor's score
distribution, not a design choice, and it will be acknowledged in the analysis
rather than engineered around.

## 8. Inference budget

| phase | inferences |
|---|---|
| Stage A | 184 × 6 = **1104** |
| Stage B | 184 × 6 = **1104** |
| **total** | **2208** |

TruFor evidence for all 368 images was precomputed in 2.5 min (checkpoint md5
verified at runtime).

## 9. Metrics (fixed now)

**Stage A, fake:** GT localization rate, cue-map overlap, joint grounding; T1 vs T0,
T1 vs T2; under T2 whether the model follows the donor cue and whether it still hits
the target GT.
**Stage A, real:** descriptive — rate of reported suspicious regions; whether correct
real evidence induces spurious localization.
**Stage B:** fake recall and real FPR for T0/T1/T2, specificity, **Youden J = TPR −
FPR**.
**Contrasts:** fake T1−T0, T1−T2, T2−T0; real FPR(T1)−FPR(T0), FPR(T2)−FPR(T0);
J(T1)−J(T0), J(T2)−J(T0).
**Statistics:** effect size → **source-level bootstrap 95 % CI** → paired McNemar p
(supplementary).

## 10. Pre-registered cases

- **A — reliable evidence integrates:** T1 ≫ T0, T1 ≫ T2, contained real-FPR rise,
  J(T1) improves
- **B — context bias persists:** T1 and T2 both raise fake verdicts, real FPR rises
  too, T1 ≈ T2 → framing bias survives a highly discriminative generator
- **C — tool quality matters:** T1 improves, T2 no longer induces fake, clear
  correct-vs-wrong separation → the ELA failure was substantially an
  evidence-quality problem
- **D — MLLM degrades TruFor:** final J drops and real FPR worsens versus the tool
  alone → the reasoning layer damages an already strong detector. **Report as-is; do
  not prompt-tune.**

## 11. Forbidden this round

gate, verifier, confidence threshold, score calibration, prompt debiasing, LoRA,
training, tool ensemble. This round is replication and diagnosis only.

## 12. Hashes

| item | sha1 |
|---|---|
| protocol | `3a9bb651dcded055f0e82bd6bce7ef2fda09f11b` |
| config | `a6dffa803e31c18bf5abe47e8d4a9c7a5f351c31` |
| V4 frozen set | `f3e4…` (see JSON) |
| trufor_adapter.py | `f1c362f242fc96685e32338b65a9a93c753e2f92` |
| trufor_env_frozen.json | `6d23c8b1365eca88d951b0523498d174ee02be96` |
| TruFor repo commit | `ae54475df6f41a491d7615100feb19263dec13f7` |
| TruFor checkpoint md5 | `55d7075dd1ff945e9c0f9437c5df9495` |
