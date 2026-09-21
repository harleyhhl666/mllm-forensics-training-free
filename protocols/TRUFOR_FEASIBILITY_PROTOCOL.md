# TRUFOR_FEASIBILITY_PROTOCOL.md

Frozen protocol for the TruFor image-level feasibility evaluation.
Frozen **before** any TruFor score on this set was computed.

Machine-readable twin: `runs/trufor_feasibility/trufor_feasibility_frozen.json`
(protocol sha1 `1e11d06ca1cd8461d10140974aabba114e9e9bbe`)

## Question

Can TruFor's whole-image `score` separate fake from **paired real** on TGIF sd2-sp?

## Sample

| field | value |
|---|---|
| sources | **200 unique coco_id** |
| per source | 1 fake variant + its paired real original |
| images | **400** (200 fake, 200 real) |
| pool | testing + validation sd2-sp indices (684 unique coco_id on disk) |
| seed | 20260918 |
| rule | one variant per coco_id, deterministic; **no** tamper_ratio or category filtering |
| clustering | `n_samples == n_unique_coco_id == 200`, so no within-source variant clustering |
| composition | bbox 110 / segm 90; 24 categories; tamper_ratio min 0.0010, median 0.0344, max 0.5057 |
| missing files | 0 |

**Data-reuse justification.** These images were previously seen by Qwen and by the
ELA pipeline. That is irrelevant to TruFor: TruFor was trained on other data
(tampCOCO, compRAISE, FantasticReality, CASIA2, IMD), its checkpoint is fixed and
public, and this evaluation performs **no checkpoint selection and no threshold
tuning**. No information about TruFor leaked from prior stages.

## Fixed configuration (from the deployment freeze)

| field | value |
|---|---|
| repo commit | `ae54475df6f41a491d7615100feb19263dec13f7` (pristine) |
| checkpoint | `trufor.pth.tar`, md5 `55d7075dd1ff945e9c0f9437c5df9495` (asserted at runtime) |
| preprocessing | official: RGB → CHW float ÷ 256.0, **no resize, no crop** |
| score | `sigmoid(det)` from the confpool detection head, [0,1], `higher_is_fake` |
| map | `softmax(pred,0)[1]`, [0,1], input resolution |
| conf | `sigmoid(conf)`, [0,1], pixel-level |
| env | conda `trufor`, python 3.10.20, torch 2.5.1+cu121 |

**Forbidden:** changing input resize, changing checkpoint, tuning post-processing,
fitting any parameter on TGIF, selecting a threshold from results, defining a new
ELA read-out.

## Metrics (fixed before running)

- **Primary:** `AUROC(score, fake vs real)`, **source-pair bootstrap 10 000×**, 95 % CI
- **Controlled FPR:** `TPR@FPR=0.05`, `TPR@FPR=0.10` — reported thresholds are
  **evaluation-derived operating points, NOT frozen deployment thresholds**, and must
  not be reused as tuned thresholds on this same data
- **Distribution:** mean, median, std, p10/p25/p75/p90 per class
- **Paired:** `Δscore = score_fake − score_real`, `P(fake>real)`, paired bootstrap CI
- **Secondary localization** (fakes only, never replaces the primary): pixel AUROC,
  IoU, pixel F1, 3×3 grid hit rate, stratified by tamper_ratio and bbox/segm;
  mask binarized at `>127` (existing frozen choice)
- **Confidence map: descriptive only** — mean conf, conf inside/outside GT, real-image
  mean conf. **No pooled image-level reliability score is constructed from it.**
- **ELA comparison:** existing frozen `blind_anomaly_score` on the **same** 200
  sources; no new read-out defined

GT masks are **never** TruFor inputs; they are used only for secondary scoring after
inference.

## Go / No-Go (pre-registered)

- **Hard No-Go:** AUROC CI includes 0.5, or discrimination near chance → TruFor is a
  useful localizer but unsuitable as an image-level evidence source here; stop the
  TruFor→MLLM line even if localization is strong
- **Yellow:** AUROC clearly above chance but poor TPR at low FPR → pause
- **Go:** AUROC clearly above chance **and** usable TPR at 5/10 % FPR **and** paired
  fake usually exceeds paired real **and** localization still effective

No numeric cutoff (e.g. AUROC > 0.75) is treated as a field standard.

## Hashes

| item | sha1 |
|---|---|
| config | `a6dffa803e31c18bf5abe47e8d4a9c7a5f351c31` |
| index (testing) | `b2b2d360789a461ffee93b31c87d04d0b23b6099` |
| index (validation) | `d07601a2a382f0c827fd7a2395737ea753a34c11` |
| adapter | `f1c362f242fc96685e32338b65a9a93c753e2f92` |
| forensic_tools | `adaf763cc54cc4cd4c292e625225ea2560aa8378` |
| trufor_env_frozen.json | `6d23c8b1365eca88d951b0523498d174ee02be96` |
