# PROTOCOL v3 — FROZEN (Confirmatory)

**Status: frozen.** From this point, no Primary-validation output may change
eligibility, cue representation, primary metrics, the donor rule, tamper bins, or
the statistical tests. If Primary fails to support the hypotheses, the protocol is
not edited and re-run on the same Primary set.

## Research question

> When an MLLM is given an external forensic visualization, does it verify that the
> cue is actually grounded in the current image, or does it merely follow visually
> salient regions in the supplied cue?

Working name: **Forensic Cue Grounding Failure**.

## Why the previous line was abandoned

The Stage-A pilot (180 samples, 720 inferences, zero protocol violations) showed
`evidence_detected` = 0.994 on fake+correct-cue and 1.000 on paired-real+own-ELA
(difference −0.006, McNemar p = 1.0). A near-constant acknowledgment signal cannot
establish that the model understood forensic evidence, so the Gate-2/Gate-4
"evidence acknowledged but rejected" construction has no testable premise with this
tool and model. `evidence_detected` is retained as a descriptive field only.

The pilot instead produced a different, testable pattern: correct cue improved GT
localization over no-tool (+0.256, p = 3.2e-10), but a mismatched donor cue also
moved localization substantially (0.500 vs no-tool 0.333), leaving only +0.089
[+0.006, +0.170] for correct-over-wrong, with 19.4% of samples following the donor
cue into a wrong region.

## Hypotheses (frozen)

- **H1** Correct forensic cue improves spatial localization relative to no-tool.
- **H2** Mismatched/wrong forensic cue also materially influences localization.
- **H3** The model's ability to follow a supplied cue is substantially stronger than
  its ability to verify whether that cue is grounded in the current image.
- **H4** A wrong cue can actively induce errors in cases that were correctly
  localized without the cue.

## Data roles (binding language)

| Set | n | coco_ids | Role |
|---|---|---|---|
| Discovery (pilot) | 180 | 129 | Already observed. Generated the hypotheses. Never used for confirmation. |
| **Primary confirmation** | **124** | **45** | coco_ids fully disjoint from the pilot. Core conclusions judged here. |
| Secondary expanded | 521 | 159 | Shares coco_ids with the pilot. **Not** independent confirmation. Power and subgroup analysis only. |

Eligibility, identical across all three and involving no model output: ELA valid;
`B_region_p99` candidate non-degenerate; candidate hits GT; candidate centre inside
GT; GT occupies ≤ 6 of 9 cells. Audit on the 1644 test fakes: 701 eligible;
exclusions — candidate missed GT 821, region degenerate 65, centre outside GT 56,
localization gate uninformative 1.

Primary's natural tamper_ratio distribution is accepted without balancing:
`<1%` 7, `1–3%` 43, `3–10%` 62, `>10%` 12.

## Conditions (frozen)

Per fake: `correct_ela`, `no_tool`, `wrong_donor_ela`. Plus `paired_real_own_ela`
per source image — descriptive only, not a hypothesis test.

`correct_ela`, `wrong_donor_ela` and `paired_real_own_ela` share a byte-identical
prompt (verified by SHA1 at runtime); only the second image's pixels differ. The
tool description contains no per-image measurement. No GT, candidate bbox,
eligibility fact, tamper_ratio, or condition name may enter a prompt; a blacklist
assertion re-checks every prompt before generation.

Donor rule: sort the set's pair_ids, rotate by half the list, and on a coco_id
collision walk forward to the first candidate with a different coco_id. Frozen
before inference; verified 0 self-donors and 0 same-coco donors in both sets.

Generic salient cue is **excluded** this round. The self-image version was measured
and rejected: its appearance does not match ELA (median map mean 0.186 vs 0.046,
p99 0.727 vs 0.256) and it carries real localization information (accidental GT hit
0.583, centre-in-mask 0.483, median IoU 0.030 — higher than ELA's 0.019). A
`donor generic` variant is the correct follow-up to separate forensic-specific from
general visual-cue following, and is deferred until H1–H4 replicate.

## Primary metrics (frozen)

- **A. GT localization** — predicted cells ∩ GT cells ≠ ∅.
- **B. Cue adherence** — predicted cells ∩ supplied-cue cells ≠ ∅, recorded
  separately for the correct and the wrong cue.
- **C. Grounding advantage** — `adherence_correct − adherence_wrong`. This measures
  discrimination between a matched and a mismatched cue. It is **not** a
  real/fake detection ability and must not be described as one.
- **D. Joint correct grounding** — in the correct condition, prediction overlaps GT
  **and** overlaps the correct cue.
- **E. Cue-induced error** — denominator: samples where the no-tool condition
  localized GT correctly. Numerator: the wrong-cue condition's GT localization
  becomes incorrect **and** its prediction overlaps the supplied wrong cue. Report
  `P(cue_induced_error | no-tool originally correct)` with CI.

## Statistical tests (frozen)

- Cochran's Q across the three fake conditions on GT localization.
- McNemar (exact binomial, two-sided) for: correct vs no-tool, correct vs wrong,
  wrong vs no-tool.
- 95% CIs for absolute differences by bootstrap.
- Cue-induced-error proportion with 95% CI.

**Clustering, decided in advance:** Primary's 124 samples come from 45 coco_ids
(up to 6 samples per source), so samples are not mutually independent within a
source. Per §5 of the directive, the **coco_id-cluster bootstrap CI is the primary
interval** and sample-level paired CIs are reported alongside for comparison.
McNemar p-values are reported as computed on paired samples, with the clustering
caveat stated explicitly wherever they appear. Secondary analysis is
cluster-bootstrap only.

Effect size and CI are required everywhere; `p < 0.05` alone never constitutes a
finding.

## Pre-registered interpretation rules

The pilot phenomenon is supported if the Primary set reproduces this whole pattern:
1. correct cue has a clear positive localization effect vs no-tool;
2. wrong cue also significantly changes/induces localization vs no-tool;
3. correct-vs-wrong discrimination is clearly weaker than the overall cue-following
   effect;
4. cue-induced error is non-zero with a material effect size on unseen coco_ids.

If only (1) holds and the wrong cue has no effect, the better explanation is that
the model uses the forensic cue effectively — **not** grounding failure. If neither
correct nor wrong cue has a stable effect, the pilot phenomenon did not replicate
and the direction stops.

## Tamper-ratio

Frozen bins `<1% / 1–3% / 3–10% / >10%`, boundaries unchanged. In Primary these are
descriptive replication only, with CIs; `<1%` (n=7) and `>10%` (n=12) carry no
strong conclusions. The formal test of "smaller manipulations show worse grounding"
belongs to the Secondary expanded analysis with coco_id-cluster-aware CIs.

## Frozen artifacts

`runs/validation/validation_primary_frozen.json`,
`runs/validation/validation_secondary_frozen.json` — each carries its sample list,
donor mapping, seed (20260918), the eligibility rule verbatim, the donor rule, and
SHA1 hashes of the windows table, dataset index, config, pilot freeze, and split
freeze. Both freeze scripts refuse to overwrite an existing file.

## This round

Primary Stage-A only. No Stage B. No Secondary run. No TruFor/MMFusion. Stop and
report when Primary completes.
