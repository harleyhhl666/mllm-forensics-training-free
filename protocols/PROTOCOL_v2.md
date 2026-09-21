# Protocol v2 — Forensic Cue Acknowledgment / Verdict Inconsistency

Supersedes the Gate-1 definition used in v1. The change is forced by a measured
result, not by convenience, and the reason is recorded here so the final write-up
cannot overstate what Gate 1 means.

## What the measurements forced

| Claim | Evidence | Verdict |
|---|---|---|
| ELA discriminates real vs fake at image level on TGIF-sp | grouped-CV AUROC 0.531–0.581 (fold std 0.024–0.085); TPR@5%FPR 0.025–0.078 across all 6 frozen read-outs | **REJECTED** |
| ELA localizes the manipulation blindly | pixel AUROC median 0.799 (frac>0.7 = 0.772); blind candidate-region hit 0.529–0.826 vs random ~0.24; median IoU up to 0.192 | **SUPPORTED** |

Root cause of the image-level failure: ELA responds strongly to texture, edges and
high-frequency content in *authentic* images too, so the *absolute* strength of the
strongest local response is distributed almost identically for real and fake
(B_p995: real median 6.72 / fake median 7.43, real p95 11.22 / fake p95 10.13).
The manipulated region is anomalous *relative to its own image*, and that relation
does not survive being compressed into a cross-image scalar. Three independent
read-out families (whole-image statistics, multi-scale scan, real-null calibration)
failed at the same magnitude, so this is treated as a structural property of ELA,
not a missing heuristic.

## Terminology (binding)

Gate 1 does NOT establish that an image is fake. What it establishes is the
presence of a

> **GT-validated blind forensic localization cue** — a spatial region proposed by
> the tool without any access to ground truth, which post-hoc evaluation confirms
> corresponds to the true manipulated region.

Forbidden phrasings in any output: "objective evidence proves the image is fake",
"the tool detected the forgery", "the model rejected conclusive evidence".
Required phrasing for the core finding:

> The model correctly read a forensic cue that corresponds to the real manipulation
> region, yet its final authenticity judgment is inconsistent with the manipulation
> evidence that cue points to.

## Gates

**Gate 0 — Tool validity.** Unchanged. ELA output must not be degenerate
(variance, dynamic range, near-constant ratio, saturation thresholds in config).
double-JPEG stays permanently `not_applicable` on TGIF (all PNG).

**Gate 1 — Blind spatial cue validity.** Two steps, strictly ordered:
1. *Blind*: ELA produces the anomaly map and a candidate region. Phase-1 default
   read-out is `B_region_p99` (frozen: 99th-percentile threshold, largest
   connected component). No GT, no image-level threshold, no scalar gate.
2. *Post-hoc evaluation only*: GT mask scores that candidate region —
   `candidate_gt_hit`, `candidate_center_in_mask`, `candidate_iou`,
   `candidate_precision`, `candidate_recall`.

   A fake sample is eligible for the core analysis when the tool is valid AND the
   blind candidate region genuinely corresponds to the manipulated region.

   **Acknowledged asymmetry:** using GT to decide eligibility means Gate 1 is not a
   gate a deployed system could apply — it is a *sample-selection criterion for
   studying MLLM behaviour under a known-good cue*. This is legitimate because the
   research question is conditional ("given a correct cue, what does the model do?"),
   but it forbids two claims: (a) that these rates describe in-the-wild behaviour,
   and (b) that Gate 1 constitutes automatic forgery detection. Candidate
   generation itself never sees GT, so there is no leakage into the cue the model
   is shown.

**Gate 2 — Evidence acknowledgment.** Stage A sees the image, the ELA
visualization, and a short tool description. It reports whether a local forensic
anomaly is present and where. Verdicts are forbidden at this stage.

**Gate 3 — Spatial agreement (three-way).** Records all three relations:
- `mllm_vs_gt` — model region ∩ GT region
- `mllm_vs_cue` — model region ∩ ELA candidate region
- `cue_vs_gt` — already established by Gate 1

Core samples require all three to agree. Requiring `mllm_vs_cue` is what
distinguishes "the model read the tool" from "the model independently guessed a
salient object that happens to be the edited one" — the latter is a real
confounder, because TGIF edits are objects and objects are salient.

**Gate 4 — Final arbitration.** Stage B runs only for samples passing Gates 0–3,
receives Stage A's own summary verbatim, and cannot modify it. `ground_truth=fake`
with `final_verdict=real` is a candidate *cue-acknowledgment / verdict
inconsistency*.

## Controls (all Stage A, run BEFORE any Stage B)

| Control | Input | Purpose |
|---|---|---|
| A — no tool | fake image only | does the tool add anything? |
| B — wrong tool | fake image + a mismatched ELA map | does the model use *this* image's cue, or just any heatmap? |
| C — paired real | TGIF original + its own ELA map | does the model call anomaly on every ELA heatmap? |

Control B implementation, frozen before running (see `wrong_tool.md`): the
mismatched map is the ELA map of a **different pilot image**, paired by a fixed
deterministic rule (sorted pilot order, rotate by half the list), resized to the
target image's dimensions. Rationale for donor-swap over spatial shuffle: a
block-shuffled map is trivially identifiable as corrupted (visible seams), so the
model could reject it for the wrong reason; a real ELA map from another photograph
is statistically plausible yet carries no information about the current image. The
rule is fixed in advance and not chosen by inspecting results.

## Decision point after the pilot's Stage A

Proceed to Stage B **only if** correct-tool localization is clearly above BOTH
no-tool and wrong-tool. If not, stop and report: without that, Gate 2/3 passing
cannot be attributed to the model reading the forensic cue.

## Pilot

150–200 fakes from the untouched test split, plus their paired originals.
Eligibility is decided by tool-layer facts only — ELA valid, candidate GT-hit,
localization gate informative — never by MLLM output. Pre-stratified by
tamper_ratio (<1%, 1–3%, 3–10%, >10%). IDs frozen to
`runs/pilot/pilot_frozen.json` and not changed afterwards.
