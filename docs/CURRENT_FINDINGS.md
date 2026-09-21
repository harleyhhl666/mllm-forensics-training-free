# CURRENT_FINDINGS.md

Internal research summary at the close of the training-free stage. Not a paper. Every
number traces to `docs/EXPERIMENT_INDEX.md` → a frozen protocol → raw verdicts.

Three layers are kept separate on purpose, because the project's main lesson is that
they behave differently: **tool**, **decision**, **explanation**.

---

## CONFIRMED

### Tool layer

| finding | evidence |
|---|---|
| ELA has some spatial localization ability but weak image-level discrimination | E02–E06. It is usable as a weak spatial cue, not as image-level evidence |
| TruFor is strong on TGIF sd2-sp | T01: image AUROC **0.9845** [0.9700, 0.9951], TPR@5%FPR **0.955**, pixel AUROC 0.993, IoU 0.758, F1 0.846 |
| TruFor's structured regions are precise but incomplete | X01: Evidence→GT grid **precision 0.979**, **recall 0.531**. It points at real tampering but covers only about half of it |

### Decision layer

| finding | evidence |
|---|---|
| The scalar score is the most stable decision signal available | T04: score-only J **+0.745** vs map-only **+0.065** |
| The raw heatmap gives no stable classification gain — it actively costs accuracy when a score is present | T04: ΔJ **−0.228**, interaction **−0.272**, both CIs exclude zero. 56 cases flip fake→real on adding the map, 0 in reverse |
| Neither model checks spatial correspondence of the map | T05, T06: shifting the map (identical value multiset, broken alignment) moves recall by **−0.027**, CI contains zero — at *both* 7B and 32B |
| 7B is extremely sensitive to framing and task semantics | S03: the structured prompt with **no payload at all** already collapses recall. S04: one sentence defining the label as binary restores it (J +0.114 → +0.902) |
| The 32B framing collapse is essentially absent | F01: `B2 − B0` recall **+0.000** [−0.016, +0.016], vs 7B's −0.723. Δeffect **+0.723** [+0.658, +0.788] |
| At 32B the binary rule mainly inflates FPR | F01: `B1 − B0` J **−0.038**, `B3 − B2` J **−0.239**, both CIs exclude zero. FPR +0.049 / +0.261 |
| Structured evidence provides **no** additional classification gain, at either scale | 7B `A3 − A1`: recall −0.011, FPR −0.016, J +0.005, all CIs contain zero. 32B `B3 − B1`: J **−0.217**, CI excludes zero |
| The best 32B configuration is the plainest one | F01: `B0` score-only J **0.918**, higher than any 32B evidence configuration and than any 7B cell |

### Explanation layer

| finding | evidence |
|---|---|
| Structured evidence makes spatial explanation near-perfectly consistent with the tool | X01: Explanation→Evidence hit **1.000**, unsupported-region rate **0.000**, vs raw map **0.404** |
| Structured evidence also moves the explanation closer to GT, not just to the tool | X01 paired: Explanation→GT **+0.340** [+0.262, +0.418] over the raw map; GT precision 0.657 → 0.995 |
| Explanation correctness is inherited from evidence correctness | X01 mirror: Explanation→Evidence stays **1.000** (paired Δ exactly 0.000) while Explanation→GT falls to **0.014** (7B) / **0.029** (32B) |
| The explanation inherits the evidence's incompleteness too | Explanation→GT recall ≈ 0.52, essentially the evidence's 0.531 |
| 7B can be spatially right and semantically wrong | X03: on 100 fake, spatial hit is 100/100, yet **79.1%** of its manipulation claims are unsupported and overall support is `unsupported` for **0.610** |
| 32B's fake explanations are far better supported, with no extra verbosity | X03: overall supported 0.490 vs 0.040; manipulation unsupported **0.044** vs 0.791; paired 81/100 fewer unsupported claims; claims per explanation 5.1 vs 5.4 |
| Neither model hallucinates objects much | X03: HOR 0.030 (7B) / 0.010 (32B) on fake, 0.000 on real |
| 7B's semantic failure worsens as the manipulated region shrinks | X03: claim UCR 0.373 (large) → 0.559 (small). 32B shows no such gradient |
| 32B is markedly more reticent on real images | X01: names a region on 12–34% of real images vs 7B's ~100%. On fake its citation coverage matches 7B |
| Models barely produce fine-grained forensic reasoning at all | X03: across 300 explanations, evaluable texture / lighting / boundary / geometry / physics claims number **0–3 each**. Explanations are almost entirely location + bare manipulation assertions |

### The one cross-scale invariant

> **Scalar score supports the decision; structured spatial evidence supports a faithful
> explanation.**

This role separation holds at both 7B and 32B: near-perfect spatial faithfulness
(agreement 1.000, 32B exact 0.950) with **no** discriminative gain (7B J +0.005,
32B J −0.217).

---

## NOT YET TESTED

- other datasets — everything rests on TGIF sd2-sp
- JPEG compression and other post-processing robustness
- other detectors — only ELA and TruFor were tried
- other MLLM families — only Qwen2.5-VL
- genuinely fine-grained forensic explanation (texture/lighting/boundary/geometry/physics), which this data barely elicited
- anything involving post-training; that is the next stage

---

## FALSIFIED OR CORRECTED

Full table with evidence in `docs/INVALID_AND_SUPERSEDED_RUNS.md` §4. In brief:

| discarded claim | what replaced it |
|---|---|
| ELA as strong image-level evidence | weak spatial cue only |
| the 7B failure is a capacity limit | 32B collapsed to the opposite constant (E09) |
| a verification step can gate bad evidence | M1 verifier collapsed (E08) |
| the map hurts via a weak-evidence **veto** | blank map is *worse* than own map and shift has zero effect — insensitivity, not veto |
| `area_ratio` drives the collapse | falsified; adding area to location *helped* (`S4 − S1 = +0.071`) |
| "generic structured overload" | mostly framing: `P1` with no payload already collapses |
| an evidence-**arbitration** failure needing presence/extent decomposition | decomposition over-detects (presence FPR ≈ 0.95); the binary rule alone fixed it |
| structured evidence improves decisions | it does not, at either scale |
| a larger model will check spatial correspondence | it does not |
| the final 7B mechanism conclusions are family-general | **two of three are capacity-dependent** |

One self-correction worth keeping visible: an earlier note credited D1's J = 0.902 to
structured evidence by comparing against `A0` (no rule, 0.826). The correct comparator is
`A1` (rule, no structure, 0.897). The gain came from the rule.

---

## LIMITATIONS THAT CONSTRAIN EVERY NUMBER ABOVE

1. **Single dataset, single manipulation family.** TGIF sd2-sp local splices, median
   tampered area under 4%.
2. **Evaluation-derived operating points.** Every threshold used for reporting is an
   evaluation-derived operating point, **not a frozen deployment threshold**.
3. **The X3 annotator is not an independent third party.** No Anthropic API key or Claude
   CLI existed in this environment, so annotation was done by blinded subagents of the
   *same model family as the assistant* — they could not see the conversation, the M1/M2
   mapping, the verdicts, or the TruFor scores, but they are not an independent
   annotator. Claims of third-party ground truth would be false.
4. **`manipulation_claim_supported` repeat agreement is only 0.650**, and it carries the
   headline 7B-vs-32B semantic comparison. The direction survives because the gap
   (0.128 vs 0.672) is far larger than the noise; the exact values should not be
   over-read.
5. **`hallucinated_object_present` κ = −0.017** despite 0.967 agreement — base-rate
   degeneracy. HOR is a low-reliability measurement.
6. **`Explanation→GT hit = 1.000` is not evidence of good explanation.** It is produced
   jointly by the evidence's own precision (0.979), near-perfect restatement, and GT
   masks spanning several grid cells (54/184 span ≥4). The informative numbers are the
   recall of ~0.52 and the mirror collapse to 0.014.
7. **Ceiling effects.** `A0` recall is already 0.837, so the +0.717 interaction in the
   7B 2×2 is substantially a ceiling artifact; head-room-normalised, the two arms are
   0.867 vs 0.963. Flagged before `A1` was run.
8. **Power limits accepted rather than engineered away.** The mirror subset is 70/184
   fake; 32B real-image FFER rests on 6 cited cases and is descriptive only.
9. **Region presence leaks label information** — fake has an empty region list 0/184
   times, real 10/184. Disclosed before any MLLM result was seen.
10. **MLLM verdicts are binary while TruFor emits a continuous score**, so
    tool-vs-integrated comparisons are decision-quality comparisons, not AUROC
    comparisons.
