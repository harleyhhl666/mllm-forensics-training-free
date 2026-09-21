# TRAINING_FREE_EXPERIMENT_REPORT.md

Internal report on the training-free stage. Written so the group can reconstruct what was
done, why each step followed from the last, and which ideas were discarded. Deliberately
not paper prose.

Every number here is cross-checked against raw artifacts by
`scripts/crosscheck_numbers.py` (`reports/CROSSCHECK.txt`, 0 mismatches).

---

## 1. Background

The research question was:

> Can an MLLM use **external forensic evidence** to detect, localize and *explain* image
> manipulation?

Training-free was chosen deliberately. The resources are 1–2 people and 1–2× 24 GB GPUs
on a 1–3 month horizon; and more importantly, the question is about whether a pretrained
model *can use* evidence, which fine-tuning would immediately confound. If a model needs
training to use a tool's output, that is itself a finding.

The chain below is long because almost every stage produced a **negative** result that
redirected the next one.

## 2. Initial hypotheses — all later revised

The first framing was built on three ideas:

- **semantic veto** — the model overrides forensic evidence when the image "looks fine";
- **acknowledgment–verdict inconsistency** — the model admits manipulation then says real;
- **evidence grounding failure** — the model does not actually read the cue.

Only the second survived contact with data, and even it turned out to be a symptom rather
than a mechanism. These are recorded as early hypotheses, not results.

## 3. ELA stage

### Why ELA first

It is training-free, deterministic, needs no weights, and is a classical forensic cue. If
an MLLM cannot use ELA, the whole premise is in doubt.

### Why CocoGlide was dropped

Its manipulations did not give the clean local-splice structure the mechanism
experiments require. TGIF `sd2-sp` was adopted instead: Phase 0 measured median mean
absolute difference **outside** the mask at 0.0030 — pixel-exact local splicing, so the
mask really is the only altered region.

### What was run

Calibration of the window/quality readout with no GT mask (E02, E03), then a pilot (E04),
then the frozen V4 validation: **V1** for grounding and **V2** for verdict (E05, E06).

### Results

- **V1**: the model *does* ground on the supplied cue. H1/H2 supported. This killed the
  "grounding failure" hypothesis.
- **V2**: **Case D dominant** — grounding and verdict are *disconnected*. The model looks
  at the right place and still answers wrongly.
- **E07 mechanism 2×2**: the fake-decision bias is a **pure interaction** of forensic
  framing × visualization. `C1−C0 = C2−C0 = +0.000`; `C3−C2 = +0.267/+0.287`. Neither
  factor alone does anything.
- **E08 M1 verification gating**: failed. The verifier's output collapsed. A negative
  result, kept.
- **E09 32B Stage-V**: **Case 2** — the larger model collapsed to the *opposite*
  constant. Capacity was not the cause.

### Conclusion

> ELA is better treated as a **weak spatial cue** than as strong image-level evidence.

Hence the switch to a learned detector.

## 4. Turning to TruFor

Rationale: a learned forensic model with both an image-level score and a pixel-level
localization map, so "score vs map" becomes an experimentally separable question.

Deployment was pinned hard: repo commit `ae54475df6f41a491d7615100feb19263dec13f7`,
checkpoint md5 `55d7075dd1ff945e9c0f9437c5df9495`, and the adapter was verified against
official outputs with **delta = 0**.

### Standalone feasibility (T01, 200 sources)

| metric | value |
|---|---|
| image AUROC | **0.9845** [0.9700, 0.9951] |
| TPR @ FPR ≤ 0.05 | **0.955** |
| pixel AUROC (mean) | 0.9930 |
| IoU @ map>0.5 (mean) | 0.7578 |
| pixel F1 @ map>0.5 (mean) | 0.8459 |

GO decision. Note the framing required whenever this is described: *the TGIF training
split was used only as an untouched evaluation pool for the external forensic model; no
model was trained or tuned on these images.*

## 5. TruFor → 7B

### T02/T03 — Stage T1 then T2

The first **positive** integration result of the whole project:

| condition | recall | FPR | J |
|---|---|---|---|
| T0 no tool | 0.201 | 0.125 | +0.076 |
| **T1 correct tool** | **0.804** | **0.011** | **+0.793** |
| T2 donor (wrong) tool | 0.717 | 0.011 | +0.707 |

Two limits were stated at the time: correct-vs-donor separation is weak and fragile
(T1−T2 = +0.087, CI lower bound +0.005, p = 0.048 — all three barely clear), and the gain
probably comes from the score rather than the map.

Also already visible: when Stage-A grounding *hit* the GT, recall was 0.800; when it
missed, **0.839**. Grounding and verdict again disconnected.

### T04 — score-map 2×2

A prerequisite had to be fixed first: the old Stage-B prompts **embedded a Stage-A
summary**, which leaks map-derived information into map-absent conditions. So all 1472
inferences were re-run rather than reusing cells.

| cell | recall | FPR | J |
|---|---|---|---|
| C00 image only | 0.038 | 0.016 | +0.022 |
| **C10 score only** | **0.755** | — | **+0.745** |
| C01 map only | 0.065 | — | +0.065 |
| C11 score + map | 0.527 | 0.011 | +0.516 |

**Case 4**: the score dominates and the map is a significant *negative* contributor when
a score is present — ΔJ **−0.228**, interaction **−0.272**. 56 individual cases flip
fake→real when the map is added; **zero** flip the other way.

### T05 — conflict interventions

Blank / donor / **shifted** / amplified / binarized maps. The shifted map is the cleanest
intervention available: `np.roll` preserves the value multiset element-for-element and
only destroys alignment.

| effect | value |
|---|---|
| map cost | −0.228 |
| blank map | −0.255 |
| **shifted map** | **−0.027, CI contains zero** |

The headline:

> **Shifting the map to the wrong location barely changes the verdict. The 7B is not
> checking spatial correspondence at all.**

The earlier "weak-visual-evidence veto" story also died here: a *blank* map (0.266) is
worse than the model's own map (0.522), which is inconsistent with a veto triggered by
weak local response.

(The shift audit initially appeared to fail. Investigation showed the *assertion* was
wrong — it demanded `|Δmean| < 1e-9`, below float32 summation precision; the actual
difference is ≤5.96e-08 with active fraction identical to 0.0. See
`INVALID_AND_SUPERSEDED_RUNS.md` §3.)

## 6. T06 — first 32B capacity check

Larger capacity **reduces interference** but does **not** introduce
spatial-correspondence checking: the shifted-map effect stays at −0.027 with a CI
containing zero.

## 7. Structured evidence (S01)

If the model will not read a heatmap spatially, give it the spatial facts in words.

The **evidence abstraction layer** is fully deterministic and adds no model: threshold
`map > 0.5`, 4-connected components, minimum area ratio 0.001, top-K 3 sorted by
integrated probability, normalised coordinates, 3×3 grid naming, centroid as primary
location. Connected components are implemented with numpy plus a stdlib BFS specifically
to avoid installing scipy into the frozen environment.

Result: **spatial agreement 1.000** versus **0.358** for the raw map — but decision
utility *collapsed* (score-only J 0.826 → structured J 0.163).

Two limitations were computed and disclosed **before** any MLLM result: the mirror subset
is only 70/184 fake (a horizontal mirror leaves `center` and the centre column
invariant), and region presence itself leaks label information.

## 8. Field ablation (S02) → framing control (S03)

### The wrong turn, and how it was caught

S02 was designed to test whether `area_ratio` caused the collapse — a hypothesis the
assistant had formed from reason-text patterns. It was **falsified**:

| condition | recall | J |
|---|---|---|
| S0 score only | 0.837 | +0.826 |
| S1 location | 0.141 | +0.141 |
| S2 area | 0.196 | +0.196 |
| S3 probability | 0.103 | +0.103 |
| S4 location+area | 0.212 | +0.212 |
| S5 full | 0.114 | +0.114 |

Every single field hurts by a similar amount (−0.696 / −0.641 / −0.734) and, decisively,
`S4 − S1 = **+0.071**` — *adding* area to location **improved** things. The judgement was
changed to "generic structured overload".

That too was wrong, and S03 is what proved it. P0–P5 separate prompt framing from
container presence from real values:

- **`P1` — the structured prompt with the payload slot empty — already collapses.**

So the cause is a **framing-induced decision-policy shift**, not the measurements. The
model produces "localized / confined / limited manipulation" narrative even when it has
been given no location, no area and no probability at all.

Diagnostic that pinned the failure to reasoning rather than perception:
`acknowledged_manipulation_but_real` is **0.000** under score-only and 0.25–0.60 under
every structured condition.

## 9. Task semantics (S04)

If the model misreads "small local edit" as "the image is still basically real", then
*defining the task* should fix it.

| cell | recall | FPR | J |
|---|---|---|---|
| D0 baseline | 0.114 | 0.000 | +0.114 |
| **D1 binary rule** | **0.967** | 0.065 | **+0.902** |
| D2 presence/extent decomposition | — | — | — |
| D3 presence-first arbitration | — | — | — |

One sentence — *"the authenticity label is binary; if any part has been genuinely
manipulated, classify as fake regardless of how small the region is"* — restored almost
all decision utility.

D2/D3 **falsified the arbitration hypothesis**: explicit presence/extent decomposition
produced presence FPR ≈ 0.95 and over-detection. Structural decomposition was not the
answer; task definition was.

(This round's first execution was **voided**: the runner duplicated
`image_manipulation_score` in the payload, D0 measured 0.592 instead of 0.114, and all
1472 inferences were discarded rather than interpreted. See
`INVALID_AND_SUPERSEDED_RUNS.md` §1.1.)

## 10. Score × semantics 2×2 (S05) — the 7B endpoint

D1's success could mean either "the rule clarifies the task generally" or "the rule
specifically repairs structured framing". The missing cell `A1` (rule, no structure)
decides it.

| cell | structured | rule | recall | FPR | J |
|---|---|---|---|---|---|
| A0 | no | no | 0.837 | 0.011 | +0.826 |
| A1 | no | **yes** | 0.978 | 0.082 | **+0.897** |
| A2 | yes | no | 0.114 | 0.000 | +0.114 |
| A3 | yes | **yes** | 0.967 | 0.065 | **+0.902** |

- The rule works in **both** arms (`A1−A0` J +0.071, CI excludes zero), so it is not a
  framing-specific patch.
- **`A3 ≈ A1`**: `A3−A1` gives recall −0.011, FPR −0.016, J +0.005 — **all three CIs
  contain zero**. Structured evidence adds **no** discriminative value once the task is
  stated correctly.
- The +0.717 interaction is substantially a **ceiling artifact** (disclosed before running
  A1): `A0` recall was already 0.837, and head-room-normalised the arms are 0.867 vs
  0.963.

So the 7B conclusion became the role separation: score for decisions, structured evidence
for faithful explanation.

## 11. 32B final capacity replication (F01)

The question was explicitly *not* "is 32B more accurate" but whether the three final
mechanism conclusions survive a scale change. Prompts were reused **byte-for-byte**
(hashes asserted), so the model is the only variable.

| cell | structured | rule | recall | FPR | spec | J |
|---|---|---|---|---|---|---|
| **B0** | no | no | 0.978 | 0.060 | 0.940 | **+0.918** |
| B1 | no | yes | 0.989 | 0.109 | 0.891 | +0.880 |
| B2 | yes | no | 0.978 | 0.076 | 0.924 | +0.902 |
| B3 | yes | yes | 1.000 | **0.337** | 0.663 | +0.663 |

Δeffect versus 7B, paired per source — **11 of 12 exclude zero**:

| effect | 32B | 7B | Δ |
|---|---|---|---|
| rule without structure (J) | −0.038 | +0.071 | **−0.109** [−0.179, −0.043] |
| rule with structure (J) | −0.239 | +0.788 | **−1.027** [−1.120, −0.929] |
| structure without rule (recall) | +0.000 | −0.723 | **+0.723** [+0.658, +0.788] |
| structure with rule (J) | −0.217 | +0.005 | **−0.223** [−0.288, −0.158] |

**Two of the three mechanism conclusions are capacity-dependent:**

1. Structured-framing collapse **vanishes** at 32B.
2. The binary rule becomes **harmful** at 32B — it only pushes decisions further toward
   fake, and with recall already 0.978 that lands entirely on real images.
3. Structured evidence still provides no extra decision gain — and now actively hurts.

And the plainest configuration wins: `B0` score-only (J 0.918) beats every 32B evidence
configuration and every 7B cell.

The honest reading is that the earlier rounds' mechanism findings are **7B-specific**, and
the intended promotion to "stable across the model family" did **not** happen. The single
invariant is the role separation.

## 12. Explainability retrospective (X01)

No new inference; all from existing outputs plus GT masks.

| relation | value |
|---|---|
| Evidence→GT any-cell hit | 1.000 |
| Evidence→GT top-1 == GT primary | 0.940 |
| **Evidence→GT grid precision** | **0.979** |
| **Evidence→GT grid recall** | **0.531** |
| Explanation→Evidence hit (structured) | 1.000 |
| unsupported-region rate | 0.000 |
| Explanation→GT hit (structured) | 1.000 |
| Explanation→GT **recall** | ~0.52 |

> The structured explanation inherits **both** the evidence's high precision **and** its
> low recall. The model restates what the tool found, including what the tool missed.

Mirror intervention, on the pre-frozen eligible subset:

| | Explanation→Evidence | Explanation→GT |
|---|---|---|
| 7B paired Δ (mirrored − correct) | **+0.000** [+0.000, +0.000] | **−0.986** [−1.000, −0.957] |
| 32B paired Δ | **+0.000** [+0.000, +0.000] | **−0.971** [−1.000, −0.929] |

Faithfulness to the evidence is perfectly preserved while correctness is destroyed.
Explanation correctness is a function of evidence correctness.

Raw map vs structured, paired: Explanation→Evidence **+0.596**, Explanation→GT
**+0.340**, GT precision 0.657 → 0.995. Structured evidence does not merely make the model
better at *restating* — it also moves the explanation closer to the truth.

Unexpected asymmetry: on **real** images 7B names a suspicious region ~100% of the time
while 32B does so on only 12–34%. The previously noted "32B low citation rate" is
restraint on real images, not incapacity — on fake, its coverage matches 7B.

## 13. Semantic audit (X02/X03)

X02 froze everything before annotation: 100 fake + 50 real, the rubric, the schema, the
30-source repeat subset, and the metric definitions. The canonical conditions are 7B `D0`
and 32B `B2`, whose prompts are **byte-identical**, so the model is the only variable. The
real set excludes any source used as a fake unit, because fake units show the annotator
that source's real image as a comparison.

X03 produced 300 primary + 60 repeat annotations, 0 schema violations.

### Fake

| | 7B | 32B |
|---|---|---|
| overall supported | 0.040 | **0.490** |
| partially supported | 0.310 | 0.500 |
| **overall unsupported** | **0.610** | **0.010** |
| **manipulation claims unsupported** | **0.791** | **0.044** |
| claims per explanation | 5.4 | 5.1 |
| hallucinated object rate | 0.030 | 0.010 |

Paired on 100 identical sources: 32B has fewer unsupported claims in **81** cases, 7B in 3,
tie 16; overall support favours 32B in **86**, 7B in 1.

The key structural result:

> **7B is spatially right and semantically wrong.** Spatial hit is 100/100, yet 79.1% of
> its manipulation claims are unsupported — it acknowledges the manipulated region and
> then reasons to "real", or reads a 0.99 manipulation score as evidence of authenticity
> (`contradictory_claim_present` in 14/50 real explanations, versus **0/50** for 32B).

That is the earlier `acknowledged_manipulation_but_real` statistic, confirmed in natural
language.

Stratified by tampered area, 7B degrades monotonically as the region shrinks (claim UCR
0.373 large → 0.559 small); 32B shows no gradient.

### Real

FFER 0.060 (7B) vs 0.100 (32B) overall; on the cited subset 32B is 0.667 but **n = 6** and
is reported as descriptive only, with no inference, per instruction.

### Annotation reliability — and what it limits

| field | agreement | κ |
|---|---|---|
| object_correct | 0.933 | 0.868 |
| unsupported_claim_present | 0.917 | **0.815** (binary, valid) |
| contradictory_claim_present | 0.867 | 0.645 (binary, valid) |
| hallucinated_object_present | 0.967 | **−0.017** (degenerate) |
| overall_semantic_support | 0.717 | 0.589 |
| **manipulation_claim_supported** | **0.650** | 0.549 |

Overall agreement 0.825. Three caveats that must travel with these numbers:

1. **`manipulation_claim_supported` agreement is only 0.650**, and it carries the headline
   comparison. The direction survives because 0.128 vs 0.672 dwarfs the noise; the precise
   values should not be over-read.
2. **`hallucinated_object_present` κ is uninterpretable** (base-rate degeneracy), so HOR is
   low-reliability.
3. **The annotator is not an independent third party.** No Anthropic endpoint existed in
   this environment; annotation used blinded subagents of the same model family as the
   assistant. They saw no conversation history, no model identity, no verdicts and no
   TruFor scores — but this is not third-party ground truth and must never be described as
   such.

### A finding hiding in the sparsity

Across 300 explanations, evaluable **texture / lighting / boundary / geometry / physics**
claims number **0–3 each**. Both models talk almost exclusively about *location* and bare
*manipulation* assertions. Prompting alone did not elicit fine-grained forensic reasoning —
which is precisely the gap the next stage targets.

## 14. Stage summary

**Decision.** An external detector's scalar score is already the stable signal. Neither
the raw map nor a structured description of it adds discriminative value at either scale,
and at 32B the plainest prompt — score only — is the best configuration measured
(J 0.918).

**Spatial explanation.** Structured evidence buys near-perfect, auditable citation of the
tool's regions (agreement 1.000, unsupported-region rate 0.000), and it also moves the
explanation closer to ground truth. But that correctness is **borrowed** from the
evidence: mirror the evidence and the explanation follows it into the wrong place.

**Semantic explanation.** Model capacity matters a great deal here, unlike for decisions.
7B routinely produces a correct location with an unsupported manipulation explanation;
32B largely does not.

**The limit.** Prompting alone did not produce rich, reliable, fine-grained forensic
reasoning. The models name places and assert manipulation; they rarely support it with
texture, lighting, boundary, geometry or physics evidence that survives audit.

Hence the training-free stage ends here. The open question — *how to make an MLLM explain
**why** a region is forged* — is a supervision problem, and is scoped in
`docs/NEXT_STAGE.md`.
