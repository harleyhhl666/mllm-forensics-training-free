# EXPLAINABILITY_SEMANTIC_AUDIT_PROTOCOL.md

Phase X2 freeze. Frozen **before** any annotation is produced. Machine-readable twin:
`runs/explainability/x2_semantic_audit_frozen.json` (sha1 `b3a26bb7e48cbeacbe6ad524392edb4f56dd85db`).
No Qwen inference and no annotator call is part of this phase.

## 1. Question

Phase X1 settled the spatial questions. Structured evidence reaches GT with grid
precision 0.979; the MLLM reproduces that evidence almost exactly (Explanation→Evidence
1.000, unsupported-region rate 0.000); mirrored evidence moves the explanation with it
(Explanation→GT falls to 0.014 / 0.029). So spatial *faithfulness* and spatial
*correctness* are separable and both already measured.

This phase asks only:

> When the model states that a region has some anomaly, is that natural-language claim
> actually supported by the image?

Not classification accuracy. Not verdict correctness.

## 2. Canonical conditions

| model | condition | prompt sha1 |
|---|---|---|
| 7B | `D0_baseline` | `22b9ad9a944812ef…` |
| 32B | `B2_structured` | `22b9ad9a944812ef…` |

**These two prompts are byte-identical**, so the only variable between the two
explanation sets is the model. Asserted at freeze time.

The abstraction-round `E2` prompt (`4600f0a5807159cb…`) is **rejected**: it carries an
extra sentence explaining `area_ratio` and coordinate normalisation, which would make
the 7B text non-comparable to 32B.

Neither canonical condition contains the binary rule. The rule changes decision policy
and its wording could colour the reason text; this phase studies the explanation alone.

## 3. Samples

100 fake + 50 real = **150 sources**, seed `20260918`, frozen before annotation.

### Fake (100 of 184)

Stratified proportionally over **GT area quartile × grid spread**. Quartile cuts
`0.0149 / 0.0383 / 0.1089`.

| | selected |
|---|---|
| area quartile | Q1 25, Q2 26, Q3 24, Q4 25 |
| grid spread | 1 cell 23, 2–3 cells 48, ≥4 cells 29 |
| mask_type | bbox 40, segm 60 (population 73 / 111) |
| categories | 12 distinct |

**Not selected on verdict, explanation quality, or spatial correctness.** The resulting
fake-verdict rates (7B 0.110, 32B 0.980) are a consequence of the sample, not a
criterion applied to it.

### Real (50)

Drawn only from the **84 sources not used as fake units**. A fake unit shows the
annotator that source's real image as "IMAGE B, a source photograph"; reusing it as a
real unit would mean the annotator had already seen it framed as a source.

One shared real set audited for **both** models — never a per-model set. Resulting
citation status: 7B cited 49/50, 32B cited 6/50. Both 32B classes present, asserted.

## 4. Annotation units

Every selected source contributes one 7B and one 32B explanation:

| | units |
|---|---|
| fake | 200 |
| real | 100 |
| **primary total** | **300** |

This gives paired 7B-vs-32B comparison on identical sources.

## 5. Anonymisation

IDs `EXPL-0001`…`EXPL-0300`, order shuffled with seed `20260920`. Model codes `M1`/`M2`
live only in the mapping file. **Each explanation is annotated independently** — the
annotator never sees two explanations of one image together and therefore cannot make a
relative comparison.

Leak scan at freeze time found no occurrence of `7b`, `32b`, `qwen`, `trufor`,
`baseline`, `condition`, `verdict`, `score`, or `larger model` in either prompt, and
none of the banned leading phrasings ("verify whether the model is correct", "this is a
fake", "models often hallucinate", "structured evidence is accurate").

## 6. What the annotator sees

**Fake**: the photograph (IMAGE A), the paired source (IMAGE B), the binary mask
(IMAGE C), an outlined overlay (IMAGE D), the explanation, `referenced_regions`, the
3×3 grid definition.

**Real**: the photograph, the explanation, `referenced_regions`, the grid definition.
**No paired counterpart** — a comparison image could induce suspicion, and the real
question is whether the explanation makes unsupported claims about *this* image.

**Withheld in both**: TruFor score, structured evidence JSON, final verdict, model
identity, condition name, every prior metric.

## 7. Claim extraction and schema

The annotator must first decompose the explanation into atomic claims, then type and
judge each one; an overall impression alone is rejected. Nine fixed `claim_type` values
(location, object, texture, lighting, boundary, geometry, physics, manipulation, other)
and five `support` values (supported, partially_supported, unsupported, uncertain,
not_applicable), all defined verbatim in the prompt.

Schema sha1 `4d9d3df7479d798fe62839c0985c2c060f9c4d71`.

### GT usage limits

The mask may be used to check a stated location and, against IMAGE B, whether content
differs. **The mask never establishes a texture, lighting, geometry or physics
anomaly** — those must be visible. A specific mechanism claim (e.g. copy-move
provenance) requires evidence for that mechanism, not merely a difference from the
source.

## 8. Uncertainty policy

`uncertain` is first-class: never merged into `unsupported`, never counted as an error.
Hedged wording about something genuinely ambiguous is `uncertain`. The prompt states
that guessing is worse than `uncertain`.

`not_applicable` is **excluded from denominators** and never counted as correct.

**The annotator's output is an assistive measurement, not ground truth.** Every reported
metric carries that caveat; reliability is quantified by the repeat subset.

## 9. Repeat subset

20 fake + 10 real sources (seed `20260921`) → **60 repeat units** (both models per
source). Re-annotated in a fresh independent context, identical rubric and image
preprocessing, first annotation never shown.

If a second annotator configuration is used, **same-model repeat consistency and
cross-model annotator agreement are reported separately**, never merged. Disagreeing
items are listed for optional human spot-check and are never silently resolved.

Exclusion: an explanation is dropped only after unparseable JSON twice; exclusions are
counted and reported, never backfilled.

## 10. Metrics

Per claim type, **discrete proportions** of supported / partially_supported /
unsupported / uncertain. `partial` is *not* collapsed to 0.5 in the primary report.
Denominator for each type = explanations that actually made a claim of that type.

- **UCR** = unsupported / evaluable, overall and per type
- **Explanation-level** support proportions
- **HOR** = explanations containing a nonexistent object / all explanations
- **FFER** (real only) = real explanations asserting an unsupported forensic anomaly /
  all real explanations
- **Paired 7B vs 32B**: means *and* paired counts of 7B-better / tie / 32B-better
- **Citation split**: cited vs no-citation reported separately; a no-citation
  explanation is **not** an explanation error
- **Stratified by spatial correctness** (Phase X1 Explanation→GT hit) — does getting the
  location right go with getting the semantics right?
- **Stratified by GT extent** (small/medium/large) — small edits may be harder to verify

## 11. Budget

| | |
|---|---|
| primary annotations | 300 |
| repeat annotations | 60 |
| **total calls** | **360** |
| images passed | 900 (4 per fake unit, 1 per real unit) |
| prompt size | ~1140 tokens + images |
| explanation length | mean 245 chars, max 497 |

The 4-image fake units dominate cost.

## 12. Freeze audit

All twelve checks PASS: sample counts; fake/real disjointness; both models matched for
every unit; all 4 area quartiles; all 3 spread classes; both mask types; repeat subset
fixed; anonymisation hides model identity; no verdict/score/condition in prompts; schema
fixed; uncertain and not_applicable non-punitive; real set shared across models.

A first freeze attempt **failed** the disjointness check, which surfaced the IMAGE B
contamination described in §3; the real pool was restricted rather than the check
weakened.

## 13. Stop condition

Phase X2 ends at freeze. No annotation started, no Qwen run, no explanation modified, no
case hand-picked, no sample changed on the basis of reason content.
