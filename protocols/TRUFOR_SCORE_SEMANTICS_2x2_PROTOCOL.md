# TRUFOR_SCORE_SEMANTICS_2x2_PROTOCOL.md

Frozen protocol for the **Score × Task-Semantics 2×2 causal control**. Frozen
**before** any inference. Machine-readable twin:
`runs/trufor_2x2/trufor_score_semantics_2x2_frozen.json`
(protocol sha1 `6f3fea39549b67ff8a5b00db7f7ff97934fc1001`)

## Question

> Is D1's success generic binary task clarification, or does clarification
> **specifically** repair the policy shift induced by structured-evidence framing?

Answered by the factorial interaction `(A3 − A2) − (A1 − A0)`.

## The 2×2

| cell | score line | structured JSON | binary rule | source |
|---|---|---|---|---|
| **A0** | yes | no | no | reuse frozen S0 / P0 cell |
| **A1** | yes | no | **yes** | **NEW — 368 inferences** |
| **A2** | yes | **yes** | no | reuse D0 |
| **A3** | yes | **yes** | **yes** | reuse D1 |

The TruFor score line is present in **all four** cells. The two manipulated factors are
(a) structured-evidence JSON present/absent, (b) binary rule present/absent.

## A0 provenance

Two earlier rounds produced this cell independently. Verified: **rendered prompt
identical 368/368, verdict agreement 1.000**. A0 is unambiguous; this freeze designates
the **field-round `S0_score_only`** cell as canonical.

## Prompts

| cell | sha1 |
|---|---|
| A0 | `8775b797feccec20afffa2ca799cd491bbae50e0` |
| **A1** | `ef68309f567c7a61517447641701add20f546217` |
| A2 | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` |
| A3 | `9167aa08c1a09d683974d68ac10821f428ee6658` |

A0/A2/A3 hashes match their frozen originals.

**A1 is built by the same transformation that produced A3 from A2** — the rule text is
inserted immediately before the anchor `Report strictly as JSON`:

- `A1 = A0 + rule` at that anchor
- `A3 = A2 + rule` at that anchor

**Verified:** the rule text is byte-identical in A1 and A3, and *removing* the rule
returns `A1 → A0` and `A3 → A2` byte-exactly. So rule text and placement are constant
across both rule arms — the only thing that differs between arms is the presence of
structured evidence.

Rule text (unchanged from D1):
> For this task, the authenticity label is binary. If there is credible evidence that
> any part of the image has been genuinely manipulated, the image should be classified
> as fake, regardless of how small or localized the manipulated region is.

## Contrasts

| name | definition |
|---|---|
| rule effect without structured | `A1 − A0` |
| rule effect with structured | `A3 − A2` |
| structured effect without rule | `A2 − A0` |
| structured effect with rule | `A3 − A1` |
| **interaction** | `(A3 − A2) − (A1 − A0)` |
| equivalent form | `(A3 − A1) − (A2 − A0)` — algebraically identical |

Metrics per cell: fake recall, real FPR, specificity, Youden J. Paired effect size →
source-level bootstrap 95 % CI; McNemar supplementary.

## How the interaction is read

- **≈ 0** → the rule helps about equally with and without structured evidence: D1's gain
  is **generic task clarification**
- **≫ 0** → the rule helps far more when structured evidence is present: clarification
  **specifically repairs** the framing-induced shift
- **≪ 0** → the rule helps mainly *without* structured evidence

**Ceiling caveat (pre-registered):** A0 already sits at recall **0.837**, so the
score-only arm has limited head-room. A small `A1 − A0` could reflect that ceiling
rather than an absent rule effect. J and FPR are reported alongside recall so a ceiling
artefact stays visible — in particular, the rule could still move FPR upward in the
score-only arm even with recall pinned near the ceiling.

## Known cells (three of four already measured)

| cell | recall | FPR | J |
|---|---|---|---|
| A0 | 0.837 | 0.011 | +0.826 |
| A2 | 0.114 | 0.000 | +0.114 |
| A3 | 0.967 | 0.065 | +0.902 |
| A1 | — | — | to be measured |

Illustrative arithmetic only (**not predictions**): if A1 landed at A0's recall the
interaction would be +0.853; if A1 reached A3's recall it would be +0.723.

## Evidence package

Structured cells (A2/A3): `grid_location`, `area_ratio`, `mean_probability`,
`max_probability`, `centroid`, produced by `abstract()` then `strip_fields()`, imported
unchanged. Score-only cells (A0/A1): the frozen score line only, no JSON. **No raw map
anywhere.**

## Samples, model, budget

184 fake + 184 paired real, 184 unique coco_id, inherited from
`trufor_task_semantics_frozen.json` (sha1 `d0617e952729…`). Qwen2.5-VL-**7B**
`cc594898137f460b…`, decoding, score formatting and visual budget unchanged.

**368 new inferences (cell A1 only)**, single-image, ≈15 min. **A0/A2/A3 are not
re-run**; their frozen verdicts are read from disk.

## Forbidden

prompt tuning, re-running A0/A2/A3, raw map, 32B, schema change, new samples.
