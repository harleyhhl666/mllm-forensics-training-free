# TRUFOR_STRUCTURED_FRAMING_PROTOCOL.md

Frozen protocol for the **structured-evidence framing / region-presence control**.
Frozen **before** any MLLM inference. Machine-readable twin:
`runs/trufor_framing/trufor_structured_framing_frozen.json`
(protocol sha1 `15e146d7d994da76cffceb7b47ab2752f21d3186`)

## Confound being addressed

In the field-ablation round, **S0 used a score-only prompt while S1–S5 used a
structured-evidence prompt**. So three things were confounded:

1. structured prompt framing,
2. presence of a region-level evidence block,
3. actual region field values.

This round separates them and does nothing else. Explicitly out of scope: area
magnitude, probability magnitude, spatial correctness, raw map, 32B.

## Conditions

| id | prompt | payload |
|---|---|---|
| **P0** score only | score-only template | *(none)* |
| **P1** framing only | structured template | *(none)* |
| **P2** empty regions | structured template | `{"regions": []}` |
| **P3** placeholder | structured template | `{"regions":[{"region_id":"region_1"},…]}` |
| **P4** true location | structured template | `{"regions":[{"grid_location":"center-right"},…]}` |
| **P5** neutral metadata | structured template | `{"regions":[{"region_id":1,"order":1},…]}` |

All conditions are **single-image**.

## Prompt parity

| template | sha1 |
|---|---|
| P0 score-only | `8775b797feccec20afffa2ca799cd491bbae50e0` (identical to S0) |
| **P1–P5 structured** | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` (identical to S1–S5) |

Both templates are reused **byte-identically**, so **P0 replicates S0** and **P4
replicates S1** exactly.

**P1 rendering detail (disclosed):** the structured template contains a payload
placeholder. For P1 that placeholder is replaced by the **empty string**, so every
instruction word is byte-identical to P2–P5 and only the payload area is blank. This
is the closest realisable form of "structured prompt, no structured block".

**Verified:** P2–P5 instruction text is identical once the payload is removed.

## Payload rules

Region **count** in P3/P4/P5 is inherited from the frozen abstraction layer (thr 0.5,
min area 0.001, top-K 3, ranked by `integrated_probability`), so container size is held
fixed across those three.

- **P3** `region_id` is a *string entry label* only — no location, size, probability or
  confidence. The prompt never ascribes forensic meaning to it.
- **P5** uses two *numeric* fields (`region_id`, `order`) to control numeric-token
  presence and JSON length. **No random numbers.**

### Payload audit (all 368 evidence packages)

**0 problems** — counts preserved, key sets exact.
Region count distribution: `{0: 10, 1: 218, 2: 77, 3: 63}`.

| condition | mean JSON chars |
|---|---|
| P0 / P1 | 0.0 |
| P2 | 15.0 |
| P3 | 54.2 |
| P5 | 58.8 |
| P4 | 63.2 |

P5 > P3 by construction (two fields vs one); P4 sits between. **Exact length matching
is not claimed** — this is a control on numeric-token presence and structural
complexity, not a byte-length match.

## Causal chain

| contrast | reads as |
|---|---|
| `P1 − P0` | prompt framing effect |
| `P2 − P1` | empty-container effect |
| `P3 − P2` | region-object effect |
| `P4 − P3` | actual spatial-information effect |
| `P5 − P3` | generic metadata effect |
| supplementary | `P4 − P0` (should reproduce S1 − S0 = −0.696), `P5 − P4` |

Primary metrics per condition: fake recall, real FPR, specificity, Youden J, with
paired source-level bootstrap 95 % CI and McNemar as supplementary.

## Pre-registered cases

- **F1 — prompt framing drives collapse:** `P1 ≪ P0` while P2/P3/P4 differ little from
  P1
- **F2 — region presence drives collapse:** `P1 ≈ P0` but `P2` or `P3 ≪ P1`
- **F3 — actual forensic fields drive collapse:** P1 ≈ P2 ≈ P3 ≈ P0 but `P4 ≪ P3`
- **F4 — generic structured-token overload:** P3 and P5 both fall similarly

## Reason audit

Deterministic keyword rules (no LLM judge): `score_high`,
`manipulation_acknowledged`, `locality` (13 patterns), `evidence_weak`.

Key metrics:

- **`acknowledged_manipulation_but_real`** — verdict *real* **and** manipulation
  acknowledged **and** (locality OR evidence_weak)
- **`locality_without_locality_data`** — locality language in **P1/P2/P3/P5**, which
  carry *no* location, area or probability value. If present, the locality narrative is
  activated by framing/schema rather than derived from evidence values.

## Schema artifact policy

The output schema is **unchanged** to preserve comparability, so
`referenced_regions` still enumerates the nine cells. The all-9 echo artifact is
recorded separately and is **not used for any mechanism judgement** this round —
primary is decision, not faithfulness.

## Scope limit

**No numeric counterfactual.** `area_ratio`, `mean_probability` and `max_probability`
are never altered. A numeric counterfactual is warranted only if P1/P2/P3 prove
near-harmless and P4 alone collapses.

## Replication checks

**P0 must reproduce 0.837 and P4 must reproduce 0.141** within sampling noise. Both are
exact prompt+payload replicates, so a large deviation indicates a pipeline problem, not
a finding.

## Samples and budget

184 fake + 184 paired real, 184 unique coco_id, inherited from
`trufor_structured_field_ablation_frozen.json` (sha1 `b5722533ed21…`).
Qwen2.5-VL-**7B** `cc594898137f460b…`, decoding and visual budget unchanged.

6 × 2 × 184 = **2208 inferences**, all single-image, ≈99 min.

## Prior results

| | recall |
|---|---|
| S0 score only | 0.837 |
| S1 location | 0.141 |
| S2 area | 0.196 |
| S3 probability | 0.103 |
| S4 loc+area | 0.212 |
| S5 full | 0.114 |

All structured FPR = 0.000; spatial agreement 1.000; shift responsiveness 1.000.
The `area_ratio` hypothesis was **refuted** (S4 − S1 = **+0.071**).

## Forbidden

prompt tuning, numeric counterfactual, schema change, raw map, 32B, random placeholder
values.
