# TRUFOR_TASK_SEMANTICS_PROTOCOL.md

Frozen protocol for the **task-semantics intervention / verdict decomposition**
experiment. Frozen **before** any MLLM inference. Machine-readable twin:
`runs/trufor_semantics/trufor_task_semantics_frozen.json`
(protocol sha1 `d0617e9527291cf151dbbc7382ac3c4f23068c3e`)

## Question

> Why does structured forensic framing make the model concede manipulation yet still
> output Real?

This round separates four candidate causes: **binary-rule misunderstanding**,
**presence-judgement failure**, **extent/existence confusion**, and **final
arbitration failure**.

Out of scope: raw map, confidence map, new detector, numeric counterfactual, threshold
tuning, LoRA, training, gate, verifier, 32B, new dataset, and any further field
ablation.

## Evidence package — constant across all four conditions

**Full structured payload**: `grid_location`, `area_ratio`, `mean_probability`,
`max_probability`, `centroid`. Extraction inherited unchanged from the frozen
abstraction layer (thr 0.5, min area 0.001, top-K 3, ranked by
`integrated_probability`). **No raw map.** The payload is a constant — only the
decision instruction changes.

## Conditions

| id | added instruction | schema |
|---|---|---|
| **D0** baseline | *(none)* | base |
| **D1** binary rule | binary authenticity rule | base |
| **D2** decomposition | rule + presence/extent fields | decomposed |
| **D3** presence-first | rule + fields + reasoning order | decomposed |

| condition | prompt sha1 |
|---|---|
| D0 | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` |
| D1 | `9167aa08c1a09d683974d68ac10821f428ee6658` |
| D2 | `91a25fb567947755d2bf31668f79178c644d2c8f` |
| D3 | `04e101c80c10e59beb4de4e1b519d2e25b7aa517` |

**Verified parity:**

- **D0 is byte-identical to the frozen `S5_full` prompt** — the replication check is
  exact, not approximate
- all four share the same opening **599 characters**
- **`D3` minus the ordered block == `D2`** exactly
- bias/leak wording scan: **none**

### The three inserted blocks, verbatim

**Binary rule (D1, D2, D3):**
> For this task, the authenticity label is binary. If there is credible evidence that
> any part of the image has been genuinely manipulated, the image should be classified
> as fake, regardless of how small or localized the manipulated region is.

**Decomposition (D2, D3):**
> Report manipulation existence and manipulation extent as separate judgements. The
> field manipulation_present states whether any genuine manipulation exists. The field
> manipulation_extent only describes how far it spreads. Extent does not substitute for
> existence: a small extent is not a reason to set manipulation_present to false. If
> manipulation_present is true, then under the binary authenticity rule final_verdict
> is fake.

**Ordered arbitration (D3):**
> Decide in this order. Step 1: judge whether any genuine manipulation exists and set
> manipulation_present. Step 2: only if it exists, judge how far it spreads and set
> manipulation_extent. Step 3: apply the binary authenticity rule to set final_verdict.
> Do not let the extent judgement from Step 2 revise the existence judgement from
> Step 1.

The rule states the **task definition only**. It never says the tool is reliable, never
asserts this evidence is genuine, and never reveals an answer.

## Output schemas

- **D0/D1** — `final_verdict`, `referenced_regions`, `reason`
- **D2/D3** — plus `manipulation_present`, `manipulation_extent`

**The model always produces `final_verdict` itself.** No external program ever derives
or overrides it from `manipulation_present`.

## Metrics

Primary decision, per condition: fake recall, real FPR, specificity, Youden J.

| contrast | |
|---|---|
| rule clarification | `D1 − D0` |
| decomposition | `D2 − D0` |
| presence-first | `D3 − D0` |
| ordered arbitration benefit | `D3 − D2` |

All with paired effect size → source-level bootstrap 95 % CI → McNemar
(supplementary).

### Presence-level (D2/D3 only)

`presence_tpr = P(present | fake)`, `presence_fpr = P(present | real)`,
`presence_j = tpr − fpr`.

**Diagnostic logic:** if presence is accurate but the verdict is poor, the failure sits
in **final arbitration**; if presence is also poor, the problem remains in **evidence
interpretation**.

### Presence–verdict consistency

- **A (primary)** — `manipulation_present = true AND final_verdict = real`. This is the
  structured-field version of last round's `acknowledged_manipulation_but_real` —
  **measurable directly, no regex**.
- **B** — `manipulation_present = false AND final_verdict = fake`

### Extent

Distribution of `none` / `localized` / `widespread` per label, and the key quantity
**`P(final_verdict = real | present = true, extent = localized)`**. If high in D0/D2 but
clearly lower in D1/D3, the model was treating "localized" as an authenticity
exemption.

### Faithfulness (secondary)

Citation rate, spatial agreement, unsupported-region rate. The goal is recall recovery
**without** losing the spatial faithfulness already achieved (agreement was 1.000 in the
abstraction round).

## Reason audit

Deterministic keyword rules, no LLM judge: `locality`, `score_high`,
`manipulation_acknowledged`, `evidence_weak`, plus two new classes:

- **`local_exemption`** — "does not necessarily make the whole image fake", "may still
  be considered real"
- **`rule_consistent`** — "any genuine manipulation", "regardless of how small", "even
  if localized"; expected mainly in D1/D3

## Pre-registered cases

- **T1 task-semantics misunderstanding** — `D1 ≫ D0`, recall largely restored, FPR not
  clearly up, present-true-but-real sharply down
- **T2 arbitration-layer failure** — presence accurate in D2/D3 but verdict stays poor
  and present-true-but-real persists
- **T3 decomposition fixes arbitration** — `D3 > D2` with presence preserved and
  consistency clearly improved
- **T4 framing failure persists** — D1/D2/D3 all stay at low recall → the policy shift
  is deeper than a task-definition misunderstanding

## Replication check

**D0 must land near recall 0.114** (field round `S5_full`). D0 uses that exact prompt,
so a large deviation means **pipeline drift, not a finding**.

## Samples and budget

184 fake + 184 paired real, 184 unique coco_id, inherited from
`trufor_structured_field_ablation_frozen.json` (sha1 `b5722533ed21…`).
Qwen2.5-VL-**7B** `cc594898137f460b…`, decoding and visual budget unchanged.

4 × 2 × 184 = **1472 inferences**, all single-image, ≈66 min. D2/D3 emit more fields, so
expect somewhat longer outputs.

## Outputs

`verdicts.jsonl`, `run_meta.json`, `analysis.txt`,
`presence_extent_analysis.json`, `inconsistency_analysis.json`

## Prior results

| | recall | FPR | J |
|---|---|---|---|
| score-only | 0.837 | 0.011 | 0.826 |
| structured framing, no payload | 0.125 | 0.005 | 0.120 |
| full structured (S5) | 0.114 | — | — |

framing effect on recall **−0.712**; locality-without-locality-data P1 0.489 / P2
0.989; `acknowledged_manipulation_but_real` P0 0.000 / P2 0.957.

## Goal

**Mechanism verification, not prompt search.** Only if D1/D3 restore decision utility
*while preserving* spatial faithfulness is a formal **Evidence Arbitration Layer**
justified.
