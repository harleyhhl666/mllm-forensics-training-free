# TRUFOR_STRUCTURED_FIELD_ABLATION_PROTOCOL.md

Frozen protocol for the **structured forensic evidence field ablation**. Frozen
**before** any MLLM inference. Machine-readable twin:
`runs/trufor_fields/trufor_structured_field_ablation_frozen.json`
(protocol sha1 `b5722533ed218eff9d48c990a540229d1bd5d645`)

## Question

> Which structured forensic field causes the MLLM to underweight an otherwise highly
> discriminative image-level score?

`area_ratio` is treated as a **hypothesis under test, not an established cause**. It
gets its own isolated condition alongside location and probability.

## No raw map anywhere

Every condition is **single-image** (original image only). The localization
visualization is withheld so the structured fields are decomposed in isolation.

## Conditions — only which KEYS survive differs

| id | fields in each region |
|---|---|
| **S0** score only | *(no structured JSON)* |
| **S1** location | `grid_location` |
| **S2** area | `area_ratio` |
| **S3** probability | `mean_probability`, `max_probability` |
| **S4** location + area | `grid_location`, `area_ratio` |
| **S5** full | `grid_location`, `area_ratio`, `mean_probability`, `max_probability`, `centroid` |

Region identity, `top_k = 3` and ranking by `integrated_probability` are inherited
**unchanged** from the frozen abstraction layer (`abstract()` imported directly, thr
0.5, min area 0.001). Decimal precision inherited: score 3, area_ratio 4, prob 3,
centroid 3.

### Rendered JSON, first fake source (score 0.999, 2 regions)

```
S1: {"regions": [{"grid_location": "center-right"}, {"grid_location": "top-right"}]}
S2: {"regions": [{"area_ratio": 0.0202}, {"area_ratio": 0.0012}]}
S3: {"regions": [{"mean_probability": 0.92, "max_probability": 0.998},
                 {"mean_probability": 0.743, "max_probability": 0.963}]}
S4: {"regions": [{"grid_location": "center-right", "area_ratio": 0.0202}, ...]}
```

## Prompt parity

| condition | sha1 |
|---|---|
| S0 | `8775b797feccec20afffa2ca799cd491bbae50e0` |
| **S1–S5** | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` (**one byte-identical prompt**) |

Only the JSON content differs across S1–S5; S0 differs only by having no structured
block. Field-interpretation / bias wording scan: **no hits**. The prompt never says
what `area_ratio` or the probabilities mean, and never says whether small regions
matter.

**Disclosed protocol change:** the previous round's E2 prompt explained that
`area_ratio` is "the fraction of the image covered by the region" and that coordinates
are normalised. **That explanatory text is removed here** to satisfy prompt parity and
the ban on explaining field meaning. **S5 is therefore a field-set replicate of E2,
not a prompt-identical one** — cross-round comparison is descriptive only; the
within-round S5 − S0 contrast is the valid one.

## Field-presence audit

Over all 368 evidence packages (184 fake + 184 real): **region counts preserved and
key sets exact, 0 problems**. No condition puts the image-level score inside the JSON —
the score is always the separate frozen score line.

## Metrics

Primary: **decision** — fake recall, real FPR, specificity, Youden J.

| contrast | |
|---|---|
| location effect | `S1 − S0` |
| **area effect** | `S2 − S0` |
| probability effect | `S3 − S0` |
| location + area | `S4 − S0` |
| full penalty | `S5 − S0` |
| supplementary | `S4 − S1`, `S4 − S2`, `S5 − S4` |

All with paired effect size → source-level bootstrap 95 % CI → McNemar
(supplementary).

Secondary faithfulness (`referenced_regions`) is interpretable **only** for the
location-bearing conditions: **S1, S4, S5**.

## Pre-registered cases

- **area-driven collapse** — `S2 ≪ S0` and `S1 ≈ S0`, with S4/S5 also down
- **generic structured overload** — S1, S2, S3 all fall similarly
- **probability conflict** — mainly `S3 ≪ S0`; inspect whether mean/max are misread
- **interaction only** — S1 ≈ S2 ≈ S3 ≈ S0 but `S5 ≪ S0`

## Reason audit (deterministic, no LLM judge)

Patterns: `area_small`, `area_limited`, `evidence_weak`, `score_high`,
`manipulation_acknowledged`.

**Key metric — `acknowledged_manipulation_but_real`:** verdict is *real* **and** the
reason acknowledges manipulation **and** it calls the area small/limited or the
evidence weak. This is a forensic reasoning inconsistency: the model concedes
localized manipulation yet still returns Real.

## Scope limit

**Field ablation only.** No counterfactual `area_ratio` values are substituted. If S2
implicates area, the counterfactual intervention is a **separate later round**, so the
two are not confounded.

## Samples, model, budget

184 fake + 184 paired real, 184 unique coco_id, inherited unchanged from
`trufor_evidence_abstraction_frozen.json` (sha1 `ec1118a2f9b8…`).
Qwen2.5-VL-**7B** snapshot `cc594898137f460b…`, decoding and visual budget unchanged.

6 conditions × 2 labels × 184 = **2208 inferences**, all single-image, ≈88 min at
2.4 s.

## Prior results for reference

| | recall | FPR | J |
|---|---|---|---|
| score only (E0) | 0.837 | 0.011 | **0.826** |
| score + raw map (E1) | 0.565 | 0.016 | 0.549 |
| full structured (E2) | 0.174 | 0.011 | **0.163** |

E2 spatial agreement 1.000, mirrored-evidence following 1.000.
Region `area_ratio` median: fake 0.0251, real 0.00395.

## Forbidden

prompt tuning, explaining what fields mean, counterfactual area values, raw map, new
forensic knowledge, threshold changes, schema changes.
