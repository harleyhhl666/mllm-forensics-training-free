# TRUFOR_EVIDENCE_ABSTRACTION_PROTOCOL.md

Frozen protocol for the **Evidence Abstraction Layer** experiment. Frozen **before**
any MLLM inference. Machine-readable twin:
`runs/trufor_abstraction/trufor_evidence_abstraction_frozen.json`
(protocol sha1 `ec1118a2f9b8fe17bad237892630752d0ade4d2f`)

## Question

> Does converting the low-level forensic map into explicit structured spatial evidence
> let the MLLM use spatial evidence more effectively and more faithfully?

Motivation: both 7B and 32B show a spatial-correspondence effect of **−0.027** with
`Δ = +0.000 [−0.065, +0.065]`, and 32B still claims the map supports its verdict on
deliberately misaligned maps. Rather than perturbing raw heatmaps further, this round
changes the **representation** of the evidence.

## Abstraction layer

Deterministic, training-free, **no ground truth, no second LLM, never reads the
authenticity label**. It only converts a TruFor map into structured statistics.

| parameter | frozen value |
|---|---|
| binary threshold | `map > 0.5` |
| connectivity | 4-connected |
| minimum component area | `area / image ≥ 0.001` |
| top-K | 3 |
| ranking | `integrated_probability` = `sum(map probability in component)` |
| coordinates | normalised `[0,1]`, origin top-left |
| grid | 3×3, `CELL_NAMES` reused from `forensic_tools.py` |
| grid rule | component **centroid** sets the primary location even if the component spans cells |
| empty case | `regions: []` — regions are **never** invented |
| rounding | score 3, area_ratio 4, prob 3, centroid 3 |

Connected components use a self-contained numpy/stdlib BFS: **scipy is absent from the
frozen `qwen_vl` environment and is deliberately not installed.**

### Example output

```json
{
 "image_manipulation_score": 0.999,
 "regions": [
  {"grid_location": "center-right", "area_ratio": 0.0202,
   "mean_probability": 0.92, "max_probability": 0.998, "centroid": [0.908, 0.574]},
  {"grid_location": "top-right", "area_ratio": 0.0012,
   "mean_probability": 0.743, "max_probability": 0.963, "centroid": [0.997, 0.262]}
 ]
}
```

## Conditions

| id | image | score | raw map | structured | 
|---|---|---|---|---|
| **E0** score only | ✓ | ✓ | | |
| **E1** raw map | ✓ | ✓ | ✓ | |
| **E2** structured | ✓ | ✓ | | ✓ |
| **E3** both | ✓ | ✓ | ✓ | ✓ |
| **E4** mirrored structured | ✓ | ✓ | | ✓ (mirrored) |

E0/E2/E4 are single-image; E1/E3 are two-image.

## Prompts

Neutral wording as specified; **no** AUROC, benchmark, GT, threshold, "trusted" or
"accurate". Bias/leak word scan: **no hits**. No Stage-A summary anywhere.

| condition | sha1 |
|---|---|
| E0 | `8775b797feccec20afffa2ca799cd491bbae50e0` |
| E1 | `0b6f915a193ec98af25bc1272bcc2e8b39e511ed` |
| E2 | `4600f0a5807159cb4f1fe5fc2d2bf068f027d3a6` |
| E3 | `d28dca6b28091e9eadad94d0b6eaa840bfb050a6` |
| E4 | `4600f0a5807159cb4f1fe5fc2d2bf068f027d3a6` (**byte-identical to E2**) |

E2 and E4 share one prompt; only the JSON content differs. This is what makes E4 a
clean intervention.

## Output schema

```json
{"final_verdict": "real|fake",
 "referenced_regions": ["3x3 grid location(s), [] if none"],
 "reason": "one or two sentences"}
```

`referenced_regions` is compared **mechanically** against the evidence's
`grid_location` values — no LLM judge.

## Mirror control (E4)

| | |
|---|---|
| grid | left ↔ right; centre column unchanged |
| centroid | `x' = 1 − x` |
| unchanged | `area_ratio`, `mean_probability`, `max_probability`, score |
| never mirrored | **the original image** |

Only *where the evidence says the anomaly is* changes; evidence strength is identical.

## Metrics

Primary: fake recall, real FPR, specificity, Youden J.

| contrast | meaning |
|---|---|
| `E2 − E0` | structured benefit |
| `E1 − E0` | raw-map cost (replication) |
| **`E2 − E1`** | **structured vs raw — most important** |
| `E3 − E2` | raw map after abstraction: helps, inert, or harms |
| `E4` vs `E2` | primarily on `referenced_regions`, not verdict |

All with paired effect size → source-level bootstrap 95 % CI → McNemar
(supplementary).

### Faithfulness

- **evidence citation rate** — `P(referenced non-empty | evidence has ≥1 region)`
- **spatial agreement** — referenced set intersects evidence locations
- **exact agreement** — referenced set equals evidence locations
- **unsupported-region rate** — cites a location absent from the evidence
- **shift responsiveness** — `E2→E4`, `P(referenced region moves with the evidence)`
- **hallucinated grounding** — cites a region while the evidence list is empty

## Full-corpus extraction stats (computed before freezing)

| | fake | real |
|---|---|---|
| region-count distribution | 1:163, 2:15, 3:6 | 0:10, 1:55, 2:62, 3:57 |
| empty region lists | **0/184** | **10/184** |
| mean regions | 1.147 | **1.902** |
| area_ratio median | 0.0251 | **0.00395** |
| top locations | center 78, bottom-center 38 | center 66, bottom-center 57 |
| **E4-eligible** | **70/184** | **80/184** |

**Two limitations disclosed up front:**

1. **The 3×3 mirror leaves the centre column and `center` invariant**, so
   shift-responsiveness is measurable only on the eligible subset (fake 70, real 80).
   This eligibility rule is frozen here, before inference, and is not selected from
   results.
2. **Region presence carries some label information** — fake yields a non-empty list
   184/184 vs real 174/184, and real images average *more but much smaller* regions
   (1.90 regions, median area 0.004) than fake (1.15, 0.025). Any E2 gain must be
   read against this, not as pure spatial reasoning.

## Pre-registered cases

- **A — representation bottleneck**: `E2 > E1`, `E2 ≳ E0`, high spatial agreement, E4
  explanations follow the shift → the limitation is in how evidence is *represented*
- **B — spatial evidence intrinsically unused**: `E2 ≈ E0`, low agreement, no shift
  response
- **C — explanation-only gain**: `J(E2) ≈ J(E0)` but faithfulness clearly improves
- **D — structured evidence harms**: `E2 < E0` → report directly, no prompt tuning

## Success criterion

> Preserve most of the scalar discrimination while substantially improving spatially
> faithful explanations — **accuracy preservation + explanation faithfulness**.

Structured evidence is **not** required to beat score-only accuracy; the TruFor scalar
score is already strong.

## Samples, model, budget

184 fake + 184 paired real, inherited unchanged from
`trufor_score_map_conflict_frozen.json` (sha1 `f65614514b09…`); paired real evidence
available 184/184; `n_samples == n_sources`.

**7B primary** (snapshot `cc594898…`); 32B deferred pending 7B results.

5 conditions × 2 labels × 184 = **1840 inferences**, ≈83 min at 2.7 s.

## Forbidden

prompt tuning, GT in the abstraction layer, second LLM, training, threshold tuning on
MLLM results, confidence map.
