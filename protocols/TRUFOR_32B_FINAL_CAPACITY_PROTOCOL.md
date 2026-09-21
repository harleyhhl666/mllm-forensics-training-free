# TRUFOR_32B_FINAL_CAPACITY_PROTOCOL.md

Frozen protocol for the **32B capacity replication of the final TruFor integration
findings**. Frozen **before** any inference. Machine-readable twin:
`runs/trufor_32b_final/trufor_32b_final_capacity_frozen.json`
(protocol sha1 `9cf55f9fe3ea461507f60fec0a2febbeff25e3ca`)

## Question

**Not** "is 32B more accurate than 7B." The question is whether three mechanism
conclusions established on 7B survive a 4.5× capacity increase:

1. binary task semantics matter for decision quality
2. structured forensic evidence badly harms decisions under wrong framing
3. under correct semantics, structured evidence buys **spatially faithful explanation**,
   not extra classification gain

**Promotion criterion:** if all three hold at 32B, the finding moves from
"7B-specific phenomenon" to "stable across Qwen2.5-VL 7B and 32B within the same model
family."

## Model and parity (asserted at freeze)

| | |
|---|---|
| snapshot | `7cfb30d71a1f4f49a57592323337a4a4727301da` (pinned, matches requirement) |
| env | `qwen_vl_32b`, frozen record sha1 `5a669edd8d76…` |
| dtype | **bfloat16**, no quantization, no CPU offload |
| device_map | `auto`, 4× RTX 3090 (GPUs 0–3) |
| inference path | `mllm_probe_mgpu.QwenVLProbeMultiGPU` |

| parameter | 7B | 32B | required | OK |
|---|---|---|---|---|
| `min_pixels` | 200704 | 200704 | 200704 | ✓ |
| `max_pixels` | 802816 | 802816 | 802816 | ✓ |
| `do_sample` | False | False | False | ✓ |
| `temperature` | 0.0 | 0.0 | 0.0 | ✓ |
| `max_new_tokens` | 320 | 320 | 320 | ✓ |
| `repetition_penalty` | 1.0 | 1.0 | 1.0 | ✓ |

The freezer **aborts** if any of these differ. Visual token budget unchanged.

## Samples

184 fake + 184 paired real, 184 unique coco_id, **identical source set to 7B A0–A3**,
inherited from the 2×2 freeze (sha1 `6f3fea39549b…`). Not reselected, not filtered on
any 7B outcome.

## Evidence — frozen, not recomputed

Structured fields: `grid_location`, `area_ratio`, `mean_probability`,
`max_probability`, `centroid`. Extraction via `abstract()` + `strip_fields()`, imported
unchanged: threshold **0.5**, **4-connected**, min area ratio **0.001**, top-K **3**,
sorted by **integrated probability**.

**Corpus-wide structured payload sha1 `8d06347c476f846d660d70388a005e2a355fb013`** —
the runner must reproduce this hash or refuse to start. (This guard exists because a
payload-construction slip silently invalidated 1472 inferences in the task-semantics
round.)

No raw map anywhere.

## Cells

| cell | structured | rule | ↔ 7B | prompt sha1 |
|---|---|---|---|---|
| **B0** | no | no | A0 | `8775b797feccec20afffa2ca799cd491bbae50e0` |
| **B1** | no | **yes** | A1 | `ef68309f567c7a61517447641701add20f546217` |
| **B2** | **yes** | no | A2 | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` |
| **B3** | **yes** | **yes** | A3 | `9167aa08c1a09d683974d68ac10821f428ee6658` |
| **B4** | **yes** (mirrored) | **yes** | — | same as B3 |

**All four hashes match their 7B originals byte-for-byte**, so the only intended
variable is the model. B4 shares B3's prompt exactly. Binary rule present in B1/B3 only
(verified).

Rule text, verbatim and unchanged:
> For this task, the authenticity label is binary. If there is credible evidence that
> any part of the image has been genuinely manipulated, the image should be classified
> as fake, regardless of how small or localized the manipulated region is.

## Output schema — identical for all cells

`final_verdict`, `referenced_regions`, `reason`.

**No `manipulation_present`, no `manipulation_extent`** — 7B showed explicit presence
decomposition causes over-detection (presence FPR 0.951, verdict J 0.049), so it is
deliberately excluded from this final mechanism replication. Asserted absent from every
prompt.

## B4 mirror control (secondary)

Frozen eligible subset: **fake 70, real 80 → 150 inferences**. Transform: grid
left↔right, centre column unchanged, centroid `x' = 1 − x`; `area_ratio`,
`mean_probability`, `max_probability` and the score **unchanged** (verified). The
original image is never mirrored.

**Eligibility was frozen in the abstraction round before any 7B mirror result** and is
reused unchanged — not reselected for 32B. Purpose: does 32B actually *consume*
structured spatial evidence?

## Effects

| effect | 32B | 7B | 7B recall | 7B J |
|---|---|---|---|---|
| rule without structured | `B1 − B0` | `A1 − A0` | +0.141 | +0.071 |
| rule with structured | `B3 − B2` | `A3 − A2` | +0.853 | +0.788 |
| structured without rule | `B2 − B0` | `A2 − A0` | **−0.723** | −0.712 |
| structured with rule | `B3 − B1` | `A3 − A1` | −0.011 | +0.005 |

Interaction `(B3 − B2) − (B1 − B0)` on recall / FPR / J. 7B reference: recall **+0.712**,
J **+0.717**.

**Primary capacity analysis: `Δeffect = effect_32B − effect_7B`, paired per source,
bootstrap 95 % CI.** Both models ran the same 184 sources. Comparing significance
between models is explicitly **not** used.

### 7B cells

| cell | recall | FPR | J |
|---|---|---|---|
| A0 | 0.837 | 0.011 | +0.826 |
| A1 | 0.978 | 0.082 | +0.897 |
| A2 | 0.114 | 0.000 | +0.114 |
| A3 | 0.967 | 0.065 | +0.902 |

## Capacity hypotheses

- **C1 capacity-invariant task semantics** — B1 > B0 and B3 > B2, rule effect still
  clearly present, Δrule-effect not significantly changed
- **C2 32B reduces framing collapse** — `B2 − B0` clearly less negative than −0.723 with
  Δ CI excluding zero
- **C3 32B uses structured evidence for decision** — `B3 − B1 > 0`, CI excludes zero, and
  clearly above the 7B ≈ 0
- **C4 role separation capacity-invariant** — `B3 ≈ B1` while structured-condition
  faithfulness stays high

## Faithfulness and reason audit

B2/B3/B4: citation rate, spatial agreement, exact agreement, unsupported-region rate,
compared against 7B A2/A3. 7B reference: spatial agreement **1.000**, mirror shift
responsiveness **1.000**.

Reason audit (deterministic rules, no LLM judge): `score_high`, `locality`,
`manipulation_acknowledged`, `rule_consistent`. Focus on **B0 vs B1** and **B2 vs B3** —
does the binary rule raise the weight of the scalar score in the stated reasoning, as it
did on 7B (`score_high` 0.668 → 0.946)?

## Budget and runtime

| | inferences |
|---|---|
| main 4 × 2 × 184 | 1472 |
| B4 eligible subset | 150 |
| **total** | **1622** |

Measured 32B rate **8.59 s/inf** (previous 32B round, single-image) → **≈3.9 h**; allow
up to **≈5.1 h** at the slower historical rate of 11.4 s. tmux, GPUs 0–3, historical
peaks 16.3/18.2/18.2/16.8 GB.

## Forbidden

raw map, confidence map, presence decomposition, extent decomposition, prompts outside
B0–B4, prompt tuning, threshold tuning, LoRA, training, new detector, new dataset.
