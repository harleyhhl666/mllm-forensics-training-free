# TRUFOR_32B_CAPACITY_PROTOCOL.md

Frozen protocol for the **Qwen2.5-VL-32B capacity replication** of TruFor evidence
integration. Frozen **before** any inference. Machine-readable twin:
`runs/trufor_32b/trufor_32b_capacity_frozen.json`
(protocol sha1 `af95a7583ae40ca1951fd1e0d4acfe810d44b5ff`)

## Question

> Does the weak coupling between spatial forensic evidence and the final verdict,
> observed on 7B, persist at 32B?

Capacity validation only. **Not** an accuracy comparison between 7B and 32B, and not
a rerun of the full pipeline.

## Model

| | |
|---|---|
| path | `/mnt/disk3/borui/hf_cache/models--Qwen--Qwen2.5-VL-32B-Instruct/snapshots/7cfb30d71a1f4f49a57592323337a4a4727301da` |
| snapshot | `7cfb30d71a1f4f49a57592323337a4a4727301da` (pinned, not `main`) |
| env | `qwen_vl_32b` |
| dtype | `bfloat16` — no quantization |
| device_map | `auto`, 4× RTX 3090, no CPU offload |
| env record | `runs/qwen32b/qwen32b_env_frozen.json` (sha1 `5a669edd8d76…`) |
| 7B reference | snapshot `cc594898137f460bfe9f0759e9844b3ce807cfb5` |

**Parity asserted at freeze time** (only the model may differ):

| | 7B | 32B | same |
|---|---|---|---|
| `min_pixels` | 200704 | 200704 | ✓ |
| `max_pixels` | 802816 | 802816 | ✓ |
| generation | `do_sample=False, temperature=0.0, max_new_tokens=320, repetition_penalty=1.0` | identical | ✓ |

## Samples

**184 fake sources, identical to the 7B rounds**, inherited from
`trufor_score_map_conflict_frozen.json` (sha1 `f65614514b09…`).
`n_samples == n_sources == 184`, so no variant clustering. Selection is **not**
conditioned on any 7B outcome.

`tamper_ratio` q1/med/q3 = 0.0149 / 0.0383 / 0.1089; mask type bbox 73 / segm 111.

**Fake targets only.** Real controls are deliberately deferred — this round tests the
map's effect on fake recall and on spatial correspondence.

## Conditions

| id | input | ↔ 7B |
|---|---|---|
| **B0** score only | image + own score | C10 |
| **B1** score + own map | image + own score + raw map | F1 / C11 |
| **B2** score + blank map | image + own score + all-zero map | F2 |
| **B3** score + shifted map | image + own score + own map rolled 50 % width | F4 |

The score shown is **always the target's own** TruFor score.

## Prompts

Reused **byte-identically** from the frozen 7B protocols; no Stage A, no Stage-A
summary, and the prompt never names the transform.

| condition | sha1 |
|---|---|
| B0 | `c38c0d0d0ae27cc34780ea69b028dc5488d3e764` (= 2×2 C10) |
| B1 / B2 / B3 | `2f1f3ef0de63100006844f491c9d027efa0a9ef1` (= 2×2 C11 = conflict F1) |

## Map transform audit (corrected)

The previous round reported `shifted preserves mean and active fraction: False`.
**That was a bad assertion, not a bad transform.** It compared float32 `mean()` at
`1e-9`, below float32 summation precision, and falsely failed on 50/184.

Corrected audit over all 184 maps:

| check | result |
|---|---|
| sorted-value multiset identical | **184/184** |
| `np.allclose(atol=0)` on sorted values | **184/184** |
| 50-bin histogram identical | **184/184** |
| max &#124;mean diff&#124; | 5.96e-08 (summation order only) |
| max &#124;active-fraction diff&#124; | **0.0** |
| odd-width maps | 30/184 (`int()` shift not exactly half-width there; values still permuted exactly) |

**AUDIT PASS.** No renormalisation is applied. Blank maps are `np.zeros_like`
through the identical rendering pipeline.

## Rendering and score formatting

Inherited unchanged: `inferno`, **fixed 0–1 scale (no min-max stretch)**, RGB PNG,
BILINEAR resize only if sizes differ; score as
`TruFor image-level manipulation score: X.XXX` — raw continuous value, 3 decimals, no
label, no threshold, no percentage.

## Contrasts

| name | definition |
|---|---|
| map cost | `B1 − B0` |
| map-content effect | `B2 − B1` |
| **spatial correspondence** | `B3 − B1` — **primary capacity test** |
| **cross-model** | `Δeffect = effect_32B − effect_7B`, bootstrap CI — **primary comparison, not absolute recall** |

### 7B reference

recalls: C10 **0.755**, F1 **0.522**, F2 **0.266**, F4 **0.495**
effects: map cost **−0.228**, blank **−0.255**, shift **−0.027** (null)

The 7B shift null is what this round must replicate or overturn.

## Interpretation (pre-registered)

- **Capacity-invariant failure** — `B1 < B0` and `B3 ≈ B1` → increasing capacity does
  not resolve the weak coupling
- **Capacity-dependent spatial grounding** — `B3 ≪ B1` while the 7B shift effect ≈ 0
  → larger capacity increases sensitivity to spatial correspondence
- **Capacity reduces map interference** — `B1 ≈ B0` while 7B `C11 ≪ C10`
- **Failure-mode shift** — near-constant verdicts or collapse → report that capacity
  scaling changes the failure mode without solving integration

## Explanation audit (secondary)

Same deterministic keyword rules as the 7B rounds. Focus on **B1 vs B3**: if the
shifted map no longer corresponds to the image yet the model still produces
spatially-grounded-sounding explanations, flag as **potential
explanation-faithfulness failure**.

## Budget and runtime

184 × 4 = **736 inferences**. At the measured 11.42 s/inf (32B, single-image
Stage-V): **≈2.3 h**, up to **≈3.2 h** if two-image inputs cost ~35 % more.

GPUs 0–3 (historical 32B assignment); all seven cards currently idle at 16 MiB.
Historical 32B peaks: 16.3 / 18.2 / 18.2 / 16.8 GB.

## Forbidden this round

prompt modification, Stage A, gate, verifier, calibration, confidence map, threshold
tuning, LoRA, training, new dataset, quantization.
