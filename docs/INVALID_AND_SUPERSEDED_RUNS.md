# INVALID_AND_SUPERSEDED_RUNS.md

What must **not** be cited, and why. Nothing listed here was deleted — every artifact
is still on disk so the record stays auditable.

Three categories are kept strictly apart:

| category | meaning |
|---|---|
| **INVALID DATA** | the run itself is corrupt; its numbers mean nothing |
| **SUPERSEDED** | the run was correct but a later run replaces it |
| **FALSIFIED HYPOTHESIS** | the data are valid; the *explanation* we attached was wrong |

The third category is not a defect. It is the normal output of a mechanism study and
the reason several rounds exist at all.

---

## 1. INVALID DATA

### 1.1 Task-semantics first run — 1472 inferences discarded

| | |
|---|---|
| artifacts | `runs/trufor_semantics/verdicts_INVALID_score_dup.jsonl` (1472 rows), `analysis_INVALID_score_dup.txt`, `run_meta_INVALID_score_dup.json` |
| cause | the runner serialised the payload with `json.dumps(full)`, where `full` still carried an extra `image_manipulation_score` key. The image-level score therefore appeared **twice** in the prompt. |
| how it surfaced | D0 was supposed to reproduce the field-ablation S5 recall of 0.114. It measured **0.592**. |
| how it was confirmed | the rendered prompts matched the field round in **0 of 368** cases; rebuilding the payload from the declared field list reproduced the frozen hash `2e6f72826f493fdf…` exactly. |
| ruling | pipeline drift, not a finding. **All 1472 rows are void and no number from them may be quoted** — including the ones that looked favourable. |
| replacement | `runs/trufor_semantics/verdicts.jsonl` (the re-run, D0 = 0.114 as predicted). |

The prompt-hash guard added later to the 32B final runner
(`corpus payload sha1 8d06347c476f846d…`, refuses to start on mismatch) exists
because of this incident.

### 1.2 ERRATUM 001 — ELA `real-own` condition mislabelled

| | |
|---|---|
| cause | `cue_vis(label, which)` cached without `label` in the key, and the ELA source was hard-coded to `s["fake"]`. |
| consequence | every historical result labelled **"real-own"** actually paired a *real image* with the *ELA of its fake counterpart*. It was never a real-own condition. |
| what cannot be cited | any pre-erratum "real-own" number in the pilot and early validation runs. |
| replacement | condition renamed, and `runs/qwen7b_stagev_cown/` (experiment **E10**) re-ran the genuinely-own case. |
| deliberately left untouched | the frozen JSONL and `mitigation_run.py` were **not** retro-edited, so the historical record still shows what was actually executed. |
| recorded in | `docs/EXPERIMENTAL_HISTORY.md` under ERRATUM 001. |

---

## 2. SUPERSEDED

| run | superseded by | reason |
|---|---|---|
| `runs/pilot_stageA/` (E04) | E05 / E06 (`v4_stageA`, `v4_stageB`) | pilot design, replaced by the frozen V4 validation protocol |
| Phase T0/T1 Stage-B cells for the score-map 2×2 | full 1472-inference re-run in `runs/trufor_score_map/` | the old Stage-B prompts **embedded a Stage-A summary**, which leaks map-derived information into conditions that are supposed to have no map. Verified by grep before reuse; `C00`/`C11` were therefore not reusable. |
| abstraction-round `E2` prompt as the 7B explanation source for X2/X3 | `D0_baseline` | `E2` contains an extra sentence explaining `area_ratio` and coordinate normalisation, so it is not byte-comparable to the 32B `B2` prompt. `D0` is byte-identical to `B2`. |

The field-ablation round also dropped two explanatory sentences from the `E2` prompt to
achieve S1–S5 parity. Consequence, stated at freeze time and repeated here: **`S5 vs E2`
(0.174) is descriptive background only**, never a strict reproduction comparison.

---

## 3. A FALSE ALARM THAT WAS NOT A BUG

### Shift audit "failure" — the assertion was wrong, not the transform

The shifted-map intervention (`F4`, later `B3`) is built with `np.roll`. An audit
assertion initially failed and the transform was suspected.

Investigation showed the transform is correct:

| check | result |
|---|---|
| sorted value multiset preserved | **184/184 maps** |
| histogram preserved | **184/184 maps** |
| active-fraction difference | **0.0 exactly** |
| mean difference | ≤ **5.96e-08** |

The assertion demanded `|mean difference| < 1e-9`, which is **below float32 summation
precision** — reordering the same values changes the sum in the last bits. The audit
threshold was at fault.

Secondary note kept in the protocol: 30 of 184 maps have odd width, so
`int(0.5 * W)` is not an exact half-width shift for those.

**`F4` and every conclusion resting on it remain valid.**

---

## 4. FALSIFIED HYPOTHESES — valid data, wrong explanation

These were our interpretations, and later experiments refuted them. Listed so nobody
revives them from an old note.

| hypothesis | refuted by | what is true instead |
|---|---|---|
| ELA can serve as strong image-level evidence | E02–E06 | ELA is a weak spatial cue; its image-level discrimination is poor |
| The 7B failure is a capacity limit | E09 (32B Stage-V) | Case 2 — 32B collapsed to the *opposite* constant |
| A verification step will gate out bad evidence | E08 (M1) | verifier output collapsed; the intervention failed |
| The map hurts because of a weak-visual-evidence **veto** on real local response | T05 | blank map (0.266) is *worse* than own map (0.522), and shifted map has **zero** effect — consistent with insensitivity to the map's content, not a veto |
| `area_ratio` drives the structured collapse | S02 | falsified. `S4 − S1 = +0.071`, i.e. **adding** area to location *improved* things. All single fields hurt similarly (−0.696 / −0.641 / −0.734) |
| "Generic structured overload" — the measurements themselves overload the model | S03 | mostly wrong. `P1`, with the structured prompt and **no payload at all**, already collapses. It is a **framing-induced decision-policy shift** |
| The failure is an evidence-**arbitration** failure needing presence/extent decomposition | S04 | `D2`/`D3` decomposition produced presence FPR ≈ 0.95 and over-detection. The binary rule alone (`D1`) fixed the decision |
| Structured evidence improves decision performance | S05, F01 | no. `A3 ≈ A1` (ΔJ +0.005, CI contains zero) at 7B; at 32B `B3 − B1` is **negative** (J −0.217) |
| A larger model will check spatial correspondence | T06, F01 | no. The shifted-map effect stays −0.027 with CI containing zero at 32B |
| The three final 7B mechanism conclusions are model-family-general | F01 | **two of three are capacity-dependent.** Framing collapse vanishes at 32B (Δrecall +0.723) and the binary rule becomes *harmful* (ΔJ −1.027) |

### One interpretation the assistant corrected itself

An earlier write-up claimed *"D1's J = 0.902 exceeds the score-only baseline 0.826, so
structured evidence brings a net gain under correct semantics."* That is **wrong**: the
correct comparator is `A1` (rule, no structure, J = 0.897), not `A0` (no rule). The gain
came from the rule, not from the structured evidence. Recorded here because the wrong
version was stated before it was caught.

---

## 5. Known limitations that are not defects

| item | status |
|---|---|
| `E4` mirror subset is only 70/184 fake | pre-registered. A 3×3 horizontal mirror leaves `center` and the whole centre column invariant, and fake regions sit most often in `center` (78) / `bottom-center` (38). Accepted rather than redesigned, to protect interpretability |
| region presence itself carries label information | disclosed **before** any MLLM result: fake has an empty region list 0/184 times, real 10/184; mean region count 1.147 vs 1.902 |
| `P1`'s empty payload still differs from `P2` by 15 characters | unavoidable — "no block" means a different character count. Approved and documented |
| `A0` recall is already 0.837, leaving little head-room | the `ceiling_caveat`, disclosed before running `A1`. A near-zero `A1 − A0` must **not** be read as "the rule does nothing" |
| 32B real-image FFER rests on 6 cited cases | descriptive only, no inference. Not oversampled, by instruction |
| X3 annotator is not an independent third party | see `docs/CURRENT_FINDINGS.md` §limitations. No Anthropic endpoint was available; blinded subagents of the same model family were used |
| `manipulation_claim_supported` repeat agreement is 0.650 | the reliability ceiling on the X3 headline comparison |
| `hallucinated_object_present` κ = −0.017 despite 0.967 agreement | base-rate degeneracy; that κ is uninterpretable and the HOR is a low-reliability measurement |
| texture / lighting / boundary / geometry / physics claim counts are 0–3 | those rows of the X3 table cannot be cited. That scarcity is itself a finding |
