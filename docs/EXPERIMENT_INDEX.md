# EXPERIMENT_INDEX.md

Every formal experiment of the training-free stage, in execution order.
Inference counts and protocol hashes in this file are read off the artifacts by `scripts/gen_docs.py`, not typed by hand.

Status values: **VALID** (citable) / **SUPERSEDED** (correct but replaced) / **INVALID** (must not be cited, see `INVALID_AND_SUPERSEDED_RUNS.md`) / **EXPLORATORY** (smoke or debug).

## Index

| ID | experiment | tool | MLLM | inferences | status |
|---|---|---|---|---|---|
| [E01](#e01) | TGIF index + source split | - | - | - | VALID |
| [E02](#e02) | ELA calibration | ELA | - | 2484 | VALID |
| [E03](#e03) | ELA readout CV | ELA | - | - | VALID |
| [E04](#e04) | Pilot Stage A/B | ELA | 7B | 720 | SUPERSEDED |
| [E05](#e05) | Validation V1 (Stage A grounding) | ELA | 7B | 1600 | VALID |
| [E06](#e06) | Validation V2 (Stage B verdict) | ELA | 7B | 1600 | VALID |
| [E07](#e07) | Framing x visualization mechanism 2x2 | ELA | 7B | 1200 | VALID |
| [E08](#e08) | M1 verification gating | ELA | 7B | 2000 | VALID |
| [E09](#e09) | 32B Stage-V capacity ablation | ELA | 32B | 400 | VALID |
| [E10](#e10) | 7B Stage-V C_own recheck | ELA | 7B | 100 | VALID |
| [T00](#t00) | TruFor install + smoke | TruFor | - | - | VALID |
| [T01](#t01) | TruFor standalone feasibility | TruFor | - | 400 | VALID |
| [T02](#t02) | Stage T1 localization | TruFor | 7B | 1104 | VALID |
| [T03](#t03) | Stage T2 verdict | TruFor | 7B | 1104 | VALID |
| [T04](#t04) | Score-map 2x2 | TruFor | 7B | 1472 | VALID |
| [T05](#t05) | Score-map conflict interventions | TruFor | 7B | 1656 | VALID |
| [T06](#t06) | 32B conflict replication | TruFor | 32B | 736 | VALID |
| [S01](#s01) | Evidence abstraction layer E0-E4 | TruFor | 7B | 1840 | VALID |
| [S02](#s02) | Structured field ablation S0-S5 | TruFor | 7B | 2208 | VALID |
| [S03](#s03) | Framing / region-presence control P0-P5 | TruFor | 7B | 2208 | VALID |
| [S04](#s04) | Task semantics D0-D3 | TruFor | 7B | 1472 | VALID |
| [S05](#s05) | Score x semantics 2x2 (A0-A3) | TruFor | 7B | 368 | VALID |
| [F01](#f01) | 32B final capacity replication B0-B4 | TruFor | 32B | 1622 | VALID |
| [X01](#x01) | Spatial correctness retrospective | - | 7B+32B | - | VALID |
| [X02](#x02) | Semantic audit protocol freeze | - | - | - | VALID |
| [X03](#x03) | Semantic explanation annotation | - | 7B+32B | - | VALID |

## Details

### E01
**TGIF index + source split**

| field | value |
|---|---|
| tool | - |
| MLLM | - |
| dataset / split | TGIF sd2-sp |
| purpose | Build the usable pair index and freeze a source-level split |
| protocol | `protocols/PROTOCOL_v2.md` |
| run directory | `runs/phase0/` |
| analysis | `phase0` |
| inferences | - |
| status | **VALID** |
| note | 2058 triples, 343 coco_id; source-level split so no coco_id crosses subsets |

### E02
**ELA calibration**

| field | value |
|---|---|
| tool | ELA |
| MLLM | - |
| dataset / split | TGIF sd2-sp |
| purpose | Calibrate ELA window/quality readout without any GT mask |
| protocol | `protocols/PROTOCOL_v2.md` |
| run directory | `runs/calibration_tgif/` |
| analysis | `readout_tgif` |
| inferences | 2484 |
| conditions | `?` (2484) |
| status | **VALID** |
| note | Gate 1 objective anomaly score; calibration never reused for confirmation |

### E03
**ELA readout CV**

| field | value |
|---|---|
| tool | ELA |
| MLLM | - |
| dataset / split | TGIF sd2-sp |
| purpose | Cross-validated window readout |
| protocol | `protocols/PROTOCOL_v2.md` |
| run directory | `runs/readout_cv/` |
| analysis | `readout_cv` |
| inferences | - |
| status | **VALID** |

### E04
**Pilot Stage A/B**

| field | value |
|---|---|
| tool | ELA |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp |
| purpose | First MLLM-with-tool probe: does an ELA cue change grounding or verdict? |
| protocol | `protocols/PROTOCOL_v3_FROZEN.md` |
| run directory | `runs/pilot_stageA/` |
| analysis | `pilot_stageA` |
| inferences | 720 |
| conditions | `fake_correct` (180), `fake_notool` (180), `fake_wrong` (180), `real_own` (180) |
| status | **SUPERSEDED** |
| note | Superseded by V4 validation; retained for history |

### E05
**Validation V1 (Stage A grounding)**

| field | value |
|---|---|
| tool | ELA |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp |
| purpose | Does the model ground on the supplied ELA cue? |
| protocol | `protocols/PROTOCOL_v4_VALIDATION_FROZEN.md` |
| run directory | `runs/v4_stageA/` |
| analysis | `v4_stageA` |
| inferences | 1600 |
| conditions | `fake_correct` (320), `fake_notool` (320), `fake_wrong` (320), `real_notool` (320), `real_own` (320) |
| status | **VALID** |
| note | H1/H2 supported: the model does use the cue |

### E06
**Validation V2 (Stage B verdict)**

| field | value |
|---|---|
| tool | ELA |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp |
| purpose | Does correct grounding translate into a correct verdict? |
| protocol | `protocols/PROTOCOL_v4_VALIDATION_FROZEN.md` |
| run directory | `runs/v4_stageB/` |
| analysis | `v4_stageB` |
| inferences | 1600 |
| conditions | `fake_correct` (320), `fake_notool` (320), `fake_wrong` (320), `real_notool` (320), `real_own` (320) |
| status | **VALID** |
| note | Case D dominant: grounding and verdict are disconnected |

### E07
**Framing x visualization mechanism 2x2**

| field | value |
|---|---|
| tool | ELA |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp |
| purpose | Is the fake-decision bias caused by forensic framing, the heatmap, or both? |
| protocol | `protocols/PROTOCOL_v4_VALIDATION_FROZEN.md` |
| run directory | `runs/mechanism_run/` |
| analysis | `mechanism_run` |
| inferences | 1200 |
| conditions | `C0` (300), `C1` (300), `C2` (300), `C3` (300) |
| status | **VALID** |
| note | Pure interaction: C1-C0 = C2-C0 = 0.000, C3-C2 = +0.267/+0.287 |

### E08
**M1 verification gating**

| field | value |
|---|---|
| tool | ELA |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp |
| purpose | Can a verification step recover decision utility? |
| protocol | `protocols/PROTOCOL_v4_VALIDATION_FROZEN.md` |
| run directory | `runs/mitigation_run/` |
| analysis | `mitigation_run` |
| inferences | 2000 |
| conditions | `?` (400), `B-correct` (200), `B-none` (200), `B-wrong` (200), `M-correct` (200), `M-none` (200), `M-wrong` (200), `S-correct` (200), `S-wrong` (200) |
| status | **VALID** |
| note | FAILED as an intervention: verifier output collapsed. Negative result |

### E09
**32B Stage-V capacity ablation**

| field | value |
|---|---|
| tool | ELA |
| MLLM | 32B |
| dataset / split | TGIF sd2-sp |
| purpose | Is the 7B failure a capacity problem? |
| protocol | `protocols/PROTOCOL_v4_VALIDATION_FROZEN.md` |
| run directory | `runs/qwen32b_stagev_ablation/` |
| analysis | `qwen32b_stagev_ablation` |
| inferences | 400 |
| conditions | `A_fake_correct` (100), `B_fake_donor` (100), `C_legacy_real_fakecounterpart` (100), `C_own_real_own` (100) |
| status | **VALID** |
| note | Case 2: 32B collapsed to the opposite constant. Capacity is not the cause |

### E10
**7B Stage-V C_own recheck**

| field | value |
|---|---|
| tool | ELA |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp |
| purpose | Re-run of the real-own condition after ERRATUM 001 |
| protocol | `protocols/PROTOCOL_v4_VALIDATION_FROZEN.md` |
| run directory | `runs/qwen7b_stagev_cown/` |
| analysis | `qwen7b_stagev_cown` |
| inferences | 100 |
| conditions | `C_own_real_own` (100) |
| status | **VALID** |
| note | Fixes the cue_vis cache-key bug; see docs/INVALID_AND_SUPERSEDED_RUNS.md |

### T00
**TruFor install + smoke**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | - |
| dataset / split | - |
| purpose | Verify the adapter reproduces upstream outputs bit-for-bit |
| protocol | `environment/TRUFOR_ENV_REPORT.md` |
| run directory | `runs/trufor_smoke/` |
| analysis | `-` |
| inferences | - |
| status | **VALID** |
| note | Adapter vs official: delta = 0 |

### T01
**TruFor standalone feasibility**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | - |
| dataset / split | TGIF sd2-sp 200 src |
| purpose | Is TruFor strong enough on this data to be worth integrating? |
| protocol | `protocols/TRUFOR_FEASIBILITY_PROTOCOL.md` |
| run directory | `runs/trufor_feasibility/` |
| analysis | `trufor_feasibility/analysis.txt` |
| inferences | 400 |
| conditions | `?` (400) |
| status | **VALID** |
| note | GO. AUROC 0.9845, TPR@5%FPR 0.955, pixel AUROC 0.993, IoU 0.758, F1 0.846 |
| frozen protocol | `runs/trufor_feasibility/trufor_feasibility_frozen.json` sha1 `1e11d06ca1cd8461d10140974aabba114e9e9bbe` |

### T02
**Stage T1 localization**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp 184 src |
| purpose | Tool-guided localization stage |
| protocol | `protocols/TRUFOR_MLLM_PROTOCOL_FROZEN.md` |
| run directory | `runs/trufor_mllm_stageA/` |
| analysis | `trufor_mllm_stageA` |
| inferences | 1104 |
| conditions | `fake_T0_notool` (184), `fake_T1_correct` (184), `fake_T2_wrong` (184), `real_T0_notool` (184), `real_T1_correct` (184), `real_T2_wrong` (184) |
| status | **VALID** |

### T03
**Stage T2 verdict**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp 184 src |
| purpose | First positive integration result |
| protocol | `protocols/TRUFOR_MLLM_PROTOCOL_FROZEN.md` |
| run directory | `runs/trufor_mllm_stageB/` |
| analysis | `trufor_mllm_stageB` |
| inferences | 1104 |
| conditions | `fake_T0_notool` (184), `fake_T1_correct` (184), `fake_T2_wrong` (184), `real_T0_notool` (184), `real_T1_correct` (184), `real_T2_wrong` (184) |
| status | **VALID** |
| note | Case C: recall 0.201->0.804, FPR 0.125->0.011, J +0.076->+0.793 |

### T04
**Score-map 2x2**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp 184 src |
| purpose | Which carries the verdict: the scalar score or the heatmap? |
| protocol | `protocols/TRUFOR_SCORE_MAP_ABLATION_PROTOCOL.md` |
| run directory | `runs/trufor_score_map/` |
| analysis | `trufor_score_map` |
| inferences | 1472 |
| conditions | `C00` (368), `C01` (368), `C10` (368), `C11` (368) |
| status | **VALID** |
| note | Case 4: score dominates; the map COSTS J -0.228 when a score is present |
| frozen protocol | `runs/trufor_score_map/trufor_score_map_ablation_frozen.json` sha1 `8f65faf529215262497404328d98faf5392f93f6` |

### T05
**Score-map conflict interventions**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp 184 src |
| purpose | Blank / donor / shifted / amplified / binarized map |
| protocol | `protocols/TRUFOR_SCORE_MAP_CONFLICT_PROTOCOL.md` |
| run directory | `runs/trufor_conflict/` |
| analysis | `trufor_conflict` |
| inferences | 1656 |
| conditions | `F1_own` (184), `F2_blank` (184), `F3_donor` (184), `F4_shifted` (184), `F5_amplified` (184), `F6_binary` (184), `R1_own` (184), `R2_blank` (184), `R3_donor` (184) |
| status | **VALID** |
| note | shift effect -0.027, CI contains zero: NO spatial-correspondence checking |
| frozen protocol | `runs/trufor_conflict/trufor_score_map_conflict_frozen.json` sha1 `f65614514b090bd12eca59a139ee1963e6921021` |

### T06
**32B conflict replication**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | 32B |
| dataset / split | TGIF sd2-sp 184 src |
| purpose | Does capacity restore spatial-correspondence checking? |
| protocol | `protocols/TRUFOR_32B_CAPACITY_PROTOCOL.md` |
| run directory | `runs/trufor_32b/` |
| analysis | `trufor_32b` |
| inferences | 736 |
| conditions | `B0_score_only` (184), `B1_own_map` (184), `B2_blank_map` (184), `B3_shifted_map` (184) |
| status | **VALID** |
| note | No: the spatial-correspondence effect stays at -0.027 with CI containing zero |
| frozen protocol | `runs/trufor_32b/trufor_32b_capacity_frozen.json` sha1 `af95a7583ae40ca1951fd1e0d4acfe810d44b5ff` |

### S01
**Evidence abstraction layer E0-E4**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp 184 src |
| purpose | Replace the raw map with a deterministic structured description |
| protocol | `protocols/TRUFOR_EVIDENCE_ABSTRACTION_PROTOCOL.md` |
| run directory | `runs/trufor_abstraction/` |
| analysis | `trufor_abstraction` |
| inferences | 1840 |
| conditions | `E0_score_only` (368), `E1_raw_map` (368), `E2_structured` (368), `E3_both` (368), `E4_shifted_struct` (368) |
| status | **VALID** |
| note | spatial agreement 1.000 vs raw map 0.358; E4 eligible subset 70/184 fake |
| frozen protocol | `runs/trufor_abstraction/trufor_evidence_abstraction_frozen.json` sha1 `ec1118a2f9b8fe17bad237892630752d0ade4d2f` |

### S02
**Structured field ablation S0-S5**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp 184 src |
| purpose | Which structured field causes the decision collapse? |
| protocol | `protocols/TRUFOR_STRUCTURED_FIELD_ABLATION_PROTOCOL.md` |
| run directory | `runs/trufor_fields/` |
| analysis | `trufor_fields` |
| inferences | 2208 |
| conditions | `S0_score_only` (368), `S1_location` (368), `S2_area` (368), `S3_probability` (368), `S4_loc_area` (368), `S5_full` (368) |
| status | **VALID** |
| note | area_ratio hypothesis FALSIFIED; all single fields hurt similarly |
| frozen protocol | `runs/trufor_fields/trufor_structured_field_ablation_frozen.json` sha1 `b5722533ed218eff9d48c990a540229d1bd5d645` |

### S03
**Framing / region-presence control P0-P5**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp 184 src |
| purpose | Is the collapse from the prompt framing or from real field values? |
| protocol | `protocols/TRUFOR_STRUCTURED_FRAMING_PROTOCOL.md` |
| run directory | `runs/trufor_framing/` |
| analysis | `trufor_framing` |
| inferences | 2208 |
| conditions | `P0_score_only` (368), `P1_framing_only` (368), `P2_empty_regions` (368), `P3_placeholder` (368), `P4_true_location` (368), `P5_neutral_meta` (368) |
| status | **VALID** |
| note | Framing-induced policy shift: P1 with no payload already collapses |
| frozen protocol | `runs/trufor_framing/trufor_structured_framing_frozen.json` sha1 `15e146d7d994da76cffceb7b47ab2752f21d3186` |

### S04
**Task semantics D0-D3**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp 184 src |
| purpose | Is the collapse a task-semantics misunderstanding or arbitration failure? |
| protocol | `protocols/TRUFOR_TASK_SEMANTICS_PROTOCOL.md` |
| run directory | `runs/trufor_semantics/` |
| analysis | `trufor_semantics` |
| inferences | 1472 |
| conditions | `D0_baseline` (368), `D1_binary_rule` (368), `D2_decomposition` (368), `D3_presence_first` (368) |
| status | **VALID** |
| note | D1 binary rule restores utility. D2/D3 presence decomposition over-detects |
| frozen protocol | `runs/trufor_semantics/trufor_task_semantics_frozen.json` sha1 `d0617e9527291cf151dbbc7382ac3c4f23068c3e` |

### S05
**Score x semantics 2x2 (A0-A3)**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | 7B |
| dataset / split | TGIF sd2-sp 184 src |
| purpose | Does the rule fix framing specifically, or clarify the task generally? |
| protocol | `protocols/TRUFOR_SCORE_SEMANTICS_2x2_PROTOCOL.md` |
| run directory | `runs/trufor_2x2/` |
| analysis | `trufor_2x2` |
| inferences | 368 |
| conditions | `A1_score_rule` (368) |
| status | **VALID** |
| note | A3 ~ A1: structured evidence gives NO extra decision utility |
| frozen protocol | `runs/trufor_2x2/trufor_score_semantics_2x2_frozen.json` sha1 `6f3fea39549b67ff8a5b00db7f7ff97934fc1001` |

### F01
**32B final capacity replication B0-B4**

| field | value |
|---|---|
| tool | TruFor |
| MLLM | 32B |
| dataset / split | TGIF sd2-sp 184 src |
| purpose | Do the three final 7B mechanism conclusions hold at 32B? |
| protocol | `protocols/TRUFOR_32B_FINAL_CAPACITY_PROTOCOL.md` |
| run directory | `runs/trufor_32b_final/` |
| analysis | `trufor_32b_final` |
| inferences | 1622 |
| conditions | `B0_score_only` (368), `B1_score_rule` (368), `B2_structured` (368), `B3_structured_rule` (368), `B4_mirrored` (150) |
| status | **VALID** |
| note | Two of three are capacity-DEPENDENT; only role separation is invariant |
| frozen protocol | `runs/trufor_32b_final/trufor_32b_final_capacity_frozen.json` sha1 `9cf55f9fe3ea461507f60fec0a2febbeff25e3ca` |

### X01
**Spatial correctness retrospective**

| field | value |
|---|---|
| tool | - |
| MLLM | 7B+32B |
| dataset / split | TGIF sd2-sp 184 src |
| purpose | Evidence->GT, Explanation->Evidence, Explanation->GT |
| protocol | `-` |
| run directory | `runs/explainability/` |
| analysis | `explainability/x1.txt` |
| inferences | - |
| status | **VALID** |
| note | No new inference. Evidence->GT precision 0.979 / recall 0.531 |
| frozen protocol | `runs/explainability/x2_semantic_audit_frozen.json` sha1 `b3a26bb7e48cbeacbe6ad524392edb4f56dd85db` |

### X02
**Semantic audit protocol freeze**

| field | value |
|---|---|
| tool | - |
| MLLM | - |
| dataset / split | 100 fake + 50 real |
| purpose | Freeze samples, rubric, schema, repeat subset and metrics |
| protocol | `protocols/EXPLAINABILITY_SEMANTIC_AUDIT_PROTOCOL.md` |
| run directory | `runs/explainability/` |
| analysis | `-` |
| inferences | - |
| status | **VALID** |
| note | 7B D0 and 32B B2 prompts are byte-identical, so the model is the only variable |
| frozen protocol | `runs/explainability/x2_semantic_audit_frozen.json` sha1 `b3a26bb7e48cbeacbe6ad524392edb4f56dd85db` |

### X03
**Semantic explanation annotation**

| field | value |
|---|---|
| tool | - |
| MLLM | 7B+32B |
| dataset / split | 100 fake + 50 real |
| purpose | Audit whether each natural-language claim is supported by the image |
| protocol | `protocols/EXPLAINABILITY_SEMANTIC_AUDIT_PROTOCOL.md` |
| run directory | `runs/explainability/x3/` |
| analysis | `explainability/x3/x3_report.txt` |
| inferences | - |
| status | **VALID** |
| note | 300 primary + 60 repeat. Annotator is assistive, NOT ground truth |
| frozen protocol | `runs/explainability/x3/x2_semantic_audit_frozen.json` sha1 `b3a26bb7e48cbeacbe6ad524392edb4f56dd85db` |

## Totals

- formal experiments indexed: **26**
- VALID: **25**, SUPERSEDED: **1**
- total indexed inference rows: **26294** (excludes files marked INVALID and all smoke runs)

Smoke and debug directories (`smoke*`, `pilot`, `validation`, `splits_tgif`, `mechanism`, `mitigation`, `calibration`) are retained on disk and listed in `PROJECT_INVENTORY_BEFORE_CLEANUP.md`; they are EXPLORATORY and are not cited anywhere.