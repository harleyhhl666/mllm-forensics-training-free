# PROJECT_INVENTORY_BEFORE_CLEANUP.md

Read-only inventory taken BEFORE any reorganisation, so the starting state is recoverable.

Generated: 2026-09-21T23:27:01+08:00

## Canonical locations

| role | path | note |
|---|---|---|
| code | `/home/borui/haolin/fevi` | server; scripts were executed here |
| results | `/mnt/disk3/borui/fevi/runs` | server; all frozen protocols + verdicts |
| datasets | `/mnt/disk3/borui/fevi/data` | server; NOT for git |
| X3 annotations | `C:/Users/HarleyH/fevi/x3` | local; annotation packets and outputs |

Neither location was a git repository before this cleanup (`git rev-parse` returned nothing in both).

## Top-level directories (code tree)
```
.
./configs
./external_tools
./external_tools/TruFor
./scripts
./src
```

## File counts

**code tree** — 224 files

| ext | count |
|---|---|
| `.py` | 135 |
| `.txt` | 41 |
| `.md` | 21 |
| `.png` | 9 |
| `.yaml` | 7 |
| `<noext>` | 2 |
| `.sh` | 2 |
| `.jpg` | 2 |
| `.yml` | 1 |
| `.pth` | 1 |
| `.th` | 1 |
| `.zip` | 1 |
| `.tar` | 1 |

**runs tree** — 3032 files

| ext | count |
|---|---|
| `.npz` | 1182 |
| `.jpg` | 700 |
| `.png` | 569 |
| `.json` | 481 |
| `.log` | 43 |
| `.jsonl` | 39 |
| `.txt` | 18 |

## Disk usage
```
852M	/home/borui/haolin/fevi
1.6G	/mnt/disk3/borui/fevi/runs
12G	/mnt/disk3/borui/fevi/data

16K	/mnt/disk3/borui/fevi/runs/smokeP
20K	/mnt/disk3/borui/fevi/runs/qwen32b
20K	/mnt/disk3/borui/fevi/runs/smokeV1
20K	/mnt/disk3/borui/fevi/runs/smokeV2
24K	/mnt/disk3/borui/fevi/runs/smoke01
24K	/mnt/disk3/borui/fevi/runs/smokeA
24K	/mnt/disk3/borui/fevi/runs/smokeB
24K	/mnt/disk3/borui/fevi/runs/smokeM
24K	/mnt/disk3/borui/fevi/runs/smokeTA
24K	/mnt/disk3/borui/fevi/runs/smokeTB
44K	/mnt/disk3/borui/fevi/runs/smoke02
52K	/mnt/disk3/borui/fevi/runs/smokeMit
56K	/mnt/disk3/borui/fevi/runs/mitigation
64K	/mnt/disk3/borui/fevi/runs/splits_tgif
80K	/mnt/disk3/borui/fevi/runs/mechanism
112K	/mnt/disk3/borui/fevi/runs/pilot
140K	/mnt/disk3/borui/fevi/runs/qwen7b_stagev_cown
384K	/mnt/disk3/borui/fevi/runs/validation
592K	/mnt/disk3/borui/fevi/runs/qwen32b_stagev_ablation
612K	/mnt/disk3/borui/fevi/runs/validation_primary_stageA
632K	/mnt/disk3/borui/fevi/runs/trufor_2x2
700K	/mnt/disk3/borui/fevi/runs/validation_primary_stageB
700K	/mnt/disk3/borui/fevi/runs/validation_primary_stageB_probe
720K	/mnt/disk3/borui/fevi/runs/calibration
792K	/mnt/disk3/borui/fevi/runs/readout_cv
884K	/mnt/disk3/borui/fevi/runs/pilot_stageA
924K	/mnt/disk3/borui/fevi/runs/trufor_32b
1.1M	/mnt/disk3/borui/fevi/runs/validation_v4
1.2M	/mnt/disk3/borui/fevi/runs/mechanism_run
1.2M	/mnt/disk3/borui/fevi/runs/phase0
1.2M	/mnt/disk3/borui/fevi/runs/trufor_mllm_stageB
1.4M	/mnt/disk3/borui/fevi/runs/calibration_tgif
1.4M	/mnt/disk3/borui/fevi/runs/trufor_mllm_stageA
1.5M	/mnt/disk3/borui/fevi/runs/trufor_score_map
1.9M	/mnt/disk3/borui/fevi/runs/trufor_conflict
2.0M	/mnt/disk3/borui/fevi/runs/v4_stageA
2.1M	/mnt/disk3/borui/fevi/runs/mitigation_run
2.2M	/mnt/disk3/borui/fevi/runs/v4_stageB
2.3M	/mnt/disk3/borui/fevi/runs/trufor_32b_final
2.6M	/mnt/disk3/borui/fevi/runs/trufor_abstraction
2.8M	/mnt/disk3/borui/fevi/runs/trufor_framing
3.0M	/mnt/disk3/borui/fevi/runs/trufor_fields
3.8M	/mnt/disk3/borui/fevi/runs/trufor_semantics
11M	/mnt/disk3/borui/fevi/runs/readout_tgif
140M	/mnt/disk3/borui/fevi/runs/explainability
157M	/mnt/disk3/borui/fevi/runs/ela_cache
608M	/mnt/disk3/borui/fevi/runs/trufor_mllm
611M	/mnt/disk3/borui/fevi/runs/trufor_feasibility
1.6G	/mnt/disk3/borui/fevi/runs
```

## Large files

| size | path | purpose |
|---|---|---|
| 2G | `/mnt/disk3/borui/fevi/data/TGIF/sd2-sp_validation.tar.gz` |  |
| 2G | `/mnt/disk3/borui/fevi/data/TGIF/sd2-sp_testing.tar.gz` |  |
| 820M | `/mnt/disk3/borui/fevi/data/TGIF/orig_validation.tar.gz` |  |
| 770M | `/mnt/disk3/borui/fevi/data/TGIF/orig_testing.tar.gz` |  |
| 268M | `/home/borui/haolin/fevi/external_tools/TruFor/test_docker/weights/trufor.pth.tar` | TruFor inference checkpoint |
| 249M | `/home/borui/haolin/fevi/external_tools/TruFor/test_docker/TruFor_weights.zip` | TruFor weights archive as downloaded |
| 117M | `/mnt/disk3/borui/fevi/data/CocoGlide.zip` |  |
| 94M | `/home/borui/haolin/fevi/external_tools/TruFor/TruFor_train_test/pretrained_models/segformers/mit_b2.pth` | TruFor upstream SegFormer backbone (vendored repo) |
| 69M | `/mnt/disk3/borui/fevi/data/probe/test_autosplice.parquet` |  |
| 28M | `/home/borui/haolin/fevi/external_tools/TruFor/TruFor_train_test/dataset/data/bcmc_COCO_train_list.txt` | TruFor upstream training list (unused here) |
| 21M | `/home/borui/haolin/fevi/external_tools/TruFor/TruFor_train_test/dataset/data/bcm_COCO_train_list.txt` | TruFor upstream training list (unused here) |
| 21M | `/home/borui/haolin/fevi/external_tools/TruFor/TruFor_train_test/dataset/data/sp_COCO_train_list.txt` | TruFor upstream training list (unused here) |
| 21M | `/home/borui/haolin/fevi/external_tools/TruFor/TruFor_train_test/dataset/data/cm_COCO_train_list.txt` | TruFor upstream training list (unused here) |

Total files over 10M: 13. Everything above is either a model checkpoint, a vendored upstream repo, or raw dataset imagery; none of it belongs in git.

## Run directories

| directory | size | files | has frozen protocol | has verdicts |
|---|---|---|---|---|
| `calibration` | 705K | 6 | yes | - |
| `calibration_tgif` | 1M | 2 | - | - |
| `ela_cache` | 155M | 829 | - | - |
| `explainability` | 138M | 904 | yes | - |
| `mechanism` | 72K | 1 | yes | - |
| `mechanism_run` | 1M | 1 | - | - |
| `mitigation` | 50K | 1 | yes | - |
| `mitigation_run` | 2M | 2 | - | - |
| `phase0` | 1M | 3 | - | - |
| `pilot` | 105K | 1 | yes | - |
| `pilot_stageA` | 873K | 1 | - | - |
| `qwen32b` | 11K | 2 | yes | - |
| `qwen32b_stagev_ablation` | 577K | 3 | - | - |
| `qwen7b_stagev_cown` | 129K | 2 | - | - |
| `readout_cv` | 783K | 2 | - | - |
| `readout_tgif` | 10M | 4 | - | - |
| `smoke01` | 16K | 2 | - | - |
| `smoke02` | 31K | 2 | - | - |
| `smokeA` | 20K | 1 | - | - |
| `smokeB` | 17K | 1 | - | - |
| `smokeM` | 16K | 1 | - | - |
| `smokeMit` | 41K | 2 | - | - |
| `smokeP` | 9K | 1 | - | - |
| `smokeTA` | 16K | 2 | - | - |
| `smokeTB` | 13K | 2 | - | - |
| `smokeV1` | 13K | 1 | - | - |
| `smokeV2` | 14K | 1 | - | - |
| `splits_tgif` | 60K | 1 | yes | - |
| `trufor_2x2` | 611K | 5 | yes | yes |
| `trufor_32b` | 910K | 4 | yes | yes |
| `trufor_32b_final` | 2M | 5 | yes | yes |
| `trufor_abstraction` | 3M | 4 | yes | yes |
| `trufor_conflict` | 2M | 4 | yes | yes |
| `trufor_feasibility` | 609M | 405 | yes | - |
| `trufor_fields` | 3M | 4 | yes | yes |
| `trufor_framing` | 3M | 4 | yes | yes |
| `trufor_mllm` | 606M | 737 | yes | - |
| `trufor_mllm_stageA` | 1M | 3 | - | - |
| `trufor_mllm_stageB` | 1M | 3 | - | - |
| `trufor_score_map` | 1M | 5 | yes | yes |
| `trufor_semantics` | 4M | 9 | yes | yes |
| `v4_stageA` | 2M | 1 | - | - |
| `v4_stageB` | 2M | 1 | - | - |
| `validation` | 374K | 2 | yes | - |
| `validation_primary_stageA` | 601K | 1 | - | - |
| `validation_primary_stageB` | 687K | 2 | - | - |
| `validation_primary_stageB_probe` | 694K | 1 | - | - |
| `validation_v4` | 1M | 2 | yes | - |

## Protocol / report markdown in code tree

| file | bytes | sha1 |
|---|---|---|
| `EXPERIMENTAL_HISTORY.md` | 4072 | `6e37981313ab8d88…` |
| `EXPLAINABILITY_SEMANTIC_AUDIT_PROTOCOL.md` | 8125 | `f9a97a4f8b68689e…` |
| `PROTOCOL_v2.md` | 6236 | `240d03a3be195fae…` |
| `PROTOCOL_v3_FROZEN.md` | 7474 | `8b480ddaff008f2b…` |
| `PROTOCOL_v4_VALIDATION_FROZEN.md` | 9685 | `fcc20e935f129cac…` |
| `QWEN32B_ENV_REPORT.md` | 7460 | `76d2f25a720ac43f…` |
| `TRUFOR_32B_CAPACITY_PROTOCOL.md` | 5536 | `cbf17c7fc59335d5…` |
| `TRUFOR_32B_FINAL_CAPACITY_PROTOCOL.md` | 6778 | `99c1b6f9579baa0f…` |
| `TRUFOR_ENV_REPORT.md` | 10750 | `385bc5abf61c2a04…` |
| `TRUFOR_EVIDENCE_ABSTRACTION_PROTOCOL.md` | 6892 | `6e173237ec522c1d…` |
| `TRUFOR_FEASIBILITY_PROTOCOL.md` | 4210 | `7ed738495015e05c…` |
| `TRUFOR_MLLM_PROTOCOL_FROZEN.md` | 7755 | `030e14df6917f1b7…` |
| `TRUFOR_SCORE_MAP_CONFLICT_PROTOCOL.md` | 6042 | `eef786ce58422423…` |
| `TRUFOR_SCORE_SEMANTICS_2x2_PROTOCOL.md` | 4766 | `bbee90a5d945900c…` |
| `TRUFOR_STRUCTURED_FIELD_ABLATION_PROTOCOL.md` | 5484 | `47b6c6a6887aa6e8…` |
| `TRUFOR_STRUCTURED_FRAMING_PROTOCOL.md` | 5929 | `4ed1afe9cffa1410…` |
| `TRUFOR_TASK_SEMANTICS_PROTOCOL.md` | 7262 | `74a39ff262367191…` |

## Scripts in src/

**analyze** (20)

```
analyze_2x2.py
analyze_32b_capacity.py
analyze_32b_final.py
analyze_abstraction.py
analyze_capacity_ablation.py
analyze_conflict.py
analyze_fields.py
analyze_framing.py
analyze_mechanism.py
analyze_mitigation.py
analyze_primary.py
analyze_score_map.py
analyze_semantics.py
analyze_stage_a.py
analyze_stage_b.py
analyze_trufor_feasibility.py
analyze_trufor_mllm_stage_a.py
analyze_trufor_mllm_stage_b.py
analyze_v4_stage_a.py
analyze_v4_stage_b.py
```

**freeze (protocol freezers)** (17)

```
freeze_2x2.py
freeze_32b_capacity.py
freeze_32b_env.py
freeze_32b_final.py
freeze_abstraction.py
freeze_conflict.py
freeze_fields.py
freeze_framing.py
freeze_pilot.py
freeze_score_map_ablation.py
freeze_semantics.py
freeze_split.py
freeze_trufor_env.py
freeze_trufor_feasibility.py
freeze_trufor_mllm.py
freeze_validation.py
freeze_x2_semantic_audit.py
```

**run (inference drivers)** (14)

```
abstraction_run.py
conflict_run.py
fields_run.py
framing_run.py
main_run.py
mechanism_run.py
mitigation_run.py
run_2x2_a1.py
run_32b_final.py
score_map_run.py
semantics_run.py
stage_b_run.py
trufor_32b_run.py
trufor_feasibility_run.py
```

**shared modules / utilities** (32)

```
analyze.py
audit_official_trufor.py
cache_ela.py
calibrate.py
calibrate_tgif.py
eta.py
extract_trufor_official.py
forensic_tools.py
generic_cue.py
m0_audit.py
mitigation_freeze.py
mllm_probe.py
mllm_probe_mgpu.py
pilot_stage_a.py
readout.py
readout_cv.py
readout_precompute.py
select_readout.py
stagev_capacity_ablation_v2.py
tgif_source_audit.py
tool_diagnostic.py
trufor_adapter.py
trufor_mllm_stage_a.py
trufor_mllm_stage_b.py
twosided_diagnostic.py
v0_audit.py
verify_dataset.py
verify_generic_cue.py
verify_tgif.py
x1_spatial.py
x3_build_packets.py
xlsx_read.py
```

**smoke / debug** (3)

```
smoke32b.py
smoke_tools.py
trufor_smoke_verify.py
```

Total `.py` in src/: 86

## Git state before cleanup

```
code tree : fatal: not a git repository (or any of the parent directories): .git
```

No commits, no remote, no `.gitignore` existed anywhere in the project before this stage.
