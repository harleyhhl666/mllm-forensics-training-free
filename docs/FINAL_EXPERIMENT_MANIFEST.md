# FINAL_EXPERIMENT_MANIFEST.md

The ledger of the training-free stage. Every hash below is computed from a file, by `scripts/gen_manifest.py` — none is transcribed.

Generated: 2026-09-21T23:51:37+08:00

## Models

| model | snapshot | env | precision | GPUs |
|---|---|---|---|---|
| Qwen2.5-VL-7B-Instruct | `cc594898137f460bfe9f0759e9844b3ce807cfb5` | `qwen_vl` | BF16 | 1x RTX3090 |
| Qwen2.5-VL-32B-Instruct | `7cfb30d71a1f4f49a57592323337a4a4727301da` | `qwen_vl_32b` | BF16 | 4x RTX3090 |

7B snapshot verified present in the semantics protocol: `True`

32B: no quantization, `device_map=auto`, no CPU offload. Decoding for both models: `do_sample=False`, `temperature=0.0`, `max_new_tokens=320`, `repetition_penalty=1.0`, `min_pixels=200704`, `max_pixels=802816`.

## TruFor

| item | value | source |
|---|---|---|
| repo | GRIP-UNINA/TruFor | vendored, not committed |
| commit | `ae54475df6f41a491d7615100feb19263dec13f7` | `environment/TRUFOR_ENV_REPORT.md` |
| runtime checkpoint md5 | `55d7075dd1ff945e9c0f9437c5df9495` | `runs/trufor_feasibility/run_meta.json` |
| weights zip md5 | `7bee48f3476c75616c3c5721ab256ff8` | `environment/TRUFOR_ENV_REPORT.md` |
| feasibility protocol sha1 | `1e11d06ca1cd8461d10140974aabba114e9e9bbe` | same run_meta |
| adapter vs official output | delta = 0 | `T00` smoke |

## Frozen protocol hashes

| protocol file | sha1 |
|---|---|
| `runs/calibration/frozen_gate1.json` | `947fe56c0b2e0c5e4143639282f417ef95254ccf` |
| `runs/explainability/x2_semantic_audit_frozen.json` | `b3a26bb7e48cbeacbe6ad524392edb4f56dd85db` |
| `runs/explainability/x3/x2_semantic_audit_frozen.json` | `b3a26bb7e48cbeacbe6ad524392edb4f56dd85db` |
| `runs/mechanism/mechanism_frozen.json` | `99b39ec38cdced98b886fadffcec4414abeaa3a6` |
| `runs/mitigation/mitigation_frozen.json` | `432463c562a28f099eb0486743a693563b63acca` |
| `runs/pilot/pilot_frozen.json` | `0455ba1c081a0859e89082fd32ee9eb1eb0ddece` |
| `runs/qwen32b/qwen32b_env_frozen.json` | `5a669edd8d76e50abf5d8948b500465a0afb6410` |
| `runs/splits_tgif/split_frozen.json` | `d2210c58f89262512fc4640462b6a3d886c2e600` |
| `runs/trufor_2x2/trufor_score_semantics_2x2_frozen.json` | `6f3fea39549b67ff8a5b00db7f7ff97934fc1001` |
| `runs/trufor_32b/trufor_32b_capacity_frozen.json` | `af95a7583ae40ca1951fd1e0d4acfe810d44b5ff` |
| `runs/trufor_32b_final/trufor_32b_final_capacity_frozen.json` | `9cf55f9fe3ea461507f60fec0a2febbeff25e3ca` |
| `runs/trufor_abstraction/trufor_evidence_abstraction_frozen.json` | `ec1118a2f9b8fe17bad237892630752d0ade4d2f` |
| `runs/trufor_conflict/trufor_score_map_conflict_frozen.json` | `f65614514b090bd12eca59a139ee1963e6921021` |
| `runs/trufor_feasibility/trufor_feasibility_frozen.json` | `1e11d06ca1cd8461d10140974aabba114e9e9bbe` |
| `runs/trufor_fields/trufor_structured_field_ablation_frozen.json` | `b5722533ed218eff9d48c990a540229d1bd5d645` |
| `runs/trufor_framing/trufor_structured_framing_frozen.json` | `15e146d7d994da76cffceb7b47ab2752f21d3186` |
| `runs/trufor_mllm/trufor_mllm_frozen.json` | `3a9bb651dcded055f0e82bd6bce7ef2fda09f11b` |
| `runs/trufor_score_map/trufor_score_map_ablation_frozen.json` | `8f65faf529215262497404328d98faf5392f93f6` |
| `runs/trufor_semantics/trufor_task_semantics_frozen.json` | `d0617e9527291cf151dbbc7382ac3c4f23068c3e` |
| `runs/validation/validation_primary_frozen.json` | `50e4e822526533c5d126fc9a6871a789000ba8e9` |
| `runs/validation/validation_secondary_frozen.json` | `4c778fd33d83f3a18da55d4840f3d511bc5af0ae` |
| `runs/validation_v4/validation_v4_frozen.json` | `fa84c82ce3e99561cb71f931dcaad01cf725c53f` |

22 frozen protocol files.

### Markdown protocols

| file | sha1 |
|---|---|
| `protocols/EXPLAINABILITY_SEMANTIC_AUDIT_PROTOCOL.md` | `f9a97a4f8b68689e0abb5ee1eb2c470cee4e8482` |
| `protocols/PROTOCOL_v2.md` | `240d03a3be195faebb76767ca8c7d872a212c929` |
| `protocols/PROTOCOL_v3_FROZEN.md` | `8b480ddaff008f2bfc9a27bf2c61c9ff21f06508` |
| `protocols/PROTOCOL_v4_VALIDATION_FROZEN.md` | `fcc20e935f129cac371fa82508d4a0441d07ad33` |
| `protocols/TRUFOR_32B_CAPACITY_PROTOCOL.md` | `cbf17c7fc59335d5aec2d05f5904f066cbd8d4c3` |
| `protocols/TRUFOR_32B_FINAL_CAPACITY_PROTOCOL.md` | `99c1b6f9579baa0fef78399b4f316d1c4b5510e0` |
| `protocols/TRUFOR_EVIDENCE_ABSTRACTION_PROTOCOL.md` | `6e173237ec522c1d07f4e010af5f5cda6e0403a4` |
| `protocols/TRUFOR_FEASIBILITY_PROTOCOL.md` | `7ed738495015e05c7ef705e52203f7023d73b9bd` |
| `protocols/TRUFOR_MLLM_PROTOCOL_FROZEN.md` | `030e14df6917f1b7547a05856bfc7c0fb2c77ddf` |
| `protocols/TRUFOR_SCORE_MAP_CONFLICT_PROTOCOL.md` | `eef786ce5842242310875865ace47682b2962a4e` |
| `protocols/TRUFOR_SCORE_SEMANTICS_2x2_PROTOCOL.md` | `bbee90a5d945900c4f1dc757631d2ed3cf7e4198` |
| `protocols/TRUFOR_STRUCTURED_FIELD_ABLATION_PROTOCOL.md` | `47b6c6a6887aa6e83c17fda12f9b1a787dc92de2` |
| `protocols/TRUFOR_STRUCTURED_FRAMING_PROTOCOL.md` | `4ed1afe9cffa1410430cc6e3fbb1ebb471a29739` |
| `protocols/TRUFOR_TASK_SEMANTICS_PROTOCOL.md` | `74a39ff262367191395b664a53e31909eb066270` |

## Key prompt hashes

Taken from the frozen protocols. Identical hashes across rounds are intentional: prompts were reused byte-for-byte so the model or the payload is the only variable.

| round | condition | sha1 |
|---|---|---|
| field ablation | `S0_score_only` | `8775b797feccec20afffa2ca799cd491bbae50e0` |
| field ablation | `S1_location` | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` |
| field ablation | `S2_area` | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` |
| field ablation | `S3_probability` | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` |
| field ablation | `S4_loc_area` | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` |
| field ablation | `S5_full` | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` |
| task semantics | `D0_baseline` | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` |
| task semantics | `D1_binary_rule` | `9167aa08c1a09d683974d68ac10821f428ee6658` |
| task semantics | `D2_decomposition` | `91a25fb567947755d2bf31668f79178c644d2c8f` |
| task semantics | `D3_presence_first` | `04e101c80c10e59beb4de4e1b519d2e25b7aa517` |
| score x semantics 2x2 | `A0_score_only` | `8775b797feccec20afffa2ca799cd491bbae50e0` |
| score x semantics 2x2 | `A1_score_rule` | `ef68309f567c7a61517447641701add20f546217` |
| score x semantics 2x2 | `A2_structured` | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` |
| score x semantics 2x2 | `A3_structured_rule` | `9167aa08c1a09d683974d68ac10821f428ee6658` |
| 32B final capacity | `B0_score_only` | `8775b797feccec20afffa2ca799cd491bbae50e0` |
| 32B final capacity | `B1_score_rule` | `ef68309f567c7a61517447641701add20f546217` |
| 32B final capacity | `B2_structured` | `22b9ad9a944812ef48723581a35c01bc3eaba7f0` |
| 32B final capacity | `B3_structured_rule` | `9167aa08c1a09d683974d68ac10821f428ee6658` |
| 32B final capacity | `B4_mirrored` | `9167aa08c1a09d683974d68ac10821f428ee6658` |
| X2 semantic audit | `fake` | `c4d81e6a4e2230e67b81aa4cf81fa4ab371426e7` |
| X2 semantic audit | `real` | `1749b3206186a8bb306effdc9f8175e968161bc7` |

## Headline results

Recomputed from raw verdicts by `scripts/crosscheck_numbers.py`; the full output with the user's audit targets is `reports/CROSSCHECK.txt` (0 mismatches).

### TruFor standalone (200 sources)

| metric | value |
|---|---|
| image AUROC | 0.9845 [0.9700, 0.9951] |
| TPR @ FPR<=0.05 | 0.9550 |
| pixel AUROC (mean) | 0.9930 |
| IoU @ map>0.5 (mean) | 0.7578 |
| pixel F1 @ map>0.5 (mean) | 0.8459 |

### Decision: 7B vs 32B final 2x2 (184 fake + 184 paired real)

| cell | structured | rule | 7B recall | 7B FPR | 7B J | 32B recall | 32B FPR | 32B J |
|---|---|---|---|---|---|---|---|---|
| A0/B0 | no | no | 0.837 | 0.011 | 0.826 | 0.978 | 0.060 | 0.918 |
| A1/B1 | no | yes | 0.978 | 0.082 | 0.897 | 0.989 | 0.109 | 0.880 |
| A2/B2 | yes | no | 0.114 | 0.000 | 0.114 | 0.978 | 0.076 | 0.902 |
| A3/B3 | yes | yes | 0.967 | 0.065 | 0.902 | 1.000 | 0.337 | 0.663 |

### Explanation

| metric | 7B | 32B |
|---|---|---|
| Evidence->GT grid precision | 0.979 | 0.979 (same evidence) |
| Evidence->GT grid recall | 0.531 | 0.531 |
| Explanation->Evidence hit | 1.000 | 1.000 |
| Explanation->GT hit (correct evidence) | 1.000 | 1.000 |
| Explanation->GT hit (mirrored evidence) | 0.014 | 0.029 |
| X3 fake overall support: supported | 0.040 | 0.490 |
| X3 fake manipulation claims unsupported | 0.791 | 0.044 |

## Frozen sample subsets

| subset | size | used by |
|---|---|---|
| 184 unique `coco_id`, 184 fake + 184 paired real | 368 images | all TruFor->MLLM rounds T02-F01, X01 |
| 200 sources standalone | 400 images | T01 feasibility |
| mirror-eligible subset | 70 fake / 80 real | S01 `E4`, F01 `B4` |
| semantic audit | 100 fake + 50 real | X02, X03 |

Cumulative distinct `coco_id` consumed: **677** (243 + 184 + 150 + 100). The TGIF training split's sd2 pool (1558 `coco_id`) has **zero** intersection with these 677.

Bootstrap: source-level, 10000 resamples, seed `20260918`. Because each source contributes exactly one variant in the TruFor rounds, `n_samples == n_sources == 184` and cluster dependence is eliminated.

## Data intentionally NOT in git

| what | location | size | why |
|---|---|---|---|
| TGIF imagery | `/mnt/disk3/borui/fevi/data` (server) | ~12 GB | redistributable dataset, not ours to ship |
| TruFor evidence npz + PNG | `runs/trufor_mllm/evidence/` (server) | ~608 MB | regenerable from the frozen protocol |
| TruFor feasibility maps | `runs/trufor_feasibility/maps/` (server) | ~611 MB | regenerable |
| ELA cache | `runs/ela_cache/` (server) | ~157 MB | regenerable |
| X3 annotation packet images | `runs/explainability/packets/` | ~144 MB | regenerable by `src/x3_build_packets.py` |
| model weights | HF cache + TruFor `weights/` | multi-GB | pinned by snapshot / md5 above |

Their checksums are in `runs/SHA256SUMS_LOCAL_ONLY.txt` so an un-pushed artifact can still be identified.
