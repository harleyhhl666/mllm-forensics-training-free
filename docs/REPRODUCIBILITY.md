# REPRODUCIBILITY.md

The goal is **not** a one-command replay of 26 000 inferences. It is that a future reader
can set the project up, understand what each formal experiment consumed and produced, and
re-run a specific one.

Every command below was checked against the script's actual argument parser. Where a
script takes no arguments beyond a config, that is stated.

---

## 0. Conventions

- **Freeze scripts dry-run by default.** Without `--freeze` they print their audit and
  write nothing. Add `--freeze` only after the audit passes. They refuse to overwrite an
  existing frozen file.
- **Run scripts read a frozen protocol** and assert against it. Several refuse to start if
  a rendered prompt or payload hash mismatches. Do not bypass those checks.
- **`--n K` limits to K samples** on every run script — use it for smoke tests.
- Output goes under `experiment.out_root` from the config.
- Long runs: use tmux, and report wall-clock start plus projected finish
  (`python src/eta.py start <total> --rate <s_per_inf>`), then
  `python src/eta.py check <jsonl> <total>`.

---

## 1. Environment

Three separate conda environments. **Do not merge them.**

```bash
conda env create -n qwen_vl     -f environment/qwen_vl_history.yml
conda env create -n qwen_vl_32b -f environment/qwen_vl_32b_history.yml
conda env create -n trufor      -f environment/trufor_history.yml
```

For exact pinning use the `*_freeze.txt` pip lists instead. `environment/SYSTEM_INFO.txt`
records the OS, driver, CUDA and library versions the results were produced on.

`qwen_vl` intentionally has **no scipy** — the evidence abstraction layer implements
connected components with numpy plus a stdlib BFS specifically so this environment stays
untouched. Installing scipy would diverge from the frozen environment.

If `huggingface.co` is unreachable (it was on this machine):

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Model snapshots — pin these, do not use `main`

| model | snapshot |
|---|---|
| Qwen2.5-VL-7B-Instruct | `cc594898137f460bfe9f0759e9844b3ce807cfb5` |
| Qwen2.5-VL-32B-Instruct | `7cfb30d71a1f4f49a57592323337a4a4727301da` |

Decoding for both, asserted at freeze time in every round: `do_sample=False`,
`temperature=0.0`, `max_new_tokens=320`, `repetition_penalty=1.0`,
`min_pixels=200704`, `max_pixels=802816`. 32B runs BF16 on 4 GPUs with
`device_map=auto`, no quantization, no CPU offload.

## 2. Data

Obtain TGIF and keep only the `sd2-sp` subset for the main line. Then set in
`configs/exp01.yaml`:

```yaml
datasets:
  tgif_sp:
    root: /your/path/to/TGIF
experiment:
  out_root: /your/path/for/results
```

`docs/data_manifest.csv` lists the frozen 184 sources with **relative** paths, so it stays
valid against your own copy. Mask binarization is `> 127` and must not be changed.
See `docs/DATASET_AND_SPLITS.md`.

## 3. TruFor

```bash
git clone https://github.com/grip-unina/TruFor external_tools/TruFor
cd external_tools/TruFor && git checkout ae54475df6f41a491d7615100feb19263dec13f7
```

Fetch the weights from `grip.unina.it` (reachable directly; Google Drive was not) and
verify:

| artifact | md5 |
|---|---|
| `TruFor_weights.zip` | `7bee48f3476c75616c3c5721ab256ff8` |
| runtime `trufor.pth.tar` | `55d7075dd1ff945e9c0f9437c5df9495` |

The adapter was verified to match official outputs exactly (delta = 0) before use.

---

## 4. Experiments

### T01 — TruFor standalone feasibility

| | |
|---|---|
| input | 200 frozen sources, fake + paired real |
| protocol | `protocols/TRUFOR_FEASIBILITY_PROTOCOL.md`, `runs/trufor_feasibility/trufor_feasibility_frozen.json` |

```bash
conda activate trufor
python src/freeze_trufor_feasibility.py configs/exp01.yaml            # audit
python src/freeze_trufor_feasibility.py configs/exp01.yaml --freeze
python src/trufor_feasibility_run.py    configs/exp01.yaml --gpu 5
python src/analyze_trufor_feasibility.py configs/exp01.yaml
```

Expected: 400 rows in `scores.jsonl`; image AUROC 0.9845, TPR@5%FPR 0.955, pixel AUROC
0.993, IoU 0.758, F1 0.846 in `analysis.txt`.

### Structured evidence generation (prerequisite for S01–F01)

The abstraction layer is deterministic and lives inside `src/freeze_abstraction.py`
(function `abstract()`): threshold `map > 0.5`, 4-connected, min area ratio 0.001,
top-K 3 sorted by integrated probability, normalised coordinates, 3×3 grid naming.
It is frozen together with the protocol — freezing the round regenerates the evidence
description from the stored TruFor maps.

```bash
conda activate qwen_vl
python src/freeze_abstraction.py configs/exp01.yaml            # audit
python src/freeze_abstraction.py configs/exp01.yaml --freeze
```

### S02 — 7B structured field ablation (the score-only baseline lives here)

```bash
conda activate qwen_vl
python src/freeze_fields.py configs/exp01.yaml --freeze
python src/fields_run.py    configs/exp01.yaml            # 2208 inferences, ~100 min
python src/analyze_fields.py configs/exp01.yaml
```

Expected: `S0_score_only` recall 0.837 / FPR 0.011 / J 0.826; `S5_full` recall 0.114.

### S04 — task semantics D0–D3

```bash
python src/freeze_semantics.py configs/exp01.yaml --freeze
python src/semantics_run.py    configs/exp01.yaml          # 1472 inferences, ~67 min
python src/analyze_semantics.py configs/exp01.yaml
```

Expected: `D0_baseline` recall **0.114**. If D0 comes out near 0.59 you have reproduced
the payload-duplication bug — stop and compare the rendered prompt hash against the
freeze rather than interpreting the numbers. See `INVALID_AND_SUPERSEDED_RUNS.md` §1.1.

`--only <cell>` runs a single condition.

### S05 — score × semantics 2×2

Only the `A1` cell is new; `A0`/`A2`/`A3` are reused from the frozen field and semantics
rounds.

```bash
python src/freeze_2x2.py configs/exp01.yaml --freeze
python src/run_2x2_a1.py configs/exp01.yaml               # 368 inferences, ~12 min
python src/analyze_2x2.py configs/exp01.yaml
```

Expected: A0 0.837/0.011/0.826, A1 0.978/0.082/0.897, A2 0.114/0.000/0.114,
A3 0.967/0.065/0.902.

### F01 — 32B final capacity replication

```bash
conda activate qwen_vl_32b
python src/freeze_32b_final.py configs/exp01.yaml configs/exp01_32b.yaml            # audit
python src/freeze_32b_final.py configs/exp01.yaml configs/exp01_32b.yaml --freeze
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/run_32b_final.py configs/exp01_32b.yaml
python src/analyze_32b_final.py configs/exp01_32b.yaml
```

1622 inferences, ~4.2 h at ~9.2 s/inference on 4×3090. Expected: B0 0.978/0.060/0.918,
B1 0.989/0.109/0.880, B2 0.978/0.076/0.902, B3 1.000/0.337/0.663.

The runner verifies the corpus payload hash `8d06347c476f846d…` and all five prompt
hashes before starting, and refuses to run on mismatch.

### X01 — spatial explainability retrospective

No inference. Reads existing verdicts plus GT masks.

```bash
conda activate qwen_vl
python src/x1_spatial.py configs/exp01.yaml
```

Expected: Evidence→GT precision 0.979, recall 0.531; Explanation→Evidence 1.000;
mirrored Explanation→GT 0.014 (7B) / 0.029 (32B).

### X02 — semantic audit freeze

```bash
python src/freeze_x2_semantic_audit.py configs/exp01.yaml            # audit, 12 checks
python src/freeze_x2_semantic_audit.py configs/exp01.yaml --freeze
```

All twelve checks must PASS. One of them (fake/real disjointness) caught a real
contamination: fake units display the paired real image to the annotator, so the real set
is drawn only from sources not used as fake units.

### X03 — annotation packets and analysis

Packet images are regenerable and not committed:

```bash
python src/x3_build_packets.py configs/exp01.yaml      # 900 images for 300 units
```

Annotation itself is not a script — it is 300 primary + 60 repeat judgements against the
frozen rubric (`runs/explainability/x3/x2_semantic_audit_frozen.json`, prompts and schema
with hashes). The outputs are committed as
`runs/explainability/x3/ann/<EXPL-id>.json` and `.../ann_repeat/<EXPL-id>.json`.

**Important**: the annotator in this project was a set of blinded agents of the same model
family as the assistant, not an independent third party. Anyone reproducing this should
either use a genuinely independent annotator or carry the same caveat.

```bash
python src/analyze_x3.py <dir containing x3/>
```

Expected: 300 primary + 60 repeat; fake 7B overall unsupported 0.610 and manipulation
unsupported 0.791; fake 32B 0.010 and 0.044; overall repeat agreement 0.825.

---

## 5. Verifying a checkout

```bash
python scripts/crosscheck_numbers.py reports/CROSSCHECK.txt   # 0 mismatches expected
python scripts/final_audit.py                                # FAIL must be 0
```

`crosscheck_numbers.py` recomputes every headline number from raw verdicts and compares
against the recorded targets. It needs the server results tree; from a bare git checkout
the committed `runs/` tree is enough for everything except the feasibility maps.

## 6. What you cannot reproduce from this repository alone

| missing | why | how it is pinned |
|---|---|---|
| TGIF imagery (~12 GB) | upstream dataset | `docs/data_manifest.csv` with relative paths |
| TruFor evidence npz/PNG (~607 MB) | regenerable | aggregate SHA256 in `runs/SHA256SUMS_LOCAL_ONLY.txt` |
| TruFor feasibility maps (~610 MB) | regenerable | same |
| ELA cache (~157 MB) | regenerable | same |
| X3 packet images (~140 MB) | regenerable via `src/x3_build_packets.py` | same |
| model weights | licence and size | snapshot ids and md5s in `docs/FINAL_EXPERIMENT_MANIFEST.md` |
