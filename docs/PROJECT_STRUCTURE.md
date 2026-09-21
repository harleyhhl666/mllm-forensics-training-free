# PROJECT_STRUCTURE.md

Where to find things. The layout follows one rule: **nothing was moved if moving it could
break reproduction.** Where old code depends on a path, the path stayed and this index
points at it.

```
.
├── README.md                     short orientation
├── .gitignore
├── configs/                      all tunable parameters live here
├── src/                          experiment code (90 files)
├── protocols/                    frozen protocol documents (markdown)
├── runs/                         results: frozen JSON, raw verdicts, analyses
├── docs/                         navigation, manifests, findings
├── reports/                      the full experiment report + crosscheck output
├── environment/                  conda exports, system info, env reports
└── scripts/                      audit and doc-generation utilities
```

## `configs/`

| file | purpose |
|---|---|
| `exp01.yaml` | the single source of tunable parameters for the 7B line, including `mask_binarize_threshold`, dataset roots and `out_root` |
| `exp01_32b.yaml` | 32B overrides. Note the key is **`dtype: bfloat16`**, not `torch_dtype` — a freeze script once read the wrong key and silently recorded the wrong precision |

Point `datasets.tgif_sp.root` and `experiment.out_root` at your own paths; nothing else
should need editing.

## `src/`

90 Python files, grouped by role rather than by directory (they were never split, because
several read each other by flat module name).

| group | naming | what it does |
|---|---|---|
| **protocol freezers** | `freeze_*.py` | compute and *freeze* a protocol: sample IDs, prompts, hashes, thresholds. Each refuses to overwrite an existing frozen file. Run with `--freeze` to write; without it they dry-run and print the audit |
| **inference drivers** | `*_run.py`, `run_*.py` | read a frozen protocol and execute inference, appending one JSON line per inference so a crash never loses completed work |
| **analysers** | `analyze_*.py` | read raw verdicts and emit metrics with source-level bootstrap CIs |
| **shared modules** | `forensic_tools.py`, `mllm_probe.py`, `mllm_probe_mgpu.py`, `trufor_adapter.py` | grid naming (`CELL_NAMES`), single-GPU and multi-GPU Qwen wrappers (`QwenVLProbeMultiGPU`), TruFor adapter |
| **utilities** | `eta.py` | wall-clock ETA for long runs: `eta.py start <total> --rate S` and `eta.py check <jsonl> <total>` |
| **smoke / debug** | `smoke*.py`, `dbg*`, `check*` | historical, exploratory. Not cited by any result |

The smoke and debug scripts were **left in place** rather than moved to `archive/`, because
some are imported by sibling scripts via flat module names. They are labelled
EXPLORATORY in `docs/EXPERIMENT_INDEX.md` instead.

### Key invariant

A freeze script is the only thing allowed to decide sample IDs, prompts, thresholds or
seeds. Run scripts *read* the frozen file and assert against it; several of them refuse to
start if a rendered prompt or payload hash does not match the freeze. That guard exists
because a payload bug once voided 1472 inferences.

## `protocols/`

Human-readable frozen protocols, one per experiment round, in rough chronological order:
`PROTOCOL_v2` → `PROTOCOL_v3_FROZEN` → `PROTOCOL_v4_VALIDATION_FROZEN` → the
`TRUFOR_*` series → `EXPLAINABILITY_SEMANTIC_AUDIT_PROTOCOL`.

Each has a machine-readable twin under `runs/<round>/*_frozen.json`; the markdown explains
the design, the JSON carries the exact IDs and hashes. **The JSON wins** if they ever
disagree.

## `runs/`

One directory per experiment. Committed contents per directory:

| file pattern | what it is |
|---|---|
| `*_frozen.json` | the machine-readable frozen protocol: sample IDs, prompts, prompt hashes, parameters, pre-registered decision rules |
| `verdicts.jsonl` / `stage_*.jsonl` / `scores.jsonl` | **raw** per-inference output, one JSON object per line |
| `analysis*.txt` / `analysis*.json` | computed metrics |
| `run_meta.json` | inference count, wall time, peak GPU memory, checkpoint md5, protocol sha1 |
| `*_INVALID_*` | **deliberately retained invalid artifacts.** Never cite these |

Excluded from git and kept on the server: `evidence/`, `maps/`, `packets/`, `ela_cache/`,
and all `.npz`/image files — about 1.5 GB. Their aggregate SHA256s are in
`runs/SHA256SUMS_LOCAL_ONLY.txt`.

## `docs/`

| file | read it when |
|---|---|
| **`EXPERIMENT_INDEX.md`** | you want to find an experiment. Start here |
| **`FINAL_EXPERIMENT_MANIFEST.md`** | you need a model snapshot, a protocol hash, or the location of un-pushed data |
| **`CURRENT_FINDINGS.md`** | you want the conclusions, split into confirmed / untested / falsified |
| **`INVALID_AND_SUPERSEDED_RUNS.md`** | you found an old number and need to know whether you may cite it |
| **`DATASET_AND_SPLITS.md`** | you need to understand TGIF, the pairing, the mask threshold, or the splits |
| `data_manifest.csv` | per-source metadata for the frozen 184 subset, with relative paths |
| `REPRODUCIBILITY.md` | you want to re-run something |
| `PROJECT_STRUCTURE.md` | this file |
| `NEXT_STAGE.md` | you are starting the post-training work |
| `EXPERIMENTAL_HISTORY.md` | historical log, including ERRATUM 001 |
| `PROJECT_INVENTORY_BEFORE_CLEANUP.md` | you want the pre-cleanup state of the tree |

## `reports/`

| file | what |
|---|---|
| `TRAINING_FREE_EXPERIMENT_REPORT.md` | the full narrative: what was done, why, what was discarded |
| `CROSSCHECK.txt` | every headline number re-derived from raw artifacts, with audit targets. 0 mismatches |

## `environment/`

Conda `--from-history` exports and `pip freeze` for all three environments, plus
`SYSTEM_INFO.txt` (OS, GPU, driver, CUDA, library versions — no hostnames or usernames)
and the two detailed env reports.

**The three environments are deliberately separate and must not be cross-installed.**
`qwen_vl` has no scipy on purpose; see `SYSTEM_INFO.txt`.

## `scripts/`

| script | what it does |
|---|---|
| `final_audit.py` | the release gate: files, hashes, snapshots, sample counts, results, git safety. Any FAIL blocks a push |
| `crosscheck_numbers.py` | re-derives every headline number from raw artifacts and compares against audit targets |
| `gen_docs.py` | generates `docs/EXPERIMENT_INDEX.md` from the artifacts |
| `gen_manifest.py` | generates `docs/FINAL_EXPERIMENT_MANIFEST.md`, hashing files rather than transcribing |
| `make_inventory.py` | generates the pre-cleanup inventory |
| `sync_small_results.py` | copies only git-appropriate result files from the server |

## `archive/`

Present for future use. Nothing was archived during this cleanup: moving the historical
debug scripts risked breaking flat-module imports, so they stayed in `src/` and are marked
EXPLORATORY in the index instead.
