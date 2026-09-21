# Training-Free MLLM Image Forensics with External Forensic Evidence

Can a pretrained multimodal LLM use an external forensic detector's output to detect,
localize and **explain** image manipulation — with no training at all?

This repository holds the completed training-free stage: 26 formal experiments, ~26 300
inferences, every protocol frozen before execution.

## Scope

Tool + MLLM, **no training, no fine-tuning, no LoRA**. Every intervention is a prompt, an
input, or a deterministic transformation of the evidence. That constraint is the point: if
a model needs training to use a tool's output, that is itself the finding.

## Components

| component | role |
|---|---|
| **ELA** | classical, training-free cue. Tested first, found to be a weak spatial cue only |
| **TruFor** (`ae54475d`) | learned forensic detector. Image AUROC 0.9845 on this data |
| **Qwen2.5-VL-7B-Instruct** (`cc594898…`) | main experimental model |
| **Qwen2.5-VL-32B-Instruct** (`7cfb30d7…`) | capacity replication, BF16 on 4×3090 |
| **evidence abstraction layer** | deterministic map → structured spatial description. No model, no training |

## Dataset

**TGIF `sd2-sp`** — Stable Diffusion 2 inpainting, *splice* variant, so pixels outside the
mask are unchanged and a spatial claim can actually be scored. Main subset: 184 unique
`coco_id`, 184 fake + 184 paired real. See [`docs/DATASET_AND_SPLITS.md`](docs/DATASET_AND_SPLITS.md).

The imagery is **not** in this repository.

## Headline results

**Decision.** The detector's scalar score is the stable signal. The raw heatmap *costs*
accuracy when a score is present (ΔJ −0.228), and neither model checks whether the map is
even in the right place — shifting it moves recall by −0.027 with a CI containing zero, at
both 7B and 32B. The best configuration measured is the plainest: 32B with score only,
J 0.918.

**Spatial explanation.** Structured evidence gives near-perfect, auditable citation of the
tool's regions (agreement 1.000, unsupported-region rate 0.000) and moves the explanation
closer to ground truth (+0.340 over the raw map). But that correctness is borrowed: mirror
the evidence and Explanation→GT collapses to 0.014.

**Semantic explanation.** Capacity matters here, unlike for decisions. 7B is spatially
right and semantically wrong — 100/100 correct locations, yet **79.1%** of its manipulation
claims are unsupported; 32B's rate is **4.4%** on identical prompts and identical sources.

**The scale surprise.** Two of the three mechanism conclusions established at 7B turned out
to be **capacity-dependent**: the structured-framing collapse vanishes at 32B
(Δrecall +0.723) and the binary task rule becomes *harmful* (ΔJ −1.027). One conclusion is
invariant:

> **Scalar score supports the decision; structured spatial evidence supports a faithful
> explanation.**

**The limit.** Across 300 audited explanations, evaluable texture / lighting / boundary /
geometry / physics claims number 0–3 each. Prompting alone did not elicit fine-grained
forensic reasoning.

Full picture, including ten falsified hypotheses:
[`docs/CURRENT_FINDINGS.md`](docs/CURRENT_FINDINGS.md).

## Repository structure

```
configs/       all tunable parameters
src/           experiment code: freeze_* / *_run / analyze_* / shared modules
protocols/     frozen protocol documents (markdown)
runs/          frozen JSON protocols, raw verdicts, analyses
docs/          index, manifests, findings, reproducibility
reports/       full experiment report + number crosscheck
environment/   conda exports, system info
scripts/       audit and documentation generators
```

Detail: [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md).

## Environment

Three separate conda environments (`qwen_vl`, `qwen_vl_32b`, `trufor`), deliberately never
cross-installed. Exports in `environment/`; `qwen_vl` has no scipy on purpose.

## Quick start

```bash
# 1. environments
conda env create -n qwen_vl -f environment/qwen_vl_history.yml

# 2. point the config at your TGIF copy and output directory
#    configs/exp01.yaml -> datasets.tgif_sp.root, experiment.out_root

# 3. verify a checkout reproduces the recorded numbers
python scripts/crosscheck_numbers.py reports/CROSSCHECK.txt   # expect 0 mismatches
python scripts/final_audit.py                                # expect 0 FAIL
```

Re-running a specific experiment: [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Finding things

| you want | go to |
|---|---|
| the list of experiments | [`docs/EXPERIMENT_INDEX.md`](docs/EXPERIMENT_INDEX.md) |
| a model snapshot, hash, or un-pushed data location | [`docs/FINAL_EXPERIMENT_MANIFEST.md`](docs/FINAL_EXPERIMENT_MANIFEST.md) |
| whether an old number may still be cited | [`docs/INVALID_AND_SUPERSEDED_RUNS.md`](docs/INVALID_AND_SUPERSEDED_RUNS.md) |
| the full narrative | [`reports/TRAINING_FREE_EXPERIMENT_REPORT.md`](reports/TRAINING_FREE_EXPERIMENT_REPORT.md) |

## Method notes

Every round freezes its protocol — sample IDs, prompts, prompt hashes, thresholds, seeds,
and the decision rules for interpreting the outcome — **before** any inference. Freeze
scripts refuse to overwrite. Run scripts assert rendered prompt and payload hashes against
the freeze and refuse to start on mismatch.

Invalid runs are **kept**, labelled, and excluded from citation rather than deleted: a
payload-duplication bug voided 1472 inferences, and those artifacts are still on disk under
`*_INVALID_score_dup.*`. Falsified hypotheses are tracked separately from invalid data,
because the first category is a result and the second is a defect.

Reported thresholds are **evaluation-derived operating points, not frozen deployment
thresholds**. The TGIF training split was used only as an untouched evaluation pool for the
external forensic model; no model was trained or tuned on these images.

## Status

**Training-free stage: complete and frozen.**

Post-training work (region-level explanation supervision, paired real/fake supervision,
LoRA / SFT) is scoped separately in [`docs/NEXT_STAGE.md`](docs/NEXT_STAGE.md) and is **not**
part of this stage's conclusions.
