# EXPERIMENTAL_HISTORY.md

Running record of protocol-level facts and corrections. Frozen artifacts are never
edited; corrections are appended here as errata.

---

## ERRATUM 001 — Stage-V third condition was mislabelled `real-own ELA`

**Date found:** during preparation of the 32B Stage-V capacity ablation, before any
32B inference was run.

### What was wrong

The frozen M1 Stage-V run (`runs/mitigation_run/stage_v.jsonl`, produced by
`src/mitigation_run.py`, sha1 `5864a7cc15a1480993922ff46ff42a0f13a419bb`) recorded
its third condition as `label=real, cue=correct`, which was described in the
protocol and in all prior reporting as **“real + its own ELA”**.

It was not. The visualization actually shown for that condition was the ELA of the
**fake counterpart**, i.e. the same visualization used for `fake + correct ELA`.

### Cause

In `mitigation_run.py` the helper was

```python
def cue_vis(label, which):        # label accepted but NOT used in the key
    if which in vis_cache:        # cache keyed on `which` alone
        return vis_cache[which]
    src = s["fake"] if which == "correct" else dn["fake"]   # always the FAKE image
```

Two independent faults compound: the ELA source is hard-coded to `s["fake"]` for
`which == "correct"`, and the cache key omits `label`, so the real-image pass could
not have produced a different visualization even if the source had been correct.

### Evidence

Verified by recomputation against the frozen record, 100/100 sample pairs:

| quantity | sha1 (example `airplane_163746_bbox_2`) | matches frozen record |
|---|---|---|
| frozen `real+correct` visualization | `48c4441235341dd0` | — |
| ELA of the **fake** image | `48c4441235341dd0` | **yes** |
| ELA of the **real** image | `98dc6cc942cd6f3b` | no |

Across all 100 pairs, `real+correct` and `fake+correct` share an identical
visualization hash (100/100 identical, 0 differing).

### Correct naming from now on

| old (wrong) | correct |
|---|---|
| `real-own ELA` | **`real + fake-counterpart ELA`** |
| `P(insufficient \| real-own ELA)` | **`P(insufficient \| real + fake-counterpart ELA)`** |
| `P(matched \| real-own)` | **`P(matched \| real + fake-counterpart ELA)`** |

The name `real-own` is reserved for the semantically correct condition, which was
measured for the first time in the capacity ablation (`C_own_real_own`, ELA
recomputed from the pristine real image).

### Which results are affected

Affected — **Stage-V** third-condition semantics only:

- the Stage-V state distribution reported for `real + own ELA` in the M1
  analysis (it describes `real + fake-counterpart ELA`);
- any `ΔSupport` computed with that condition as the reference.

**Not affected** — the M1 mitigation conclusions:

- Goals 1/2/3, the secondary diagnostics and the sham attribution are all computed
  from **Stage-D verdicts**, not from Stage-V fields.
- The Stage-D conditions (`B-*`, `S-*`, `M-*`) build their visualizations in a
  separate code path in `mitigation_run.py`'s decision loop, which selects the ELA
  source from the condition table rather than from `cue_vis`'s cache.
- The earlier V4 Stage-B finding “real + own ELA specificity 0.497” comes from
  `stage_b_run.py`, a different script and a different code path, and is unaffected.
- The M1 verdict (failure: verifier collapse, Goal 2 not met) rests on
  `P(State A) = 0/400` and `ΔMatch = +0.020`, both of which use the
  `fake+correct` vs `fake+donor` contrast and are untouched by this bug.

### Handling

- `mitigation_run.py` is **not modified**; it is retained as the execution record
  of the frozen run, with its original sha1.
- The frozen JSONL outputs are **not edited or deleted**.
- A new runner, `src/stagev_capacity_ablation_v2.py`, keys the ELA cache on the
  source file path and names the ELA source explicitly per condition
  (`analysed_image` and `ela_source` are written to every output row).
- That runner asserts, before any inference: the legacy condition reproduces the
  historical visualization hash, and the own-ELA condition differs from it.

---
