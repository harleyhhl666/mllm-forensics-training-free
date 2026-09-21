#!/usr/bin/env python3
"""Generate docs/EXPERIMENT_INDEX.md and the manifest hash tables FROM THE ARTIFACTS.

Protocol hashes, sample counts and inference counts are read off disk, never typed in.
Human-supplied fields (purpose, status, notes) live in the SPEC table below, so the
narrative is reviewable while the numbers stay machine-derived.
"""
import glob, hashlib, json, os, sys
from collections import Counter

RUNS = "runs"

# id, name, tool, mllm, dataset, purpose, protocol, rundir, analysis, status, notes
SPEC = [
 ("E01", "TGIF index + source split", "-", "-", "TGIF sd2-sp",
  "Build the usable pair index and freeze a source-level split",
  "protocols/PROTOCOL_v2.md", "phase0", "phase0", "VALID",
  "2058 triples, 343 coco_id; source-level split so no coco_id crosses subsets"),
 ("E02", "ELA calibration", "ELA", "-", "TGIF sd2-sp",
  "Calibrate ELA window/quality readout without any GT mask",
  "protocols/PROTOCOL_v2.md", "calibration_tgif", "readout_tgif", "VALID",
  "Gate 1 objective anomaly score; calibration never reused for confirmation"),
 ("E03", "ELA readout CV", "ELA", "-", "TGIF sd2-sp",
  "Cross-validated window readout",
  "protocols/PROTOCOL_v2.md", "readout_cv", "readout_cv", "VALID", ""),
 ("E04", "Pilot Stage A/B", "ELA", "7B", "TGIF sd2-sp",
  "First MLLM-with-tool probe: does an ELA cue change grounding or verdict?",
  "protocols/PROTOCOL_v3_FROZEN.md", "pilot_stageA", "pilot_stageA", "SUPERSEDED",
  "Superseded by V4 validation; retained for history"),
 ("E05", "Validation V1 (Stage A grounding)", "ELA", "7B", "TGIF sd2-sp",
  "Does the model ground on the supplied ELA cue?",
  "protocols/PROTOCOL_v4_VALIDATION_FROZEN.md", "v4_stageA", "v4_stageA", "VALID",
  "H1/H2 supported: the model does use the cue"),
 ("E06", "Validation V2 (Stage B verdict)", "ELA", "7B", "TGIF sd2-sp",
  "Does correct grounding translate into a correct verdict?",
  "protocols/PROTOCOL_v4_VALIDATION_FROZEN.md", "v4_stageB", "v4_stageB", "VALID",
  "Case D dominant: grounding and verdict are disconnected"),
 ("E07", "Framing x visualization mechanism 2x2", "ELA", "7B", "TGIF sd2-sp",
  "Is the fake-decision bias caused by forensic framing, the heatmap, or both?",
  "protocols/PROTOCOL_v4_VALIDATION_FROZEN.md", "mechanism_run", "mechanism_run",
  "VALID", "Pure interaction: C1-C0 = C2-C0 = 0.000, C3-C2 = +0.267/+0.287"),
 ("E08", "M1 verification gating", "ELA", "7B", "TGIF sd2-sp",
  "Can a verification step recover decision utility?",
  "protocols/PROTOCOL_v4_VALIDATION_FROZEN.md", "mitigation_run", "mitigation_run",
  "VALID", "FAILED as an intervention: verifier output collapsed. Negative result"),
 ("E09", "32B Stage-V capacity ablation", "ELA", "32B", "TGIF sd2-sp",
  "Is the 7B failure a capacity problem?",
  "protocols/PROTOCOL_v4_VALIDATION_FROZEN.md", "qwen32b_stagev_ablation",
  "qwen32b_stagev_ablation", "VALID",
  "Case 2: 32B collapsed to the opposite constant. Capacity is not the cause"),
 ("E10", "7B Stage-V C_own recheck", "ELA", "7B", "TGIF sd2-sp",
  "Re-run of the real-own condition after ERRATUM 001",
  "protocols/PROTOCOL_v4_VALIDATION_FROZEN.md", "qwen7b_stagev_cown",
  "qwen7b_stagev_cown", "VALID",
  "Fixes the cue_vis cache-key bug; see docs/INVALID_AND_SUPERSEDED_RUNS.md"),
 ("T00", "TruFor install + smoke", "TruFor", "-", "-",
  "Verify the adapter reproduces upstream outputs bit-for-bit",
  "environment/TRUFOR_ENV_REPORT.md", "trufor_smoke", "-", "VALID",
  "Adapter vs official: delta = 0"),
 ("T01", "TruFor standalone feasibility", "TruFor", "-", "TGIF sd2-sp 200 src",
  "Is TruFor strong enough on this data to be worth integrating?",
  "protocols/TRUFOR_FEASIBILITY_PROTOCOL.md", "trufor_feasibility",
  "trufor_feasibility/analysis.txt", "VALID",
  "GO. AUROC 0.9845, TPR@5%FPR 0.955, pixel AUROC 0.993, IoU 0.758, F1 0.846"),
 ("T02", "Stage T1 localization", "TruFor", "7B", "TGIF sd2-sp 184 src",
  "Tool-guided localization stage",
  "protocols/TRUFOR_MLLM_PROTOCOL_FROZEN.md", "trufor_mllm_stageA",
  "trufor_mllm_stageA", "VALID", ""),
 ("T03", "Stage T2 verdict", "TruFor", "7B", "TGIF sd2-sp 184 src",
  "First positive integration result",
  "protocols/TRUFOR_MLLM_PROTOCOL_FROZEN.md", "trufor_mllm_stageB",
  "trufor_mllm_stageB", "VALID",
  "Case C: recall 0.201->0.804, FPR 0.125->0.011, J +0.076->+0.793"),
 ("T04", "Score-map 2x2", "TruFor", "7B", "TGIF sd2-sp 184 src",
  "Which carries the verdict: the scalar score or the heatmap?",
  "protocols/TRUFOR_SCORE_MAP_ABLATION_PROTOCOL.md", "trufor_score_map",
  "trufor_score_map", "VALID",
  "Case 4: score dominates; the map COSTS J -0.228 when a score is present"),
 ("T05", "Score-map conflict interventions", "TruFor", "7B", "TGIF sd2-sp 184 src",
  "Blank / donor / shifted / amplified / binarized map",
  "protocols/TRUFOR_SCORE_MAP_CONFLICT_PROTOCOL.md", "trufor_conflict",
  "trufor_conflict", "VALID",
  "shift effect -0.027, CI contains zero: NO spatial-correspondence checking"),
 ("T06", "32B conflict replication", "TruFor", "32B", "TGIF sd2-sp 184 src",
  "Does capacity restore spatial-correspondence checking?",
  "protocols/TRUFOR_32B_CAPACITY_PROTOCOL.md", "trufor_32b", "trufor_32b", "VALID",
  "No: the spatial-correspondence effect stays at -0.027 with CI containing zero"),
 ("S01", "Evidence abstraction layer E0-E4", "TruFor", "7B", "TGIF sd2-sp 184 src",
  "Replace the raw map with a deterministic structured description",
  "protocols/TRUFOR_EVIDENCE_ABSTRACTION_PROTOCOL.md", "trufor_abstraction",
  "trufor_abstraction", "VALID",
  "spatial agreement 1.000 vs raw map 0.358; E4 eligible subset 70/184 fake"),
 ("S02", "Structured field ablation S0-S5", "TruFor", "7B", "TGIF sd2-sp 184 src",
  "Which structured field causes the decision collapse?",
  "protocols/TRUFOR_STRUCTURED_FIELD_ABLATION_PROTOCOL.md", "trufor_fields",
  "trufor_fields", "VALID",
  "area_ratio hypothesis FALSIFIED; all single fields hurt similarly"),
 ("S03", "Framing / region-presence control P0-P5", "TruFor", "7B",
  "TGIF sd2-sp 184 src",
  "Is the collapse from the prompt framing or from real field values?",
  "protocols/TRUFOR_STRUCTURED_FRAMING_PROTOCOL.md", "trufor_framing",
  "trufor_framing", "VALID",
  "Framing-induced policy shift: P1 with no payload already collapses"),
 ("S04", "Task semantics D0-D3", "TruFor", "7B", "TGIF sd2-sp 184 src",
  "Is the collapse a task-semantics misunderstanding or arbitration failure?",
  "protocols/TRUFOR_TASK_SEMANTICS_PROTOCOL.md", "trufor_semantics",
  "trufor_semantics", "VALID",
  "D1 binary rule restores utility. D2/D3 presence decomposition over-detects"),
 ("S05", "Score x semantics 2x2 (A0-A3)", "TruFor", "7B", "TGIF sd2-sp 184 src",
  "Does the rule fix framing specifically, or clarify the task generally?",
  "protocols/TRUFOR_SCORE_SEMANTICS_2x2_PROTOCOL.md", "trufor_2x2", "trufor_2x2",
  "VALID", "A3 ~ A1: structured evidence gives NO extra decision utility"),
 ("F01", "32B final capacity replication B0-B4", "TruFor", "32B",
  "TGIF sd2-sp 184 src",
  "Do the three final 7B mechanism conclusions hold at 32B?",
  "protocols/TRUFOR_32B_FINAL_CAPACITY_PROTOCOL.md", "trufor_32b_final",
  "trufor_32b_final", "VALID",
  "Two of three are capacity-DEPENDENT; only role separation is invariant"),
 ("X01", "Spatial correctness retrospective", "-", "7B+32B", "TGIF sd2-sp 184 src",
  "Evidence->GT, Explanation->Evidence, Explanation->GT",
  "-", "explainability", "explainability/x1.txt", "VALID",
  "No new inference. Evidence->GT precision 0.979 / recall 0.531"),
 ("X02", "Semantic audit protocol freeze", "-", "-", "100 fake + 50 real",
  "Freeze samples, rubric, schema, repeat subset and metrics",
  "protocols/EXPLAINABILITY_SEMANTIC_AUDIT_PROTOCOL.md",
  "explainability", "-", "VALID",
  "7B D0 and 32B B2 prompts are byte-identical, so the model is the only variable"),
 ("X03", "Semantic explanation annotation", "-", "7B+32B", "100 fake + 50 real",
  "Audit whether each natural-language claim is supported by the image",
  "protocols/EXPLAINABILITY_SEMANTIC_AUDIT_PROTOCOL.md",
  "explainability/x3", "explainability/x3/x3_report.txt", "VALID",
  "300 primary + 60 repeat. Annotator is assistive, NOT ground truth"),
]


def sha1f(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()


def count_rows(d):
    n, conds = 0, Counter()
    for f in glob.glob(os.path.join(RUNS, d, "*.jsonl")):
        if "INVALID" in f:
            continue
        for l in open(f, encoding="utf-8", errors="ignore"):
            n += 1
            try:
                conds[json.loads(l).get("condition", "?")] += 1
            except Exception:
                pass
    return n, conds


def frozen_of(d):
    out = []
    for f in sorted(glob.glob(os.path.join(RUNS, d, "*frozen*.json"))):
        out.append((os.path.relpath(f).replace("\\", "/"), sha1f(f)))
    return out


def main():
    L = ["# EXPERIMENT_INDEX.md\n",
         "Every formal experiment of the training-free stage, in execution order.",
         "Inference counts and protocol hashes in this file are read off the "
         "artifacts by `scripts/gen_docs.py`, not typed by hand.\n",
         "Status values: **VALID** (citable) / **SUPERSEDED** (correct but replaced) "
         "/ **INVALID** (must not be cited, see "
         "`INVALID_AND_SUPERSEDED_RUNS.md`) / **EXPLORATORY** (smoke or debug).\n"]

    L.append("## Index\n")
    L.append("| ID | experiment | tool | MLLM | inferences | status |")
    L.append("|---|---|---|---|---|---|")
    for (i, nm, tool, mllm, ds, purp, proto, rd, an, st, note) in SPEC:
        n, _ = count_rows(rd)
        L.append(f"| [{i}](#{i.lower()}) | {nm} | {tool} | {mllm} | "
                 f"{n if n else '-'} | {st} |")
    L.append("")

    L.append("## Details\n")
    for (i, nm, tool, mllm, ds, purp, proto, rd, an, st, note) in SPEC:
        n, conds = count_rows(rd)
        L.append(f"### {i}")
        L.append(f"**{nm}**\n")
        L.append("| field | value |")
        L.append("|---|---|")
        L.append(f"| tool | {tool} |")
        L.append(f"| MLLM | {mllm} |")
        L.append(f"| dataset / split | {ds} |")
        L.append(f"| purpose | {purp} |")
        L.append(f"| protocol | `{proto}` |")
        L.append(f"| run directory | `runs/{rd}/` |")
        L.append(f"| analysis | `{an if an!='-' else '-'}` |")
        L.append(f"| inferences | {n if n else '-'} |")
        if conds:
            L.append(f"| conditions | {', '.join(f'`{k}` ({v})' for k, v in sorted(conds.items()))} |")
        L.append(f"| status | **{st}** |")
        if note:
            L.append(f"| note | {note} |")
        fz = frozen_of(rd)
        if fz:
            for path, h in fz:
                L.append(f"| frozen protocol | `{path}` sha1 `{h}` |")
        L.append("")

    tot = sum(count_rows(s[7])[0] for s in SPEC)
    nv = sum(1 for s in SPEC if s[9] == "VALID")
    L.append("## Totals\n")
    L.append(f"- formal experiments indexed: **{len(SPEC)}**")
    L.append(f"- VALID: **{nv}**, SUPERSEDED: "
             f"**{sum(1 for s in SPEC if s[9]=='SUPERSEDED')}**")
    L.append(f"- total indexed inference rows: **{tot}** "
             f"(excludes files marked INVALID and all smoke runs)")
    L.append("")
    L.append("Smoke and debug directories (`smoke*`, `pilot`, `validation`, "
             "`splits_tgif`, `mechanism`, `mitigation`, `calibration`) are retained "
             "on disk and listed in `PROJECT_INVENTORY_BEFORE_CLEANUP.md`; they are "
             "EXPLORATORY and are not cited anywhere.")

    os.makedirs("docs", exist_ok=True)
    body = "\n".join(L)
    open("docs/EXPERIMENT_INDEX.md", "w", encoding="utf-8").write(body)
    print(f"wrote docs/EXPERIMENT_INDEX.md ({len(body)} bytes, {len(SPEC)} experiments,"
          f" {tot} indexed rows)")


if __name__ == "__main__":
    main()
