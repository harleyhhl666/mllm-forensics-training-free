"""Re-derive every headline number from the artifacts on disk.

Nothing here trusts a README, a filename or a chat summary. Each value is read from
a frozen protocol, a raw verdicts file, or an analysis output, and the source path is
printed next to it so a reader can check it. Cross-check targets supplied by the user
are compared, and any mismatch is reported as a MISMATCH to investigate -- never
silently overwritten.
"""
import glob, json, os, sys
from collections import Counter, defaultdict
import numpy as np

R = "/mnt/disk3/borui/fevi/runs"
OUT = []


def p(s=""):
    print(s)
    OUT.append(s)


def load_jsonl(path, cond=None):
    out = []
    if not os.path.exists(path):
        return out
    for l in open(path):
        d = json.loads(l)
        if cond is None or d.get("condition") == cond:
            out.append(d)
    return out


def metrics(rows):
    """recall / FPR / specificity / Youden J from raw verdicts."""
    f = [r for r in rows if r["label"] == "fake"]
    t = [r for r in rows if r["label"] == "real"]
    rec = np.mean([r["final_verdict"] == "fake" for r in f]) if f else float("nan")
    fpr = np.mean([r["final_verdict"] == "fake" for r in t]) if t else float("nan")
    return rec, fpr, 1 - fpr, rec - fpr, len(f), len(t)


def chk(name, got, want, tol=0.002):
    if want is None:
        return f"  {name:<34} {got:>8.3f}   (no target)"
    ok = abs(got - want) <= tol if not (np.isnan(got) or np.isnan(want)) else False
    tag = "OK" if ok else "** MISMATCH **"
    return f"  {name:<34} {got:>8.3f}   target {want:<7.3f} {tag}"


def main():
    p("=" * 88)
    p("CROSS-CHECK: every number re-derived from artifacts on disk")
    p("=" * 88)

    # ---------------- models ----------------
    p("\n## MODEL SNAPSHOTS (from frozen env / protocol files)")
    q32 = json.load(open(f"{R}/qwen32b/qwen32b_env_frozen.json"))
    p(f"  32B snapshot  {q32.get('snapshot') or q32.get('revision')}")
    p(f"    source: runs/qwen32b/qwen32b_env_frozen.json")
    fin = json.load(open(f"{R}/trufor_32b_final/trufor_32b_final_capacity_frozen.json"))
    p(f"  32B snapshot in final protocol: {json.dumps(fin.get('model'))[:200]}")
    sem = json.load(open(f"{R}/trufor_semantics/trufor_task_semantics_frozen.json"))
    p(f"  7B  snapshot in semantics protocol: "
      f"{json.dumps(sem.get('model'))[:200]}")

    # ---------------- TruFor standalone ----------------
    p("\n## TRUFOR STANDALONE FEASIBILITY")
    # The feasibility metrics live in the analysis TEXT output, not a json.
    import re
    ap = f"{R}/trufor_feasibility/analysis.txt"
    p(f"  source: {ap}")
    txt = open(ap).read() if os.path.exists(ap) else ""
    PAT = (("image AUROC", r"AUROC\s*=\s*([0-9.]+)", 0.9845),
           ("TPR@FPR<=0.05", r"TPR@FPR<=0\.05\s*=\s*([0-9.]+)", 0.955),
           ("pixel AUROC (mean)", r"pixel AUROC.*?mean\s+([0-9.]+)", 0.993),
           ("IoU @map>0.5 (mean)", r"IoU @map>0\.5\s+mean\s+([0-9.]+)", 0.758),
           ("pixel F1 @map>0.5 (mean)", r"pixel F1 @map>0\.5\s+mean\s+([0-9.]+)",
            0.846))
    for nm, rx, want in PAT:
        m = re.search(rx, txt, re.S)
        if m:
            p(chk(nm, float(m.group(1)), want))
        else:
            p(f"  {nm:<34} NOT FOUND in analysis.txt")
    rm = f"{R}/trufor_feasibility/run_meta.json"
    if os.path.exists(rm):
        j = json.load(open(rm))
        p(f"  checkpoint md5 {j.get('checkpoint_md5')}  "
          f"(target 55d7075dd1ff945e9c0f9437c5df9495 "
          f"{'OK' if j.get('checkpoint_md5')=='55d7075dd1ff945e9c0f9437c5df9495' else '** MISMATCH **'})")
        p(f"  protocol sha1  {j.get('protocol_sha1')}")

    # ---------------- 7B final 2x2 ----------------
    p("\n## 7B FINAL 2x2  (recomputed from raw verdicts)")
    TARGET7 = {"A0": (.837, .011, .826), "A1": (.978, .082, .897),
               "A2": (.114, .000, .114), "A3": (.967, .065, .902)}
    SRC7 = {"A0": (f"{R}/trufor_fields/verdicts.jsonl", "S0_score_only"),
            "A1": (f"{R}/trufor_2x2/verdicts_A1.jsonl", None),
            "A2": (f"{R}/trufor_semantics/verdicts.jsonl", "D0_baseline"),
            "A3": (f"{R}/trufor_semantics/verdicts.jsonl", "D1_binary_rule")}
    for cell in ("A0", "A1", "A2", "A3"):
        path, cond = SRC7[cell]
        rows = load_jsonl(path, cond)
        if cond is None:
            rows = [r for r in rows if r.get("condition", "").startswith("A1")] or rows
        rec_, fpr, spec, J, nf, nr = metrics(rows)
        tr, tf, tj = TARGET7[cell]
        p(f"  [{cell}] n_fake={nf} n_real={nr}   src={os.path.basename(path)}"
          f"{':'+cond if cond else ''}")
        p(chk("    recall", rec_, tr))
        p(chk("    FPR", fpr, tf))
        p(chk("    Youden J", J, tj))

    # ---------------- 32B final ----------------
    p("\n## 32B FINAL B0-B3  (recomputed from raw verdicts)")
    TARGET32 = {"B0_score_only": (.978, .060, .918),
                "B1_score_rule": (.989, .109, .880),
                "B2_structured": (.978, .076, .902),
                "B3_structured_rule": (1.000, .337, .663)}
    vp = f"{R}/trufor_32b_final/verdicts.jsonl"
    allrows = load_jsonl(vp)
    p(f"  source: {vp}  total rows {len(allrows)}")
    p(f"  conditions present: {dict(Counter(r['condition'] for r in allrows))}")
    for cond, (tr, tf, tj) in TARGET32.items():
        rows = [r for r in allrows if r["condition"] == cond]
        if not rows:
            p(f"  [{cond}] NOT FOUND")
            continue
        rec_, fpr, spec, J, nf, nr = metrics(rows)
        p(f"  [{cond}] n_fake={nf} n_real={nr}")
        p(chk("    recall", rec_, tr))
        p(chk("    FPR", fpr, tf))
        p(chk("    specificity", spec, 1 - tf))
        p(chk("    Youden J", J, tj))

    # ---------------- X1 evidence->GT ----------------
    p("\n## X1 EVIDENCE -> GT")
    x1p = f"{R}/explainability/x1_spatial_analysis.json"
    if os.path.exists(x1p):
        p(f"  source: {x1p}  (recomputing from stored GT + frozen evidence)")
        x1 = json.load(open(x1p))
        GT = x1["gt"]
        ev = load_jsonl(f"{R}/trufor_semantics/verdicts.jsonl", "D0_baseline")
        pr, rc = [], []
        for r in ev:
            if r["label"] != "fake":
                continue
            e = set(r.get("evidence_locations") or [])
            g = set(GT.get(r["sample_id"], {}).get("cells", []))
            if e and g:
                pr.append(len(e & g) / len(e))
                rc.append(len(e & g) / len(g))
        p(chk("  evidence->GT precision", float(np.mean(pr)), 0.979))
        p(chk("  evidence->GT recall", float(np.mean(rc)), 0.531))
        p(f"  n scored {len(pr)}")
    else:
        p(f"  {x1p} missing")

    # ---------------- X3 semantic audit ----------------
    p("\n## X3 SEMANTIC AUDIT  (recomputed from annotation files)")
    LOC = os.path.expanduser("C:/Users/HarleyH/fevi/x3") if os.name == "nt" \
        else "/mnt/disk3/borui/fevi/runs/explainability/x3"
    frp = os.path.join(LOC, "x2_semantic_audit_frozen.json")
    if not os.path.exists(frp):
        frp = f"{R}/explainability/x2_semantic_audit_frozen.json"
    fr = json.load(open(frp, encoding="utf-8"))
    MAP = fr["anonymization_map"]
    anndir = os.path.join(LOC, "ann")
    A = {}
    for q in glob.glob(os.path.join(anndir, "*.json")):
        try:
            A[os.path.basename(q)[:-5]] = json.load(open(q, encoding="utf-8"))
        except Exception:
            pass
    p(f"  annotations found: {len(A)}  (expected {len(fr['annotation_units'])})")
    if A:
        TG = {"M1": (.040, .310, .610, .040), "M2": (.490, .500, .010, .000)}
        for mc, nm in (("M1", "7B"), ("M2", "32B")):
            ids = [i for i in A if MAP[i]["label"] == "fake"
                   and MAP[i]["model_code"] == mc]
            if not ids:
                continue
            c = Counter(A[i]["overall_semantic_support"] for i in ids)
            n = len(ids)
            got = [c.get(k, 0) / n for k in ("supported", "partially_supported",
                                             "unsupported", "uncertain")]
            p(f"  [fake {nm}] n={n}")
            for lbl, g, w in zip(("supported", "partial", "unsupported", "uncertain"),
                                 got, TG[mc]):
                p(chk(f"    {lbl}", g, w))
            man = [cl["support"] for i in ids for cl in A[i].get("claims", [])
                   if cl.get("claim_type") == "manipulation"
                   and cl.get("support") != "not_applicable"]
            if man:
                u = float(np.mean([s == "unsupported" for s in man]))
                p(chk("    manipulation unsupported", u,
                      0.791 if mc == "M1" else 0.044))
                p(f"      n evaluable manipulation claims {len(man)}")

    # ---------------- run inventory with counts ----------------
    p("\n## RAW RESULT COUNTS (verdict rows per run directory)")
    p(f"  {'directory':<26}{'rows':>8}  conditions")
    for d in sorted(os.listdir(R)):
        fp = os.path.join(R, d)
        if not os.path.isdir(fp):
            continue
        for v in sorted(glob.glob(os.path.join(fp, "*.jsonl"))):
            rows = load_jsonl(v)
            if not rows:
                continue
            cs = Counter(r.get("condition", "?") for r in rows)
            p(f"  {d+'/'+os.path.basename(v):<26}{len(rows):>8}  "
              f"{len(cs)} cond: {', '.join(sorted(cs)[:4])}"
              f"{' ...' if len(cs) > 4 else ''}")

    p("\n" + "=" * 88)
    nm = sum(1 for l in OUT if "MISMATCH" in l)
    p(f"MISMATCHES: {nm}")
    p("Any mismatch above must be investigated, not overwritten.")
    dest = sys.argv[1] if len(sys.argv) > 1 else "CROSSCHECK.txt"
    open(dest, "w", encoding="utf-8").write("\n".join(OUT))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
