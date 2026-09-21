"""Analysis of the frozen M1 mitigation experiment.

Reports, in the order fixed by the protocol: integrity, Stage-V behaviour, the
six/eight-condition TPR/FPR/J table, the three goals, the secondary diagnostics,
and the attribution contrasts that separate deliberation (S vs B) from
verification-state gating (M vs S).

All intervals are coco_id-cluster bootstrap. Differences are always reported as a
difference with its own CI; CI overlap is never used as evidence of equivalence.
"""
import json, os, sys
from collections import defaultdict, Counter
from math import comb
import numpy as np, yaml

ORDER = ("B-none", "B-correct", "B-wrong", "S-correct", "S-wrong",
         "M-correct", "M-wrong", "M-none")
CORE = ("B-none", "B-correct", "B-wrong", "S-correct", "S-wrong",
        "M-correct", "M-wrong")
NBOOT, SEED = 10000, 20260918
PRACTICAL = 0.10          # project-internal engineering standard, not a field norm


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def cb(cl, fn, nb=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    ks = sorted(cl)
    v = [fn(cl[k]) for k in ks]
    w = [len(cl[k]) for k in ks]
    keep = [i for i, x in enumerate(v) if np.isfinite(x)]
    if not keep:
        return (float("nan"),) * 3
    obs = np.average([v[i] for i in keep], weights=[w[i] for i in keep])
    out = []
    for _ in range(nb):
        p = rng.integers(0, len(keep), len(keep))
        ii = [keep[i] for i in p]
        out.append(np.average([v[i] for i in ii], weights=[w[i] for i in ii]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    out_root = cfg["experiment"]["out_root"]
    tag = sys.argv[2] if len(sys.argv) > 2 else "mitigation_run"
    F = json.load(open(os.path.join(out_root, "mitigation", "mitigation_frozen.json")))
    D = [json.loads(l) for l in open(os.path.join(out_root, tag, "decisions.jsonl"))]
    V = [json.loads(l) for l in open(os.path.join(out_root, tag, "stage_v.jsonl"))]

    G = defaultdict(dict)
    for r in D:
        G[(r["pair_id"], r["label"])][r["condition"]] = r
    Vm = {(r["pair_id"], r["label"], r["cue"]): r for r in V}
    cid = {s["pair_id"]: s["coco_id"] for s in F["samples"]}
    tam = {s["pair_id"]: s["tamper_ratio"] for s in F["samples"]}
    ids = sorted({p for (p, l) in G
                  if len(G[(p, "fake")]) == 8 and len(G[(p, "real")]) == 8})

    print("M1 MITIGATION EXPERIMENT — verification-gated evidence, sham-controlled")
    print(f"decisions={len(D)}  stage_v={len(V)}  complete samples={len(ids)}")
    print(f"clusters={len({cid[p] for p in ids})}  bootstrap={NBOOT} seed={SEED}")
    print(f"sampling: {F['sampling_rule']}\n")

    print("=" * 86)
    print("1. INTEGRITY")
    pf = sum(1 for r in D if "json_parse_failed" in (r["notes"] or []))
    vpf = sum(1 for r in V if "stage_v_parse_failed" in (r["notes"] or []))
    vinv = sum(1 for r in V if "invalid_stage_v_to_state_C" in (r["notes"] or []))
    ph_ok = all(r["prompt_sha1"] == F["prompt_hashes"][
        f"STAGE_D_{r['state_injected']}" if r["state_injected"]
        else ("B_NONE" if r["cue"] is None else "B_CUE")] for r in D)
    bs_ok = all(G[(p, l)][f"B-{c}"]["prompt_sha1"] == G[(p, l)][f"S-{c}"]["prompt_sha1"]
                for p in ids for l in ("fake", "real") for c in ("correct", "wrong"))
    s_clean = all(G[(p, l)][f"S-{c}"]["state_injected"] is None
                  for p in ids for l in ("fake", "real") for c in ("correct", "wrong"))
    m_inj = all(G[(p, l)][f"M-{c}"]["state_injected"] in ("A", "B", "C")
                for p in ids for l in ("fake", "real") for c in ("correct", "wrong"))
    print(f"  decision JSON parse failures     : {pf}/{len(D)}")
    print(f"  Stage-V parse failures           : {vpf}/{len(V)}")
    print(f"  Stage-V invalid -> forced State C: {vinv}")
    print(f"  all prompt hashes match protocol : {ph_ok}")
    print(f"  B and S share Stage-D prompt     : {bs_ok}")
    print(f"  S never received a state         : {s_clean}")
    print(f"  M always received a state        : {m_inj}")
    print(f"  M-none == B-none prompt          : "
          f"{F['prompt_hashes']['M_NONE'] == F['prompt_hashes']['B_NONE']}")

    # ---------------------------------------------------------------- Stage V
    print("\n" + "=" * 86)
    print("2. STAGE-V VERIFIER BEHAVIOUR  (conservative, prompt-induced criterion)")
    print("   NOTE: the asymmetric 'supported only if' wording is imposed by the")
    print("   prompt; this is NOT naturally emergent model calibration.")
    cells = [("fake", "correct", "fake + correct ELA"),
             ("fake", "donor", "fake + donor ELA"),
             ("real", "correct", "real + own ELA  <-- key stress test"),
             ("real", "donor", "real + donor ELA")]
    print(f"\n  {'cell':<34}{'matched':>9}{'mismatch':>10}{'supported':>11}"
          f"{'insuff':>9}   states A/B/C")
    SV = {}
    for lab, cue, nm in cells:
        rs = [Vm[(p, lab, cue)] for p in ids if (p, lab, cue) in Vm]
        n = len(rs)
        m = sum(1 for r in rs if r["image_consistency"] == "matched")
        su = sum(1 for r in rs if r["forensic_support"] == "supported")
        st = Counter(r["state"] for r in rs)
        SV[(lab, cue)] = dict(n=n, matched=m / n if n else float("nan"),
                              supported=su / n if n else float("nan"), states=st)
        print(f"  {nm:<34}{m/n:>9.3f}{1-m/n:>10.3f}{su/n:>11.3f}{1-su/n:>9.3f}   "
              f"{st.get('A',0)}/{st.get('B',0)}/{st.get('C',0)}")

    def cbv(lab, cue, fn):
        cl = defaultdict(list)
        for p in ids:
            if (p, lab, cue) in Vm:
                cl[cid[p]].append(Vm[(p, lab, cue)])
        return cb(cl, fn)

    print("\n  headline Stage-V quantities (cluster CI):")
    o, lo, hi = cbv("fake", "donor", lambda c: np.mean(
        [int(x["image_consistency"] == "mismatched") for x in c]))
    print(f"    P(mismatched | fake, donor cue)  {o:.3f} [{lo:.3f},{hi:.3f}]")
    o, lo, hi = cbv("real", "correct", lambda c: np.mean(
        [int(x["forensic_support"] == "insufficient") for x in c]))
    print(f"    P(insufficient | real, own ELA)  {o:.3f} [{lo:.3f},{hi:.3f}]")
    o, lo, hi = cbv("real", "correct", lambda c: np.mean(
        [int(x["state"] == "A") for x in c]))
    print(f"    P(State A | real, own ELA)       {o:.3f} [{lo:.3f},{hi:.3f}]"
          f"   <- real image kept as evidence")
    # discrimination: matched(correct) - matched(donor), paired within fake images
    cl = defaultdict(list)
    for p in ids:
        cl[cid[p]].append((Vm[(p, "fake", "correct")], Vm[(p, "fake", "donor")]))
    o, lo, hi = cb(cl, lambda c: np.mean(
        [int(a["image_consistency"] == "matched") -
         int(b["image_consistency"] == "matched") for a, b in c]))
    print(f"    P(matched|correct) - P(matched|donor), fake images: "
          f"{o:+.3f} [{lo:+.3f},{hi:+.3f}]")
    print(f"      -> {'verifier DISCRIMINATES cue validity' if lo > 0 else 'verifier does NOT discriminate cue validity (CI includes zero)'}")
    deg = all(SV[(l, c)]["matched"] < 0.05 for l, c, _ in cells)
    if deg:
        print("    WARNING: verifier rejects essentially everything -> Goal 1 could be")
        print("    met trivially by blanket rejection. Interpret Goals 2/3 accordingly.")

    # ------------------------------------------------------------- main table
    def clus(lab):
        cl = defaultdict(list)
        for p in ids:
            cl[cid[p]].append(G[(p, lab)])
        return cl
    clf, clr = clus("fake"), clus("real")
    rate = lambda c: (lambda cells: float(np.mean(
        [int(x[c]["final_verdict"] == "fake") for x in cells])))
    clj = defaultdict(list)
    for p in ids:
        clj[cid[p]].append((G[(p, "fake")], G[(p, "real")]))

    print("\n" + "=" * 86)
    print("3. TPR / FPR / YOUDEN J BY CONDITION")
    print(f"  {'condition':<11}{'TPR':>7} {'TPR 95% CI':<18}{'FPR':>7} "
          f"{'FPR 95% CI':<18}{'J':>7} {'J 95% CI'}")
    T, Fp, J = {}, {}, {}
    for c in ORDER:
        t, tl, th = cb(clf, rate(c))
        f, fl, fh = cb(clr, rate(c))
        j, jl, jh = cb(clj, lambda cs, c=c: float(np.mean(
            [int(a[c]["final_verdict"] == "fake") - int(b[c]["final_verdict"] == "fake")
             for a, b in cs])))
        T[c], Fp[c], J[c] = t, f, j
        mark = "  (sanity only)" if c == "M-none" else ""
        print(f"  {c:<11}{t:>7.3f} [{tl:.3f},{th:.3f}]   {f:>7.3f} [{fl:.3f},{fh:.3f}]"
              f"   {j:>7.3f} [{jl:+.3f},{jh:+.3f}]{mark}")

    def diff(lab, a, b, metric="rate"):
        if metric == "J":
            d, lo, hi = cb(clj, lambda cs: float(np.mean(
                [(int(x[a]["final_verdict"] == "fake") - int(y[a]["final_verdict"] == "fake"))
                 - (int(x[b]["final_verdict"] == "fake") - int(y[b]["final_verdict"] == "fake"))
                 for x, y in cs])))
            return d, lo, hi, None
        cl = clf if lab == "fake" else clr
        d, lo, hi = cb(cl, lambda cs: float(np.mean(
            [int(x[a]["final_verdict"] == "fake") - int(x[b]["final_verdict"] == "fake")
             for x in cs])))
        _, _, p = mcnemar([int(G[(q, lab)][a]["final_verdict"] == "fake") for q in ids],
                          [int(G[(q, lab)][b]["final_verdict"] == "fake") for q in ids])
        return d, lo, hi, p

    def show(label, lab, a, b, metric="rate", target=""):
        d, lo, hi, p = diff(lab, a, b, metric)
        sig = "EXCLUDES zero" if (lo > 0 or hi < 0) else "includes zero"
        ps = f"  McNemar p={p:.4g}" if p is not None else ""
        print(f"  {label:<44}{d:+.3f}  [{lo:+.3f},{hi:+.3f}] {sig}{ps}")
        if target:
            print(f"      target: {target}")
        return d, lo, hi

    print("\n" + "=" * 86)
    print("4. PRIMARY GOALS")
    g1 = show("Goal1 FPR(M-wrong) - FPR(B-wrong)", "real", "M-wrong", "B-wrong",
              target="negative, with a clear practical effect")
    g2 = show("Goal2 J(M-correct) - J(B-correct)", None, "M-correct", "B-correct",
              "J", target="positive  <-- CORE MITIGATION METRIC")
    g3f = show("Goal3 FPR(M-none) - FPR(B-none)", "real", "M-none", "B-none",
               target="near zero (identical prompts; pure sampling noise)")
    g3t = show("Goal3 TPR(M-none) - TPR(B-none)", "fake", "M-none", "B-none",
               target="near zero")

    print("\n" + "=" * 86)
    print("5. SECONDARY DIAGNOSTICS")
    s1 = show("TPR(M-correct) - TPR(B-none)", "fake", "M-correct", "B-none",
              target="positive: useful correct-cue benefit preserved")
    s2 = show("J(M-wrong) - J(B-none)", None, "M-wrong", "B-none", "J",
              target="near zero: wrong cue approximately harmless")
    s3 = show("dJ_correct = J(M-correct) - J(B-correct)", None, "M-correct",
              "B-correct", "J")
    s4 = show("dJ_wrong  = J(M-wrong) - J(B-wrong)", None, "M-wrong", "B-wrong", "J")

    print("\n" + "=" * 86)
    print("6. ATTRIBUTION — deliberation (S vs B) versus gating (M vs S)")
    print("\n  a) deliberation effect: does the extra Stage-V pass alone help?")
    a1 = show("FPR(S-wrong) - FPR(B-wrong)", "real", "S-wrong", "B-wrong")
    a2 = show("J(S-correct) - J(B-correct)", None, "S-correct", "B-correct", "J")
    print("\n  b) verification-state gating effect: does injecting the state help")
    print("     BEYOND having deliberated?")
    a3 = show("FPR(M-wrong) - FPR(S-wrong)", "real", "M-wrong", "S-wrong",
              target="negative")
    a4 = show("J(M-correct) - J(S-correct)", None, "M-correct", "S-correct", "J",
              target="positive, or at least not worse")

    print("\n" + "=" * 86)
    print("7. STATE DISTRIBUTION ACTUALLY INJECTED INTO M CONDITIONS")
    for lab in ("fake", "real"):
        for cue in ("correct", "wrong"):
            st = Counter(G[(p, lab)][f"M-{cue}"]["state_injected"] for p in ids)
            print(f"  M-{cue:<8} {lab:<5} A={st.get('A',0):<4}B={st.get('B',0):<4}"
                  f"C={st.get('C',0)}")

    print("\n" + "=" * 86)
    print("8. PRE-REGISTERED VERDICT")
    neg = lambda t: t[2] < 0          # CI upper bound below zero
    pos = lambda t: t[1] > 0          # CI lower bound above zero
    small = lambda t: abs(t[0]) < 0.05
    print(f"  Goal1 wrong-cue FPR suppressed      : "
          f"{'MET' if neg(g1) else 'NOT MET'}  ({g1[0]:+.3f})")
    print(f"  Goal2 J(M-correct) > J(B-correct)   : "
          f"{'MET' if pos(g2) else 'NOT MET'}  ({g2[0]:+.3f})")
    print(f"  Goal3 structure introduces no shift : "
          f"{'MET' if small(g3f) and small(g3t) else 'NOT MET'}  "
          f"(FPR {g3f[0]:+.3f}, TPR {g3t[0]:+.3f})")
    print(f"  residual FPR(M-wrong)={Fp['M-wrong']:.3f} vs project-internal "
          f"engineering standard {PRACTICAL:.2f}: "
          f"{'below' if Fp['M-wrong'] <= PRACTICAL else 'above'}")
    print("    (project-internal only; NOT a field standard)")
    print()
    delib_w, delib_c = neg(a1), pos(a2)
    gate_w, gate_c = neg(a3), pos(a4)
    if not (delib_w or delib_c) and (gate_w or gate_c):
        print("  -> S ~ B and M > S: VERIFICATION-STATE GATING produces the benefit.")
    elif (delib_w or delib_c) and not (gate_w or gate_c):
        print("  -> S > B and M ~ S: benefit is mainly EXTRA DELIBERATION.")
        print("     Do NOT attribute the effect to evidence gating.")
    elif (delib_w or delib_c) and (gate_w or gate_c):
        print("  -> S > B and M > S: BOTH deliberation and explicit gating contribute.")
    else:
        print("  -> no improvement attributable to either component.")
    if not (neg(g1) or pos(g2)):
        print("  -> M1 MITIGATION FAILS on the primary goals.")
    if deg:
        print("\n  CAVEAT: the verifier rejected nearly all cues, so any wrong-cue")
        print("  suppression may reflect blanket rejection rather than verification.")
        print("  Check TPR(M-correct) - TPR(B-none) before claiming mitigation.")

    print("\n" + "=" * 86)
    print("9. TAMPER-RATIO BANDS (descriptive)")
    qs = np.quantile([tam[p] for p in ids], [0.25, 0.5, 0.75])
    def band(t):
        return ("q1" if t <= qs[0] else "q2" if t <= qs[1]
                else "q3" if t <= qs[2] else "q4")
    print(f"  quartile cuts: {qs[0]:.4f} / {qs[1]:.4f} / {qs[2]:.4f}")
    print(f"  {'band':<6}{'n':>4}  " + "".join(f"{c:>12}" for c in
          ("B-corr TPR", "M-corr TPR", "B-wrong FPR", "M-wrong FPR")))
    for b in ("q1", "q2", "q3", "q4"):
        sub = [p for p in ids if band(tam[p]) == b]
        if not sub:
            continue
        f = lambda c, lab: np.mean([int(G[(p, lab)][c]["final_verdict"] == "fake")
                                    for p in sub])
        print(f"  {b:<6}{len(sub):>4}  {f('B-correct','fake'):>12.3f}"
              f"{f('M-correct','fake'):>12.3f}{f('B-wrong','real'):>12.3f}"
              f"{f('M-wrong','real'):>12.3f}")


if __name__ == "__main__":
    main()
