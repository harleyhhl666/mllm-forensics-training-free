"""Phase-1 analysis: the 9 required outputs, with correct denominators.

Key rule enforced here: the core rejection rate's denominator is
N(Gate0-3 all pass), never N(all fake). Every gate's own pass rate is reported
separately so the failure stage stays identifiable.
"""
import json, os, sys
from collections import Counter
import numpy as np, yaml


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    if n == 0:
        return f"{k}/{n} (n/a)"
    return f"{k}/{n} = {p:.3f} [95% CI {lo:.3f}-{hi:.3f}]"


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    tag = sys.argv[2] if len(sys.argv) > 2 else "run"
    path = os.path.join(cfg["experiment"]["out_root"], tag, "samples.jsonl")
    rows = [json.loads(l) for l in open(path)]
    fake = [r for r in rows if r["ground_truth"] == "fake"]
    real = [r for r in rows if r["ground_truth"] == "real"]
    grid = cfg["localization"]["grid"]
    print(f"records: {len(rows)}  (fake {len(fake)}, real {len(real)})\n")

    # ---- 1. baseline real/fake performance (Stage B verdict)
    print("=" * 70)
    print("1. BASELINE REAL/FAKE PERFORMANCE (Stage B verdict, all samples)")
    tp = sum(1 for r in fake if r["stage_b"] and r["stage_b"]["final_verdict"] == "fake")
    tn = sum(1 for r in real if r["stage_b"] and r["stage_b"]["final_verdict"] == "real")
    unp = sum(1 for r in rows if not r["stage_b"] or not r["stage_b"]["final_verdict"])
    print(f"  recall on fake (detected as fake): {fmt(tp, len(fake))}")
    print(f"  specificity on real (called real): {fmt(tn, len(real))}")
    print(f"  accuracy: {fmt(tp + tn, len(rows))}")
    print(f"  unparseable/absent verdicts: {unp}")
    vc = Counter((r["ground_truth"], (r["stage_b"] or {}).get("final_verdict")) for r in rows)
    print(f"  confusion (gt, verdict): {dict(vc)}")

    # ---- 2. tool validity rates
    print("\n" + "=" * 70)
    print("2. FORENSIC TOOL VALID / INVALID RATE (Gate 0, per tool)")
    tnames = sorted({t for r in rows for t in r["tool_validity"]})
    for t in tnames:
        k = sum(1 for r in rows if r["tool_validity"].get(t))
        print(f"  {t:<18} valid {fmt(k, len(rows))}")
        rs = Counter(x for r in rows for x in r["tool_invalid_reasons"].get(t, []))
        for reason, c in rs.most_common(4):
            print(f"      invalid reason: {reason} ({c})")
    print(f"  Gate 0 (>=1 valid tool): {fmt(sum(1 for r in rows if r['gate0']), len(rows))}")

    # ---- 3-5, 8: the gate cascade, on FAKE samples
    print("\n" + "=" * 70)
    print("3-5,8. GATE CASCADE ON FAKE SAMPLES (each rate conditional on the previous gate)")
    g0 = [r for r in fake if r["gate0"]]
    g1 = [r for r in g0 if r["gate1"]]
    g2 = [r for r in g1 if r["gate2"]]
    g2i = [r for r in g2 if not r.get("localization_gate_uninformative")]
    g3 = [r for r in g2i if r["gate3"]]
    print(f"  all fake                          : {len(fake)}")
    print(f"  Gate 0 tool valid                 : {fmt(len(g0), len(fake))}")
    print(f"  Gate 1 objective evidence | G0    : {fmt(len(g1), len(g0))}")
    print(f"  Gate 2 MLLM acknowledges  | G1    : {fmt(len(g2), len(g1))}")
    print(f"  (excluded: GT blankets grid, Gate 3 vacuous): {len(g2) - len(g2i)}")
    print(f"  Gate 3 localization ok    | G2*   : {fmt(len(g3), len(g2i))}")
    print(f"  Gate 0-3 all pass                 : {len(g3)}")

    # ---- Gate 2 specificity: does acknowledgment mean anything?
    print("\n" + "=" * 70)
    print("2b. GATE-2 SPECIFICITY (critical: is acknowledgment informative at all?)")
    rv = [r for r in real if r["gate0"]]
    ack_fake = sum(1 for r in g0 if r["gate2"])
    ack_real = sum(1 for r in rv if r["gate2"])
    print(f"  evidence_detected on FAKE (Gate 0 passed): {fmt(ack_fake, len(g0))}")
    print(f"  evidence_detected on REAL (Gate 0 passed): {fmt(ack_real, len(rv))}")
    if len(g0) and len(rv):
        d = ack_fake / len(g0) - ack_real / len(rv)
        print(f"  difference = {d:+.3f}")
        if abs(d) < 0.10:
            print("  -> WARNING: the model acknowledges 'evidence' on real and fake at nearly")
            print("     the same rate. Gate 2 is then near-vacuous: passing it does not show")
            print("     the model read genuine forensic evidence, and any downstream")
            print("     'rejection' statistic must not be interpreted as evidence rejection.")

    # ---- 6. random localization baseline
    print("\n" + "=" * 70)
    print("6. RANDOM LOCALIZATION BASELINE (Control 2)")
    # expected hit rate of a random prediction set of the model's own size
    exp_hits, obs = [], []
    for r in g2i:
        ngt = len(r["gt_regions"])
        npred = max(1, len(r["stage_a"]["predicted_regions"])) if r["stage_a"] else 1
        cells = grid * grid
        # P(at least one hit) for npred draws without replacement from `cells`
        p_miss = 1.0
        for i in range(npred):
            p_miss *= max(0.0, (cells - ngt - i)) / max(1, (cells - i))
        exp_hits.append(1 - p_miss)
        obs.append(1 if r["gate3"] else 0)
    if obs:
        e = float(np.mean(exp_hits)); o = float(np.mean(obs)); n = len(obs)
        z = (o - e) / np.sqrt(max(e * (1 - e) / n, 1e-12))
        print(f"  model localization accuracy (Gate 3 | Gate 2): {o:.3f}  (n={n})")
        print(f"  random baseline, matched prediction-set sizes: {e:.3f}")
        print(f"  z = {z:+.2f}   -> {'ABOVE chance' if z > 1.96 else 'NOT above chance'}")
    else:
        print("  no Gate-2 samples; baseline undefined")

    # ---- 7. no-tool localization (Control 1)
    print("\n" + "=" * 70)
    print("7. NO-TOOL LOCALIZATION (Control 1: does the model use the tools?)")
    both = [r for r in fake if r["stage_a"] and r["stage_a_no_tool"]
            and not r.get("localization_gate_uninformative")]
    wt = sum(1 for r in both if r["localization_correct"])
    nt = sum(1 for r in both if r["localization_correct_no_tool"])
    print(f"  with tools   : {fmt(wt, len(both))}")
    print(f"  without tools: {fmt(nt, len(both))}")
    # paired McNemar
    b = sum(1 for r in both if r["localization_correct"] and not r["localization_correct_no_tool"])
    c = sum(1 for r in both if not r["localization_correct"] and r["localization_correct_no_tool"])
    if b + c > 0:
        chi = (abs(b - c) - 1) ** 2 / (b + c)
        print(f"  McNemar: with-only={b}, without-only={c}, chi2(cc)={chi:.2f} "
              f"-> {'significant at .05' if chi > 3.84 else 'not significant'}")
    else:
        print("  McNemar: no discordant pairs")
    ack_wt = sum(1 for r in both if r["gate2"])
    ack_nt = sum(1 for r in both if r["stage_a_no_tool"]["evidence_detected"] is True)
    print(f"  evidence_detected rate with tools   : {fmt(ack_wt, len(both))}")
    print(f"  evidence_detected rate without tools: {fmt(ack_nt, len(both))}")

    # ---- 9. the core statistic
    print("\n" + "=" * 70)
    print("9. CORE: EVIDENCE ACKNOWLEDGED BUT REJECTED")
    rej = [r for r in g3 if r["evidence_rejection_case"]]
    print(f"  R_reject = N(G0-3 pass AND verdict=real) / N(G0-3 pass)")
    print(f"           = {fmt(len(rej), len(g3))}")
    print(f"  (for contrast, the WRONG denominator would give "
          f"{len(rej)}/{len(fake)} = {len(rej)/max(len(fake),1):.3f})")

    # stability (Control 3)
    print("\n  Control 3: stability of the reversal under repeated sampling")
    stable, flip, none_ = 0, 0, 0
    for r in g3:
        reps = [x for x in (r["stage_b_repeats"] or []) if x]
        if not reps:
            none_ += 1; continue
        if all(x == "real" for x in reps):
            stable += 1
        elif any(x == "real" for x in reps):
            flip += 1
    print(f"    G0-3 samples with all repeats = real (persistent reversal): {stable}")
    print(f"    sampling-sensitive (mixed real/fake across repeats)       : {flip}")
    print(f"    no usable repeats                                        : {none_}")

    # evidence strength stratification
    print("\n" + "=" * 70)
    print("10. REJECTION RATE BY EVIDENCE STRENGTH (denominator = G0-3 pass in that stratum)")
    for s in ("weak", "medium", "strong", None):
        sub = [r for r in g3 if r["evidence_strength"] == s]
        k = sum(1 for r in sub if r["evidence_rejection_case"])
        print(f"  {str(s):<8}: {fmt(k, len(sub))}")

    print("\n11. GATE RATES BY TAMPER_RATIO (explanatory variable)")
    for lo, hi in [(0, .05), (.05, .15), (.15, .35), (.35, 1.01)]:
        sub = [r for r in fake if lo <= r["tamper_ratio"] < hi]
        s0 = [r for r in sub if r["gate0"]]; s1 = [r for r in s0 if r["gate1"]]
        s2 = [r for r in s1 if r["gate2"]]; s3 = [r for r in s2 if r["gate3"]]
        rj = sum(1 for r in s3 if r["evidence_rejection_case"])
        print(f"  [{lo:.2f},{hi:.2f}) n={len(sub):>4}  G1|G0={len(s1)}/{len(s0)}  "
              f"G2|G1={len(s2)}/{len(s1)}  G3|G2={len(s3)}/{len(s2)}  rej={rj}/{len(s3)}")

    print("\n12. PROTOCOL INTEGRITY CHECKS")
    leak = sum(1 for r in rows if r["stage_a"] and r["stage_a"].get("verdict_leak_in_stage_a"))
    pf_a = sum(1 for r in rows if "json_parse_failed" in (r["stage_a_notes"] or []))
    pf_b = sum(1 for r in rows if "json_parse_failed" in (r["stage_b_notes"] or []))
    print(f"  Stage A outputs containing verdict-like language: {leak}/{len(rows)}")
    print(f"  Stage A JSON parse failures: {pf_a}/{len(rows)}")
    print(f"  Stage B JSON parse failures: {pf_b}/{len(rows)}")


if __name__ == "__main__":
    main()
