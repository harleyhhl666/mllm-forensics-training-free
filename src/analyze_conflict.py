"""Analysis of the score-map conflict intervention.

Reports each fake condition against F1, the real controls against R1, recovery in
the 56 suppression cases, the shifted-map correspondence test, and a deterministic
explanation audit. Amplified/binary are visualization interventions only and are
labelled as such throughout.
"""
import json, os, re, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np, yaml

NBOOT, SEED = 10000, 20260918
FK = ("F1_own", "F2_blank", "F3_donor", "F4_shifted", "F5_amplified", "F6_binary")
RL = ("R1_own", "R2_blank", "R3_donor")
LAB = {"F1_own": "F1 own map (baseline)", "F2_blank": "F2 blank map",
       "F3_donor": "F3 donor map", "F4_shifted": "F4 shifted own map",
       "F5_amplified": "F5 amplified [vis]", "F6_binary": "F6 binary [vis]",
       "R1_own": "R1 own map", "R2_blank": "R2 blank map", "R3_donor": "R3 donor map"}
PRIOR = dict(C10_score_only=0.755, C11_score_map=0.527, C00_image_only=0.038)


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return 0, 0, 1.0
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(vals, nb=NBOOT, seed=SEED):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    d = v[rng.integers(0, len(v), (nb, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def sig(t):
    return "EXCLUDES zero" if (t[1] > 0 or t[2] < 0) else "includes zero"


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    R = cfg["experiment"]["out_root"]
    od = os.path.join(R, "trufor_conflict")
    F = json.load(open(os.path.join(od, "trufor_score_map_conflict_frozen.json")))
    rows = [json.loads(l) for l in open(os.path.join(od, "verdicts.jsonl"))]
    G = defaultdict(dict)
    for r in rows:
        G[r["sample_id"]][r["condition"]] = r
    ids = sorted(p for p in G if len(G[p]) == 9)
    sup = set(F["suppression_subgroup"]["ids"])

    print("TRUFOR SCORE-MAP CONFLICT INTERVENTION")
    print(f"rows {len(rows)}  complete sources {len(ids)}  bootstrap {NBOOT} "
          f"seed {SEED}")
    print("NOTE: F5 amplified and F6 binary are VISUALIZATION interventions. They are")
    print("not new TruFor outputs and are never evidence of tool improvement.")

    print("\n" + "=" * 92)
    print("1. INTEGRITY")
    pf = sum(1 for r in rows if "json_parse_failed" in (r["notes"] or []))
    print(f"  parse failures       : {pf}/{len(rows)}")
    print(f"  missing verdicts     : {sum(1 for r in rows if not r['final_verdict'])}")
    print(f"  cells                : {dict(Counter(r['condition'] for r in rows))}")
    print(f"  unique coco_id       : {len({G[p]['F1_own']['coco_id'] for p in ids})}")
    one = all(len({G[p][c]["score_shown"] for c in FK}) == 1 for p in ids)
    print(f"  own score identical across all fake conditions: {one}")
    bz = all(G[p][c]["map_mean"] == 0 for p in ids for c in ("F2_blank", "R2_blank"))
    print(f"  blank maps are exactly zero                   : {bz}")
    shok = all(abs(G[p]["F4_shifted"]["map_mean"] - G[p]["F1_own"]["map_mean"]) < 1e-9
               and abs(G[p]["F4_shifted"]["map_active_frac"]
                       - G[p]["F1_own"]["map_active_frac"]) < 1e-9 for p in ids)
    print(f"  shifted preserves mean and active fraction    : {shok}")
    print(f"  all conditions used 2 images                  : "
          f"{all(r['n_images'] == 2 for r in rows)}")
    m = json.load(open(os.path.join(od, "run_meta.json")))
    print(f"  protocol sha1        : {m['protocol_sha1']}")

    fk = lambda p, c: int(G[p][c]["final_verdict"] == "fake")

    print("\n" + "=" * 92)
    print("2. FAKE RECALL BY CONDITION")
    print(f"  {'condition':<24}{'recall':>9}{'95% CI':>18}{'map mean':>11}"
          f"{'active frac':>13}")
    S = {}
    for c in FK:
        o, lo, hi = boot([fk(p, c) for p in ids])
        S[c] = o
        mm = np.mean([G[p][c]["map_mean"] for p in ids])
        af = np.mean([G[p][c]["map_active_frac"] for p in ids])
        print(f"  {LAB[c]:<24}{o:>9.3f} [{lo:.3f},{hi:.3f}]{mm:>11.5f}{af:>13.5f}")
    print(f"\n  reference from the 2x2 round (same prompt for C11/F1):")
    print(f"    score only  (C10) {PRIOR['C10_score_only']:.3f}")
    print(f"    score + map (C11) {PRIOR['C11_score_map']:.3f}   vs F1 {S['F1_own']:.3f} "
          f"-> replication delta {S['F1_own']-PRIOR['C11_score_map']:+.3f}")

    print("\n" + "=" * 92)
    print("3. CONTRASTS vs F1 (effect -> source bootstrap CI -> McNemar p)")
    E = {}
    for c in FK[1:]:
        d = boot([fk(p, c) - fk(p, "F1_own") for p in ids])
        x01, x10, pv = mcnemar([fk(p, "F1_own") for p in ids], [fk(p, c) for p in ids])
        E[c] = d
        tagv = " [vis]" if c in ("F5_amplified", "F6_binary") else ""
        print(f"  {c} - F1{tagv}")
        print(f"    {d[0]:+.3f}  [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}   p={pv:.4g}   "
              f"(F1-only={x01}, {c.split('_')[0]}-only={x10})")

    print("\n" + "=" * 92)
    print("4. REAL CONTROLS (FPR)")
    rl = lambda p, c: int(G[p][c]["final_verdict"] == "fake")
    for c in RL:
        o, lo, hi = boot([rl(p, c) for p in ids])
        print(f"  {LAB[c]:<24}FPR {o:.3f} [{lo:.3f},{hi:.3f}]")
    for c in RL[1:]:
        d = boot([rl(p, c) - rl(p, "R1_own") for p in ids])
        _, _, pv = mcnemar([rl(p, "R1_own") for p in ids], [rl(p, c) for p in ids])
        print(f"  {c} - R1: {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}  p={pv:.4g}")

    print("\n" + "=" * 92)
    print(f"5. SUPPRESSION SUBGROUP RECOVERY (n={len(sup & set(ids))})")
    sids = sorted(sup & set(ids))
    print("  these are cases where score-only said Fake but score+map said Real")
    print(f"  {'condition':<24}{'recovered to Fake':>19}{'rate':>8}{'95% CI':>18}")
    for c in FK:
        v = [fk(p, c) for p in sids]
        o, lo, hi = boot(v)
        print(f"  {LAB[c]:<24}{sum(v):>13}/{len(sids):<5}{o:>8.3f} [{lo:.3f},{hi:.3f}]")
    print("\n  non-suppression fakes for comparison:")
    oth = [p for p in ids if p not in sup]
    for c in FK:
        o, _, _ = boot([fk(p, c) for p in oth])
        print(f"    {LAB[c]:<24}{o:.3f}  (n={len(oth)})")
    print("\n  suppression-case properties vs the rest:")
    for nm, sel in (("suppressed", sids), ("others", oth)):
        tr = np.mean([G[p]["F1_own"]["tamper_ratio"] for p in sel])
        mm = np.mean([G[p]["F1_own"]["map_mean"] for p in sel])
        af = np.mean([G[p]["F1_own"]["map_active_frac"] for p in sel])
        mx = np.mean([G[p]["F1_own"]["map_max"] for p in sel])
        sc = np.mean([G[p]["F1_own"]["score_shown"] for p in sel])
        print(f"    {nm:<12} n={len(sel):<4} tamper {tr:.4f}  map_mean {mm:.5f}  "
              f"active {af:.5f}  map_max {mx:.4f}  score {sc:.4f}")

    print("\n" + "=" * 92)
    print("6. F4 SHIFTED — SPATIAL-CORRESPONDENCE TEST")
    d4 = E["F4_shifted"]
    print(f"  F4 - F1 = {d4[0]:+.3f} [{d4[1]:+.3f},{d4[2]:+.3f}] {sig(d4)}")
    print("  F4 has the SAME value distribution and global intensity as F1; only the")
    print("  alignment with the image is broken. So a null result here means the")
    print("  verdict does not depend on map-image correspondence.")
    d3 = E["F3_donor"]
    print(f"  for comparison F3 - F1 = {d3[0]:+.3f} [{d3[1]:+.3f},{d3[2]:+.3f}] {sig(d3)}")
    print("  (donor differs in texture too, so F4 is the cleaner test)")

    print("\n" + "=" * 92)
    print("7. EXPLANATION AUDIT (deterministic rules; secondary)")
    pats = {k: re.compile("|".join(re.escape(w) for w in v), re.I)
            for k, v in F["explanation_audit"]["patterns"].items()}
    print(f"  {'condition':<24}" + "".join(f"{k[:11]:>13}" for k in pats))
    for c in FK:
        line = f"  {LAB[c]:<24}"
        for k in pats:
            line += f"{np.mean([1 if pats[k].search(G[p][c]['reason'] or '') else 0 for p in ids]):>13.3f}"
        print(line)
    print("\n  among fakes judged REAL, how often the reason says the map shows little:")
    for c in FK:
        wrong = [p for p in ids if not fk(p, c)]
        if not wrong:
            print(f"    {LAB[c]:<24}n=0")
            continue
        v = np.mean([1 if pats["map_shows_little"].search(G[p][c]["reason"] or "")
                     else 0 for p in wrong])
        w = np.mean([1 if pats["region_small"].search(G[p][c]["reason"] or "")
                     else 0 for p in wrong])
        print(f"    {LAB[c]:<24}n={len(wrong):<4} map_shows_little {v:.3f}   "
              f"region_small {w:.3f}")
    print("\n  example reasons per condition (fake targets):")
    for c in FK:
        ex = next((G[p][c]["reason"] for p in ids if not fk(p, c)), None)
        print(f"    {c} (judged Real): {str(ex)[:140]}")

    print("\n" + "=" * 92)
    print("8. MECHANISM JUDGEMENT")
    f1 = S["F1_own"]
    blank_up = E["F2_blank"][1] > 0
    amp_up = E["F5_amplified"][1] > 0
    bin_up = E["F6_binary"][1] > 0
    shift_dn = E["F4_shifted"][2] < 0
    donor_dn = E["F3_donor"][2] < 0
    allclose = all(abs(E[c][0]) < 0.05 for c in FK[1:])
    print(f"  F1 {f1:.3f} | F2 {S['F2_blank']:.3f} | F3 {S['F3_donor']:.3f} | "
          f"F4 {S['F4_shifted']:.3f} | F5 {S['F5_amplified']:.3f} | F6 {S['F6_binary']:.3f}")
    print(f"  score-only reference {PRIOR['C10_score_only']:.3f}")
    print()
    mech = []
    if blank_up and (amp_up or bin_up) and not shift_dn:
        mech.append("A weak-visual-evidence veto")
    if shift_dn or donor_dn:
        mech.append("B spatial-mismatch sensitivity")
    if allclose and f1 < PRIOR["C10_score_only"] - 0.05:
        mech.append("C generic second-image interference")
    if blank_up and (shift_dn or donor_dn):
        mech = ["D mixed: " + " + ".join(
            ["A weak-visual-evidence veto", "B spatial-mismatch sensitivity"])]
    if not mech:
        mech = ["indeterminate — report descriptively"]
    for x in mech:
        print(f"  -> MECHANISM {x}")
    if blank_up:
        print(f"\n  blank > own by {E['F2_blank'][0]:+.3f}: removing the real map's")
        print("  content RAISES recall, so the map itself suppresses the score rather")
        print("  than merely adding a second image.")
    if amp_up or bin_up:
        print("  making the same evidence more visually salient raises recall, which")
        print("  supports the 'small response looks unconvincing' account. This is a")
        print("  property of the MLLM's reading, NOT a TruFor improvement.")
    if not shift_dn:
        print("  breaking spatial alignment while preserving map statistics changes")
        print("  little, so the effect is driven by appearance rather than by")
        print("  map-image correspondence.")

    print(f"\n9. RUNTIME  {m['n_inferences']} inf  {m['minutes']} min  "
          f"mean {m['mean_seconds']}s  peak {m['peak_GB']} GB")


if __name__ == "__main__":
    main()
