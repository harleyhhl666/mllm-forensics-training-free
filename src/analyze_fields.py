"""Analysis of the structured-field ablation.

Decides between area-driven collapse, generic structured overload, probability
conflict and interaction-only using the five primary contrasts plus the three
supplementary ones. Also reports the reason audit and the
acknowledged_manipulation_but_real inconsistency rate.
"""
import json, os, re, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np, yaml

NBOOT, SEED = 10000, 20260918
S = ("S0_score_only", "S1_location", "S2_area", "S3_probability", "S4_loc_area",
     "S5_full")
LAB = {"S0_score_only": "S0 score only", "S1_location": "S1 + location",
       "S2_area": "S2 + area", "S3_probability": "S3 + probability",
       "S4_loc_area": "S4 + location & area", "S5_full": "S5 full structured"}
PRIMARY = [("location_effect", "S1_location"), ("area_effect", "S2_area"),
           ("probability_effect", "S3_probability"),
           ("location_plus_area", "S4_loc_area"), ("full_penalty", "S5_full")]
SUPP = [("S4-S1  adding area on top of location", "S4_loc_area", "S1_location"),
        ("S4-S2  adding location on top of area", "S4_loc_area", "S2_area"),
        ("S5-S4  adding probability and centroid", "S5_full", "S4_loc_area")]


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return 0, 0, 1.0
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(v, seed=SEED):
    a = np.asarray([x for x in v if x is not None], float)
    if not len(a):
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    d = a[rng.integers(0, len(a), (NBOOT, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def sig(t):
    return "EXCLUDES zero" if (t[1] > 0 or t[2] < 0) else "includes zero"


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    R = cfg["experiment"]["out_root"]
    od = os.path.join(R, "trufor_fields")
    F = json.load(open(os.path.join(
        od, "trufor_structured_field_ablation_frozen.json")))
    rows = [json.loads(l) for l in open(os.path.join(od, "verdicts.jsonl"))]
    G = defaultdict(dict)
    for r in rows:
        G[(r["sample_id"], r["label"])][r["condition"]] = r
    ids = sorted({p for (p, l) in G if len(G[(p, "fake")]) == 6
                  and len(G[(p, "real")]) == 6})
    m = json.load(open(os.path.join(od, "run_meta.json")))
    fk = lambda p, l, c: int(G[(p, l)][c]["final_verdict"] == "fake")

    print("STRUCTURED FIELD ABLATION — 7B")
    print(f"rows {len(rows)}  complete sources {len(ids)}  bootstrap {NBOOT} "
          f"seed {SEED}")
    print(f"{m['n_inferences']} inf  {m['minutes']} min  mean {m['mean_seconds']}s")

    print("\n" + "=" * 94)
    print("1. INTEGRITY")
    print(f"  parse failures {sum(1 for r in rows if 'json_parse_failed' in r['notes'])}"
          f"/{len(rows)}   bad verdicts "
          f"{sum(1 for r in rows if 'bad_verdict' in r['notes'])}")
    print(f"  cells {dict(Counter(r['condition'] for r in rows))}")
    print(f"  unique coco_id {len({G[(p,'fake')]['S0_score_only']['coco_id'] for p in ids})}")
    print(f"  all single-image: {all(r['n_images']==1 for r in rows)}")
    ok = True
    for p in ids:
        for l in ("fake", "real"):
            for c in S[1:]:
                st = G[(p, l)][c]["structured"]
                for r in st["regions"]:
                    if set(r) != set(F["conditions"][c]["fields"]):
                        ok = False
    print(f"  key sets exactly match declared fields: {ok}")
    print(f"  score identical across conditions: "
          f"{all(len({G[(p,l)][c]['score_shown'] for c in S})==1 for p in ids for l in ('fake','real'))}")
    print(f"  region count identical across structured conditions: "
          f"{all(len({len(G[(p,l)][c]['structured']['regions']) for c in S[1:]})==1 for p in ids for l in ('fake','real'))}")
    print(f"  protocol sha1 {m['protocol_sha1']}")

    print("\n" + "=" * 94)
    print("2. DECISION BY CONDITION")
    print(f"  {'condition':<24}{'recall':>9}{'95% CI':>18}{'FPR':>8}{'95% CI':>18}"
          f"{'spec':>7}{'J':>8}")
    SC = {}
    for c in S:
        rc = boot([fk(p, "fake", c) for p in ids])
        fp = boot([fk(p, "real", c) for p in ids])
        jj = boot([fk(p, "fake", c) - fk(p, "real", c) for p in ids])
        SC[c] = dict(recall=rc, fpr=fp, j=jj)
        print(f"  {LAB[c]:<24}{rc[0]:>9.3f} [{rc[1]:.3f},{rc[2]:.3f}]"
              f"{fp[0]:>8.3f} [{fp[1]:.3f},{fp[2]:.3f}]{1-fp[0]:>7.3f}{jj[0]:>8.3f}")
    print(f"\n  descriptive background (different prompt, NOT a replication):")
    print(f"    previous round E0 0.837 / E2 0.174")

    print("\n" + "=" * 94)
    print("3. PRIMARY CONTRASTS vs S0")
    PC = {}
    for nm, c in PRIMARY:
        print(f"\n  {nm}   {c.split('_')[0]} - S0")
        for w, l in (("fake recall", "fake"), ("real FPR", "real")):
            d = boot([fk(p, l, c) - fk(p, l, "S0_score_only") for p in ids])
            _, _, pv = mcnemar([fk(p, l, "S0_score_only") for p in ids],
                               [fk(p, l, c) for p in ids])
            if w.startswith("fake"):
                PC[c] = d
            print(f"    {w:<12} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}  "
                  f"p={pv:.4g}")
        dj = boot([(fk(p, "fake", c) - fk(p, "real", c))
                   - (fk(p, "fake", "S0_score_only") - fk(p, "real", "S0_score_only"))
                   for p in ids])
        print(f"    {'J':<12} {dj[0]:+.3f} [{dj[1]:+.3f},{dj[2]:+.3f}] {sig(dj)}")

    print("\n" + "=" * 94)
    print("4. SUPPLEMENTARY CONTRASTS (field addition, holding the rest fixed)")
    SU = {}
    for nm, hi, lo in SUPP:
        d = boot([fk(p, "fake", hi) - fk(p, "fake", lo) for p in ids])
        _, _, pv = mcnemar([fk(p, "fake", lo) for p in ids],
                           [fk(p, "fake", hi) for p in ids])
        SU[nm] = d
        print(f"  {nm:<42} recall {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}"
              f"  p={pv:.4g}")

    print("\n" + "=" * 94)
    print("5. REASON AUDIT (deterministic rules)")
    pats = {k: re.compile("|".join(re.escape(w) for w in v), re.I)
            for k, v in F["reason_audit"]["patterns"].items()}
    print(f"  fake targets")
    print(f"  {'condition':<24}" + "".join(f"{k[:13]:>15}" for k in pats))
    for c in S:
        line = f"  {LAB[c]:<24}"
        for k in pats:
            line += f"{np.mean([1 if pats[k].search(G[(p,'fake')][c]['reason'] or '') else 0 for p in ids]):>15.3f}"
        print(line)
    print(f"\n  real targets")
    print(f"  {'condition':<24}" + "".join(f"{k[:13]:>15}" for k in pats))
    for c in S:
        line = f"  {LAB[c]:<24}"
        for k in pats:
            line += f"{np.mean([1 if pats[k].search(G[(p,'real')][c]['reason'] or '') else 0 for p in ids]):>15.3f}"
        print(line)

    print("\n" + "=" * 94)
    print("6. ACKNOWLEDGED MANIPULATION BUT REAL (forensic inconsistency)")
    print(f"  {F['reason_audit']['key_metric']['definition']}")
    print(f"\n  {'condition':<24}{'fake':>10}{'95% CI':>18}{'real':>10}")
    AMR = {}
    for c in S:
        vals = {}
        for l in ("fake", "real"):
            v = []
            for p in ids:
                r = G[(p, l)][c]
                txt = r["reason"] or ""
                bad = (r["final_verdict"] == "real"
                       and pats["manipulation_acknowledged"].search(txt)
                       and (pats["area_small"].search(txt)
                            or pats["area_limited"].search(txt)
                            or pats["evidence_weak"].search(txt)))
                v.append(1 if bad else 0)
            vals[l] = boot(v)
        AMR[c] = vals
        print(f"  {LAB[c]:<24}{vals['fake'][0]:>10.3f} "
              f"[{vals['fake'][1]:.3f},{vals['fake'][2]:.3f}]{vals['real'][0]:>10.3f}")
    print("\n  among fake targets judged Real, share showing the inconsistency:")
    for c in S:
        wrong = [p for p in ids if not fk(p, "fake", c)]
        if not wrong:
            print(f"    {LAB[c]:<24}n=0")
            continue
        v = np.mean([1 if (pats["manipulation_acknowledged"].search(
            G[(p, 'fake')][c]["reason"] or "")
            and (pats["area_small"].search(G[(p, 'fake')][c]["reason"] or "")
                 or pats["area_limited"].search(G[(p, 'fake')][c]["reason"] or "")
                 or pats["evidence_weak"].search(G[(p, 'fake')][c]["reason"] or "")))
            else 0 for p in wrong])
        print(f"    {LAB[c]:<24}n={len(wrong):<5}{v:.3f}")

    print("\n" + "=" * 94)
    print("7. AREA DOSE-RESPONSE (descriptive, S2 only)")
    ar = np.array([max(G[(p, "fake")]["S2_area"]["evidence_areas"]) for p in ids])
    qs = np.percentile(ar, [25, 50, 75])
    bins = np.digitize(ar, qs)
    print(f"  largest region area_ratio quartile cuts {np.round(qs,4)}")
    print(f"  {'quartile':<12}{'n':>5}{'S0 recall':>12}{'S2 recall':>12}{'delta':>9}")
    for q in range(4):
        sel = [p for p, b in zip(ids, bins) if b == q]
        if not sel:
            continue
        r0 = np.mean([fk(p, "fake", "S0_score_only") for p in sel])
        r2 = np.mean([fk(p, "fake", "S2_area") for p in sel])
        print(f"  Q{q+1:<11}{len(sel):>5}{r0:>12.3f}{r2:>12.3f}{r2-r0:>+9.3f}")

    print("\n" + "=" * 94)
    print("8. SECONDARY FAITHFULNESS (location-bearing conditions only)")
    for c in ("S1_location", "S4_loc_area", "S5_full"):
        inter, exact, a9 = [], [], []
        for p in ids:
            for l in ("fake", "real"):
                r = G[(p, l)][c]
                ev = set(r["evidence_locations"] or [])
                rf = set(r["referenced_regions"])
                a9.append(1 if len(rf) == 9 else 0)
                if len(rf) == 9 or not ev or not rf:
                    continue
                inter.append(1 if rf & ev else 0)
                exact.append(1 if rf == ev else 0)
        print(f"  {LAB[c]:<24} agreement {np.mean(inter):.3f}  exact "
              f"{np.mean(exact):.3f}  all-9 echo {np.mean(a9):.3f}")

    print("\n" + "=" * 94)
    print("9. PATTERN JUDGEMENT")
    s0 = SC["S0_score_only"]["recall"][0]
    d1, d2, d3 = PC["S1_location"], PC["S2_area"], PC["S3_probability"]
    d5 = PC["S5_full"]
    s41 = SU["S4-S1  adding area on top of location"]
    s42 = SU["S4-S2  adding location on top of area"]
    print(f"  recall  S0 {s0:.3f}  S1 {SC['S1_location']['recall'][0]:.3f}  "
          f"S2 {SC['S2_area']['recall'][0]:.3f}  S3 {SC['S3_probability']['recall'][0]:.3f}"
          f"  S4 {SC['S4_loc_area']['recall'][0]:.3f}  S5 {SC['S5_full']['recall'][0]:.3f}")
    print(f"  S1-S0 {d1[0]:+.3f} {sig(d1)}")
    print(f"  S2-S0 {d2[0]:+.3f} {sig(d2)}")
    print(f"  S3-S0 {d3[0]:+.3f} {sig(d3)}")
    print(f"  S4-S1 {s41[0]:+.3f} {sig(s41)}   S4-S2 {s42[0]:+.3f} {sig(s42)}")
    print()
    neg = lambda d: d[2] < 0
    flat = lambda d: not (d[1] > 0 or d[2] < 0)
    out = []
    if neg(d2) and flat(d1) and neg(s41):
        out.append("AREA-DRIVEN COLLAPSE: area alone hurts, location alone does not, "
                   "and adding area on top of location hurts further")
    if neg(d1) and neg(d2) and neg(d3) and \
            max(abs(d1[0]), abs(d2[0]), abs(d3[0])) - \
            min(abs(d1[0]), abs(d2[0]), abs(d3[0])) < 0.15:
        out.append("GENERIC STRUCTURED OVERLOAD: all single fields fall by a similar "
                   "amount")
    if neg(d3) and abs(d3[0]) > abs(d1[0]) and abs(d3[0]) > abs(d2[0]):
        out.append("PROBABILITY CONFLICT dominates among single fields")
    if flat(d1) and flat(d2) and flat(d3) and neg(d5):
        out.append("INTERACTION-ONLY: single fields are harmless, the combination is "
                   "not")
    if not out:
        out.append("no single pre-registered pattern matched cleanly — report "
                   "descriptively")
    for o in out:
        print(f"  -> {o}")


if __name__ == "__main__":
    main()
