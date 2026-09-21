"""Analysis of the framing / region-presence control.

Decomposes the collapse into prompt framing (P1-P0), empty container (P2-P1),
region object (P3-P2), actual spatial content (P4-P3) and generic metadata (P5-P3).
Includes the pre-registered replication checks: P0 ~ 0.837 and P4 ~ 0.141.
"""
import json, os, re, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np, yaml

NBOOT, SEED = 10000, 20260918
P = ("P0_score_only", "P1_framing_only", "P2_empty_regions", "P3_placeholder",
     "P4_true_location", "P5_neutral_meta")
LAB = {"P0_score_only": "P0 score-only prompt", "P1_framing_only": "P1 framing, no block",
       "P2_empty_regions": "P2 empty regions", "P3_placeholder": "P3 placeholder id",
       "P4_true_location": "P4 true location", "P5_neutral_meta": "P5 neutral metadata"}
CHAIN = [("prompt_framing_effect", "P1_framing_only", "P0_score_only"),
         ("empty_container_effect", "P2_empty_regions", "P1_framing_only"),
         ("region_object_effect", "P3_placeholder", "P2_empty_regions"),
         ("actual_spatial_information", "P4_true_location", "P3_placeholder"),
         ("generic_metadata_effect", "P5_neutral_meta", "P3_placeholder")]
SUPP = [("P4 - P0  (expect ~ -0.696)", "P4_true_location", "P0_score_only"),
        ("P5 - P4  (neutral vs real location)", "P5_neutral_meta", "P4_true_location")]
NO_LOC_DATA = ("P1_framing_only", "P2_empty_regions", "P3_placeholder",
               "P5_neutral_meta")


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
    od = os.path.join(R, "trufor_framing")
    F = json.load(open(os.path.join(od, "trufor_structured_framing_frozen.json")))
    rows = [json.loads(l) for l in open(os.path.join(od, "verdicts.jsonl"))]
    G = defaultdict(dict)
    for r in rows:
        G[(r["sample_id"], r["label"])][r["condition"]] = r
    ids = sorted({p for (p, l) in G if len(G[(p, "fake")]) == 6
                  and len(G[(p, "real")]) == 6})
    m = json.load(open(os.path.join(od, "run_meta.json")))
    fk = lambda p, l, c: int(G[(p, l)][c]["final_verdict"] == "fake")

    print("STRUCTURED FRAMING / REGION-PRESENCE CONTROL — 7B")
    print(f"rows {len(rows)}  complete sources {len(ids)}  bootstrap {NBOOT} "
          f"seed {SEED}")
    print(f"{m['n_inferences']} inf  {m['minutes']} min  mean {m['mean_seconds']}s  "
          f"{m.get('started','?')} -> {m.get('finished','?')}")

    print("\n" + "=" * 94)
    print("1. INTEGRITY")
    print(f"  parse failures "
          f"{sum(1 for r in rows if 'json_parse_failed' in r['notes'])}/{len(rows)}"
          f"   bad verdicts {sum(1 for r in rows if 'bad_verdict' in r['notes'])}")
    print(f"  cells {dict(Counter(r['condition'] for r in rows))}")
    print(f"  unique coco_id {len({G[(p,'fake')]['P0_score_only']['coco_id'] for p in ids})}")
    print(f"  all single-image: {all(r['n_images']==1 for r in rows)}")
    print(f"  P0 on score-only template, P1-P5 on structured: "
          f"{all(G[(p,l)]['P0_score_only']['prompt_template']=='score_only' and all(G[(p,l)][c]['prompt_template']=='structured' for c in P[1:]) for p in ids for l in ('fake','real'))}")
    print(f"  P0/P1 payload empty: "
          f"{all(G[(p,l)][c]['payload_chars']==0 for p in ids for l in ('fake','real') for c in ('P0_score_only','P1_framing_only'))}")
    print(f"  score identical across conditions: "
          f"{all(len({G[(p,l)][c]['score_shown'] for c in P})==1 for p in ids for l in ('fake','real'))}")
    print(f"  region count identical in P3/P4/P5 payloads: "
          f"{all(len({len(G[(p,l)][c]['payload']['regions']) for c in ('P3_placeholder','P4_true_location','P5_neutral_meta')})==1 for p in ids for l in ('fake','real'))}")
    print(f"  protocol sha1 {m['protocol_sha1']}")

    print("\n" + "=" * 94)
    print("2. DECISION BY CONDITION")
    print(f"  {'condition':<26}{'recall':>9}{'95% CI':>18}{'FPR':>8}{'95% CI':>18}"
          f"{'spec':>7}{'J':>8}")
    SC = {}
    for c in P:
        rc = boot([fk(p, "fake", c) for p in ids])
        fp = boot([fk(p, "real", c) for p in ids])
        jj = boot([fk(p, "fake", c) - fk(p, "real", c) for p in ids])
        SC[c] = dict(recall=rc, fpr=fp, j=jj)
        print(f"  {LAB[c]:<26}{rc[0]:>9.3f} [{rc[1]:.3f},{rc[2]:.3f}]"
              f"{fp[0]:>8.3f} [{fp[1]:.3f},{fp[2]:.3f}]{1-fp[0]:>7.3f}{jj[0]:>8.3f}")

    print("\n" + "=" * 94)
    print("3. REPLICATION CHECKS (pre-registered)")
    e0, e4 = F["replication_checks"]["P0_expect"], F["replication_checks"]["P4_expect"]
    g0, g4 = SC["P0_score_only"]["recall"], SC["P4_true_location"]["recall"]
    print(f"  P0 expected {e0:.3f}  got {g0[0]:.3f} [{g0[1]:.3f},{g0[2]:.3f}]  "
          f"delta {g0[0]-e0:+.3f}  {'OK' if g0[1]<=e0<=g0[2] else 'OUTSIDE CI'}")
    print(f"  P4 expected {e4:.3f}  got {g4[0]:.3f} [{g4[1]:.3f},{g4[2]:.3f}]  "
          f"delta {g4[0]-e4:+.3f}  {'OK' if g4[1]<=e4<=g4[2] else 'OUTSIDE CI'}")
    print(f"  these are exact prompt+payload replicates of S0/S1; a large deviation")
    print(f"  would indicate a pipeline problem rather than a finding")

    print("\n" + "=" * 94)
    print("4. CAUSAL CHAIN (each step paired within source)")
    CH = {}
    for nm, hi, lo in CHAIN:
        print(f"\n  {nm}   {hi.split('_')[0]} - {lo.split('_')[0]}")
        for w, l in (("fake recall", "fake"), ("real FPR", "real")):
            d = boot([fk(p, l, hi) - fk(p, l, lo) for p in ids])
            _, _, pv = mcnemar([fk(p, l, lo) for p in ids], [fk(p, l, hi) for p in ids])
            if w.startswith("fake"):
                CH[nm] = d
            print(f"    {w:<12} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}  "
                  f"p={pv:.4g}")
        dj = boot([(fk(p, "fake", hi) - fk(p, "real", hi))
                   - (fk(p, "fake", lo) - fk(p, "real", lo)) for p in ids])
        print(f"    {'J':<12} {dj[0]:+.3f} [{dj[1]:+.3f},{dj[2]:+.3f}] {sig(dj)}")

    print("\n" + "=" * 94)
    print("5. SUPPLEMENTARY CONTRASTS")
    for nm, hi, lo in SUPP:
        d = boot([fk(p, "fake", hi) - fk(p, "fake", lo) for p in ids])
        _, _, pv = mcnemar([fk(p, "fake", lo) for p in ids],
                           [fk(p, "fake", hi) for p in ids])
        print(f"  {nm:<40} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}  p={pv:.4g}")

    print("\n" + "=" * 94)
    print("6. REASON AUDIT")
    pats = {k: re.compile("|".join(re.escape(w) for w in v), re.I)
            for k, v in F["reason_audit"]["patterns"].items()}
    for lb in ("fake", "real"):
        print(f"\n  {lb} targets")
        print(f"  {'condition':<26}" + "".join(f"{k[:13]:>15}" for k in pats))
        for c in P:
            line = f"  {LAB[c]:<26}"
            for k in pats:
                line += f"{np.mean([1 if pats[k].search(G[(p,lb)][c]['reason'] or '') else 0 for p in ids]):>15.3f}"
            print(line)

    print("\n" + "=" * 94)
    print("7. LOCALITY WITHOUT LOCALITY DATA")
    print("  P1/P2/P3/P5 carry NO location, area or probability value. Locality")
    print("  language there is activated by framing/schema, not derived from evidence.")
    print(f"  {'condition':<26}{'fake':>9}{'95% CI':>18}{'real':>9}{'has loc data':>14}")
    for c in P:
        f_ = boot([1 if pats["locality"].search(G[(p, 'fake')][c]["reason"] or "")
                   else 0 for p in ids])
        r_ = boot([1 if pats["locality"].search(G[(p, 'real')][c]["reason"] or "")
                   else 0 for p in ids])
        tag = "no" if c in NO_LOC_DATA else ("n/a" if c == "P0_score_only" else "YES")
        print(f"  {LAB[c]:<26}{f_[0]:>9.3f} [{f_[1]:.3f},{f_[2]:.3f}]{r_[0]:>9.3f}"
              f"{tag:>14}")
    d = boot([(1 if pats["locality"].search(G[(p, 'fake')]["P1_framing_only"]["reason"] or "") else 0)
              - (1 if pats["locality"].search(G[(p, 'fake')]["P0_score_only"]["reason"] or "") else 0)
              for p in ids])
    print(f"\n  P1 - P0 locality language (fake): {d[0]:+.3f} "
          f"[{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}")

    print("\n" + "=" * 94)
    print("8. ACKNOWLEDGED MANIPULATION BUT REAL")
    print(f"  {F['reason_audit']['key_metrics']['acknowledged_manipulation_but_real']}")
    print(f"\n  {'condition':<26}{'fake':>9}{'95% CI':>18}{'real':>9}"
          f"{'among fake->Real':>18}")
    for c in P:
        def bad(p, l):
            r = G[(p, l)][c]
            t = r["reason"] or ""
            return (r["final_verdict"] == "real"
                    and pats["manipulation_acknowledged"].search(t)
                    and (pats["locality"].search(t) or pats["evidence_weak"].search(t)))
        f_ = boot([1 if bad(p, "fake") else 0 for p in ids])
        r_ = boot([1 if bad(p, "real") else 0 for p in ids])
        wrong = [p for p in ids if not fk(p, "fake", c)]
        sub = (np.mean([1 if bad(p, "fake") else 0 for p in wrong])
               if wrong else float("nan"))
        print(f"  {LAB[c]:<26}{f_[0]:>9.3f} [{f_[1]:.3f},{f_[2]:.3f}]{r_[0]:>9.3f}"
              f"{sub:>13.3f} (n={len(wrong)})")

    print("\n" + "=" * 94)
    print("9. SCHEMA ARTIFACT (recorded, NOT used for judgement)")
    for c in P:
        a9 = np.mean([G[(p, l)][c]["all9_echo"] for p in ids for l in ("fake", "real")])
        em = np.mean([1 if not G[(p, l)][c]["referenced_regions"] else 0
                      for p in ids for l in ("fake", "real")])
        print(f"  {LAB[c]:<26} all-9 echo {a9:.3f}   empty {em:.3f}")

    print("\n" + "=" * 94)
    print("10. CASE JUDGEMENT")
    r = {c: SC[c]["recall"][0] for c in P}
    print("  recall  " + "  ".join(f"{c.split('_')[0]} {r[c]:.3f}" for c in P))
    f1, f2, f3, f4, f5 = (CH["prompt_framing_effect"], CH["empty_container_effect"],
                          CH["region_object_effect"], CH["actual_spatial_information"],
                          CH["generic_metadata_effect"])
    for nm, d in (("P1-P0 framing", f1), ("P2-P1 container", f2),
                  ("P3-P2 region obj", f3), ("P4-P3 spatial", f4),
                  ("P5-P3 metadata", f5)):
        print(f"  {nm:<20} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}")
    print()
    neg = lambda d: d[2] < 0
    flat = lambda d: not (d[1] > 0 or d[2] < 0)
    out = []
    if neg(f1) and abs(f1[0]) > 0.15:
        out.append("F1 PROMPT FRAMING drives the collapse: the structured instruction "
                   "alone, with no payload at all, already shifts the decision policy")
    if flat(f1) and (neg(f2) or neg(f3)):
        out.append("F2 REGION PRESENCE drives the collapse: an explicit region-level "
                   "representation changes policy without meaningful values")
    if flat(f1) and flat(f2) and flat(f3) and neg(f4):
        out.append("F3 ACTUAL FORENSIC FIELDS drive the collapse")
    if neg(f3) and neg(f5) and abs(abs(f3[0]) - abs(f5[0])) < 0.10:
        out.append("F4 GENERIC STRUCTURED-TOKEN OVERLOAD: placeholder and neutral "
                   "metadata fall by a similar amount")
    if not out:
        out.append("no single pre-registered case matched cleanly — report "
                   "descriptively")
    for o in out:
        print(f"  -> {o}")
    tot = r["P0_score_only"] - r["P4_true_location"]
    if tot:
        print(f"\n  share of the total P0->P4 drop ({tot:+.3f}) attributable to each "
              f"step:")
        for nm, d in (("framing P1-P0", f1), ("container P2-P1", f2),
                      ("region object P3-P2", f3), ("spatial content P4-P3", f4)):
            print(f"    {nm:<24} {100*(-d[0])/tot:+6.1f}%")


if __name__ == "__main__":
    main()
