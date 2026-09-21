"""Analysis of the task-semantics / verdict-decomposition experiment.

Reports the four decision contrasts, presence-level accuracy for D2/D3, the
presence-verdict inconsistencies measured from structured fields (no regex), the
localized-exemption quantity, and the reason audit. Also writes the two extra JSON
artifacts required by the protocol.
"""
import json, os, re, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np, yaml

NBOOT, SEED = 10000, 20260918
D = ("D0_baseline", "D1_binary_rule", "D2_decomposition", "D3_presence_first")
LAB = {"D0_baseline": "D0 baseline", "D1_binary_rule": "D1 + binary rule",
       "D2_decomposition": "D2 + decomposition", "D3_presence_first": "D3 presence-first"}
DEC = ("D2_decomposition", "D3_presence_first")
CONTRASTS = [("rule_clarification", "D1_binary_rule", "D0_baseline"),
             ("decomposition", "D2_decomposition", "D0_baseline"),
             ("presence_first", "D3_presence_first", "D0_baseline"),
             ("ordered_benefit", "D3_presence_first", "D2_decomposition")]


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
    od = os.path.join(R, "trufor_semantics")
    F = json.load(open(os.path.join(od, "trufor_task_semantics_frozen.json")))
    rows = [json.loads(l) for l in open(os.path.join(od, "verdicts.jsonl"))]
    G = defaultdict(dict)
    for r in rows:
        G[(r["sample_id"], r["label"])][r["condition"]] = r
    ids = sorted({p for (p, l) in G if len(G[(p, "fake")]) == 4
                  and len(G[(p, "real")]) == 4})
    m = json.load(open(os.path.join(od, "run_meta.json")))
    fk = lambda p, l, c: int(G[(p, l)][c]["final_verdict"] == "fake")

    print("TASK-SEMANTICS / VERDICT DECOMPOSITION — 7B")
    print(f"rows {len(rows)}  complete sources {len(ids)}  bootstrap {NBOOT} "
          f"seed {SEED}")
    print(f"{m['n_inferences']} inf  {m['minutes']} min  mean {m['mean_seconds']}s  "
          f"{m.get('started')} -> {m.get('finished')}")

    print("\n" + "=" * 94)
    print("1. INTEGRITY")
    nt = Counter(x for r in rows for x in r["notes"])
    print(f"  parse failures "
          f"{sum(1 for r in rows if 'json_parse_failed' in r['notes'])}/{len(rows)}"
          f"   bad verdicts {sum(1 for r in rows if 'bad_verdict' in r['notes'])}")
    print(f"  note counts {dict(nt) or 'none'}")
    print(f"  cells {dict(Counter(r['condition'] for r in rows))}")
    print(f"  unique coco_id {len({G[(p,'fake')]['D0_baseline']['coco_id'] for p in ids})}")
    ndec = [r for r in rows if r["decomposed"]]
    print(f"  D2/D3 manipulation_present parsed "
          f"{sum(1 for r in ndec if r['manipulation_present'] is not None)}/{len(ndec)}")
    print(f"  D2/D3 manipulation_extent parsed "
          f"{sum(1 for r in ndec if r['manipulation_extent'] is not None)}/{len(ndec)}")
    print(f"  D0/D1 carry no presence field: "
          f"{all(r['manipulation_present'] is None for r in rows if not r['decomposed'])}")
    print(f"  evidence payload identical across conditions: "
          f"{all(len({tuple(G[(p,l)][c]['evidence_locations']) for c in D})==1 for p in ids for l in ('fake','real'))}")
    print(f"  protocol sha1 {m['protocol_sha1']}")

    print("\n" + "=" * 94)
    print("2. DECISION BY CONDITION")
    print(f"  {'condition':<24}{'recall':>9}{'95% CI':>18}{'FPR':>8}{'95% CI':>18}"
          f"{'spec':>7}{'J':>8}{'J 95% CI':>18}")
    SC = {}
    for c in D:
        rc = boot([fk(p, "fake", c) for p in ids])
        fp = boot([fk(p, "real", c) for p in ids])
        jj = boot([fk(p, "fake", c) - fk(p, "real", c) for p in ids])
        SC[c] = dict(recall=rc, fpr=fp, j=jj, spec=1 - fp[0])
        print(f"  {LAB[c]:<24}{rc[0]:>9.3f} [{rc[1]:.3f},{rc[2]:.3f}]"
              f"{fp[0]:>8.3f} [{fp[1]:.3f},{fp[2]:.3f}]{1-fp[0]:>7.3f}"
              f"{jj[0]:>8.3f} [{jj[1]:+.3f},{jj[2]:+.3f}]")
    print(f"\n  reference: score-only prompt recall 0.837 / FPR 0.011 / J 0.826")

    print("\n" + "=" * 94)
    print("3. REPLICATION CHECK")
    e = F["replication_check"]["D0_expect_recall"]
    g = SC["D0_baseline"]["recall"]
    print(f"  D0 expected ~{e:.3f} (field round S5, same prompt bytes)")
    print(f"  D0 observed  {g[0]:.3f} [{g[1]:.3f},{g[2]:.3f}]   delta {g[0]-e:+.3f}")
    print(f"  {'OK — within CI' if g[1] <= e <= g[2] else 'OUTSIDE CI — check pipeline before interpreting'}")

    print("\n" + "=" * 94)
    print("4. DECISION CONTRASTS")
    CH = {}
    for nm, hi, lo in CONTRASTS:
        print(f"\n  {nm}   {hi.split('_')[0]} - {lo.split('_')[0]}")
        for w, l in (("fake recall", "fake"), ("real FPR", "real")):
            d = boot([fk(p, l, hi) - fk(p, l, lo) for p in ids])
            _, _, pv = mcnemar([fk(p, l, lo) for p in ids], [fk(p, l, hi) for p in ids])
            CH[(nm, l)] = d
            print(f"    {w:<12} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}  "
                  f"p={pv:.4g}")
        dj = boot([(fk(p, "fake", hi) - fk(p, "real", hi))
                   - (fk(p, "fake", lo) - fk(p, "real", lo)) for p in ids])
        CH[(nm, "J")] = dj
        print(f"    {'J':<12} {dj[0]:+.3f} [{dj[1]:+.3f},{dj[2]:+.3f}] {sig(dj)}")

    print("\n" + "=" * 94)
    print("5. PRESENCE-LEVEL METRICS (D2/D3)")
    pres = lambda p, l, c: G[(p, l)][c]["manipulation_present"]
    PRES = {}
    print(f"  {'condition':<24}{'TPR':>9}{'95% CI':>18}{'FPR':>9}{'95% CI':>18}{'J':>8}")
    for c in DEC:
        tp = boot([1 if pres(p, "fake", c) else 0 for p in ids
                   if pres(p, "fake", c) is not None])
        fp = boot([1 if pres(p, "real", c) else 0 for p in ids
                   if pres(p, "real", c) is not None])
        PRES[c] = dict(tpr=tp, fpr=fp, j=tp[0] - fp[0])
        print(f"  {LAB[c]:<24}{tp[0]:>9.3f} [{tp[1]:.3f},{tp[2]:.3f}]"
              f"{fp[0]:>9.3f} [{fp[1]:.3f},{fp[2]:.3f}]{tp[0]-fp[0]:>8.3f}")
    print(f"\n  compare with the same condition's VERDICT J:")
    for c in DEC:
        print(f"    {LAB[c]:<24} presence J {PRES[c]['j']:+.3f}   "
              f"verdict J {SC[c]['j'][0]:+.3f}   gap "
              f"{PRES[c]['j']-SC[c]['j'][0]:+.3f}")
    print(f"  a large positive gap localises the failure to final arbitration")

    print("\n" + "=" * 94)
    print("6. PRESENCE-VERDICT INCONSISTENCY (structured fields, no regex)")
    INC = {}
    for c in DEC:
        print(f"\n  {LAB[c]}")
        row = {}
        for l in ("fake", "real"):
            use = [p for p in ids if pres(p, l, c) is not None]
            a = boot([1 if (pres(p, l, c) is True and not fk(p, l, c)) else 0
                      for p in use])
            b = boot([1 if (pres(p, l, c) is False and fk(p, l, c)) else 0
                      for p in use])
            row[l] = dict(A=a, B=b, n=len(use))
            print(f"    [{l}] A present=true & verdict=real  {a[0]:.3f} "
                  f"[{a[1]:.3f},{a[2]:.3f}]")
            print(f"    [{l}] B present=false & verdict=fake {b[0]:.3f} "
                  f"[{b[1]:.3f},{b[2]:.3f}]   n={len(use)}")
        # among present=true only
        for l in ("fake", "real"):
            sub = [p for p in ids if pres(p, l, c) is True]
            if sub:
                v = np.mean([1 if not fk(p, l, c) else 0 for p in sub])
                print(f"    [{l}] P(verdict=real | present=true) {v:.3f} "
                      f"(n={len(sub)})")
                row[l]["P_real_given_present"] = float(v)
                row[l]["n_present_true"] = len(sub)
        INC[c] = row
    print(f"\n  last round's regex proxy (acknowledged_manipulation_but_real):")
    print(f"    P0 0.000  ->  P2 0.957. This round measures it directly.")

    print("\n" + "=" * 94)
    print("7. EXTENT ANALYSIS (D2/D3)")
    EXT = {}
    ext = lambda p, l, c: G[(p, l)][c]["manipulation_extent"]
    for c in DEC:
        print(f"\n  {LAB[c]}")
        row = {}
        for l in ("fake", "real"):
            cnt = Counter(ext(p, l, c) for p in ids)
            tot = sum(cnt.values())
            row[l] = {k: cnt.get(k, 0) / tot for k in ("none", "localized",
                                                       "widespread")}
            print(f"    [{l}] " + "  ".join(
                f"{k} {cnt.get(k,0)}/{tot} ({cnt.get(k,0)/tot:.3f})"
                for k in ("none", "localized", "widespread")))
        for l in ("fake", "real"):
            sub = [p for p in ids if pres(p, l, c) is True
                   and ext(p, l, c) == "localized"]
            if sub:
                v = boot([1 if not fk(p, l, c) else 0 for p in sub])
                print(f"    [{l}] P(verdict=real | present=true, extent=localized) "
                      f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]  n={len(sub)}")
                row[f"{l}_P_real_given_localized"] = dict(
                    value=v[0], ci=[v[1], v[2]], n=len(sub))
        EXT[c] = row
    print(f"\n  reading: if this is high in D2 but low in D3, the presence-first order")
    print(f"  removed the 'localized = authenticity exemption' shortcut")

    print("\n" + "=" * 94)
    print("8. REASON AUDIT")
    pats = {k: re.compile("|".join(re.escape(w) for w in v), re.I)
            for k, v in F["reason_audit"]["patterns"].items()}
    for l in ("fake", "real"):
        print(f"\n  {l} targets")
        print(f"  {'condition':<24}" + "".join(f"{k[:12]:>14}" for k in pats))
        for c in D:
            line = f"  {LAB[c]:<24}"
            for k in pats:
                line += f"{np.mean([1 if pats[k].search(G[(p,l)][c]['reason'] or '') else 0 for p in ids]):>14.3f}"
            print(line)
    print(f"\n  key columns: local_exemption (localized => still real) and")
    print(f"  rule_consistent (explicit adherence to the binary rule)")
    for c in D:
        le = boot([1 if pats["local_exemption"].search(
            G[(p, 'fake')][c]["reason"] or "") else 0 for p in ids])
        rc = boot([1 if pats["rule_consistent"].search(
            G[(p, 'fake')][c]["reason"] or "") else 0 for p in ids])
        print(f"    {LAB[c]:<24} local_exemption {le[0]:.3f} "
              f"[{le[1]:.3f},{le[2]:.3f}]   rule_consistent {rc[0]:.3f} "
              f"[{rc[1]:.3f},{rc[2]:.3f}]")

    print("\n" + "=" * 94)
    print("9. FAITHFULNESS (secondary — must not regress)")
    for c in D:
        inter, exact, a9 = [], [], []
        for p in ids:
            for l in ("fake", "real"):
                r = G[(p, l)][c]
                ev = set(r["evidence_locations"] or [])
                rf = set(r["referenced_regions"])
                a9.append(r["all9_echo"])
                if r["all9_echo"] or not ev or not rf:
                    continue
                inter.append(1 if rf & ev else 0)
                exact.append(1 if rf == ev else 0)
        print(f"  {LAB[c]:<24} agreement {np.mean(inter):.3f}  exact "
              f"{np.mean(exact):.3f}  all-9 echo {np.mean(a9):.3f}  n={len(inter)}")

    print("\n" + "=" * 94)
    print("10. CASE JUDGEMENT")
    r0, r1 = SC["D0_baseline"]["recall"][0], SC["D1_binary_rule"]["recall"][0]
    r2, r3 = SC["D2_decomposition"]["recall"][0], SC["D3_presence_first"]["recall"][0]
    print("  recall  " + "  ".join(f"{c.split('_')[0]} {SC[c]['recall'][0]:.3f}"
                                   for c in D))
    print("  J       " + "  ".join(f"{c.split('_')[0]} {SC[c]['j'][0]:+.3f}"
                                   for c in D))
    for nm, _, _ in CONTRASTS:
        print(f"  {nm:<20} recall {CH[(nm,'fake')][0]:+.3f} "
              f"{sig(CH[(nm,'fake')])}   FPR {CH[(nm,'real')][0]:+.3f}   "
              f"J {CH[(nm,'J')][0]:+.3f} {sig(CH[(nm,'J')])}")
    print()
    pos = lambda d: d[1] > 0
    flat = lambda d: not (d[1] > 0 or d[2] < 0)
    out = []
    d1 = CH[("rule_clarification", "fake")]
    dj1 = CH[("rule_clarification", "J")]
    ord_b = CH[("ordered_benefit", "fake")]
    if pos(d1) and d1[0] > 0.2:
        out.append("T1 TASK-SEMANTICS MISUNDERSTANDING: stating the binary rule alone "
                   "restores a large part of decision utility")
    gaps = [PRES[c]["j"] - SC[c]["j"][0] for c in DEC]
    if max(gaps) > 0.2:
        out.append("T2 ARBITRATION-LAYER FAILURE: presence discrimination clearly "
                   "exceeds verdict discrimination, so the loss occurs when the "
                   "verdict is formed")
    if pos(ord_b):
        out.append("T3 DECOMPOSITION FIXES ARBITRATION: presence-first ordering beats "
                   "plain field decomposition")
    if all(SC[c]["recall"][0] < 0.35 for c in D[1:]):
        out.append("T4 FRAMING FAILURE PERSISTS: none of the interventions restores "
                   "recall, so the policy shift is deeper than a task-definition "
                   "misunderstanding")
    if not out:
        out.append("no single pre-registered case matched cleanly — report "
                   "descriptively")
    for o in out:
        print(f"  -> {o}")

    json.dump(dict(presence=PRES, extent=EXT,
                   verdict=dict((c, dict(recall=SC[c]["recall"],
                                         fpr=SC[c]["fpr"], j=SC[c]["j"]))
                                for c in D)),
              open(os.path.join(od, "presence_extent_analysis.json"), "w"), indent=1)
    json.dump(dict(inconsistency=INC,
                   definition=F["inconsistency"],
                   note="A is the primary metric; measured from structured fields, "
                        "not regex"),
              open(os.path.join(od, "inconsistency_analysis.json"), "w"), indent=1)
    print(f"\n  wrote presence_extent_analysis.json and inconsistency_analysis.json")


if __name__ == "__main__":
    main()
