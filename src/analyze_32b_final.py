"""Analysis of the 32B final capacity replication.

Primary output is Delta-effect = effect_32B - effect_7B, paired per source, with
bootstrap CI. Absolute accuracy comparison between models is deliberately secondary:
the question is whether capacity changes the MECHANISM.
"""
import json, os, re, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np, yaml

NBOOT, SEED = 10000, 20260918
B = ("B0_score_only", "B1_score_rule", "B2_structured", "B3_structured_rule")
A = {"B0_score_only": "A0", "B1_score_rule": "A1", "B2_structured": "A2",
     "B3_structured_rule": "A3"}
LAB = {"B0_score_only": "B0 score only", "B1_score_rule": "B1 score + rule",
       "B2_structured": "B2 structured", "B3_structured_rule": "B3 struct + rule"}
EFF = [("rule_without_structured", "B1_score_rule", "B0_score_only"),
       ("rule_with_structured", "B3_structured_rule", "B2_structured"),
       ("structured_without_rule", "B2_structured", "B0_score_only"),
       ("structured_with_rule", "B3_structured_rule", "B1_score_rule")]
MIRROR = {"top-left": "top-right", "top-center": "top-center",
          "top-right": "top-left", "center-left": "center-right",
          "center": "center", "center-right": "center-left",
          "bottom-left": "bottom-right", "bottom-center": "bottom-center",
          "bottom-right": "bottom-left"}


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return 0, 0, 1.0
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot_stat(fn, ids, seed=SEED, nb=NBOOT):
    rng = np.random.default_rng(seed)
    arr = np.asarray(ids, dtype=object)
    idx = rng.integers(0, len(ids), (nb, len(ids)))
    vals = np.array([fn(list(arr[idx[i]])) for i in range(nb)])
    return float(fn(ids)), float(np.percentile(vals, 2.5)), \
        float(np.percentile(vals, 97.5))


def sig(t):
    return "EXCLUDES zero" if (t[1] > 0 or t[2] < 0) else "includes zero"


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    R = cfg["experiment"]["out_root"]
    od = os.path.join(R, "trufor_32b_final")
    F = json.load(open(os.path.join(od, "trufor_32b_final_capacity_frozen.json")))
    rows = [json.loads(l) for l in open(os.path.join(od, "verdicts.jsonl"))]
    G = defaultdict(dict)
    for r in rows:
        G[(r["sample_id"], r["label"])][r["condition"]] = r

    def load7(path, cond):
        out = {}
        for l in open(path):
            d = json.loads(l)
            if d["condition"] == cond:
                out[(d["sample_id"], d["label"])] = d
        return out
    V7 = {
        "A0": load7(os.path.join(R, "trufor_fields", "verdicts.jsonl"),
                    "S0_score_only"),
        "A1": load7(os.path.join(R, "trufor_2x2", "verdicts_A1.jsonl"),
                    "A1_score_rule"),
        "A2": load7(os.path.join(R, "trufor_semantics", "verdicts.jsonl"),
                    "D0_baseline"),
        "A3": load7(os.path.join(R, "trufor_semantics", "verdicts.jsonl"),
                    "D1_binary_rule"),
    }
    ids = sorted({p for (p, l) in G if all(c in G[(p, "fake")] for c in B)
                  and all(c in G[(p, "real")] for c in B)})
    ids = [p for p in ids if all((p, l) in V7[a] for a in V7
                                 for l in ("fake", "real"))]
    m = json.load(open(os.path.join(od, "run_meta.json")))

    print("32B FINAL CAPACITY REPLICATION — Qwen2.5-VL-32B-Instruct")
    print(f"complete sources {len(ids)}  bootstrap {NBOOT} seed {SEED}")
    print(f"{m['n_inferences']} inf  {m['minutes']} min  mean {m['mean_seconds']}s  "
          f"{m['started']} -> {m['finished']}")
    print(f"the question is whether capacity changes the MECHANISM, not whether 32B "
          f"is more accurate")

    print("\n" + "=" * 94)
    print("1. INTEGRITY")
    print(f"  parse failures "
          f"{sum(1 for r in rows if 'json_parse_failed' in r['notes'])}/{len(rows)}"
          f"   bad verdicts {sum(1 for r in rows if 'bad_verdict' in r['notes'])}")
    print(f"  cells {dict(Counter(r['condition'] for r in rows))}")
    print(f"  unique coco_id {len({G[(p,'fake')]['B0_score_only']['coco_id'] for p in ids})}")
    print(f"  payload sha1 {m['payload_sha1'][:16]} (frozen "
          f"{F['evidence']['corpus_payload_sha1'][:16]}) "
          f"{'OK' if m['payload_sha1']==F['evidence']['corpus_payload_sha1'] else 'MISMATCH'}")
    print(f"  B0/B1 carry no structured evidence: "
          f"{all(G[(p,l)][c]['evidence_locations'] is None for p in ids for l in ('fake','real') for c in ('B0_score_only','B1_score_rule'))}")
    print(f"  score identical across cells: "
          f"{all(len({G[(p,l)][c]['score_shown'] for c in B})==1 for p in ids for l in ('fake','real'))}")
    print(f"  shared sources with 7B: {len(ids)}")
    print(f"  protocol sha1 {m['protocol_sha1']}")

    fk = lambda c, p, l: int(G[(p, l)][c]["final_verdict"] == "fake")
    f7 = lambda a, p, l: int(V7[a][(p, l)]["final_verdict"] == "fake")
    rec = lambda c: (lambda ii: float(np.mean([fk(c, p, "fake") for p in ii])))
    fpr = lambda c: (lambda ii: float(np.mean([fk(c, p, "real") for p in ii])))
    jj = lambda c: (lambda ii: float(np.mean([fk(c, p, "fake") for p in ii])
                                     - np.mean([fk(c, p, "real") for p in ii])))
    rec7 = lambda a: (lambda ii: float(np.mean([f7(a, p, "fake") for p in ii])))
    fpr7 = lambda a: (lambda ii: float(np.mean([f7(a, p, "real") for p in ii])))
    jj7 = lambda a: (lambda ii: float(np.mean([f7(a, p, "fake") for p in ii])
                                      - np.mean([f7(a, p, "real") for p in ii])))

    print("\n" + "=" * 94)
    print("2. 32B CELLS (7B in parentheses)")
    print(f"  {'cell':<22}{'recall':>9}{'95% CI':>18}{'FPR':>8}{'95% CI':>18}"
          f"{'spec':>7}{'J':>8}{'   7B recall/FPR/J'}")
    SC = {}
    for c in B:
        r = boot_stat(rec(c), ids)
        f = boot_stat(fpr(c), ids)
        j = boot_stat(jj(c), ids)
        SC[c] = dict(recall=r, fpr=f, j=j)
        s7 = F["seven_b_reference"]["cells"][A[c]]
        print(f"  {LAB[c]:<22}{r[0]:>9.3f} [{r[1]:.3f},{r[2]:.3f}]"
              f"{f[0]:>8.3f} [{f[1]:.3f},{f[2]:.3f}]{1-f[0]:>7.3f}{j[0]:>8.3f}"
              f"   ({s7['recall']:.3f}/{s7['fpr']:.3f}/{s7['j']:+.3f})")

    print("\n" + "=" * 94)
    print("3. 32B SIMPLE EFFECTS")
    E32 = {}
    for nm, hi, lo in EFF:
        print(f"\n  {nm}   {hi.split('_')[0]} - {lo.split('_')[0]}")
        for w, g in (("fake recall", rec), ("real FPR", fpr), ("J", jj)):
            d = boot_stat(lambda ii, h=hi, l=lo, gg=g: gg(h)(ii) - gg(l)(ii), ids)
            E32[(nm, w)] = d
            line = f"    {w:<12} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}"
            if w != "J":
                lab = "fake" if w.startswith("fake") else "real"
                _, _, pv = mcnemar([fk(lo, p, lab) for p in ids],
                                   [fk(hi, p, lab) for p in ids])
                line += f"  p={pv:.4g}"
            print(line)

    print("\n" + "=" * 94)
    print("4. 32B FACTORIAL INTERACTION  (B3-B2) - (B1-B0)")
    I32 = {}
    for w, g in (("fake recall", rec), ("real FPR", fpr), ("J", jj)):
        d = boot_stat(lambda ii, gg=g: (gg("B3_structured_rule")(ii)
                                        - gg("B2_structured")(ii))
                      - (gg("B1_score_rule")(ii) - gg("B0_score_only")(ii)), ids)
        I32[w] = d
        s7 = F["seven_b_reference"]["interaction"]
        k7 = {"fake recall": "recall", "real FPR": "fpr", "J": "j"}[w]
        print(f"  {w:<12} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}"
              f"   7B {s7[k7]:+.3f}")

    print("\n" + "=" * 94)
    print("5. DELTA EFFECT = effect_32B - effect_7B  (PRIMARY capacity analysis)")
    print("  paired per source; both models ran the same sources")
    print("  a capacity change requires the delta CI to exclude zero")
    DE = {}
    for nm, hi, lo in EFF:
        a_hi, a_lo = A[hi], A[lo]
        print(f"\n  {nm}")
        for w, g32, g7 in (("fake recall", rec, rec7), ("real FPR", fpr, fpr7),
                           ("J", jj, jj7)):
            e32 = E32[(nm, w)]
            e7 = boot_stat(lambda ii, h=a_hi, l=a_lo, gg=g7: gg(h)(ii) - gg(l)(ii),
                           ids)
            d = boot_stat(lambda ii, h=hi, l=lo, ah=a_hi, al=a_lo, g=g32, h7=g7:
                          (g(h)(ii) - g(l)(ii)) - (h7(ah)(ii) - h7(al)(ii)), ids)
            DE[(nm, w)] = d
            print(f"    {w:<12} 32B {e32[0]:+.3f}  7B {e7[0]:+.3f}  "
                  f"DELTA {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}")

    print("\n" + "=" * 94)
    print("6. THE TWO DECISIVE EFFECTS")
    d31 = E32[("structured_with_rule", "J")]
    d31r = E32[("structured_with_rule", "fake recall")]
    dd31 = DE[("structured_with_rule", "J")]
    print(f"  B3 - B1 (does structured evidence add decision utility under correct")
    print(f"  semantics?)  recall {d31r[0]:+.3f} [{d31r[1]:+.3f},{d31r[2]:+.3f}] "
          f"{sig(d31r)}")
    print(f"                J      {d31[0]:+.3f} [{d31[1]:+.3f},{d31[2]:+.3f}] "
          f"{sig(d31)}")
    print(f"     7B A3 - A1 J was +0.005 (includes zero)")
    print(f"     DELTA {dd31[0]:+.3f} [{dd31[1]:+.3f},{dd31[2]:+.3f}] {sig(dd31)}")
    d20 = E32[("structured_without_rule", "fake recall")]
    dd20 = DE[("structured_without_rule", "fake recall")]
    print(f"\n  B2 - B0 (does the framing collapse weaken with capacity?)")
    print(f"     32B recall {d20[0]:+.3f} [{d20[1]:+.3f},{d20[2]:+.3f}]   "
          f"7B was -0.723")
    print(f"     DELTA {dd20[0]:+.3f} [{dd20[1]:+.3f},{dd20[2]:+.3f}] {sig(dd20)}")

    print("\n" + "=" * 94)
    print("7. FAITHFULNESS (B2/B3, vs 7B A2/A3)")
    def faith(src, cells, key=None):
        out = {}
        for c in cells:
            cite, inter, exact, unsup, a9 = [], [], [], [], []
            for p in ids:
                for l in ("fake", "real"):
                    r = src[c][(p, l)] if key else G[(p, l)][c]
                    ev = set(r["evidence_locations"] or [])
                    rf = set(r["referenced_regions"])
                    a9.append(1 if len(rf) == 9 else 0)
                    if len(rf) == 9 or not ev:
                        continue
                    cite.append(1 if rf else 0)
                    if rf:
                        inter.append(1 if rf & ev else 0)
                        exact.append(1 if rf == ev else 0)
                        unsup.append(1 if rf - ev else 0)
            out[c] = (np.mean(cite), np.mean(inter), np.mean(exact), np.mean(unsup),
                      np.mean(a9), len(inter))
        return out
    f32 = faith(None, ("B2_structured", "B3_structured_rule"))
    print(f"  {'condition':<24}{'cite':>8}{'agree':>8}{'exact':>8}{'unsup':>8}"
          f"{'all9':>8}{'n':>6}")
    for c, v in f32.items():
        print(f"  32B {LAB[c]:<20}{v[0]:>8.3f}{v[1]:>8.3f}{v[2]:>8.3f}{v[3]:>8.3f}"
              f"{v[4]:>8.3f}{v[5]:>6}")
    f7d = faith(V7, ("A2", "A3"), key=True)
    for c, v in f7d.items():
        print(f"  7B  {c:<20}{v[0]:>8.3f}{v[1]:>8.3f}{v[2]:>8.3f}{v[3]:>8.3f}"
              f"{v[4]:>8.3f}{v[5]:>6}")
    print(f"  7B reference spatial agreement was "
          f"{F['seven_b_reference']['faithfulness']['spatial_agreement']:.3f}")

    print("\n" + "=" * 94)
    print("8. B4 MIRROR CONTROL — frozen eligible subset only")
    elig = F["b4"]["eligible_ids"]
    B4 = {}
    for l in ("fake", "real"):
        use = [p for p in elig[l] if (p, l) in G and "B4_mirrored" in G[(p, l)]
               and "B3_structured_rule" in G[(p, l)]]
        foll, keep, chg = [], [], []
        for p in use:
            r4, r3 = G[(p, l)]["B4_mirrored"], G[(p, l)]["B3_structured_rule"]
            ev4 = set(r4["evidence_locations"] or [])
            ev3 = set(r3["evidence_locations"] or [])
            rf4 = set(r4["referenced_regions"])
            if len(rf4) == 9 or not ev4:
                continue
            foll.append(1 if rf4 & ev4 else 0)
            keep.append(1 if rf4 & (ev3 - ev4) else 0)
            chg.append(1 if r4["final_verdict"] != r3["final_verdict"] else 0)
        print(f"\n  [{l}] frozen eligible {len(elig[l])}  scored {len(foll)}")
        if foll:
            fo = boot_stat(lambda ii, mp=dict(zip(use, foll)):
                           float(np.mean([mp[p] for p in ii])), use)
            print(f"    follows mirrored evidence   {fo[0]:.3f} "
                  f"[{fo[1]:.3f},{fo[2]:.3f}]")
        else:
            print(f"    follows mirrored evidence   n/a")
        print(f"    still cites original side   {np.mean(keep):.3f}")
        print(f"    B3 -> B4 verdict changed    {np.mean(chg):.3f} "
              f"({sum(chg)}/{len(chg)})")
        B4[l] = dict(n_eligible=len(elig[l]), n_scored=len(foll),
                     follows=float(np.mean(foll)) if foll else None,
                     keeps_original=float(np.mean(keep)) if keep else None,
                     verdict_changed=float(np.mean(chg)) if chg else None)
    print(f"\n  7B mirror shift responsiveness was "
          f"{F['seven_b_reference']['faithfulness']['mirror_shift_responsiveness']:.3f}")
    print(f"  a verdict change is NOT required: relocating the stated evidence need")
    print(f"  not alter authenticity")

    print("\n" + "=" * 94)
    print("9. REASON AUDIT")
    pats = {k: re.compile("|".join(re.escape(w) for w in v), re.I)
            for k, v in F["reason_audit"]["patterns"].items()}
    for l in ("fake", "real"):
        print(f"\n  {l} targets")
        print(f"  {'condition':<24}" + "".join(f"{k[:12]:>14}" for k in pats))
        for c in B:
            line = f"  {LAB[c]:<24}"
            for k in pats:
                line += f"{np.mean([1 if pats[k].search(G[(p,l)][c]['reason'] or '') else 0 for p in ids]):>14.3f}"
            print(line)
    print(f"\n  7B pattern: the rule raised score_high from 0.668 to 0.946 on fakes")
    for a, bb in (("B0_score_only", "B1_score_rule"),
                  ("B2_structured", "B3_structured_rule")):
        s_a = np.mean([1 if pats["score_high"].search(G[(p, 'fake')][a]["reason"] or "")
                       else 0 for p in ids])
        s_b = np.mean([1 if pats["score_high"].search(G[(p, 'fake')][bb]["reason"] or "")
                       else 0 for p in ids])
        print(f"  32B score_high  {a.split('_')[0]} {s_a:.3f} -> "
              f"{bb.split('_')[0]} {s_b:.3f}  ({s_b-s_a:+.3f})")

    print("\n" + "=" * 94)
    print("10. CAPACITY HYPOTHESES C1-C4")
    r = {c: SC[c]["recall"][0] for c in B}
    print("  recall  " + "  ".join(f"{c.split('_')[0]} {r[c]:.3f}" for c in B))
    print("  J       " + "  ".join(f"{c.split('_')[0]} {SC[c]['j'][0]:+.3f}"
                                   for c in B))
    rw = E32[("rule_without_structured", "J")]
    rws = E32[("rule_with_structured", "J")]
    drw = DE[("rule_without_structured", "J")]
    drws = DE[("rule_with_structured", "J")]
    print(f"\n  rule effect J: without struct {rw[0]:+.3f} {sig(rw)}   "
          f"with struct {rws[0]:+.3f} {sig(rws)}")
    print(f"  delta vs 7B:   without {drw[0]:+.3f} {sig(drw)}   "
          f"with {drws[0]:+.3f} {sig(drws)}")
    print()
    out = []
    if rw[1] > 0 or rws[1] > 0:
        tag = ("unchanged from 7B" if not (drws[1] > 0 or drws[2] < 0)
               else "but its size DID change vs 7B")
        out.append(f"C1 capacity-invariant task semantics: the rule effect persists at "
                   f"32B ({tag})")
    if dd20[1] > 0:
        out.append("C2 32B REDUCES framing collapse: B2-B0 is less negative than 7B "
                   "with the delta CI excluding zero")
    elif not (dd20[1] > 0 or dd20[2] < 0):
        out.append("C2 NOT supported: the framing collapse is statistically unchanged "
                   "from 7B")
    if d31[1] > 0 and dd31[1] > 0:
        out.append("C3 32B gains decision utility from structured evidence: B3-B1 > 0 "
                   "and clearly above the 7B near-zero")
    if not (d31[1] > 0 or d31[2] < 0):
        out.append("C4 role separation is capacity-invariant: B3 ~ B1, so structured "
                   "evidence still buys faithful explanation rather than "
                   "discrimination")
    if d31[2] < 0:
        out.append("structured evidence REDUCES discrimination even at 32B under "
                   "correct semantics")
    for o in out:
        print(f"  -> {o}")

    json.dump(dict(cells={c: SC[c] for c in B},
                   effects_32b={f"{k[0]}|{k[1]}": v for k, v in E32.items()},
                   interaction_32b=I32,
                   delta_effects={f"{k[0]}|{k[1]}": v for k, v in DE.items()},
                   faithfulness_32b={k: list(v) for k, v in f32.items()},
                   faithfulness_7b={k: list(v) for k, v in f7d.items()},
                   b4=B4, n_sources=len(ids)),
              open(os.path.join(od, "analysis_32b_final.json"), "w"), indent=1)
    print(f"\n  wrote analysis_32b_final.json")


if __name__ == "__main__":
    main()
