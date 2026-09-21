"""Analysis of the Evidence Abstraction Layer experiment.

Decision utility and spatial faithfulness are reported SEPARATELY and must not be
conflated: an E2 gain cannot be read as improved spatial reasoning because region
count and area themselves carry label information. Whether spatial evidence is
actually used rests on referenced-region agreement and on shift responsiveness within
the frozen eligible subset only.
"""
import json, os, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np, yaml

NBOOT, SEED = 10000, 20260918
E = ("E0_score_only", "E1_raw_map", "E2_structured", "E3_both", "E4_shifted_struct")
LAB = {"E0_score_only": "E0 score only", "E1_raw_map": "E1 score + raw map",
       "E2_structured": "E2 score + structured", "E3_both": "E3 structured + raw",
       "E4_shifted_struct": "E4 mirrored structured"}
MIRROR = {"top-left": "top-right", "top-center": "top-center",
          "top-right": "top-left", "center-left": "center-right",
          "center": "center", "center-right": "center-left",
          "bottom-left": "bottom-right", "bottom-center": "bottom-center",
          "bottom-right": "bottom-left"}
DEC = (("structured_benefit", "E2_structured", "E0_score_only"),
       ("raw_map_cost", "E1_raw_map", "E0_score_only"),
       ("structured_vs_raw", "E2_structured", "E1_raw_map"),
       ("raw_after_abstraction", "E3_both", "E2_structured"))


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
    od = os.path.join(R, "trufor_abstraction")
    F = json.load(open(os.path.join(od, "trufor_evidence_abstraction_frozen.json")))
    rows = [json.loads(l) for l in open(os.path.join(od, "verdicts.jsonl"))]
    G = defaultdict(dict)
    for r in rows:
        G[(r["sample_id"], r["label"])][r["condition"]] = r
    ids = sorted({p for (p, l) in G if len(G[(p, "fake")]) == 5
                  and len(G[(p, "real")]) == 5})
    m = json.load(open(os.path.join(od, "run_meta.json")))

    print("EVIDENCE ABSTRACTION LAYER — 7B")
    print(f"rows {len(rows)}  complete sources {len(ids)}  bootstrap {NBOOT} "
          f"seed {SEED}")
    print(f"{m['n_inferences']} inf  {m['minutes']} min  mean {m['mean_seconds']}s  "
          f"peak {m['peak_GB']} GB")

    print("\n" + "=" * 94)
    print("1. INTEGRITY")
    pf = sum(1 for r in rows if "json_parse_failed" in r["notes"])
    bv = sum(1 for r in rows if "bad_verdict" in r["notes"])
    jr = sum(1 for r in rows if r["junk_regions"])
    print(f"  parse failures {pf}/{len(rows)}   bad verdicts {bv}   "
          f"unparsed region tokens {jr}")
    print(f"  cells {dict(Counter(r['condition'] for r in rows))}")
    print(f"  unique coco_id {len({G[(p,'fake')]['E0_score_only']['coco_id'] for p in ids})}")
    print(f"  E0/E2/E4 single image: "
          f"{all(G[(p,l)][c]['n_images']==1 for p in ids for l in ('fake','real') for c in ('E0_score_only','E2_structured','E4_shifted_struct'))}")
    print(f"  E1/E3 two images     : "
          f"{all(G[(p,l)][c]['n_images']==2 for p in ids for l in ('fake','real') for c in ('E1_raw_map','E3_both'))}")
    ok4 = all(G[(p, l)]["E4_shifted_struct"]["evidence_n_regions"]
              == G[(p, l)]["E2_structured"]["evidence_n_regions"]
              and G[(p, l)]["E4_shifted_struct"]["score_shown"]
              == G[(p, l)]["E2_structured"]["score_shown"]
              for p in ids for l in ("fake", "real"))
    print(f"  E4 preserves E2 region count and score: {ok4}")
    mir = all([MIRROR[x] for x in G[(p, l)]["E2_structured"]["evidence_locations"]]
              == G[(p, l)]["E4_shifted_struct"]["evidence_locations"]
              for p in ids for l in ("fake", "real"))
    print(f"  E4 locations are exactly the mirror of E2: {mir}")
    print(f"  protocol sha1 {m['protocol_sha1']}")

    print("\n" + "=" * 94)
    print("2. DEGENERATE-REFERENCE ARTIFACT (must be read before faithfulness)")
    print("  the output schema enumerates the nine grid names, and without structured")
    print("  evidence the model sometimes echoes ALL nine back. Such answers carry no")
    print("  spatial information and are counted separately.")
    print(f"  {'condition':<26}{'all-9 echo':>12}{'empty':>8}{'mean cited':>12}")
    for c in E:
        a9 = np.mean([1 if len(G[(p, l)][c]["referenced_regions"]) == 9 else 0
                      for p in ids for l in ("fake", "real")])
        em = np.mean([1 if not G[(p, l)][c]["referenced_regions"] else 0
                      for p in ids for l in ("fake", "real")])
        mc = np.mean([len(G[(p, l)][c]["referenced_regions"])
                      for p in ids for l in ("fake", "real")])
        print(f"  {LAB[c]:<26}{a9:>12.3f}{em:>8.3f}{mc:>12.2f}")

    fk = lambda p, l, c: int(G[(p, l)][c]["final_verdict"] == "fake")

    print("\n" + "=" * 94)
    print("=== PART A: DECISION UTILITY ===")
    print("=" * 94)
    print(f"  {'condition':<26}{'recall':>9}{'95% CI':>18}{'FPR':>8}{'95% CI':>18}{'J':>9}")
    SC = {}
    for c in E:
        rc = boot([fk(p, "fake", c) for p in ids])
        fp = boot([fk(p, "real", c) for p in ids])
        jj = boot([fk(p, "fake", c) - fk(p, "real", c) for p in ids])
        SC[c] = dict(recall=rc, fpr=fp, j=jj)
        print(f"  {LAB[c]:<26}{rc[0]:>9.3f} [{rc[1]:.3f},{rc[2]:.3f}]"
              f"{fp[0]:>8.3f} [{fp[1]:.3f},{fp[2]:.3f}]{jj[0]:>9.3f}")
    print(f"\n  reference (earlier round, prompt without referenced_regions):")
    print(f"    C10 score only 0.755   C11 score+map 0.527")

    print("\n" + "=" * 94)
    print("3. PRIMARY DECISION CONTRASTS")
    for nm, hi, lo in DEC:
        print(f"\n  {nm}   {hi.split('_')[0]} - {lo.split('_')[0]}")
        for w, l in (("fake recall", "fake"), ("real FPR", "real")):
            d = boot([fk(p, l, hi) - fk(p, l, lo) for p in ids])
            _, _, pv = mcnemar([fk(p, l, lo) for p in ids],
                               [fk(p, l, hi) for p in ids])
            print(f"    {w:<12} {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}  "
                  f"p={pv:.4g}")
        dj = boot([(fk(p, "fake", hi) - fk(p, "real", hi))
                   - (fk(p, "fake", lo) - fk(p, "real", lo)) for p in ids])
        print(f"    {'J':<12} {dj[0]:+.3f} [{dj[1]:+.3f},{dj[2]:+.3f}] {sig(dj)}")

    print("\n" + "=" * 94)
    print("4. CONFOUND CHECK — does structured evidence leak the label?")
    print("  frozen corpus stats: fake non-empty 184/184 vs real 174/184;")
    print("  mean regions fake 1.147 vs real 1.902; area median 0.0251 vs 0.00395")
    for l in ("fake", "real"):
        nr = [G[(p, l)]["E2_structured"]["evidence_n_regions"] for p in ids]
        print(f"  {l:<5} mean regions {np.mean(nr):.3f}  empty "
              f"{sum(1 for x in nr if x==0)}/{len(nr)}")
    emp = [p for p in ids if G[(p, "real")]["E2_structured"]["evidence_n_regions"] == 0]
    non = [p for p in ids if G[(p, "real")]["E2_structured"]["evidence_n_regions"] > 0]
    if emp:
        print(f"  real FPR | empty evidence  {np.mean([fk(p,'real','E2_structured') for p in emp]):.3f} (n={len(emp)})")
        print(f"  real FPR | non-empty       {np.mean([fk(p,'real','E2_structured') for p in non]):.3f} (n={len(non)})")
    print("  -> any E2 decision gain may ride on these scalar-like cues, NOT on")
    print("     spatial reasoning. Read Part B before interpreting Part A.")

    print("\n" + "=" * 94)
    print("=== PART B: SPATIAL FAITHFULNESS ===")
    print("=" * 94)
    print("5. CITATION AND AGREEMENT (structured conditions; all-9 echoes excluded")
    print("   from agreement as non-informative, and counted in the artifact row)")
    print(f"  {'condition':<26}{'cite':>8}{'inter':>8}{'exact':>8}{'unsup':>8}{'halluc':>9}")
    FAITH = {}
    for c in ("E2_structured", "E3_both", "E4_shifted_struct"):
        cite, inter, exact, unsup, halluc = [], [], [], [], []
        for p in ids:
            for l in ("fake", "real"):
                r = G[(p, l)][c]
                ev = set(r["evidence_locations"] or [])
                rf = set(r["referenced_regions"])
                if len(rf) == 9:
                    continue
                if ev:
                    cite.append(1 if rf else 0)
                    if rf:
                        inter.append(1 if rf & ev else 0)
                        exact.append(1 if rf == ev else 0)
                        unsup.append(1 if rf - ev else 0)
                else:
                    halluc.append(1 if rf else 0)
        FAITH[c] = dict(cite=boot(cite), inter=boot(inter), exact=boot(exact),
                        unsup=boot(unsup), halluc=boot(halluc))
        f = FAITH[c]
        print(f"  {LAB[c]:<26}{f['cite'][0]:>8.3f}{f['inter'][0]:>8.3f}"
              f"{f['exact'][0]:>8.3f}{f['unsup'][0]:>8.3f}{f['halluc'][0]:>9.3f}")
    for k, nm in (("inter", "spatial agreement (intersects)"),
                  ("exact", "exact agreement"), ("unsup", "unsupported-region rate"),
                  ("cite", "citation rate")):
        f = FAITH["E2_structured"]
        print(f"  E2 {nm:<32} {f[k][0]:.3f} [{f[k][1]:.3f},{f[k][2]:.3f}]")

    print("\n  baseline: agreement without structured evidence (E1 raw map),")
    print("  scored against what the abstraction WOULD have said:")
    bi, bc = [], []
    for p in ids:
        for l in ("fake", "real"):
            ev = set(G[(p, l)]["E2_structured"]["evidence_locations"] or [])
            rf = set(G[(p, l)]["E1_raw_map"]["referenced_regions"])
            if len(rf) == 9 or not ev:
                continue
            bc.append(1 if rf else 0)
            if rf:
                bi.append(1 if rf & ev else 0)
    b = boot(bi)
    print(f"  E1 spatial agreement {b[0]:.3f} [{b[1]:.3f},{b[2]:.3f}]  (n={len(bi)})")
    print(f"  E2 - E1 agreement is reported as a difference of independent rates,")
    print(f"  not paired, because the E1 denominator differs; treat descriptively.")

    print("\n" + "=" * 94)
    print("6. SHIFT RESPONSIVENESS — frozen eligible subset ONLY")
    print("  eligible = primary region's grid location changes under the mirror")
    print("  non-eligible samples are NOT in the denominator and the subset is never")
    print("  expanded post hoc")
    for l in ("fake", "real"):
        el = [p for p in ids if G[(p, l)]["E2_structured"]["e4_eligible"]]
        frozen_n = len(F["e4_eligible_ids"][l])
        foll, stay, part = [], [], []
        for p in el:
            e2, e4 = G[(p, l)]["E2_structured"], G[(p, l)]["E4_shifted_struct"]
            ev2 = set(e2["evidence_locations"] or [])
            ev4 = set(e4["evidence_locations"] or [])
            r4 = set(e4["referenced_regions"])
            if len(r4) == 9 or not ev4:
                continue
            foll.append(1 if r4 & ev4 else 0)
            stay.append(1 if (r4 & (ev2 - ev4)) else 0)
        fo, st = boot(foll), boot(stay)
        print(f"\n  [{l}] eligible {len(el)} (frozen {frozen_n}), scored {len(foll)}")
        print(f"    follows mirrored evidence      {fo[0]:.3f} [{fo[1]:.3f},{fo[2]:.3f}]")
        print(f"    still cites the ORIGINAL side  {st[0]:.3f} [{st[1]:.3f},{st[2]:.3f}]")
        ver = boot([fk(p, l, "E4_shifted_struct") - fk(p, l, "E2_structured")
                    for p in el])
        print(f"    verdict change E4-E2           {ver[0]:+.3f} "
              f"[{ver[1]:+.3f},{ver[2]:+.3f}] {sig(ver)}")
        print(f"    (a verdict change is NOT required: moving the stated location need")
        print(f"     not alter authenticity)")

    print("\n" + "=" * 94)
    print("7. CASE JUDGEMENT")
    e2r, e0r, e1r = SC["E2_structured"], SC["E0_score_only"], SC["E1_raw_map"]
    d21 = boot([fk(p, "fake", "E2_structured") - fk(p, "fake", "E1_raw_map")
                for p in ids])
    d20 = boot([fk(p, "fake", "E2_structured") - fk(p, "fake", "E0_score_only")
                for p in ids])
    agree = FAITH["E2_structured"]["inter"][0]
    fol_f = None
    el = [p for p in ids if G[(p, "fake")]["E2_structured"]["e4_eligible"]]
    vv = [1 if (set(G[(p,'fake')]["E4_shifted_struct"]["referenced_regions"])
                & set(G[(p,'fake')]["E4_shifted_struct"]["evidence_locations"] or []))
          else 0 for p in el
          if len(G[(p,'fake')]["E4_shifted_struct"]["referenced_regions"]) != 9]
    fol_f = float(np.mean(vv)) if vv else float("nan")
    print(f"  recall  E0 {e0r['recall'][0]:.3f}  E1 {e1r['recall'][0]:.3f}  "
          f"E2 {e2r['recall'][0]:.3f}  E3 {SC['E3_both']['recall'][0]:.3f}")
    print(f"  J       E0 {e0r['j'][0]:.3f}  E1 {e1r['j'][0]:.3f}  "
          f"E2 {e2r['j'][0]:.3f}  E3 {SC['E3_both']['j'][0]:.3f}")
    print(f"  E2-E1 recall {d21[0]:+.3f} {sig(d21)}   E2-E0 recall {d20[0]:+.3f} "
          f"{sig(d20)}")
    print(f"  E2 spatial agreement {agree:.3f}   fake shift-following {fol_f:.3f}")
    print()
    cases = []
    if d21[1] > 0 and d20[2] >= 0 and agree > 0.6 and fol_f > 0.6:
        cases.append("A representation bottleneck")
    if not (d20[1] > 0 or d20[2] < 0) and agree < 0.4 and fol_f < 0.4:
        cases.append("B spatial evidence intrinsically unused")
    if (not (boot([(fk(p,'fake','E2_structured')-fk(p,'real','E2_structured'))
                   -(fk(p,'fake','E0_score_only')-fk(p,'real','E0_score_only'))
                   for p in ids])[1] > 0) and agree > 0.6):
        cases.append("C structured abstraction improves explanation faithfulness "
                     "without materially changing discrimination")
    if d20[2] < 0:
        cases.append("D structured evidence harms the scalar decision")
    if not cases:
        cases.append("no pre-registered case matched cleanly — report descriptively")
    for c in cases:
        print(f"  -> CASE {c}")
    print("\n  success criterion was accuracy PRESERVATION + faithfulness gain,")
    print("  not beating score-only accuracy.")


if __name__ == "__main__":
    main()
