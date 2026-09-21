"""Analysis of the 2x2 score/map decomposition.

Primary: recall / FPR / Youden J per cell, the four simple effects, and the
factorial interaction in both algebraic forms. Secondary: a deterministic
keyword audit of the saved reason strings -- no LLM judges explanations.
"""
import json, os, re, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np, yaml

NBOOT, SEED = 10000, 20260918
CONDS = ("C00", "C10", "C01", "C11")
LAB = {"C00": "C00 image only", "C10": "C10 score only",
       "C01": "C01 map only", "C11": "C11 score + map"}
PREV = dict(T0=dict(recall=0.201, fpr=0.125, j=0.076),
            T1=dict(recall=0.804, fpr=0.011, j=0.793))


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return 1.0
    k = min(b01, b10)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


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
    od = os.path.join(R, "trufor_score_map")
    F = json.load(open(os.path.join(od, "trufor_score_map_ablation_frozen.json")))
    rows = [json.loads(l) for l in open(os.path.join(od, "verdicts.jsonl"))]
    G = defaultdict(dict)
    for r in rows:
        G[r["sample_id"]][(r["label"], r["condition"])] = r
    ids = sorted(p for p in G if len(G[p]) == 8)

    print("TRUFOR SCORE-MAP CAUSAL DECOMPOSITION (2x2, Stage-B only)")
    print(f"rows {len(rows)}  complete sources {len(ids)}  bootstrap {NBOOT} "
          f"seed {SEED}")

    print("\n" + "=" * 92)
    print("1. INTEGRITY")
    pf = sum(1 for r in rows if "json_parse_failed" in (r["notes"] or []))
    mv = sum(1 for r in rows if not r["final_verdict"])
    print(f"  parse failures          : {pf}/{len(rows)}")
    print(f"  missing verdicts        : {mv}/{len(rows)}")
    print(f"  cells                   : {dict(Counter((r['label'],r['condition']) for r in rows))}")
    print(f"  unique coco_id          : {len({G[p][('fake','C00')]['coco_id'] for p in ids})}")
    print(f"  n_samples == n_sources  : "
          f"{len({G[p][('fake','C00')]['coco_id'] for p in ids}) == len(ids)}")
    ok_img = all(G[p][(l, c)]["n_images"] == (2 if c in ("C01", "C11") else 1)
                 for p in ids for l in ("fake", "real") for c in CONDS)
    print(f"  image counts per condition correct: {ok_img}")
    ok_sc = all((G[p][(l, c)]["evidence_score"] is not None) == (c in ("C10", "C11"))
                for p in ids for l in ("fake", "real") for c in CONDS)
    print(f"  score supplied only in C10/C11    : {ok_sc}")
    m = json.load(open(os.path.join(od, "run_meta.json")))
    print(f"  protocol sha1           : {m['protocol_sha1']}")
    print(f"  Stage-A summary used    : NO (absent from all four prompts)")

    fk = lambda p, c: int(G[p][("fake", c)]["final_verdict"] == "fake")
    rl = lambda p, c: int(G[p][("real", c)]["final_verdict"] == "fake")

    print("\n" + "=" * 92)
    print("2. FOUR CELLS")
    print(f"  {'condition':<20}{'score':>6}{'map':>5}{'recall':>9}{'95% CI':>17}"
          f"{'FPR':>8}{'95% CI':>17}{'spec':>7}{'J':>8}")
    S = {}
    for c in CONDS:
        r, rlo, rhi = boot([fk(p, c) for p in ids])
        f, flo, fhi = boot([rl(p, c) for p in ids])
        j, jlo, jhi = boot([fk(p, c) - rl(p, c) for p in ids])
        S[c] = dict(recall=r, fpr=f, j=j, jlo=jlo, jhi=jhi)
        sc = "yes" if c in ("C10", "C11") else "no"
        mp = "yes" if c in ("C01", "C11") else "no"
        print(f"  {LAB[c]:<20}{sc:>6}{mp:>5}{r:>9.3f} [{rlo:.3f},{rhi:.3f}]"
              f"{f:>8.3f} [{flo:.3f},{fhi:.3f}]{1-f:>7.3f}{j:>8.3f}")
    print(f"\n  {'':<20}J with CI:")
    for c in CONDS:
        print(f"  {LAB[c]:<20}{S[c]['j']:+.3f} [{S[c]['jlo']:+.3f},{S[c]['jhi']:+.3f}]")

    def eff(fn, c1, c0):
        d = boot([fn(p, c1) - fn(p, c0) for p in ids])
        pv = mcnemar([fn(p, c0) for p in ids], [fn(p, c1) for p in ids])
        return d, pv

    def jeff(c1, c0):
        return boot([(fk(p, c1) - rl(p, c1)) - (fk(p, c0) - rl(p, c0)) for p in ids])

    print("\n" + "=" * 92)
    print("3. SIMPLE EFFECTS  (effect -> source bootstrap CI -> McNemar p)")
    SE = [("score effect WITHOUT map", "C10", "C00"),
          ("score effect WITH map", "C11", "C01"),
          ("map effect WITHOUT score", "C01", "C00"),
          ("map effect WITH score", "C11", "C10")]
    E = {}
    for nm, c1, c0 in SE:
        print(f"\n  {nm}   ({c1} - {c0})")
        dr, pr = eff(fk, c1, c0)
        df, pf_ = eff(rl, c1, c0)
        dj = jeff(c1, c0)
        E[nm] = dict(recall=dr, fpr=df, j=dj)
        print(f"    recall {dr[0]:+.3f} [{dr[1]:+.3f},{dr[2]:+.3f}] {sig(dr)}  p={pr:.4g}")
        print(f"    FPR    {df[0]:+.3f} [{df[1]:+.3f},{df[2]:+.3f}] {sig(df)}  p={pf_:.4g}")
        print(f"    J      {dj[0]:+.3f} [{dj[1]:+.3f},{dj[2]:+.3f}] {sig(dj)}")

    print("\n" + "=" * 92)
    print("4. INTERACTION")
    for nm, fn in (("recall", fk), ("FPR", rl)):
        a = boot([(fn(p, "C11") - fn(p, "C01")) - (fn(p, "C10") - fn(p, "C00"))
                  for p in ids])
        b = boot([(fn(p, "C11") - fn(p, "C10")) - (fn(p, "C01") - fn(p, "C00"))
                  for p in ids])
        print(f"  {nm:<7} (C11-C01)-(C10-C00) = {a[0]:+.3f} [{a[1]:+.3f},{a[2]:+.3f}] {sig(a)}")
        print(f"  {nm:<7} (C11-C10)-(C01-C00) = {b[0]:+.3f} [{b[1]:+.3f},{b[2]:+.3f}] {sig(b)}")
    ja = boot([((fk(p, "C11") - rl(p, "C11")) - (fk(p, "C01") - rl(p, "C01")))
               - ((fk(p, "C10") - rl(p, "C10")) - (fk(p, "C00") - rl(p, "C00")))
               for p in ids])
    print(f"  J       (C11-C01)-(C10-C00) = {ja[0]:+.3f} [{ja[1]:+.3f},{ja[2]:+.3f}] {sig(ja)}")
    print("  (the two algebraic forms are identical by construction for recall/FPR)")

    print("\n" + "=" * 92)
    print("5. FAKE AND REAL SEPARATELY")
    print(f"  {'condition':<20}{'fake->fake':>12}{'real->fake':>12}")
    for c in CONDS:
        print(f"  {LAB[c]:<20}{sum(fk(p,c) for p in ids):>8}/{len(ids)}"
              f"{sum(rl(p,c) for p in ids):>8}/{len(ids)}")
    print("\n  verdict flips vs C00 (fake images):")
    for c in ("C10", "C01", "C11"):
        g = sum(1 for p in ids if fk(p, c) and not fk(p, "C00"))
        l = sum(1 for p in ids if not fk(p, c) and fk(p, "C00"))
        print(f"    {c}: real->fake {g:>4}   fake->real {l:>4}")
    print("  verdict flips vs C00 (real images):")
    for c in ("C10", "C01", "C11"):
        g = sum(1 for p in ids if rl(p, c) and not rl(p, "C00"))
        l = sum(1 for p in ids if not rl(p, c) and rl(p, "C00"))
        print(f"    {c}: real->fake {g:>4}   fake->real {l:>4}")

    # does the verdict track the withheld score in map-only?
    print("\n  sanity: in C01 the score is NOT shown; does the verdict still track it?")
    for c in ("C01", "C00"):
        sf = [G[p][("fake", c)]["actual_score"] for p in ids if fk(p, c)]
        sn = [G[p][("fake", c)]["actual_score"] for p in ids if not fk(p, c)]
        print(f"    {c} fake: mean actual score when verdict=fake {np.mean(sf) if sf else float('nan'):.3f} "
              f"(n={len(sf)})  vs verdict=real {np.mean(sn) if sn else float('nan'):.3f} (n={len(sn)})")

    print("\n" + "=" * 92)
    print("6. EXPLANATION AUDIT (deterministic keyword rules; secondary)")
    CATS = F["explanation_audit"]["categories"]
    pat = {k: re.compile("|".join(re.escape(w) for w in v), re.I)
           for k, v in CATS.items()}
    print(f"  {'condition':<20}" + "".join(f"{k.replace('mentions_',''):>14}"
                                           for k in CATS))
    for c in CONDS:
        line = f"  {LAB[c]:<20}"
        for k in CATS:
            hits = np.mean([1 if pat[k].search(
                G[p][(l, c)]["reason"] or "") else 0
                for p in ids for l in ("fake", "real")])
            line += f"{hits:>14.3f}"
        print(line)
    # coupling: region AND semantic content in the same reason
    print("\n  coupled region+content mentions (a spatial claim tied to image content):")
    for c in CONDS:
        v = np.mean([1 if (pat["mentions_region"].search(G[p][(l, c)]["reason"] or "")
                           and pat["mentions_semantic_content"].search(
                               G[p][(l, c)]["reason"] or "")) else 0
                     for p in ids for l in ("fake", "real")])
        print(f"    {LAB[c]:<20}{v:.3f}")
    print("\n  example reasons:")
    for c in CONDS:
        r = G[ids[0]][("fake", c)]["reason"]
        print(f"    {c}: {str(r)[:150]}")

    print("\n" + "=" * 92)
    print("7. PRE-REGISTERED CASE JUDGEMENT")
    sw = E["score effect WITHOUT map"]
    swm = E["score effect WITH map"]
    mw = E["map effect WITHOUT score"]
    mws = E["map effect WITH score"]
    print(f"  score effect  no-map {sw['j'][0]:+.3f} {sig(sw['j'])}   "
          f"with-map {swm['j'][0]:+.3f} {sig(swm['j'])}")
    print(f"  map   effect  no-score {mw['j'][0]:+.3f} {sig(mw['j'])}   "
          f"with-score {mws['j'][0]:+.3f} {sig(mws['j'])}")
    print(f"  J: C00 {S['C00']['j']:+.3f}  C10 {S['C10']['j']:+.3f}  "
          f"C01 {S['C01']['j']:+.3f}  C11 {S['C11']['j']:+.3f}")
    print()
    score_big = sw["j"][1] > 0
    map_small = not (mw["j"][1] > 0)
    map_big = mw["j"][1] > 0 and mws["j"][1] > 0
    harms = (S["C10"]["j"] > S["C11"]["j"]) or (mws["fpr"][1] > 0)
    if harms and score_big:
        print("  -> CASE 4: adding the map interferes with an otherwise useful scalar")
        print("     signal (C10 outperforms C11, or the map raises real FPR).")
    elif score_big and map_small:
        print("  -> CASE 1: SCORE-DOMINATED. The verdict improvement is primarily")
        print("     driven by the image-level scalar score; spatial evidence adds")
        print("     little to the final authenticity decision.")
    elif map_big and score_big and ja[1] > 0:
        print("  -> CASE 3: SYNERGY. Scalar and spatial evidence are jointly")
        print("     integrated; the combination exceeds either alone.")
    elif map_big:
        print("  -> CASE 2: spatial localization contributes independently of the")
        print("     scalar score.")
    else:
        print("  -> mixed; report descriptively without forcing a case.")

    print("\n" + "=" * 92)
    print("8. CONSISTENCY WITH THE PREVIOUS ROUND (different protocol)")
    print(f"  prev T0 (with Stage-A summary): recall {PREV['T0']['recall']:.3f} "
          f"FPR {PREV['T0']['fpr']:.3f} J {PREV['T0']['j']:+.3f}")
    print(f"  now  C00 (no summary)         : recall {S['C00']['recall']:.3f} "
          f"FPR {S['C00']['fpr']:.3f} J {S['C00']['j']:+.3f}")
    print(f"  prev T1 correct (score+map+summary): recall {PREV['T1']['recall']:.3f} "
          f"FPR {PREV['T1']['fpr']:.3f} J {PREV['T1']['j']:+.3f}")
    print(f"  now  C11 (score+map, no summary)   : recall {S['C11']['recall']:.3f} "
          f"FPR {S['C11']['fpr']:.3f} J {S['C11']['j']:+.3f}")
    print("  NOTE: the previous round fed Stage B a Stage-A summary, so these are")
    print("  different protocols; treat this as a descriptive consistency check, not")
    print("  a paired contrast.")
    print(f"\n9. RUNTIME  {m['n_inferences']} inf  {m['minutes']} min  "
          f"mean {m['mean_seconds']}s  peak {m['peak_GB']} GB")


if __name__ == "__main__":
    main()
