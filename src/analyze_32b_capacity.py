"""Analysis of the 32B capacity replication.

Primary: per-condition recall, the three effects, and delta_effect = effect_32B -
effect_7B with bootstrap CI. The capacity conclusion rests on the delta_effect CI,
NOT on comparing significance between models.

Both models ran the same 184 sources, so delta_effect is computed per source and
bootstrapped paired across models.
"""
import json, os, re, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np, yaml

NBOOT, SEED = 10000, 20260918
B = ("B0_score_only", "B1_own_map", "B2_blank_map", "B3_shifted_map")
LAB = {"B0_score_only": "B0 score only", "B1_own_map": "B1 score + own map",
       "B2_blank_map": "B2 score + blank map",
       "B3_shifted_map": "B3 score + shifted map"}
# 7B counterpart of each 32B condition
MAP7 = {"B0_score_only": ("2x2", "C10"), "B1_own_map": ("conflict", "F1_own"),
        "B2_blank_map": ("conflict", "F2_blank"),
        "B3_shifted_map": ("conflict", "F4_shifted")}
EFFECTS = (("map_cost", "B1_own_map", "B0_score_only"),
           ("content_effect", "B2_blank_map", "B1_own_map"),
           ("spatial_correspondence", "B3_shifted_map", "B1_own_map"))


def mcnemar(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return 0, 0, 1.0
    k = min(b01, b10)
    return b01, b10, min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def boot(v, seed=SEED):
    a = np.asarray(v, float)
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
    od = os.path.join(R, "trufor_32b")
    F = json.load(open(os.path.join(od, "trufor_32b_capacity_frozen.json")))
    rows = [json.loads(l) for l in open(os.path.join(od, "verdicts.jsonl"))]
    G = defaultdict(dict)
    for r in rows:
        G[r["sample_id"]][r["condition"]] = r
    ids = sorted(p for p in G if len(G[p]) == 4)

    # 7B verdicts on the same sources
    c7 = defaultdict(dict)
    for l in open(os.path.join(R, "trufor_conflict", "verdicts.jsonl")):
        d = json.loads(l)
        if d["label"] == "fake":
            c7[d["sample_id"]][d["condition"]] = d
    a7 = defaultdict(dict)
    for l in open(os.path.join(R, "trufor_score_map", "verdicts.jsonl")):
        d = json.loads(l)
        if d["label"] == "fake":
            a7[d["sample_id"]][d["condition"]] = d

    print("QWEN2.5-VL-32B CAPACITY REPLICATION")
    print(f"rows {len(rows)}  complete sources {len(ids)}  bootstrap {NBOOT} "
          f"seed {SEED}")
    m = json.load(open(os.path.join(od, "run_meta.json")))
    print(f"model {F['model']['snapshot'][:16]}  {m['dtype']}  "
          f"{m['n_inferences']} inf  {m['minutes']} min  mean {m['mean_seconds']}s")

    print("\n" + "=" * 92)
    print("1. INTEGRITY")
    pf = sum(1 for r in rows if "json_parse_failed" in (r["notes"] or []))
    print(f"  parse failures    : {pf}/{len(rows)}")
    print(f"  missing verdicts  : {sum(1 for r in rows if not r['final_verdict'])}")
    print(f"  cells             : {dict(Counter(r['condition'] for r in rows))}")
    print(f"  unique coco_id    : {len({G[p]['B0_score_only']['coco_id'] for p in ids})}")
    print(f"  own score identical across conditions: "
          f"{all(len({G[p][c]['score_shown'] for c in B})==1 for p in ids)}")
    print(f"  B0 single image / others two images  : "
          f"{all(G[p]['B0_score_only']['n_images']==1 for p in ids)} / "
          f"{all(G[p][c]['n_images']==2 for p in ids for c in B[1:])}")
    shok = all(abs(G[p]['B3_shifted_map']['map_mean']
                   - G[p]['B1_own_map']['map_mean']) < 1e-6 for p in ids)
    print(f"  B3 preserves B1 map mean (1e-6)     : {shok}")
    print(f"  B2 blank exactly zero               : "
          f"{all(G[p]['B2_blank_map']['map_mean']==0 for p in ids)}")
    print(f"  protocol sha1 {m['protocol_sha1']}")
    print(f"  7B sources available: conflict {len(c7)}  2x2 {len(a7)}")
    print(f"  shared sources with 7B: {len(set(ids) & set(c7) & set(a7))}")

    print("\n" + "=" * 92)
    print("2. COLLAPSE CHECK (failure-mode screen)")
    for c in B:
        vs = [G[p][c]["final_verdict"] for p in ids]
        cnt = Counter(vs)
        top = cnt.most_common(1)[0]
        print(f"  {LAB[c]:<26} {dict(cnt)}   most common {top[0]} "
              f"{top[1]/len(vs):.3f}")
    allv = Counter(r["final_verdict"] for r in rows)
    const = max(allv.values()) / len(rows)
    print(f"  overall verdict mix {dict(allv)}  max share {const:.3f}")
    print(f"  near-constant output (>0.95 one label): {const > 0.95}")
    rl = [len(r["reason"] or "") for r in rows]
    print(f"  reason length mean {np.mean(rl):.0f} chars  min {min(rl)}  max {max(rl)}")

    fk = lambda g, p, c: int(g[p][c]["final_verdict"] == "fake")

    print("\n" + "=" * 92)
    print("3. RECALL BY CONDITION (32B)")
    S32 = {}
    print(f"  {'condition':<26}{'recall':>9}{'95% CI':>18}   7B counterpart")
    for c in B:
        o, lo, hi = boot([fk(G, p, c) for p in ids])
        S32[c] = o
        src, cc = MAP7[c]
        g7 = c7 if src == "conflict" else a7
        o7 = np.mean([fk(g7, p, cc) for p in ids if cc in g7[p]])
        print(f"  {LAB[c]:<26}{o:>9.3f} [{lo:.3f},{hi:.3f}]   {cc} {o7:.3f}")

    print("\n" + "=" * 92)
    print("4. EFFECTS (32B), each paired within source")
    E32, E7 = {}, {}
    for nm, hi_c, lo_c in EFFECTS:
        d = boot([fk(G, p, hi_c) - fk(G, p, lo_c) for p in ids])
        x01, x10, pv = mcnemar([fk(G, p, lo_c) for p in ids],
                               [fk(G, p, hi_c) for p in ids])
        E32[nm] = d
        print(f"  {nm:<24} {hi_c.split('_')[0]} - {lo_c.split('_')[0]}  "
              f"{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] {sig(d)}  p={pv:.4g}")

    print("\n" + "=" * 92)
    print("5. SAME EFFECTS ON 7B (same 184 sources)")
    for nm, hi_c, lo_c in EFFECTS:
        sh, ch = MAP7[hi_c]
        sl, cl = MAP7[lo_c]
        gh, gl = (c7 if sh == "conflict" else a7), (c7 if sl == "conflict" else a7)
        use = [p for p in ids if ch in gh[p] and cl in gl[p]]
        d = boot([fk(gh, p, ch) - fk(gl, p, cl) for p in use])
        E7[nm] = (d, use, gh, ch, gl, cl)
        print(f"  {nm:<24} {cl} -> {ch}  {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}] "
              f"{sig(d)}   n={len(use)}")

    print("\n" + "=" * 92)
    print("6. DELTA EFFECT = effect_32B - effect_7B  (PRIMARY capacity comparison)")
    print("  computed per source, bootstrapped paired across models")
    print("  a capacity difference requires the delta CI to exclude zero; comparing")
    print("  significance between models is NOT used as evidence")
    DE = {}
    for nm, hi_c, lo_c in EFFECTS:
        d7, use, gh, ch, gl, cl = E7[nm]
        per = [(fk(G, p, hi_c) - fk(G, p, lo_c)) - (fk(gh, p, ch) - fk(gl, p, cl))
               for p in use]
        dd = boot(per)
        DE[nm] = dd
        print(f"\n  {nm}")
        print(f"    32B {E32[nm][0]:+.3f} [{E32[nm][1]:+.3f},{E32[nm][2]:+.3f}]")
        print(f"    7B  {d7[0]:+.3f} [{d7[1]:+.3f},{d7[2]:+.3f}]")
        print(f"    DELTA {dd[0]:+.3f} [{dd[1]:+.3f},{dd[2]:+.3f}] {sig(dd)}  "
              f"n={len(use)}")

    print("\n" + "=" * 92)
    print("7. EXPLANATION-FAITHFULNESS: B1 vs B3")
    pats = {k: re.compile("|".join(re.escape(w) for w in v), re.I) for k, v in
            json.load(open(os.path.join(
                R, "trufor_conflict",
                "trufor_score_map_conflict_frozen.json")))[
                "explanation_audit"]["patterns"].items()}
    SPAT = re.compile(r"\b(top|bottom|left|right|center|centre|upper|lower|corner|"
                      r"region|area|portion|section)\b", re.I)
    print(f"  {'condition':<26}" + "".join(f"{k[:11]:>13}" for k in pats)
          + f"{'spatial':>10}")
    for c in B:
        line = f"  {LAB[c]:<26}"
        for k in pats:
            line += f"{np.mean([1 if pats[k].search(G[p][c]['reason'] or '') else 0 for p in ids]):>13.3f}"
        line += f"{np.mean([1 if SPAT.search(G[p][c]['reason'] or '') else 0 for p in ids]):>10.3f}"
        print(line)
    s1 = np.mean([1 if SPAT.search(G[p]["B1_own_map"]["reason"] or "") else 0
                  for p in ids])
    s3 = np.mean([1 if SPAT.search(G[p]["B3_shifted_map"]["reason"] or "") else 0
                  for p in ids])
    dsp = boot([(1 if SPAT.search(G[p]["B3_shifted_map"]["reason"] or "") else 0)
                - (1 if SPAT.search(G[p]["B1_own_map"]["reason"] or "") else 0)
                for p in ids])
    print(f"\n  spatial-language rate  B1 {s1:.3f}  B3 {s3:.3f}  "
          f"diff {dsp[0]:+.3f} [{dsp[1]:+.3f},{dsp[2]:+.3f}] {sig(dsp)}")
    print("  B3's map is deliberately misaligned with the image. Spatial language at")
    print("  a similar rate means the explanation's apparent grounding does not track")
    print("  whether the evidence actually corresponds to the image.")
    if not (dsp[1] > 0 or dsp[2] < 0):
        print("  -> FLAG: potential explanation-faithfulness failure")
    print("\n  example B1 / B3 reasons for the same source:")
    for p in ids[:2]:
        print(f"    [{p}]")
        print(f"      B1: {str(G[p]['B1_own_map']['reason'])[:150]}")
        print(f"      B3: {str(G[p]['B3_shifted_map']['reason'])[:150]}")

    print("\n" + "=" * 92)
    print("8. INTERPRETATION")
    mc, ce, sc = E32["map_cost"], E32["content_effect"], E32["spatial_correspondence"]
    print(f"  32B recalls  B0 {S32['B0_score_only']:.3f}  B1 {S32['B1_own_map']:.3f}"
          f"  B2 {S32['B2_blank_map']:.3f}  B3 {S32['B3_shifted_map']:.3f}")
    print(f"  7B  recalls  C10 0.755  F1 0.522  F2 0.266  F4 0.495")
    print()
    verdicts = []
    if const > 0.95:
        verdicts.append("FAILURE-MODE SHIFT: near-constant verdicts")
    if mc[2] < 0 and not (sc[1] > 0 or sc[2] < 0):
        verdicts.append("CAPACITY-INVARIANT FAILURE: B1 < B0 and B3 ~ B1")
    if sc[2] < 0 and DE["spatial_correspondence"][2] < 0:
        verdicts.append("CAPACITY-DEPENDENT SPATIAL GROUNDING: B3 < B1 and the delta "
                        "vs 7B excludes zero")
    if not (mc[1] > 0 or mc[2] < 0) and DE["map_cost"][1] > 0:
        verdicts.append("CAPACITY REDUCES MAP INTERFERENCE: B1 ~ B0 and the delta vs "
                        "7B excludes zero")
    if not verdicts:
        verdicts.append("no pre-registered pattern matched cleanly — report "
                        "descriptively")
    for v in verdicts:
        print(f"  -> {v}")
    print("\n  delta CIs decide capacity claims:")
    for nm in DE:
        print(f"    {nm:<24} {DE[nm][0]:+.3f} [{DE[nm][1]:+.3f},{DE[nm][2]:+.3f}] "
              f"{sig(DE[nm])}")

    print(f"\n9. RESOURCES  peaks {m['gpu_mem']}  device_map {m['device_map_resolved'][:60]}...")


if __name__ == "__main__":
    main()
