"""Phase T1 analysis: TruFor -> MLLM Stage-A grounding.

Answers the five questions fixed before the run:
  1 does correct TruFor evidence raise fake GT localization (T1 vs T0)
  2 is T1 better than donor evidence (T1 vs T2)
  3 does the model follow the donor map under T2
  4 under T2, can it still hit the target GT from the image alone
  5 does real + evidence induce spurious localization

All intervals are source-level bootstrap (one sample per coco_id, so this is also
the cluster bootstrap). Effect size -> CI -> McNemar p, in that order.
"""
import json, os, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np, yaml

NBOOT, SEED = 10000, 20260918
FK = ("fake_T0_notool", "fake_T1_correct", "fake_T2_wrong")
RL = ("real_T0_notool", "real_T1_correct", "real_T2_wrong")
LAB = {"fake_T0_notool": "T0 no-tool", "fake_T1_correct": "T1 correct TruFor",
       "fake_T2_wrong": "T2 donor TruFor", "real_T0_notool": "T0 no-tool",
       "real_T1_correct": "T1 own TruFor", "real_T2_wrong": "T2 donor TruFor"}


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


def boot(vals, nb=NBOOT, seed=SEED):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    d = v[rng.integers(0, len(v), (nb, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    R = cfg["experiment"]["out_root"]
    tag = sys.argv[2] if len(sys.argv) > 2 else "trufor_mllm_stageA"
    F = json.load(open(os.path.join(R, "trufor_mllm", "trufor_mllm_frozen.json")))
    rows = [json.loads(l) for l in open(os.path.join(R, tag, "stage_a.jsonl"))]
    G = defaultdict(dict)
    for r in rows:
        G[r["sample_id"]][r["condition"]] = r
    ids = sorted(p for p in G if len(G[p]) == 6)

    print("PHASE T1 — TruFor -> MLLM STAGE-A GROUNDING")
    print(f"rows {len(rows)}  complete sources {len(ids)}  "
          f"bootstrap {NBOOT} seed {SEED}")

    print("\n" + "=" * 90)
    print("1. INTEGRITY")
    pf = sum(1 for r in rows if "parse_failed" in (r["notes"] or []))
    print(f"  parse failures            : {pf}/{len(rows)}")
    print(f"  per-condition counts      : {dict(Counter(r['condition'] for r in rows))}")
    print(f"  unique coco_id            : {len({G[p][FK[0]]['coco_id'] for p in ids})}")
    print(f"  n_samples == n_coco_id    : "
          f"{len({G[p][FK[0]]['coco_id'] for p in ids}) == len(ids)}")
    ph = {r["prompt_sha1"] for r in rows if r["evidence"] is not None}
    print(f"  distinct tool-prompt hashes: {len(ph)} "
          f"(differ only by the 3-decimal score, as frozen)")
    ni = {r["condition"]: r["n_images"] for r in rows}
    print(f"  images per condition      : {ni}")
    dn = sum(1 for p in ids
             if G[p][FK[0]]["coco_id"] ==
             G[G[p][FK[0]]["donor_sample_id"]][FK[0]]["coco_id"]
             if G[p][FK[0]]["donor_sample_id"] in G)
    print(f"  same-coco donors           : {dn}")

    # ---------------------------------------------------------------- fake
    print("\n" + "=" * 90)
    print("2. FAKE — GROUNDING RATES")
    print(f"  {'condition':<20}{'GT hit':>9}{'95% CI':>18}{'cue hit':>10}"
          f"{'joint':>9}{'n_pred':>9}")
    res = {}
    for c in FK:
        gh = [int(G[p][c]["gt_hit"]) for p in ids if G[p][c]["gt_hit"] is not None]
        o, lo, hi = boot(gh)
        ch = [G[p][c]["cue_hit"] for p in ids if G[p][c]["cue_hit"] is not None]
        jh = [G[p][c]["joint_hit"] for p in ids if G[p][c]["joint_hit"] is not None]
        npd = [G[p][c]["n_pred"] for p in ids]
        res[c] = dict(gt=o, lo=lo, hi=hi,
                      cue=(np.mean(ch) if ch else float("nan")),
                      joint=(np.mean(jh) if jh else float("nan")),
                      npred=np.mean(npd))
        print(f"  {LAB[c]:<20}{o:>9.3f}  [{lo:.3f},{hi:.3f}]   "
              f"{res[c]['cue']:>9.3f}{res[c]['joint']:>9.3f}{res[c]['npred']:>9.2f}")

    def contrast(c1, c0, field="gt_hit"):
        v = [int(G[p][c1][field]) - int(G[p][c0][field]) for p in ids
             if G[p][c1][field] is not None and G[p][c0][field] is not None]
        o, lo, hi = boot(v)
        a = [int(G[p][c0][field]) for p in ids if G[p][c0][field] is not None
             and G[p][c1][field] is not None]
        b = [int(G[p][c1][field]) for p in ids if G[p][c0][field] is not None
             and G[p][c1][field] is not None]
        x01, x10, pv = mcnemar(a, b)
        return o, lo, hi, pv, x01, x10

    print("\n  CONTRASTS on GT localization (effect -> CI -> McNemar p)")
    CC = [("fake_T1_correct", "fake_T0_notool", "T1 - T0"),
          ("fake_T1_correct", "fake_T2_wrong", "T1 - T2"),
          ("fake_T2_wrong", "fake_T0_notool", "T2 - T0")]
    store = {}
    for c1, c0, nm in CC:
        o, lo, hi, pv, x01, x10 = contrast(c1, c0)
        store[nm] = (o, lo, hi)
        sig = "EXCLUDES zero" if (lo > 0 or hi < 0) else "includes zero"
        print(f"    {nm:<10} {o:+.3f}  [{lo:+.3f},{hi:+.3f}] {sig}   "
              f"p={pv:.4g}  ({c0.split('_')[1]}-only={x01}, {c1.split('_')[1]}-only={x10})")

    # ------------------------------------------------- donor following (Q3/Q4)
    print("\n" + "=" * 90)
    print("3. Q3/Q4 — DONOR FOLLOWING vs IMAGE-BASED RECOVERY (fake, T2)")
    c2 = "fake_T2_wrong"
    c1 = "fake_T1_correct"
    # donor cue hit under T2 = followed the (wrong) map
    foll = [int(G[p][c2]["cue_hit"]) for p in ids if G[p][c2]["cue_hit"] is not None]
    o, lo, hi = boot(foll)
    print(f"  follows donor cue (T2 pred ∩ donor cue cells)   {o:.3f} [{lo:.3f},{hi:.3f}]")
    own = [int(G[p][c1]["cue_hit"]) for p in ids if G[p][c1]["cue_hit"] is not None]
    o2, l2, h2 = boot(own)
    print(f"  follows own cue   (T1 pred ∩ own   cue cells)   {o2:.3f} [{l2:.3f},{h2:.3f}]")
    d = boot([int(G[p][c1]["cue_hit"]) - int(G[p][c2]["cue_hit"]) for p in ids
              if G[p][c1]["cue_hit"] is not None and G[p][c2]["cue_hit"] is not None])
    print(f"  cue adherence T1 - T2                           {d[0]:+.3f} "
          f"[{d[1]:+.3f},{d[2]:+.3f}]")
    # Q4: under T2, still hits GT?
    still = [int(G[p][c2]["gt_hit"]) for p in ids if G[p][c2]["gt_hit"] is not None]
    o3, l3, h3 = boot(still)
    print(f"  still hits target GT under T2                   {o3:.3f} [{l3:.3f},{h3:.3f}]")
    # decompose T2 outcomes
    both = sum(1 for p in ids if G[p][c2]["gt_hit"] and G[p][c2]["cue_hit"])
    gtonly = sum(1 for p in ids if G[p][c2]["gt_hit"] and not G[p][c2]["cue_hit"])
    cueonly = sum(1 for p in ids if not G[p][c2]["gt_hit"] and G[p][c2]["cue_hit"])
    neither = sum(1 for p in ids if not G[p][c2]["gt_hit"] and not G[p][c2]["cue_hit"])
    print(f"  T2 outcome split: GT+cue {both}  GT only {gtonly}  "
          f"cue only {cueonly}  neither {neither}  (n={len(ids)})")
    print("  NOTE: donor and GT cells can coincide by chance; 'cue only' is the")
    print("  cleanest evidence of being pulled to a wrong location.")

    # ---------------------------------------------------------------- real
    print("\n" + "=" * 90)
    print("4. Q5 — REAL IMAGES (descriptive: spurious localization)")
    print(f"  {'condition':<20}{'n_pred mean':>13}{'95% CI':>18}"
          f"{'reports any':>13}{'cue hit':>10}{'evidence_used':>15}")
    for c in RL:
        npd = [G[p][c]["n_pred"] for p in ids]
        o, lo, hi = boot(npd)
        anyr = np.mean([1 if G[p][c]["n_pred"] > 0 else 0 for p in ids])
        ch = [G[p][c]["cue_hit"] for p in ids if G[p][c]["cue_hit"] is not None]
        eu = [G[p][c]["stage_a"]["evidence_used"] for p in ids
              if G[p][c]["stage_a"]["evidence_used"] is not None]
        print(f"  {LAB[c]:<20}{o:>13.2f}  [{lo:.2f},{hi:.2f}]   {anyr:>12.3f}"
              f"{(np.mean(ch) if ch else float('nan')):>10.3f}"
              f"{(np.mean(eu) if eu else float('nan')):>15.3f}")
    dn = boot([G[p]["real_T1_correct"]["n_pred"] - G[p]["real_T0_notool"]["n_pred"]
               for p in ids])
    print(f"\n  n_pred change real T1 - T0  {dn[0]:+.3f} [{dn[1]:+.3f},{dn[2]:+.3f}]")
    dn2 = boot([G[p]["real_T2_wrong"]["n_pred"] - G[p]["real_T0_notool"]["n_pred"]
                for p in ids])
    print(f"  n_pred change real T2 - T0  {dn2[0]:+.3f} [{dn2[1]:+.3f},{dn2[2]:+.3f}]")
    # spatial pattern
    print("\n  most frequent predicted cells (real):")
    for c in RL:
        cnt = Counter(x for p in ids for x in G[p][c]["predicted_cells"])
        top = ", ".join(f"{k} {v}" for k, v in cnt.most_common(4))
        print(f"    {LAB[c]:<20}{top}")

    print("\n" + "=" * 90)
    print("5. SCORE INTERPRETATION (did the model read the score field?)")
    for c in FK + RL:
        cnt = Counter(G[p][c]["stage_a"]["tool_score_interpretation"] for p in ids)
        print(f"  {c:<20}{dict(cnt)}")

    print("\n" + "=" * 90)
    print("6. STAGE-A VERDICT ON WHETHER TO PROCEED")
    t10 = store["T1 - T0"]
    t12 = store["T1 - T2"]
    t20 = store["T2 - T0"]
    print(f"  T1 - T0 {t10[0]:+.3f} [{t10[1]:+.3f},{t10[2]:+.3f}]")
    print(f"  T1 - T2 {t12[0]:+.3f} [{t12[1]:+.3f},{t12[2]:+.3f}]")
    print(f"  T2 - T0 {t20[0]:+.3f} [{t20[1]:+.3f},{t20[2]:+.3f}]")
    print()
    sep = t12[1] > 0
    tool_up = t10[1] > 0
    donor_up = t20[1] > 0
    if sep:
        print("  -> T1 > T2 with CI excluding zero: the model exploits the")
        print("     correspondence of a high-quality spatial evidence package.")
        print("     Proceeding to Stage B is justified.")
    elif tool_up and donor_up:
        print("  -> T1 and T2 both beat no-tool but do NOT separate from each other:")
        print("     record as a GENERIC TOOL-GUIDANCE effect, not successful")
        print("     grounding. Do not call this evidence-correctness sensitivity.")
    elif not tool_up and not donor_up:
        print("  -> neither tool condition beats no-tool: no grounding benefit.")
    else:
        print("  -> mixed: report descriptively, force no conclusion.")
    if not sep:
        print("\n  PER THE PROTOCOL: T1 ~ T2 means Stage A responds mainly to the")
        print("  presence of forensic context rather than to evidence correctness.")
        print("  PAUSE and consult before running Stage B.")

    mp = os.path.join(R, tag, "run_meta.json")
    if os.path.exists(mp):
        m = json.load(open(mp))
        print("\n" + "=" * 90)
        print(f"7. RUNTIME  {m['n_inferences']} inf  {m['minutes']} min  "
              f"mean {m['mean_seconds']}s  peak {m['peak_GB']}")


if __name__ == "__main__":
    main()
