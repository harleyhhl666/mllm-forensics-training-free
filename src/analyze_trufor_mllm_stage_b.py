"""Phase T2 analysis: TruFor -> MLLM Stage-B verdicts.

Reports recall / FPR / Youden J for T0/T1/T2, the three paired contrasts on each,
and an explicit standalone-TruFor vs MLLM-integrated comparison answering whether
integration preserves, reduces, or destroys the tool's existing discrimination.
"""
import json, os, sys
from collections import Counter, defaultdict
from math import comb
import numpy as np, yaml

NBOOT, SEED = 10000, 20260918
T = ("T0_notool", "T1_correct", "T2_wrong")
LAB = {"T0_notool": "T0 no-tool", "T1_correct": "T1 correct TruFor",
       "T2_wrong": "T2 donor TruFor"}
# TruFor standalone, from the frozen feasibility run (200 sources, same domain)
STANDALONE = dict(auroc=0.9845, ci=[0.9700, 0.9951], tpr05=0.955, tpr10=0.965,
                  n_sources=200)


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


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    a = np.concatenate([pos, neg])
    r = np.empty(len(a))
    o = np.argsort(a, kind="mergesort")
    sa = a[o]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        r[o[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    R = cfg["experiment"]["out_root"]
    tag = sys.argv[2] if len(sys.argv) > 2 else "trufor_mllm_stageB"
    F = json.load(open(os.path.join(R, "trufor_mllm", "trufor_mllm_frozen.json")))
    rows = [json.loads(l) for l in open(os.path.join(R, tag, "stage_b.jsonl"))]
    G = defaultdict(dict)
    for r in rows:
        G[r["sample_id"]][r["condition"]] = r
    ids = sorted(p for p in G if len(G[p]) == 6)

    print("PHASE T2 — TruFor -> MLLM STAGE-B FINAL VERDICT")
    print(f"rows {len(rows)}  complete sources {len(ids)}  bootstrap {NBOOT} "
          f"seed {SEED}")

    print("\n" + "=" * 90)
    print("1. INTEGRITY")
    pf = sum(1 for r in rows if "json_parse_failed" in (r["notes"] or []))
    mv = sum(1 for r in rows if not r["final_verdict"])
    print(f"  parse failures         : {pf}/{len(rows)}")
    print(f"  missing verdicts       : {mv}/{len(rows)}")
    print(f"  per-condition counts   : {dict(Counter(r['condition'] for r in rows))}")
    print(f"  unique coco_id         : {len({G[p]['fake_T0_notool']['coco_id'] for p in ids})}")
    print(f"  n_samples == n_coco_id : "
          f"{len({G[p]['fake_T0_notool']['coco_id'] for p in ids}) == len(ids)}")
    m = json.load(open(os.path.join(R, tag, "run_meta.json")))
    print(f"  protocol sha1          : {m['protocol_sha1']}")
    print(f"  Stage-A source         : {os.path.basename(m['stage_a_source'])} (read-only)")

    fake = lambda p, k: int(G[p][f"fake_{k}"]["final_verdict"] == "fake")
    real = lambda p, k: int(G[p][f"real_{k}"]["final_verdict"] == "fake")

    print("\n" + "=" * 90)
    print("2. PRIMARY — RECALL / FPR / SPECIFICITY / YOUDEN J")
    print(f"  {'condition':<20}{'recall':>8}{'95% CI':>17}{'FPR':>8}{'95% CI':>17}"
          f"{'spec':>8}{'J':>8}{'J 95% CI':>18}")
    S = {}
    for k in T:
        rc, rl, rh = boot([fake(p, k) for p in ids])
        fp, fl, fh = boot([real(p, k) for p in ids])
        j, jl, jh = boot([fake(p, k) - real(p, k) for p in ids])
        S[k] = dict(recall=rc, fpr=fp, j=j, spec=1 - fp)
        print(f"  {LAB[k]:<20}{rc:>8.3f} [{rl:.3f},{rh:.3f}] {fp:>8.3f} "
              f"[{fl:.3f},{fh:.3f}] {1-fp:>8.3f}{j:>8.3f} [{jl:+.3f},{jh:+.3f}]")

    def contrast(fn, k1, k0, nm):
        d, lo, hi = boot([fn(p, k1) - fn(p, k0) for p in ids])
        _, _, pv = mcnemar([fn(p, k0) for p in ids], [fn(p, k1) for p in ids])
        sig = "EXCLUDES zero" if (lo > 0 or hi < 0) else "includes zero"
        print(f"    {nm:<12}{d:+.3f}  [{lo:+.3f},{hi:+.3f}] {sig}   p={pv:.4g}")
        return d, lo, hi

    print("\n  FAKE RECALL contrasts")
    fr = {}
    for k1, k0, nm in (("T1_correct", "T0_notool", "T1 - T0"),
                       ("T2_wrong", "T0_notool", "T2 - T0"),
                       ("T1_correct", "T2_wrong", "T1 - T2")):
        fr[nm] = contrast(fake, k1, k0, nm)
    print("\n  REAL FPR contrasts")
    rr = {}
    for k1, k0, nm in (("T1_correct", "T0_notool", "T1 - T0"),
                       ("T2_wrong", "T0_notool", "T2 - T0"),
                       ("T1_correct", "T2_wrong", "T1 - T2")):
        rr[nm] = contrast(real, k1, k0, nm)
    print("\n  YOUDEN J contrasts")
    jr = {}
    for k1, k0, nm in (("T1_correct", "T0_notool", "T1 - T0"),
                       ("T2_wrong", "T0_notool", "T2 - T0"),
                       ("T1_correct", "T2_wrong", "T1 - T2")):
        d, lo, hi = boot([(fake(p, k1) - real(p, k1)) - (fake(p, k0) - real(p, k0))
                          for p in ids])
        sig = "EXCLUDES zero" if (lo > 0 or hi < 0) else "includes zero"
        print(f"    {nm:<12}{d:+.3f}  [{lo:+.3f},{hi:+.3f}] {sig}")
        jr[nm] = (d, lo, hi)

    print("\n" + "=" * 90)
    print("3. STANDALONE TRUFOR vs MLLM-INTEGRATED")
    print(f"  TruFor standalone (frozen feasibility, {STANDALONE['n_sources']} sources,")
    print(f"    same sd2-sp domain, threshold-free):")
    print(f"      AUROC      {STANDALONE['auroc']:.4f} "
          f"CI [{STANDALONE['ci'][0]:.4f},{STANDALONE['ci'][1]:.4f}]")
    print(f"      TPR@5%FPR  {STANDALONE['tpr05']:.3f}")
    print(f"      TPR@10%FPR {STANDALONE['tpr10']:.3f}")
    print(f"      implied J at the 5% operating point  "
          f"{STANDALONE['tpr05']-0.05:+.3f}")
    print(f"      implied J at the 10% operating point "
          f"{STANDALONE['tpr10']-0.10:+.3f}")
    print(f"\n  MLLM binary verdicts on this set ({len(ids)} sources):")
    for k in T:
        print(f"      {LAB[k]:<20} recall {S[k]['recall']:.3f}  "
              f"FPR {S[k]['fpr']:.3f}  J {S[k]['j']:+.3f}")
    best = max(T, key=lambda k: S[k]["j"])
    dj = S[best]["j"] - (STANDALONE["tpr05"] - 0.05)
    print(f"\n  best MLLM condition by J: {LAB[best]}  J {S[best]['j']:+.3f}")
    print(f"  J difference vs TruFor standalone @5% FPR: {dj:+.3f}")
    print("  NOTE: the MLLM emits a binary verdict, TruFor a continuous score, so")
    print("  this compares the MLLM's single operating point against TruFor's ROC")
    print("  points. It is a like-for-like comparison of DECISION quality, not of")
    print("  AUROC. Sample sets differ (184 vs 200 sources, same domain and")
    print("  generator) -> same-domain comparison, not exact paired.")
    if S[best]["j"] < (STANDALONE["tpr05"] - 0.05) - 0.10:
        print("\n  -> INTEGRATION REDUCES the discrimination already present in the")
        print("     external tool.")
    elif S[best]["j"] >= (STANDALONE["tpr05"] - 0.05) - 0.05:
        print("\n  -> INTEGRATION PRESERVES most of the tool's discrimination.")
    else:
        print("\n  -> integration partially reduces the tool's discrimination.")

    print("\n" + "=" * 90)
    print("4. DOES THE STAGE-A GROUNDING CARRY INTO THE VERDICT?")
    for k in ("T1_correct", "T2_wrong"):
        hit = [p for p in ids if G[p][f"fake_{k}"]["stage_a_gt_hit"]]
        mis = [p for p in ids if G[p][f"fake_{k}"]["stage_a_gt_hit"] is False]
        rh = np.mean([fake(p, k) for p in hit]) if hit else float("nan")
        rm = np.mean([fake(p, k) for p in mis]) if mis else float("nan")
        print(f"  {LAB[k]:<20} recall | Stage-A hit GT  {rh:.3f} (n={len(hit)})   "
              f"| Stage-A missed  {rm:.3f} (n={len(mis)})   diff {rh-rm:+.3f}")

    print("\n" + "=" * 90)
    print("5. VERDICT BY TAMPER RATIO (descriptive)")
    q = np.quantile([G[p]["fake_T0_notool"]["tamper_ratio"] for p in ids],
                    [.25, .5, .75])
    print(f"  quartile cuts {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}")
    print(f"  {'band':<6}{'n':>4}" + "".join(f"{LAB[k].split()[0]+' rec':>12}" for k in T)
          + "".join(f"{LAB[k].split()[0]+' fpr':>12}" for k in T))
    for nm, sel in (("q1", lambda t: t <= q[0]), ("q2", lambda t: q[0] < t <= q[1]),
                    ("q3", lambda t: q[1] < t <= q[2]), ("q4", lambda t: t > q[2])):
        sub = [p for p in ids if sel(G[p]["fake_T0_notool"]["tamper_ratio"])]
        if not sub:
            continue
        line = f"  {nm:<6}{len(sub):>4}"
        for k in T:
            line += f"{np.mean([fake(p,k) for p in sub]):>12.3f}"
        for k in T:
            line += f"{np.mean([real(p,k) for p in sub]):>12.3f}"
        print(line)

    print("\n" + "=" * 90)
    print("6. PRE-REGISTERED CASE JUDGEMENT")
    up = lambda t: t[1] > 0
    dn = lambda t: t[2] < 0
    r10, r20, r12 = fr["T1 - T0"], fr["T2 - T0"], fr["T1 - T2"]
    f10, f20, f12 = rr["T1 - T0"], rr["T2 - T0"], rr["T1 - T2"]
    j10 = jr["T1 - T0"]
    print(f"  recall  T1-T0 {r10[0]:+.3f} {'sig' if up(r10) else 'ns'}   "
          f"T2-T0 {r20[0]:+.3f} {'sig' if up(r20) else 'ns'}   "
          f"T1-T2 {r12[0]:+.3f} {'sig' if up(r12) else 'ns'}")
    print(f"  FPR     T1-T0 {f10[0]:+.3f} {'sig' if up(f10) else 'ns'}   "
          f"T2-T0 {f20[0]:+.3f} {'sig' if up(f20) else 'ns'}   "
          f"T1-T2 {f12[0]:+.3f} {'sig' if up(f12) else 'ns'}")
    print(f"  J       T1-T0 {j10[0]:+.3f} {'sig' if (j10[1]>0 or j10[2]<0) else 'ns'}")
    print()
    both_fake_up = up(r10) and up(r20)
    fpr_up = up(f10) or up(f20)
    t1_beats_t2 = up(r12)
    if both_fake_up and fpr_up and not t1_beats_t2:
        print("  -> CASE B: forensic-context bias persists. Both correct and donor")
        print("     evidence shift the verdict toward Fake and real FPR rises, with")
        print("     no reliable correct-vs-donor separation — the MLLM responds to the")
        print("     presence/status of forensic evidence rather than to its")
        print("     correctness, even though the tool itself is highly discriminative.")
    elif up(r10) and t1_beats_t2 and j10[1] > 0:
        print("  -> CASE C: stronger forensic evidence changes MLLM integration")
        print("     behaviour qualitatively; T1 improves recall and J with a real")
        print("     correct-vs-donor distinction.")
    elif S[best]["j"] < (STANDALONE["tpr05"] - 0.05) - 0.10:
        print("  -> CASE D: the MLLM reasoning layer degrades an already strong")
        print("     forensic detector. Reported as-is; no prompt tuning.")
    else:
        print("  -> mixed; report descriptively without forcing a case.")

    print(f"\n7. RUNTIME  {m['n_inferences']} inf  {m['minutes']} min  "
          f"mean {m['mean_seconds']}s  peak {m['peak_GB']}")


if __name__ == "__main__":
    main()
