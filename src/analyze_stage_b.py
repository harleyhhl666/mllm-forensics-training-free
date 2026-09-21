"""Exploratory Stage-B analysis: Level 1 (all 124 Primary) + Level 2 (B2 subset).

Reporting order fixed: effect size -> coco_id cluster bootstrap 95% CI ->
sample-level McNemar p (supplementary, clustering caveat attached).

B2/B3 are post-hoc behavioral subsets defined on correct-cue Stage-A behaviour, so
within-B2 condition differences are NOT unbiased causal effects; Level 1 is the clean
comparison. This constraint is printed in the output, not just documented.
"""
import json, os, sys
from collections import defaultdict, Counter
from math import comb, gamma
import numpy as np, yaml

COND = ("fake_correct", "fake_notool", "fake_wrong")
LAB = {"fake_correct": "correct ELA", "fake_notool": "no tool",
       "fake_wrong": "wrong donor ELA"}
BANDS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.10), (0.10, 1.01)]
BNAMES = ["<1%", "1-3%", "3-10%", ">10%"]
NBOOT, SEED = 10000, 20260918


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
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


def cochran_q(mat):
    X = np.asarray(mat, float)
    n, k = X.shape
    Ri, Cj, N = X.sum(1), X.sum(0), X.sum()
    den = k * N - (Ri ** 2).sum()
    if den == 0:
        return float("nan"), k - 1, 1.0
    Q = (k - 1) * (k * (Cj ** 2).sum() - N ** 2) / den
    s, x = (k - 1) / 2.0, Q / 2.0
    tot, term = 0.0, 1.0 / s
    for i in range(500):
        tot += term
        term *= x / (s + i + 1)
        if term < 1e-16 * max(tot, 1e-300):
            break
    return float(Q), k - 1, float(max(0.0, min(1.0, 1.0 - tot * x ** s * np.exp(-x) / gamma(s))))


def cb_rate(cl, fn, nb=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    keys = sorted(cl)
    v = [np.mean([fn(r) for r in cl[k]]) for k in keys]
    w = [len(cl[k]) for k in keys]
    obs = np.average(v, weights=w)
    out = []
    for _ in range(nb):
        p = rng.integers(0, len(keys), len(keys))
        out.append(np.average([v[i] for i in p], weights=[w[i] for i in p]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def cb_diff(cl, fa, fb, nb=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    keys = sorted(cl)
    a = [np.mean([fa(r) for r in cl[k]]) for k in keys]
    b = [np.mean([fb(r) for r in cl[k]]) for k in keys]
    w = [len(cl[k]) for k in keys]
    obs = np.average(a, weights=w) - np.average(b, weights=w)
    out = []
    for _ in range(nb):
        p = rng.integers(0, len(keys), len(keys))
        ww = [w[i] for i in p]
        out.append(np.average([a[i] for i in p], weights=ww) -
                   np.average([b[i] for i in p], weights=ww))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    out_root = cfg["experiment"]["out_root"]
    btag = sys.argv[2] if len(sys.argv) > 2 else "validation_primary_stageB"
    atag = "validation_primary_stageA"
    frozen = json.load(open(os.path.join(out_root, "validation",
                                         "validation_primary_frozen.json")))
    cid = {s["pair_id"]: s["coco_id"] for s in frozen["samples"]}

    A = defaultdict(dict)
    for line in open(os.path.join(out_root, atag, "stage_a.jsonl")):
        d = json.loads(line)
        A[d["pair_id"]][d["condition"]] = d
    B = defaultdict(dict)
    for line in open(os.path.join(out_root, btag, "stage_b.jsonl")):
        d = json.loads(line)
        if not d.get("stochastic"):
            B[d["pair_id"]][d["condition"]] = d

    comp = sorted(p for p in B if len(B[p]) == 4 and len(A[p]) == 4)
    print("EXPLORATORY STAGE B — Primary set (post-hoc behavioral subsets)")
    print(f"complete samples: {len(comp)}/{frozen['n_samples']}   "
          f"clusters: {len({cid[p] for p in comp})}")
    print("NOTE: B2/B3 are defined from correct-cue Stage-A behaviour, so within-")
    print("subset condition contrasts are conditional, NOT unbiased causal effects.")
    print("Level 1 (all Primary) is the clean paired condition comparison.\n")

    pf = sum(1 for p in comp for c in B[p]
             if "json_parse_failed" in (B[p][c]["stage_b_notes"] or []))
    nv = sum(1 for p in comp for c in B[p] if not B[p][c]["final_verdict"])
    print("=" * 84)
    print("0. INTEGRITY")
    print(f"  Stage-B JSON parse failures : {pf}")
    print(f"  missing/unparsed verdicts   : {nv}")

    fake = lambda c: (lambda d: int(d[c]["final_verdict"] == "fake"))
    cl_all = defaultdict(list)
    for p in comp:
        cl_all[cid[p]].append(B[p])

    print("\n" + "=" * 84)
    print("LEVEL 1 — ALL PRIMARY SAMPLES: fake recall by condition")
    print(f"  {'condition':<18}{'recall':>8}  {'cluster 95% CI':<22}{'sample Wilson'}")
    rate = {}
    for c in COND:
        o, lo, hi = cb_rate(cl_all, fake(c))
        k = sum(int(B[p][c]["final_verdict"] == "fake") for p in comp)
        _, wl, wh = wilson(k, len(comp))
        rate[c] = o
        print(f"  {LAB[c]:<18}{o:>8.3f}  [{lo:+.3f},{hi:+.3f}]      "
              f"{k}/{len(comp)} [{wl:.3f},{wh:.3f}]")
    rr = sum(int(B[p]["real_own"]["final_verdict"] == "real") for p in comp)
    print(f"\n  paired-real specificity (verdict=real): {rr}/{len(comp)} = "
          f"{rr/len(comp):.3f}")

    mat = [[int(B[p][c]["final_verdict"] == "fake") for c in COND] for p in comp]
    Q, df, pq = cochran_q(mat)
    CAV = f"(sample-level; {len(comp)} samples / {len(cl_all)} clusters — not independent)"
    print(f"\n  Cochran's Q: Q={Q:.3f} df={df} p={pq:.4g}  {CAV}\n")
    for a, b in (("fake_correct", "fake_notool"), ("fake_correct", "fake_wrong"),
                 ("fake_wrong", "fake_notool")):
        d, lo, hi = cb_diff(cl_all, fake(a), fake(b))
        x01, x10, pv = mcnemar([int(B[p][a]["final_verdict"] == "fake") for p in comp],
                               [int(B[p][b]["final_verdict"] == "fake") for p in comp])
        print(f"  {LAB[a]} vs {LAB[b]}")
        print(f"    1. effect size    : {d:+.3f}  ({rate[a]:.3f} vs {rate[b]:.3f})")
        print(f"    2. cluster 95% CI : [{lo:+.3f},{hi:+.3f}]")
        print(f"    3. McNemar p      : {pv:.4g}  discordant {a}-only={x01} "
              f"{b}-only={x10}  {CAV}\n")

    # ---------------- Level 2
    b2 = [p for p in comp if A[p]["fake_correct"]["joint_correct"]]
    b3 = [p for p in b2 if not A[p]["fake_notool"]["mllm_vs_gt"]]
    print("=" * 84)
    print("LEVEL 2 — BEHAVIORALLY GROUNDED SUBSETS")
    for nm, S in (("B2 (primary)", b2), ("B3 (sensitivity)", b3)):
        ncl = len({cid[p] for p in S})
        if not S:
            print(f"\n  {nm}: empty")
            continue
        cl = defaultdict(list)
        for p in S:
            cl[cid[p]].append(B[p])
        kf = sum(int(B[p]["fake_correct"]["final_verdict"] == "fake") for p in S)
        kr = len(S) - kf
        of, lof, hif = cb_rate(cl, fake("fake_correct"))
        orl, lor, hir = cb_rate(cl, lambda d: int(d["fake_correct"]["final_verdict"] == "real"))
        _, wl, wh = wilson(kr, len(S))
        print(f"\n  {nm}: n={len(S)}  unique coco_ids={ncl}")
        print(f"    verdict counts (correct ELA): fake={kf}  real={kr}")
        print(f"    P(Fake | subset, correct)   : {of:.3f}  cluster CI "
              f"[{lof:+.3f},{hif:+.3f}]")
        print(f"    P(Real | subset, correct)   : {orl:.3f}  cluster CI "
              f"[{lor:+.3f},{hir:+.3f}]   sample [{wl:.3f},{wh:.3f}]")
        print(f"    ^ Grounding-Verdict Inconsistency rate")
        if nm.startswith("B2"):
            for c in ("fake_notool", "fake_wrong"):
                o2, l2, h2 = cb_rate(cl, fake(c))
                print(f"    (conditional, not causal) recall {LAB[c]:<16}: {o2:.3f} "
                      f"[{l2:+.3f},{h2:+.3f}]")

    # Q3: propagation -- B2 vs complement
    comp_b2 = [p for p in comp if p not in set(b2)]
    if comp_b2 and b2:
        cl2 = defaultdict(list)
        for p in b2:
            cl2[cid[p]].append(B[p])
        clc = defaultdict(list)
        for p in comp_b2:
            clc[cid[p]].append(B[p])
        o1, l1, h1 = cb_rate(cl2, fake("fake_correct"))
        o2, l2, h2 = cb_rate(clc, fake("fake_correct"))
        print("\n" + "=" * 84)
        print("Q3 — DOES LOCALIZATION SUCCESS PROPAGATE TO THE VERDICT?")
        print(f"  recall | B2 (grounded)   : {o1:.3f} [{l1:+.3f},{h1:+.3f}]  n={len(b2)}")
        print(f"  recall | not B2          : {o2:.3f} [{l2:+.3f},{h2:+.3f}]  n={len(comp_b2)}")
        print(f"  difference               : {o1-o2:+.3f}  (unpaired, different samples)")

    # Q4: enumerate inconsistency cases
    cases = [p for p in b2 if B[p]["fake_correct"]["final_verdict"] == "real"]
    print("\n" + "=" * 84)
    print(f"Q4 — GROUNDING–VERDICT INCONSISTENCY CASES: {len(cases)}")
    for p in cases[:12]:
        r = B[p]["fake_correct"]
        a = A[p]["fake_correct"]
        print(f"\n  {p}  (coco {cid[p]}, tamper {r['tamper_ratio']:.4f})")
        print(f"    Stage-A regions={a['predicted_cells']} gt={a['gt_cells']} "
              f"cue={a['cue_cells']}")
        print(f"    Stage-B reason: {r['reason'][:200]}")
    if len(cases) > 12:
        print(f"\n  ... {len(cases)} total")

    print("\n" + "=" * 84)
    print("TAMPER-RATIO STRATA (frozen bins; small bins descriptive only)")
    print(f"  {'band':<7}{'n':>4}{'recall_corr':>12}{'recall_notool':>14}"
          f"{'B2 n':>6}{'B2 P(real)':>11}")
    for (lo, hi), nm in zip(BANDS, BNAMES):
        sub = [p for p in comp if lo <= B[p]["fake_correct"]["tamper_ratio"] < hi]
        if not sub:
            continue
        s2 = [p for p in sub if p in set(b2)]
        pr = (np.mean([int(B[p]["fake_correct"]["final_verdict"] == "real") for p in s2])
              if s2 else float("nan"))
        print(f"  {nm:<7}{len(sub):>4}"
              f"{np.mean([int(B[p]['fake_correct']['final_verdict']=='fake') for p in sub]):>12.3f}"
              f"{np.mean([int(B[p]['fake_notool']['final_verdict']=='fake') for p in sub]):>14.3f}"
              f"{len(s2):>6}{pr:>11.3f}"
              + ("  DESCRIPTIVE" if len(sub) < 20 else ""))

    if cases:
        json.dump(sorted(cases), open(os.path.join(out_root, btag,
                                                   "reversal_candidates.json"), "w"))
        print(f"\nwrote reversal_candidates.json ({len(cases)} ids) for the "
              f"stochastic stability probe")


if __name__ == "__main__":
    main()
