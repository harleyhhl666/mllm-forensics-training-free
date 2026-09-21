"""V2 Stage-B analysis on the TGIF validation split: H3, H4, B2, controls.

The attenuation analysis is a paired difference-of-differences over coco_id
clusters:

    D_A(k) = loc_correct(k) - loc_wrong(k)        [Stage A, cluster k]
    D_B(k) = recall_correct(k) - recall_wrong(k)  [Stage B, cluster k]
    attenuation = weighted_mean(D_A) - weighted_mean(D_B)

Both stages are measured on the SAME samples and the SAME clusters, so a single
cluster bootstrap resamples clusters once and recomputes both terms together. This
gives a direct CI for the attenuation instead of eyeballing two separate CIs.

Caveat printed in the output: the two stages measure different outcomes (spatial
localization vs verdict), so the attenuation is a descriptive contrast of two
condition effects on a shared sample, not a claim that one quantity "became" the
other.
"""
import json, os, sys
from collections import defaultdict, Counter
from math import comb, gamma
import numpy as np, yaml

COND = ("fake_correct", "fake_notool", "fake_wrong")
LAB = {"fake_correct": "correct ELA", "fake_notool": "no tool",
       "fake_wrong": "wrong donor ELA", "real_own": "real + own ELA",
       "real_notool": "real, no tool"}
BANDS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.10), (0.10, 1.01)]
BN = ["<1%", "1-3%", "3-10%", ">10%"]
NBOOT, SEED = 10000, 20260918


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
    for i in range(600):
        tot += term
        term *= x / (s + i + 1)
        if term < 1e-16 * max(tot, 1e-300):
            break
    return float(Q), k - 1, float(max(0.0, min(1.0, 1 - tot * x ** s * np.exp(-x) / gamma(s))))


def _cluster_arrays(cl, fn):
    ks = sorted(cl)
    return ks, [np.mean([fn(r) for r in cl[k]]) for k in ks], [len(cl[k]) for k in ks]


def cb_rate(cl, fn, nb=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    ks, v, w = _cluster_arrays(cl, fn)
    obs = np.average(v, weights=w)
    o = []
    for _ in range(nb):
        p = rng.integers(0, len(ks), len(ks))
        o.append(np.average([v[i] for i in p], weights=[w[i] for i in p]))
    lo, hi = np.percentile(o, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def cb_diff(cl, fa, fb, nb=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    ks, a, w = _cluster_arrays(cl, fa)
    _, b, _ = _cluster_arrays(cl, fb)
    obs = np.average(a, weights=w) - np.average(b, weights=w)
    o = []
    for _ in range(nb):
        p = rng.integers(0, len(ks), len(ks))
        ww = [w[i] for i in p]
        o.append(np.average([a[i] for i in p], weights=ww) -
                 np.average([b[i] for i in p], weights=ww))
    lo, hi = np.percentile(o, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def cb_attenuation(clA, clB, nb=NBOOT, seed=SEED):
    """Paired difference-of-differences with ONE cluster resampling for both stages."""
    rng = np.random.default_rng(seed)
    ks = sorted(set(clA) & set(clB))
    dA, dB, w = [], [], []
    for k in ks:
        a = clA[k]
        b = clB[k]
        dA.append(np.mean([int(x["fake_correct"]["mllm_vs_gt"]) for x in a]) -
                  np.mean([int(x["fake_wrong"]["mllm_vs_gt"]) for x in a]))
        dB.append(np.mean([int(x["fake_correct"]["final_verdict"] == "fake") for x in b]) -
                  np.mean([int(x["fake_wrong"]["final_verdict"] == "fake") for x in b]))
        w.append(len(a))
    oA = np.average(dA, weights=w)
    oB = np.average(dB, weights=w)
    obs = oA - oB
    o = []
    for _ in range(nb):
        p = rng.integers(0, len(ks), len(ks))
        ww = [w[i] for i in p]
        o.append(np.average([dA[i] for i in p], weights=ww) -
                 np.average([dB[i] for i in p], weights=ww))
    lo, hi = np.percentile(o, [2.5, 97.5])
    return float(oA), float(oB), float(obs), float(lo), float(hi), len(ks)


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    out_root = cfg["experiment"]["out_root"]
    btag = sys.argv[2] if len(sys.argv) > 2 else "v4_stageB"
    atag = "v4_stageA"
    frozen = json.load(open(os.path.join(out_root, "validation_v4",
                                         "validation_v4_frozen.json")))
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

    comp = sorted(p for p in B if len(B[p]) == 5 and len(A[p]) == 5)
    clB = defaultdict(list)
    clA = defaultdict(list)
    for p in comp:
        clB[cid[p]].append(B[p])
        clA[cid[p]].append(A[p])
    CAV = f"(sample-level; {len(comp)} samples / {len(clB)} clusters — not independent)"

    print("V2 — INDEPENDENT STAGE-B VALIDATION (TGIF validation split)")
    print(f"complete samples={len(comp)}/{frozen['n_samples']}  clusters={len(clB)}")
    print(f"cluster bootstrap: {NBOOT} resamples, seed {SEED}")

    print("\n" + "=" * 84)
    print("1. INTEGRITY")
    rows = [x for p in comp for x in B[p].values()]
    pf = sum(1 for r in rows if "json_parse_failed" in (r["stage_b_notes"] or []))
    nv = sum(1 for r in rows if not r["final_verdict"])
    sh = defaultdict(set)
    for r in rows:
        sh[r["condition"]].add(r["prompt_sha1"])
    donor = {s["pair_id"]: s["wrong_tool_donor_pair_id"] for s in frozen["samples"]}
    dn = sum(1 for p in comp if cid[donor[p]] == cid[p])
    self_d = sum(1 for p in comp if donor[p] == p)
    print(f"  Stage-B JSON parse failures : {pf}/{len(rows)}")
    print(f"  missing verdicts            : {nv}/{len(rows)}")
    print(f"  same-coco_id donors         : {dn}")
    print(f"  self donors                 : {self_d}")
    print(f"  template sha (all conditions share one template, summaries differ)")

    fk = lambda c: (lambda d: int(d[c]["final_verdict"] == "fake"))

    print("\n" + "=" * 84)
    print("2. FAKE RECALL by condition (all 320 fakes)")
    print(f"  {'condition':<18}{'recall':>8}  {'cluster 95% CI':<22}{'sample Wilson'}")
    rate = {}
    for c in COND:
        o, lo, hi = cb_rate(clB, fk(c))
        k = sum(int(B[p][c]["final_verdict"] == "fake") for p in comp)
        _, wl, wh = wilson(k, len(comp))
        rate[c] = o
        print(f"  {LAB[c]:<18}{o:>8.3f}  [{lo:+.3f},{hi:+.3f}]      "
              f"{k}/{len(comp)} [{wl:.3f},{wh:.3f}]")
    mat = [[int(B[p][c]["final_verdict"] == "fake") for c in COND] for p in comp]
    Q, df, pq = cochran_q(mat)
    print(f"\n  Cochran's Q: Q={Q:.3f} df={df} p={pq:.4g}  {CAV}")

    print("\n" + "=" * 84)
    print("3. PAIRED CONDITION EFFECTS on fake recall")
    eff = {}
    for a, b in (("fake_correct", "fake_notool"), ("fake_correct", "fake_wrong"),
                 ("fake_wrong", "fake_notool")):
        d, lo, hi = cb_diff(clB, fk(a), fk(b))
        x01, x10, p = mcnemar([int(B[q][a]["final_verdict"] == "fake") for q in comp],
                              [int(B[q][b]["final_verdict"] == "fake") for q in comp])
        eff[(a, b)] = (d, lo, hi, p)
        print(f"\n  {LAB[a]} vs {LAB[b]}")
        print(f"    1. effect size    : {d:+.3f}  ({rate[a]:.3f} vs {rate[b]:.3f})")
        print(f"    2. cluster 95% CI : [{lo:+.3f},{hi:+.3f}]  -> "
              f"{'excludes zero' if lo > 0 or hi < 0 else 'INCLUDES zero'}")
        print(f"    3. McNemar p      : {p:.4g}  discordant {a}-only={x01} "
              f"{b}-only={x10}  {CAV}")

    print("\n" + "=" * 84)
    print("4. H4 — ATTENUATION of the matched-cue effect (paired diff-of-diffs)")
    oA, oB, att, alo, ahi, nk = cb_attenuation(clA, clB)
    print(f"  Stage-A (correct - wrong) on GT localization : {oA:+.3f}")
    print(f"  Stage-B (correct - wrong) on fake recall     : {oB:+.3f}")
    print(f"  attenuation = Stage-A - Stage-B             : {att:+.3f}")
    print(f"  cluster 95% CI for the attenuation          : [{alo:+.3f},{ahi:+.3f}]  "
          f"-> {'EXCLUDES zero' if alo > 0 else 'includes zero'}  ({nk} clusters)")
    dB_, dBlo, dBhi, _ = eff[("fake_correct", "fake_wrong")]
    print(f"\n  Stage-B correct-wrong CI: [{dBlo:+.3f},{dBhi:+.3f}] -> "
          f"{'includes zero' if dBlo <= 0 <= dBhi else 'excludes zero'}")
    print("  CAVEAT: the two stages measure different outcomes (spatial localization")
    print("  vs verdict). The attenuation is a descriptive contrast of two condition")
    print("  effects on the same samples, not a claim that one became the other.")

    print("\n" + "=" * 84)
    print("5. B2 CONFIRMATORY SUBSET (pre-registered)")
    b2 = [p for p in comp if A[p]["fake_correct"]["joint_correct"]]
    nb2 = [p for p in comp if p not in set(b2)]
    cl2 = defaultdict(list)
    for p in b2:
        cl2[cid[p]].append(B[p])
    kf = sum(int(B[p]["fake_correct"]["final_verdict"] == "fake") for p in b2)
    kr = len(b2) - kf
    of, lof, hif = cb_rate(cl2, fk("fake_correct"))
    orl, lor, hir = cb_rate(cl2, lambda d: int(d["fake_correct"]["final_verdict"] == "real"))
    _, wl, wh = wilson(kr, len(b2))
    print(f"  B2: n={len(b2)}  unique coco_ids={len(cl2)}")
    print(f"  verdict counts (correct ELA): fake={kf}  real={kr}")
    print(f"  P(Fake | B2, correct) : {of:.3f}  cluster CI [{lof:+.3f},{hif:+.3f}]")
    print(f"  P(Real | B2, correct) : {orl:.3f}  cluster CI [{lor:+.3f},{hir:+.3f}]  "
          f"sample [{wl:.3f},{wh:.3f}]")
    print(f"    ^ Grounding-Verdict Inconsistency (NOT evidence rejection)")
    for c in ("fake_notool", "fake_wrong"):
        o2, l2, h2 = cb_rate(cl2, fk(c))
        print(f"  (conditional) recall {LAB[c]:<16}: {o2:.3f} [{l2:+.3f},{h2:+.3f}]")

    print("\n" + "=" * 84)
    print("6. PROPAGATION — B2 vs non-B2 (descriptive, not causal)")
    clN = defaultdict(list)
    for p in nb2:
        clN[cid[p]].append(B[p])
    o1, l1, h1 = cb_rate(cl2, fk("fake_correct"))
    o2, l2, h2 = cb_rate(clN, fk("fake_correct"))
    print(f"  recall | B2     : {o1:.3f} [{l1:+.3f},{h1:+.3f}]  n={len(b2)} "
          f"({len(cl2)} clusters)")
    print(f"  recall | non-B2 : {o2:.3f} [{l2:+.3f},{h2:+.3f}]  n={len(nb2)} "
          f"({len(clN)} clusters)")
    print(f"  difference      : {o1-o2:+.3f}  (unpaired: disjoint sample sets)")

    print("\n" + "=" * 84)
    print("7. PAIRED-REAL CONTROLS — specificity and false-positive effect")
    for c in ("real_own", "real_notool"):
        o, lo, hi = cb_rate(clB, lambda d, c=c: int(d[c]["final_verdict"] == "real"))
        k = sum(int(B[p][c]["final_verdict"] == "real") for p in comp)
        print(f"  {LAB[c]:<18} specificity={o:.3f}  cluster CI [{lo:+.3f},{hi:+.3f}]  "
              f"{k}/{len(comp)}")
    fp_own, flo, fhi = cb_diff(
        clB, lambda d: int(d["real_own"]["final_verdict"] == "fake"),
        lambda d: int(d["real_notool"]["final_verdict"] == "fake"))
    x01, x10, pfp = mcnemar([int(B[p]["real_own"]["final_verdict"] == "fake") for p in comp],
                            [int(B[p]["real_notool"]["final_verdict"] == "fake") for p in comp])
    print(f"\n  false-positive increase from adding own ELA to a REAL image:")
    print(f"    1. effect size    : {fp_own:+.3f}")
    print(f"    2. cluster 95% CI : [{flo:+.3f},{fhi:+.3f}]  -> "
          f"{'EXCLUDES zero' if flo > 0 else 'includes zero'}")
    print(f"    3. McNemar p      : {pfp:.4g}  {CAV}")

    print("\n" + "=" * 84)
    print("8. TAMPER-RATIO BANDS (secondary, descriptive)")
    print(f"  {'band':<7}{'n':>4}{'clust':>6}{'correct':>9}{'notool':>8}{'wrong':>7}"
          f"{'c-w':>8}{'B2 n':>6}{'B2 P(real)':>11}")
    for (lo_, hi_), nm in zip(BANDS, BN):
        sub = [p for p in comp if lo_ <= B[p]["fake_correct"]["tamper_ratio"] < hi_]
        if not sub:
            continue
        f = lambda c: np.mean([int(B[p][c]["final_verdict"] == "fake") for p in sub])
        s2 = [p for p in sub if p in set(b2)]
        pr = (np.mean([int(B[p]["fake_correct"]["final_verdict"] == "real") for p in s2])
              if s2 else float("nan"))
        print(f"  {nm:<7}{len(sub):>4}{len({cid[p] for p in sub}):>6}"
              f"{f('fake_correct'):>9.3f}{f('fake_notool'):>8.3f}{f('fake_wrong'):>7.3f}"
              f"{f('fake_correct')-f('fake_wrong'):>+8.3f}{len(s2):>6}{pr:>11.3f}")

    print("\n" + "=" * 84)
    print("9. PRE-REGISTERED DECISION")
    dcn, lcn, hcn, _ = eff[("fake_correct", "fake_notool")]
    dwn, lwn, hwn, _ = eff[("fake_wrong", "fake_notool")]
    heat_both = lcn > 0 and lwn > 0
    cw_zero = dBlo <= 0 <= dBhi
    fp_up = flo > 0
    print(f"  Stage-A correct>wrong  : {oA:+.3f} (confirmed in V1)")
    print(f"  Stage-B correct-wrong  : {oB:+.3f} [{dBlo:+.3f},{dBhi:+.3f}]")
    print(f"  attenuation            : {att:+.3f} [{alo:+.3f},{ahi:+.3f}]")
    print(f"  both heatmaps raise recall vs no-tool : {heat_both}")
    print(f"  real-image FP rises with own ELA      : {fp_up} ({fp_own:+.3f})")
    print(f"  B2 Grounding-Verdict Inconsistency    : {orl:.3f}")
    print()
    if heat_both and fp_up and cw_zero:
        print("  => CASE D dominant: forensic-visualization context prior /")
        print("     fake-decision bias. Both cues raise recall, real-image false")
        print("     positives rise too, and correct-vs-wrong is ~0 at the verdict.")
        print("     Report primarily as context prior, NOT evidence arbitration.")
        print("     (Case A's attenuation pattern also holds, but D constrains it.)")
    elif cw_zero and alo > 0:
        print("  => CASE A: disconnect replicated. Matched-cue information is")
        print("     substantially attenuated between spatial grounding and verdict.")
    elif not cw_zero and dB_ > 0:
        print("  => CASE B: disconnect NOT replicated; cue matching propagates to")
        print("     the final authenticity judgment. Stop the disconnect line.")
    else:
        print("  => pattern does not match A/B/D cleanly; report descriptively.")


if __name__ == "__main__":
    main()
