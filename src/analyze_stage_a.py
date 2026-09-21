"""Stage A pilot analysis: paired statistics across the three fake conditions,
plus Gate-2 specificity against the paired-real control.

Paired design is respected throughout: each fake contributes one observation per
condition, so all condition comparisons use McNemar on discordant pairs and a
paired bootstrap (resampling coco_ids, not images, because one coco_id may supply
two samples) for the CI of the difference. Cochran's Q tests all three conditions
jointly before pairwise tests are read.
"""
import json, os, sys
from collections import defaultdict, Counter
import numpy as np, yaml

COND = ("fake_correct", "fake_notool", "fake_wrong")
BANDS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.10), (0.10, 1.01)]
BAND_NAMES = ["<1%", "1-3%", "3-10%", ">10%"]


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def fmt(k, n):
    p, lo, hi = wilson(k, n)
    return f"{k:>4}/{n:<4} = {p:.3f} [{lo:.3f},{hi:.3f}]" if n else f"{k}/{n}"


def mcnemar(a, b):
    """a, b: equal-length 0/1 lists (paired). Returns b01, b10, chi2_cc, p."""
    from math import erfc, sqrt
    b01 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    b10 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = b01 + b10
    if n == 0:
        return b01, b10, float("nan"), 1.0
    chi = (abs(b01 - b10) - 1) ** 2 / n if n > 0 else 0.0
    # exact binomial two-sided p (more reliable at small n than chi-square)
    from math import comb
    k = min(b01, b10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return b01, b10, chi, p


def cochran_q(mat):
    """mat: n x k matrix of 0/1. Returns Q, df, p (chi-square approx)."""
    from math import erfc
    X = np.asarray(mat, float)
    n, k = X.shape
    Ri = X.sum(axis=1)
    Cj = X.sum(axis=0)
    N = X.sum()
    den = (k * N - (Ri ** 2).sum())
    if den == 0:
        return float("nan"), k - 1, 1.0
    Q = (k - 1) * (k * (Cj ** 2).sum() - N ** 2) / den
    # survival of chi-square with df=k-1 via regularized gamma (series, df small)
    df = k - 1
    try:
        from math import gamma, exp
        def gammainc_upper(s, x, terms=400):
            # continued-fraction-free series for lower, then complement
            tot, term = 0.0, 1.0 / s
            for i in range(terms):
                tot += term
                term *= x / (s + i + 1)
                if term < 1e-16 * max(tot, 1e-300):
                    break
            low = tot * (x ** s) * np.exp(-x)
            return 1.0 - low / gamma(s)
        p = float(gammainc_upper(df / 2.0, Q / 2.0))
    except Exception:
        p = float("nan")
    return float(Q), df, max(0.0, min(1.0, p))


def boot_diff(groups, key_a, key_b, field, n_boot=5000, seed=7):
    """Paired bootstrap over coco_id clusters for rate(a) - rate(b)."""
    rng = np.random.default_rng(seed)
    cids = sorted(groups)
    obs_a = [v for c in cids for v in groups[c][key_a]]
    obs_b = [v for c in cids for v in groups[c][key_b]]
    if not obs_a:
        return float("nan"), (float("nan"), float("nan"))
    point = np.mean(obs_a) - np.mean(obs_b)
    diffs = []
    for _ in range(n_boot):
        pick = rng.choice(len(cids), len(cids), replace=True)
        a = [v for i in pick for v in groups[cids[i]][key_a]]
        b = [v for i in pick for v in groups[cids[i]][key_b]]
        if a and b:
            diffs.append(np.mean(a) - np.mean(b))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), (float(lo), float(hi))


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    tag = sys.argv[2] if len(sys.argv) > 2 else "pilot_stageA"
    path = os.path.join(cfg["experiment"]["out_root"], tag, "stage_a.jsonl")
    rows = [json.loads(l) for l in open(path)]
    by = defaultdict(dict)
    for r in rows:
        by[r["pair_id"]][r["condition"]] = r
    complete = [p for p, d in by.items() if len(d) == 4]
    print(f"rows={len(rows)}  samples with all 4 conditions={len(complete)}")
    if not complete:
        sys.exit("no complete samples yet")

    print("\n" + "=" * 78)
    print("0. PROTOCOL INTEGRITY")
    pf = sum(1 for r in rows if "json_parse_failed" in (r["stage_a_notes"] or []))
    leak = sum(1 for r in rows if r["stage_a"] and r["stage_a"].get("verdict_leak_in_stage_a"))
    shas = defaultdict(set)
    for r in rows:
        shas[r["condition"]].add(r["prompt_sha1"])
    tool_sha = shas["fake_correct"] | shas["fake_wrong"] | shas["real_own"]
    print(f"  JSON parse failures                      : {pf}/{len(rows)}")
    print(f"  Stage A verdict leaks                    : {leak}/{len(rows)}")
    print(f"  correct/wrong/real share one prompt sha1 : {len(tool_sha) == 1}")
    empty = sum(1 for r in rows if not r["predicted_cells"])
    print(f"  responses with no predicted region       : {empty}/{len(rows)}")

    # ---- per-condition rates on fakes
    print("\n" + "=" * 78)
    print("1. FAKE CONDITIONS: rates (n = complete samples)")
    print(f"  {'condition':<14}{'acknowledge':<26}{'B: vs GT':<26}{'C: vs cue':<26}")
    acc = {}
    for c in COND:
        ack = [int(by[p][c]["evidence_detected"]) for p in complete]
        gt = [int(by[p][c]["mllm_vs_gt"]) for p in complete]
        cue = [int(by[p][c]["mllm_vs_cue"]) for p in complete]
        acc[c] = dict(ack=ack, gt=gt, cue=cue)
        print(f"  {c:<14}{fmt(sum(ack),len(ack)):<26}{fmt(sum(gt),len(gt)):<26}"
              f"{fmt(sum(cue),len(cue)):<26}")
    jc = [int(by[p]["fake_correct"]["joint_correct"]) for p in complete]
    print(f"\n  D: joint_correct (correct cond, vs GT AND vs cue): {fmt(sum(jc), len(jc))}")

    # ---- paired-real control
    print("\n" + "=" * 78)
    print("2. GATE-2 SPECIFICITY (paired real + own ELA)")
    ra = [int(by[p]["real_own"]["evidence_detected"]) for p in complete]
    fa = acc["fake_correct"]["ack"]
    print(f"  P(acknowledge | fake + correct cue) : {fmt(sum(fa), len(fa))}")
    print(f"  P(acknowledge | paired real + own)  : {fmt(sum(ra), len(ra))}")
    b01, b10, chi, p = mcnemar(fa, ra)
    print(f"  difference = {np.mean(fa)-np.mean(ra):+.3f}   "
          f"McNemar fake-only={b01} real-only={b10} p={p:.4g}")
    nreg_f = np.mean([len(by[p]['fake_correct']['predicted_cells']) for p in complete])
    nreg_r = np.mean([len(by[p]['real_own']['predicted_cells']) for p in complete])
    print(f"  mean #predicted regions: fake={nreg_f:.2f}  real={nreg_r:.2f}")
    cf = Counter(x for p in complete for x in by[p]['real_own']['predicted_cells'])
    print(f"  real_own predicted-cell distribution: {dict(cf.most_common())}")

    # ---- paired tests
    groups = defaultdict(lambda: defaultdict(list))
    for p in complete:
        c = by[p]["fake_correct"]["coco_id"]
        for cond in COND:
            groups[c][f"gt_{cond}"].append(int(by[p][cond]["mllm_vs_gt"]))
        groups[c]["joint_correct"].append(int(by[p]["fake_correct"]["joint_correct"]))
        groups[c]["joint_notool"].append(
            int(by[p]["fake_notool"]["mllm_vs_gt"]))   # no cue exists => B only

    print("\n" + "=" * 78)
    print("3. PAIRED TESTS on GT localization (B)")
    mat = [[by[p][c]["mllm_vs_gt"] for c in COND] for p in complete]
    Q, df, pq = cochran_q(mat)
    print(f"  Cochran's Q over 3 conditions: Q={Q:.3f} df={df} p={pq:.4g}")
    for other in ("fake_notool", "fake_wrong"):
        a = acc["fake_correct"]["gt"]; b = acc[other]["gt"]
        b01, b10, chi, p = mcnemar(a, b)
        d, (lo, hi) = boot_diff(groups, "gt_fake_correct", f"gt_{other}", "gt")
        print(f"\n  correct vs {other}:")
        print(f"    rates {np.mean(a):.3f} vs {np.mean(b):.3f}   diff={d:+.3f} "
              f"[95% CI {lo:+.3f},{hi:+.3f}]")
        print(f"    McNemar: correct-only={b01}, {other}-only={b10}, "
              f"chi2_cc={chi:.2f}, exact p={p:.4g}")

    # wrong-cue capture: did the model follow the donor cue?
    print("\n" + "=" * 78)
    print("4. WRONG-CUE CAPTURE (was the model pulled to the donor region?)")
    wc = [int(by[p]["fake_wrong"]["mllm_vs_cue"]) for p in complete]
    cc = [int(by[p]["fake_correct"]["mllm_vs_cue"]) for p in complete]
    print(f"  follows its OWN cue (correct cond)  : {fmt(sum(cc), len(cc))}")
    print(f"  follows DONOR cue (wrong cond)      : {fmt(sum(wc), len(wc))}")
    both = sum(1 for p in complete if by[p]["fake_wrong"]["mllm_vs_cue"]
               and not by[p]["fake_wrong"]["mllm_vs_gt"])
    print(f"  followed donor cue AND missed GT    : {fmt(both, len(complete))}")

    # ---- tamper_ratio bands
    print("\n" + "=" * 78)
    print("5. BY TAMPER_RATIO BAND (descriptive)")
    print(f"  {'band':<8}{'n':>4}  {'B correct':>10}{'B notool':>10}{'B wrong':>9}"
          f"{'D joint':>9}{'ack real':>9}")
    for (lo, hi), nm in zip(BANDS, BAND_NAMES):
        sub = [p for p in complete if lo <= by[p]["fake_correct"]["tamper_ratio"] < hi]
        if not sub:
            continue
        f = lambda c, k: np.mean([int(by[p][c][k]) for p in sub])
        print(f"  {nm:<8}{len(sub):>4}  {f('fake_correct','mllm_vs_gt'):>10.3f}"
              f"{f('fake_notool','mllm_vs_gt'):>10.3f}{f('fake_wrong','mllm_vs_gt'):>9.3f}"
              f"{f('fake_correct','joint_correct'):>9.3f}"
              f"{f('real_own','evidence_detected'):>9.3f}")

    print("\n" + "=" * 78)
    print("6. STAGE-B GO / NO-GO INPUTS")
    a = acc["fake_correct"]["gt"]
    d1, ci1 = boot_diff(groups, "gt_fake_correct", "gt_fake_notool", "gt")
    d2, ci2 = boot_diff(groups, "gt_fake_correct", "gt_fake_wrong", "gt")
    _, _, _, p1 = mcnemar(a, acc["fake_notool"]["gt"])
    _, _, _, p2 = mcnemar(a, acc["fake_wrong"]["gt"])
    _, _, _, p3 = mcnemar(fa, ra)
    print(f"  (1) correct > no-tool   : diff={d1:+.3f} CI[{ci1[0]:+.3f},{ci1[1]:+.3f}] p={p1:.4g}")
    print(f"  (2) correct > wrong-tool: diff={d2:+.3f} CI[{ci2[0]:+.3f},{ci2[1]:+.3f}] p={p2:.4g}")
    print(f"  (3) ack fake vs real    : diff={np.mean(fa)-np.mean(ra):+.3f} p={p3:.4g}")
    print(f"  joint_correct available for Stage B: {sum(jc)} samples")


if __name__ == "__main__":
    main()
