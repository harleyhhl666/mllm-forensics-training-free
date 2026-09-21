"""Primary confirmatory analysis under PROTOCOL v3 (frozen).

Reporting order is fixed by the directive:
  1. absolute effect size
  2. coco_id-cluster bootstrap 95% CI   <- PRIMARY interval
  3. sample-level McNemar p-value       <- supplementary, always with the
                                           cluster-dependence caveat

Metrics: A GT localization, B cue adherence, C grounding advantage,
D joint correct grounding, E cue-induced error.
"""
import json, os, sys
from collections import defaultdict, Counter
from math import comb
import numpy as np, yaml

COND = ("fake_correct", "fake_notool", "fake_wrong")
LABEL = {"fake_correct": "correct ELA", "fake_notool": "no tool",
         "fake_wrong": "wrong donor ELA"}
BANDS = [(0.0, 0.01), (0.01, 0.03), (0.03, 0.10), (0.10, 1.01)]
BAND_NAMES = ["<1%", "1-3%", "3-10%", ">10%"]
NBOOT = 10000
SEED = 20260918
CAVEAT = "(sample-level; 124 samples from 45 coco_ids -> not mutually independent)"


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(a, b):
    b01 = sum(1 for x, y in zip(a, b) if x and not y)
    b10 = sum(1 for x, y in zip(a, b) if y and not x)
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))
    return b01, b10, p


def cochran_q(mat):
    from math import gamma
    X = np.asarray(mat, float)
    n, k = X.shape
    Ri, Cj, N = X.sum(1), X.sum(0), X.sum()
    den = k * N - (Ri ** 2).sum()
    if den == 0:
        return float("nan"), k - 1, 1.0
    Q = (k - 1) * (k * (Cj ** 2).sum() - N ** 2) / den
    df = k - 1
    s, x = df / 2.0, Q / 2.0
    tot, term = 0.0, 1.0 / s
    for i in range(500):
        tot += term
        term *= x / (s + i + 1)
        if term < 1e-16 * max(tot, 1e-300):
            break
    p = 1.0 - tot * (x ** s) * np.exp(-x) / gamma(s)
    return float(Q), df, float(max(0.0, min(1.0, p)))


def cluster_boot_rate(clusters, field_fn, n_boot=NBOOT, seed=SEED):
    """95% CI of a single rate, resampling coco_id clusters."""
    rng = np.random.default_rng(seed)
    keys = sorted(clusters)
    vals = [np.mean([field_fn(r) for r in clusters[k]]) for k in keys]
    sizes = [len(clusters[k]) for k in keys]
    obs = np.average(vals, weights=sizes)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        v = [vals[i] for i in pick]; w = [sizes[i] for i in pick]
        out.append(np.average(v, weights=w))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def cluster_boot_diff(clusters, fa, fb, n_boot=NBOOT, seed=SEED):
    """95% CI of rate(a) - rate(b), resampling coco_id clusters (paired within)."""
    rng = np.random.default_rng(seed)
    keys = sorted(clusters)
    da, db, sz = [], [], []
    for k in keys:
        da.append(np.mean([fa(r) for r in clusters[k]]))
        db.append(np.mean([fb(r) for r in clusters[k]]))
        sz.append(len(clusters[k]))
    obs = np.average(da, weights=sz) - np.average(db, weights=sz)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        a = [da[i] for i in pick]; b = [db[i] for i in pick]; w = [sz[i] for i in pick]
        out.append(np.average(a, weights=w) - np.average(b, weights=w))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    tag = sys.argv[2] if len(sys.argv) > 2 else "validation_primary_stageA"
    setf = sys.argv[3] if len(sys.argv) > 3 else "validation/validation_primary_frozen.json"
    out_root = cfg["experiment"]["out_root"]
    rows = [json.loads(l) for l in open(os.path.join(out_root, tag, "stage_a.jsonl"))]
    frozen = json.load(open(os.path.join(out_root, setf)))

    by = defaultdict(dict)
    for r in rows:
        by[r["pair_id"]][r["condition"]] = r
    comp = sorted(p for p, d in by.items() if len(d) == 4)
    cid = {s["pair_id"]: s["coco_id"] for s in frozen["samples"]}
    clusters = defaultdict(list)
    for p in comp:
        clusters[cid[p]].append(by[p])

    print(f"PROTOCOL v3 PRIMARY CONFIRMATION — {frozen['set_name']}")
    print(f"rows={len(rows)}  complete samples={len(comp)}/{frozen['n_samples']}  "
          f"coco_id clusters={len(clusters)}")
    print(f"cluster bootstrap: {NBOOT} resamples over coco_id, seed {SEED}")

    print("\n" + "=" * 82)
    print("0. PROTOCOL INTEGRITY")
    pf = sum(1 for r in rows if "json_parse_failed" in (r["stage_a_notes"] or []))
    lk = sum(1 for r in rows if r["stage_a"] and r["stage_a"].get("verdict_leak_in_stage_a"))
    sh = defaultdict(set)
    for r in rows:
        sh[r["condition"]].add(r["prompt_sha1"])
    one = len(sh["fake_correct"] | sh["fake_wrong"] | sh["real_own"]) == 1
    dn = sum(1 for p in comp if cid[by[p]["fake_wrong"]["donor_pair_id"]] == cid[p])
    print(f"  JSON parse failures                       : {pf}/{len(rows)}")
    print(f"  Stage-A verdict leaks                     : {lk}/{len(rows)}")
    print(f"  correct/wrong/real share one prompt sha1  : {one}")
    print(f"  same-coco_id donors                       : {dn}")
    print(f"  empty predicted region                    : "
          f"{sum(1 for r in rows if not r['predicted_cells'])}/{len(rows)}")

    gt = lambda c: (lambda d: int(d[c]["mllm_vs_gt"]))
    cu = lambda c: (lambda d: int(d[c]["mllm_vs_cue"]))

    print("\n" + "=" * 82)
    print("1. A — GT LOCALIZATION, three fake conditions")
    print(f"  {'condition':<18}{'rate':>7}  {'cluster 95% CI':<22}{'sample-level Wilson'}")
    rates = {}
    for c in COND:
        o, lo, hi = cluster_boot_rate(clusters, gt(c))
        k = sum(int(by[p][c]["mllm_vs_gt"]) for p in comp)
        _, wl, wh = wilson(k, len(comp))
        rates[c] = o
        print(f"  {LABEL[c]:<18}{o:>7.3f}  [{lo:+.3f},{hi:+.3f}]      "
              f"{k}/{len(comp)} [{wl:.3f},{wh:.3f}]")

    print("\n" + "=" * 82)
    print("2. B — CUE ADHERENCE  &  C — GROUNDING ADVANTAGE")
    ac, alo, ahi = cluster_boot_rate(clusters, cu("fake_correct"))
    aw, wlo, whi = cluster_boot_rate(clusters, cu("fake_wrong"))
    print(f"  adherence to correct cue : {ac:.3f}  cluster CI [{alo:+.3f},{ahi:+.3f}]")
    print(f"  adherence to wrong cue   : {aw:.3f}  cluster CI [{wlo:+.3f},{whi:+.3f}]")
    ga, glo, ghi = cluster_boot_diff(clusters, cu("fake_correct"), cu("fake_wrong"))
    b01, b10, pm = mcnemar_exact([int(by[p]["fake_correct"]["mllm_vs_cue"]) for p in comp],
                                 [int(by[p]["fake_wrong"]["mllm_vs_cue"]) for p in comp])
    print(f"\n  C grounding_advantage    : {ga:+.3f}")
    print(f"    cluster 95% CI         : [{glo:+.3f},{ghi:+.3f}]")
    print(f"    McNemar p              : {pm:.4g}  {CAVEAT}")
    print("    interpretation: discrimination between matched and mismatched cue;")
    print("    NOT a real/fake detection ability.")

    print("\n" + "=" * 82)
    print("3. D — JOINT CORRECT GROUNDING (correct cond: overlaps GT AND its cue)")
    jo, jlo, jhi = cluster_boot_rate(clusters, lambda d: int(d["fake_correct"]["joint_correct"]))
    kj = sum(int(by[p]["fake_correct"]["joint_correct"]) for p in comp)
    _, jwl, jwh = wilson(kj, len(comp))
    print(f"  rate {jo:.3f}  cluster CI [{jlo:+.3f},{jhi:+.3f}]   "
          f"sample {kj}/{len(comp)} [{jwl:.3f},{jwh:.3f}]")

    print("\n" + "=" * 82)
    print("4. PAIRED CONDITION COMPARISONS on GT localization")
    mat = [[by[p][c]["mllm_vs_gt"] for c in COND] for p in comp]
    Q, df, pq = cochran_q(mat)
    print(f"  Cochran's Q: Q={Q:.3f} df={df} p={pq:.4g}  {CAVEAT}\n")
    for a, b in (("fake_correct", "fake_notool"), ("fake_correct", "fake_wrong"),
                 ("fake_wrong", "fake_notool")):
        d, lo, hi = cluster_boot_diff(clusters, gt(a), gt(b))
        x01, x10, p = mcnemar_exact([int(by[q][a]["mllm_vs_gt"]) for q in comp],
                                    [int(by[q][b]["mllm_vs_gt"]) for q in comp])
        print(f"  {LABEL[a]} vs {LABEL[b]}")
        print(f"    1. effect size        : {d:+.3f}  ({rates[a]:.3f} vs {rates[b]:.3f})")
        print(f"    2. cluster 95% CI     : [{lo:+.3f},{hi:+.3f}]")
        print(f"    3. McNemar p          : {p:.4g}   discordant {a}-only={x01} "
              f"{b}-only={x10}   {CAVEAT}\n")

    print("=" * 82)
    print("5. E — CUE-INDUCED ERROR")
    den = [p for p in comp if by[p]["fake_notool"]["mllm_vs_gt"]]
    num = [p for p in den if (not by[p]["fake_wrong"]["mllm_vs_gt"])
           and by[p]["fake_wrong"]["mllm_vs_cue"]]
    p_, lo_, hi_ = wilson(len(num), len(den))
    dc = defaultdict(list)
    for p in den:
        dc[cid[p]].append(dict(x=int(p in num)))
    if dc:
        co, clo, chi = cluster_boot_rate(dc, lambda d: d["x"])
    else:
        co = clo = chi = float("nan")
    print(f"  denominator (no-tool GT correct)        : {len(den)}")
    print(f"  numerator (wrong cue broke it AND")
    print(f"             prediction moved to that cue): {len(num)}")
    print(f"  1. effect size  P(cue_induced_error)    : {p_:.3f}")
    print(f"  2. cluster 95% CI                       : [{clo:+.3f},{chi:+.3f}]  "
          f"(clusters={len(dc)})")
    print(f"     sample-level Wilson CI               : [{lo_:.3f},{hi_:.3f}]")
    broke = sum(1 for p in den if not by[p]["fake_wrong"]["mllm_vs_gt"])
    print(f"  context: wrong cue broke localization in {broke}/{len(den)} "
          f"= {broke/max(len(den),1):.3f} of originally-correct cases")

    print("\n" + "=" * 82)
    print("6. EFFECT CONSISTENCY ACROSS THE 45 coco_id CLUSTERS")
    pos = neg = zero = 0
    per = []
    for k in sorted(clusters):
        ds = clusters[k]
        a = np.mean([d["fake_correct"]["mllm_vs_gt"] for d in ds])
        b = np.mean([d["fake_notool"]["mllm_vs_gt"] for d in ds])
        w = np.mean([d["fake_wrong"]["mllm_vs_gt"] for d in ds])
        per.append((k, len(ds), a, b, w))
        if a > b: pos += 1
        elif a < b: neg += 1
        else: zero += 1
    print(f"  correct > no-tool in {pos}/{len(clusters)} clusters, "
          f"< in {neg}, tied in {zero}")
    cw = sum(1 for _, _, a, _, w in per if a > w)
    cl = sum(1 for _, _, a, _, w in per if a < w)
    print(f"  correct > wrong   in {cw}/{len(clusters)} clusters, < in {cl}, "
          f"tied in {len(clusters)-cw-cl}")
    print(f"  {'coco_id':<10}{'n':>3}{'correct':>9}{'notool':>8}{'wrong':>7}")
    for k, n, a, b, w in per[:12]:
        print(f"  {k:<10}{n:>3}{a:>9.2f}{b:>8.2f}{w:>7.2f}")
    print(f"  ... ({len(per)} clusters total)")

    print("\n" + "=" * 82)
    print("7. PAIRED-REAL DESCRIPTIVE CONTROL (real + own ELA; not a hypothesis test)")
    ra = np.mean([int(by[p]["real_own"]["evidence_detected"]) for p in comp])
    fa = np.mean([int(by[p]["fake_correct"]["evidence_detected"]) for p in comp])
    print(f"  evidence_detected (descriptive only): fake {fa:.3f}   real {ra:.3f}")
    nf = np.mean([len(by[p]["fake_correct"]["predicted_cells"]) for p in comp])
    nr = np.mean([len(by[p]["real_own"]["predicted_cells"]) for p in comp])
    print(f"  mean #predicted regions            : fake {nf:.2f}   real {nr:.2f}")
    cr = Counter(x for p in comp for x in by[p]["real_own"]["predicted_cells"])
    cf = Counter(x for p in comp for x in by[p]["fake_correct"]["predicted_cells"])
    print(f"  real_own spatial prior : {dict(cr.most_common(5))}")
    print(f"  fake_correct spatial   : {dict(cf.most_common(5))}")

    print("\n" + "=" * 82)
    print("8. TAMPER-RATIO STRATA (frozen bins; small bins DESCRIPTIVE ONLY)")
    print(f"  {'band':<7}{'n':>4}{'clust':>6}{'correct':>9}{'notool':>8}{'wrong':>7}"
          f"{'grnd_adv':>10}{'cue_err':>9}")
    for (lo, hi), nm in zip(BANDS, BAND_NAMES):
        sub = [p for p in comp if lo <= by[p]["fake_correct"]["tamper_ratio"] < hi]
        if not sub:
            continue
        nc = len({cid[p] for p in sub})
        f = lambda c, k: np.mean([int(by[p][c][k]) for p in sub])
        adv = f("fake_correct", "mllm_vs_cue") - f("fake_wrong", "mllm_vs_cue")
        d2 = [p for p in sub if by[p]["fake_notool"]["mllm_vs_gt"]]
        n2 = [p for p in d2 if (not by[p]["fake_wrong"]["mllm_vs_gt"])
              and by[p]["fake_wrong"]["mllm_vs_cue"]]
        ce = f"{len(n2)}/{len(d2)}" if d2 else "n/a"
        flag = "  DESCRIPTIVE" if len(sub) < 20 else ""
        print(f"  {nm:<7}{len(sub):>4}{nc:>6}{f('fake_correct','mllm_vs_gt'):>9.3f}"
              f"{f('fake_notool','mllm_vs_gt'):>8.3f}{f('fake_wrong','mllm_vs_gt'):>7.3f}"
              f"{adv:>+10.3f}{ce:>9}{flag}")

    print("\n" + "=" * 82)
    print("9. PRE-REGISTERED INTERPRETATION CHECKS")
    d1, l1, h1 = cluster_boot_diff(clusters, gt("fake_correct"), gt("fake_notool"))
    d2, l2, h2 = cluster_boot_diff(clusters, gt("fake_wrong"), gt("fake_notool"))
    print(f"  (1) correct cue positive vs no-tool : {d1:+.3f} [{l1:+.3f},{h1:+.3f}]  "
          f"-> {'YES' if l1 > 0 else 'NO'}")
    print(f"  (2) wrong cue also changes vs no-tool: {d2:+.3f} [{l2:+.3f},{h2:+.3f}]  "
          f"-> {'YES' if l2 > 0 else 'NO'}")
    print(f"  (3) grounding advantage {ga:+.3f} [{glo:+.3f},{ghi:+.3f}] vs "
          f"cue-following {d2:+.3f}")
    print(f"      discrimination weaker than following -> "
          f"{'YES' if ga < d2 else 'NO'}")
    print(f"  (4) cue_induced_error {p_:.3f} [{clo:+.3f},{chi:+.3f}] non-zero -> "
          f"{'YES' if clo > 0 else 'NO'}")


if __name__ == "__main__":
    main()
