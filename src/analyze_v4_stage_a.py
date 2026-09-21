"""V1 Stage-A analysis on the TGIF validation split (H1/H2 only).

Reporting order fixed by protocol: effect size -> coco_id-cluster bootstrap 95% CI
-> sample-level McNemar p (supplementary, clustering caveat).

Does NOT touch Stage B.
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


def cb_rate(cl, fn, nb=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    ks = sorted(cl)
    v = [np.mean([fn(r) for r in cl[k]]) for k in ks]
    w = [len(cl[k]) for k in ks]
    obs = np.average(v, weights=w)
    o = []
    for _ in range(nb):
        p = rng.integers(0, len(ks), len(ks))
        o.append(np.average([v[i] for i in p], weights=[w[i] for i in p]))
    lo, hi = np.percentile(o, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def cb_diff(cl, fa, fb, nb=NBOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    ks = sorted(cl)
    a = [np.mean([fa(r) for r in cl[k]]) for k in ks]
    b = [np.mean([fb(r) for r in cl[k]]) for k in ks]
    w = [len(cl[k]) for k in ks]
    obs = np.average(a, weights=w) - np.average(b, weights=w)
    o = []
    for _ in range(nb):
        p = rng.integers(0, len(ks), len(ks))
        ww = [w[i] for i in p]
        o.append(np.average([a[i] for i in p], weights=ww) -
                 np.average([b[i] for i in p], weights=ww))
    lo, hi = np.percentile(o, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    tag = sys.argv[2] if len(sys.argv) > 2 else "v4_stageA"
    out_root = cfg["experiment"]["out_root"]
    frozen = json.load(open(os.path.join(out_root, "validation_v4",
                                         "validation_v4_frozen.json")))
    cid = {s["pair_id"]: s["coco_id"] for s in frozen["samples"]}
    rows = [json.loads(l) for l in
            open(os.path.join(out_root, tag, "stage_a.jsonl"))]
    by = defaultdict(dict)
    for r in rows:
        by[r["pair_id"]][r["condition"]] = r
    NEED = 5
    comp = sorted(p for p, d in by.items() if len(d) == NEED)
    cl = defaultdict(list)
    for p in comp:
        cl[cid[p]].append(by[p])

    print("V1 — INDEPENDENT STAGE-A VALIDATION (TGIF validation split)")
    print(f"rows={len(rows)}  complete samples={len(comp)}/{frozen['n_samples']}  "
          f"coco_id clusters={len(cl)}")
    print(f"sampling: {frozen.get('sampling')}")
    print(f"cluster bootstrap: {NBOOT} resamples, seed {SEED}")
    CAV = f"(sample-level; {len(comp)} samples / {len(cl)} clusters — not independent)"

    print("\n" + "=" * 84)
    print("0. INTEGRITY")
    pf = sum(1 for r in rows if "json_parse_failed" in (r["stage_a_notes"] or []))
    lk = sum(1 for r in rows if r["stage_a"] and r["stage_a"].get("verdict_leak_in_stage_a"))
    sh = defaultdict(set)
    for r in rows:
        sh[r["condition"]].add(r["prompt_sha1"])
    tool_one = len(sh["fake_correct"] | sh["fake_wrong"] | sh["real_own"]) == 1
    nt_one = len(sh["fake_notool"] | sh["real_notool"]) == 1
    dn = sum(1 for p in comp if cid[by[p]["fake_wrong"]["donor_pair_id"]] == cid[p])
    print(f"  JSON parse failures                        : {pf}/{len(rows)}")
    print(f"  Stage-A verdict leaks                      : {lk}/{len(rows)}")
    print(f"  correct/wrong/real_own share one prompt sha1: {tool_one}")
    print(f"  no-tool conditions share one prompt sha1    : {nt_one}")
    print(f"  same-coco_id donors                        : {dn}")
    print(f"  empty predicted region                     : "
          f"{sum(1 for r in rows if not r['predicted_cells'])}/{len(rows)}")

    gt = lambda c: (lambda d: int(d[c]["mllm_vs_gt"]))
    cu = lambda c: (lambda d: int(d[c]["mllm_vs_cue"]))

    print("\n" + "=" * 84)
    print("1. GT LOCALIZATION by condition")
    print(f"  {'condition':<18}{'rate':>7}  {'cluster 95% CI':<22}{'sample Wilson'}")
    rate = {}
    for c in COND:
        o, lo, hi = cb_rate(cl, gt(c))
        k = sum(int(by[p][c]["mllm_vs_gt"]) for p in comp)
        _, wl, wh = wilson(k, len(comp))
        rate[c] = o
        print(f"  {LAB[c]:<18}{o:>7.3f}  [{lo:+.3f},{hi:+.3f}]      "
              f"{k}/{len(comp)} [{wl:.3f},{wh:.3f}]")

    mat = [[by[p][c]["mllm_vs_gt"] for c in COND] for p in comp]
    Q, df, pq = cochran_q(mat)
    print(f"\n  Cochran's Q: Q={Q:.3f} df={df} p={pq:.4g}  {CAV}")

    print("\n" + "=" * 84)
    print("H1 — correct ELA vs no-tool")
    d, lo, hi = cb_diff(cl, gt("fake_correct"), gt("fake_notool"))
    x01, x10, p = mcnemar([int(by[q]["fake_correct"]["mllm_vs_gt"]) for q in comp],
                          [int(by[q]["fake_notool"]["mllm_vs_gt"]) for q in comp])
    print(f"  1. effect size    : {d:+.3f}  ({rate['fake_correct']:.3f} vs "
          f"{rate['fake_notool']:.3f})")
    print(f"  2. cluster 95% CI : [{lo:+.3f},{hi:+.3f}]  -> "
          f"{'EXCLUDES zero' if lo > 0 else 'includes zero'}")
    print(f"  3. McNemar p      : {p:.4g}  discordant correct-only={x01} "
          f"notool-only={x10}  {CAV}")
    h1 = lo > 0

    print("\n" + "=" * 84)
    print("H2 — correct ELA vs wrong donor ELA")
    d2, lo2, hi2 = cb_diff(cl, gt("fake_correct"), gt("fake_wrong"))
    y01, y10, p2 = mcnemar([int(by[q]["fake_correct"]["mllm_vs_gt"]) for q in comp],
                           [int(by[q]["fake_wrong"]["mllm_vs_gt"]) for q in comp])
    print(f"  1. effect size    : {d2:+.3f}  ({rate['fake_correct']:.3f} vs "
          f"{rate['fake_wrong']:.3f})")
    print(f"  2. cluster 95% CI : [{lo2:+.3f},{hi2:+.3f}]  -> "
          f"{'EXCLUDES zero' if lo2 > 0 else 'includes zero'}")
    print(f"  3. McNemar p      : {p2:.4g}  discordant correct-only={y01} "
          f"wrong-only={y10}  {CAV}")
    h2 = lo2 > 0

    print("\n  (supplementary) wrong donor ELA vs no-tool")
    d3, lo3, hi3 = cb_diff(cl, gt("fake_wrong"), gt("fake_notool"))
    z01, z10, p3 = mcnemar([int(by[q]["fake_wrong"]["mllm_vs_gt"]) for q in comp],
                           [int(by[q]["fake_notool"]["mllm_vs_gt"]) for q in comp])
    print(f"    effect {d3:+.3f}  cluster CI [{lo3:+.3f},{hi3:+.3f}]  p={p3:.4g}")

    print("\n" + "=" * 84)
    print("2. CUE ADHERENCE and JOINT GROUNDING")
    ac, al, ah = cb_rate(cl, cu("fake_correct"))
    aw, wl2, wh2 = cb_rate(cl, cu("fake_wrong"))
    print(f"  adherence to correct cue : {ac:.3f}  cluster CI [{al:+.3f},{ah:+.3f}]")
    print(f"  adherence to wrong cue   : {aw:.3f}  cluster CI [{wl2:+.3f},{wh2:+.3f}]")
    ga, gl, gh = cb_diff(cl, cu("fake_correct"), cu("fake_wrong"))
    print(f"  grounding advantage      : {ga:+.3f}  cluster CI [{gl:+.3f},{gh:+.3f}]")
    jo, jl, jh = cb_rate(cl, lambda d: int(d["fake_correct"]["joint_correct"]))
    kj = sum(int(by[p]["fake_correct"]["joint_correct"]) for p in comp)
    _, jwl, jwh = wilson(kj, len(comp))
    print(f"  joint correct grounding  : {jo:.3f}  cluster CI [{jl:+.3f},{jh:+.3f}]   "
          f"sample {kj}/{len(comp)} [{jwl:.3f},{jwh:.3f}]")
    print(f"  -> pre-registered B2 subset size for V2: {kj} samples, "
          f"{len({cid[p] for p in comp if by[p]['fake_correct']['joint_correct']})} clusters")

    print("\n" + "=" * 84)
    print("3. PAIRED-REAL CONTROLS (spatial behaviour; descriptive)")
    for c in ("real_own", "real_notool"):
        ack = np.mean([int(by[p][c]["evidence_detected"]) for p in comp])
        nreg = np.mean([len(by[p][c]["predicted_cells"]) for p in comp])
        cnt = Counter(x for p in comp for x in by[p][c]["predicted_cells"])
        print(f"  {LAB[c]:<18} evidence_detected={ack:.3f}  mean #regions={nreg:.2f}")
        print(f"    {dict(cnt.most_common(5))}")
    for c in ("fake_correct", "fake_notool"):
        cnt = Counter(x for p in comp for x in by[p][c]["predicted_cells"])
        print(f"  {LAB[c]:<18} spatial: {dict(cnt.most_common(5))}")

    print("\n" + "=" * 84)
    print(f"4. EFFECT CONSISTENCY ACROSS {len(cl)} coco_id CLUSTERS")
    pn = pw = 0
    nn = nw = 0
    for k in cl:
        ds = cl[k]
        a = np.mean([d["fake_correct"]["mllm_vs_gt"] for d in ds])
        b = np.mean([d["fake_notool"]["mllm_vs_gt"] for d in ds])
        w = np.mean([d["fake_wrong"]["mllm_vs_gt"] for d in ds])
        if a > b: pn += 1
        elif a < b: nn += 1
        if a > w: pw += 1
        elif a < w: nw += 1
    print(f"  correct > no-tool : {pn}/{len(cl)}   < : {nn}   tied : {len(cl)-pn-nn}")
    print(f"  correct > wrong   : {pw}/{len(cl)}   < : {nw}   tied : {len(cl)-pw-nw}")

    print("\n" + "=" * 84)
    print("5. TAMPER-RATIO BANDS (descriptive)")
    print(f"  {'band':<7}{'n':>4}{'clust':>6}{'correct':>9}{'notool':>8}{'wrong':>7}"
          f"{'c-n':>8}{'c-w':>8}{'joint':>8}")
    for (lo_, hi_), nm in zip(BANDS, BN):
        sub = [p for p in comp if lo_ <= by[p]["fake_correct"]["tamper_ratio"] < hi_]
        if not sub:
            continue
        f = lambda c: np.mean([int(by[p][c]["mllm_vs_gt"]) for p in sub])
        j = np.mean([int(by[p]["fake_correct"]["joint_correct"]) for p in sub])
        print(f"  {nm:<7}{len(sub):>4}{len({cid[p] for p in sub}):>6}"
              f"{f('fake_correct'):>9.3f}{f('fake_notool'):>8.3f}{f('fake_wrong'):>7.3f}"
              f"{f('fake_correct')-f('fake_notool'):>+8.3f}"
              f"{f('fake_correct')-f('fake_wrong'):>+8.3f}{j:>8.3f}")

    print("\n" + "=" * 84)
    print("PRE-REGISTERED DECISION")
    print(f"  H1 correct > no-tool : {d:+.3f} [{lo:+.3f},{hi:+.3f}]  -> "
          f"{'SUPPORTED' if h1 else 'NOT supported'}")
    print(f"  H2 correct > wrong   : {d2:+.3f} [{lo2:+.3f},{hi2:+.3f}]  -> "
          f"{'SUPPORTED' if h2 else 'NOT supported'}")
    if h1 and h2:
        print("  => Stage-A grounding REPLICATED. V2 Stage B is permitted.")
    else:
        print("  => Case C: grounding did not replicate. STOP; do not run Stage B.")


if __name__ == "__main__":
    main()
