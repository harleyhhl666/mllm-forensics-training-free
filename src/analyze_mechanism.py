"""Mechanism analysis: 2x2 context-bias design, frozen contrasts.

Contrasts (declared before inference, in mechanism_frozen.json):
  C1 - C0   text forensic-context effect
  C2 - C0   neutral visualization effect
  C3 - C2   forensic framing increment over byte-identical visualization
  C3 - C1   visualization effect under forensic framing
  interaction = (C3 - C1) - (C2 - C0)

Each is computed separately on fake images (TPR) and real images (FPR), with
coco_id-cluster bootstrap CIs as the primary interval.
"""
import json, os, sys
from collections import defaultdict, Counter
from math import comb, gamma
import numpy as np, yaml

CONDS = ("C0", "C1", "C2", "C3")
DESC = {"C0": "no vis + neutral", "C1": "no vis + forensic",
        "C2": "donor vis + neutral", "C3": "donor vis + forensic"}
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


def cb(cl, fn, nb=NBOOT, seed=SEED):
    """Cluster bootstrap for an arbitrary per-cluster statistic fn(list_of_cells)."""
    rng = np.random.default_rng(seed)
    ks = sorted(cl)
    v = [fn(cl[k]) for k in ks]
    w = [len(cl[k]) for k in ks]
    keep = [i for i, x in enumerate(v) if np.isfinite(x)]
    obs = np.average([v[i] for i in keep], weights=[w[i] for i in keep])
    out = []
    for _ in range(nb):
        p = rng.integers(0, len(keep), len(keep))
        ii = [keep[i] for i in p]
        out.append(np.average([v[i] for i in ii], weights=[w[i] for i in ii]))
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(obs), float(lo), float(hi)


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    out_root = cfg["experiment"]["out_root"]
    tag = sys.argv[2] if len(sys.argv) > 2 else "mechanism_run"
    F = json.load(open(os.path.join(out_root, "mechanism", "mechanism_frozen.json")))
    rows = [json.loads(l) for l in
            open(os.path.join(out_root, tag, "mechanism.jsonl"))]

    G = defaultdict(dict)
    for r in rows:
        G[(r["pair_id"], r["label"])][r["condition"]] = r
    cid = {s["pair_id"]: s["coco_id"] for s in F["samples"]}
    comp = {lab: sorted(p for (p, l) in G if l == lab and len(G[(p, lab)]) == 4)
            for lab in ("fake", "real")}
    both = sorted(set(comp["fake"]) & set(comp["real"]))

    print("MECHANISM EXPERIMENT — 2x2 forensic context bias")
    print(f"rows={len(rows)}  complete fake={len(comp['fake'])}  "
          f"complete real={len(comp['real'])}  both labels={len(both)}")
    print(f"frozen sampling: {F['sampling_rule']}")
    print(f"cluster bootstrap: {NBOOT} resamples, seed {SEED}\n")

    print("=" * 84)
    print("1. INTEGRITY")
    pf = sum(1 for r in rows if "json_parse_failed" in (r["notes"] or []))
    nv = sum(1 for r in rows if not r["final_verdict"])
    ph_ok = all(r["prompt_sha1"] == F["prompt_hashes"][r["condition"]] for r in rows)
    vis_bad = 0
    for (p, lab), d in G.items():
        if len(d) == 4 and d["C2"]["visualization_sha1"] != d["C3"]["visualization_sha1"]:
            vis_bad += 1
    dn = sum(1 for p in both if cid[G[(p, "fake")]["C0"]["donor_pair_id"]] == cid[p])
    print(f"  JSON parse failures                : {pf}/{len(rows)}")
    print(f"  missing verdicts                   : {nv}/{len(rows)}")
    print(f"  all prompt hashes match frozen     : {ph_ok}")
    print(f"  C2/C3 visualization byte-identical : {'ALL' if vis_bad == 0 else f'{vis_bad} MISMATCH'}")
    print(f"  same-coco_id donors                : {dn}")
    print(f"  unique coco_ids                    : {len({cid[p] for p in both})}")

    def clusters(lab, ids):
        cl = defaultdict(list)
        for p in ids:
            cl[cid[p]].append(G[(p, lab)])
        return cl

    clf = clusters("fake", both)
    clr = clusters("real", both)
    fake_at = lambda c: (lambda cells: float(np.mean(
        [int(x[c]["final_verdict"] == "fake") for x in cells])))

    print("\n" + "=" * 84)
    print("2. FAKE IMAGES — TPR = P(predicted Fake | actual Fake)")
    print(f"  {'cond':<5}{'framing':<22}{'TPR':>7}  {'cluster 95% CI':<22}{'sample Wilson'}")
    tpr = {}
    for c in CONDS:
        o, lo, hi = cb(clf, fake_at(c))
        k = sum(int(G[(p, "fake")][c]["final_verdict"] == "fake") for p in both)
        _, wl, wh = wilson(k, len(both))
        tpr[c] = o
        print(f"  {c:<5}{DESC[c]:<22}{o:>7.3f}  [{lo:+.3f},{hi:+.3f}]      "
              f"{k}/{len(both)} [{wl:.3f},{wh:.3f}]")
    Qf, dff, pqf = cochran_q([[int(G[(p, "fake")][c]["final_verdict"] == "fake")
                               for c in CONDS] for p in both])
    print(f"  Cochran's Q (4 conditions): Q={Qf:.3f} df={dff} p={pqf:.4g}")

    print("\n" + "=" * 84)
    print("3. REAL IMAGES — FPR = P(predicted Fake | actual Real)")
    print(f"  {'cond':<5}{'framing':<22}{'FPR':>7}  {'cluster 95% CI':<22}{'sample Wilson'}")
    fpr = {}
    for c in CONDS:
        o, lo, hi = cb(clr, fake_at(c))
        k = sum(int(G[(p, "real")][c]["final_verdict"] == "fake") for p in both)
        _, wl, wh = wilson(k, len(both))
        fpr[c] = o
        print(f"  {c:<5}{DESC[c]:<22}{o:>7.3f}  [{lo:+.3f},{hi:+.3f}]      "
              f"{k}/{len(both)} [{wl:.3f},{wh:.3f}]")
    Qr, dfr, pqr = cochran_q([[int(G[(p, "real")][c]["final_verdict"] == "fake")
                               for c in CONDS] for p in both])
    print(f"  Cochran's Q (4 conditions): Q={Qr:.3f} df={dfr} p={pqr:.4g}")

    print("\n" + "=" * 84)
    print("4. FROZEN CONTRASTS  (effect size -> cluster 95% CI -> McNemar p)")
    CONTRASTS = [("C1", "C0", "text forensic-context effect"),
                 ("C2", "C0", "neutral visualization effect"),
                 ("C3", "C2", "forensic framing over IDENTICAL visualization"),
                 ("C3", "C1", "visualization effect under forensic framing")]
    res = {}
    for lab, cl, nm in (("fake", clf, "TPR"), ("real", clr, "FPR")):
        print(f"\n  --- {lab.upper()} images ({nm}) ---")
        for a, b, desc in CONTRASTS:
            f = lambda cells, a=a, b=b: float(np.mean(
                [int(x[a]["final_verdict"] == "fake") -
                 int(x[b]["final_verdict"] == "fake") for x in cells]))
            d, lo, hi = cb(cl, f)
            x01, x10, p = mcnemar(
                [int(G[(q, lab)][a]["final_verdict"] == "fake") for q in both],
                [int(G[(q, lab)][b]["final_verdict"] == "fake") for q in both])
            res[(lab, a, b)] = (d, lo, hi, p)
            star = "EXCLUDES zero" if (lo > 0 or hi < 0) else "includes zero"
            print(f"    {a}-{b}  {desc}")
            print(f"      1. effect {d:+.3f}   2. CI [{lo:+.3f},{hi:+.3f}] {star}   "
                  f"3. p={p:.4g} ({a}-only={x01}, {b}-only={x10})")

    print("\n" + "=" * 84)
    print("5. FACTORIAL INTERACTION  (C3-C1) - (C2-C0)")
    for lab, cl in (("fake", clf), ("real", clr)):
        f = lambda cells: float(np.mean(
            [(int(x["C3"]["final_verdict"] == "fake") - int(x["C1"]["final_verdict"] == "fake"))
             - (int(x["C2"]["final_verdict"] == "fake") - int(x["C0"]["final_verdict"] == "fake"))
             for x in cells]))
        d, lo, hi = cb(cl, f)
        print(f"  {lab:<5} interaction {d:+.3f}  cluster CI [{lo:+.3f},{hi:+.3f}]  -> "
              f"{'EXCLUDES zero' if (lo > 0 or hi < 0) else 'includes zero'}")

    print("\n" + "=" * 84)
    print("6. YOUDEN J = TPR - FPR  (secondary descriptive; not a go/no-go)")
    print(f"  {'cond':<5}{'framing':<22}{'TPR':>7}{'FPR':>7}{'J':>8}  cluster CI for J")
    for c in CONDS:
        f = lambda cells, c=c: float(np.mean(
            [int(x[c]["final_verdict"] == "fake") for x in cells]))
        # J per cluster needs both labels; build paired cluster structure
        clj = defaultdict(list)
        for p in both:
            clj[cid[p]].append((G[(p, "fake")], G[(p, "real")]))
        fj = lambda cells, c=c: float(np.mean(
            [int(a[c]["final_verdict"] == "fake") - int(b[c]["final_verdict"] == "fake")
             for a, b in cells]))
        j, jlo, jhi = cb(clj, fj)
        print(f"  {c:<5}{DESC[c]:<22}{tpr[c]:>7.3f}{fpr[c]:>7.3f}{j:>8.3f}  "
              f"[{jlo:+.3f},{jhi:+.3f}]")

    print("\n" + "=" * 84)
    print("7. PRE-REGISTERED MECHANISM VERDICT")
    def sig(lab, a, b):
        d, lo, hi, _ = res[(lab, a, b)]
        return d, (lo > 0)
    d10f, s10f = sig("fake", "C1", "C0")
    d10r, s10r = sig("real", "C1", "C0")
    d20f, s20f = sig("fake", "C2", "C0")
    d20r, s20r = sig("real", "C2", "C0")
    d32f, s32f = sig("fake", "C3", "C2")
    d32r, s32r = sig("real", "C3", "C2")
    print(f"  C1-C0 (text only)      fake {d10f:+.3f} {'sig' if s10f else 'ns'}   "
          f"real {d10r:+.3f} {'sig' if s10r else 'ns'}")
    print(f"  C2-C0 (vis only)       fake {d20f:+.3f} {'sig' if s20f else 'ns'}   "
          f"real {d20r:+.3f} {'sig' if s20r else 'ns'}")
    print(f"  C3-C2 (framing added)  fake {d32f:+.3f} {'sig' if s32f else 'ns'}   "
          f"real {d32r:+.3f} {'sig' if s32r else 'ns'}")
    print()
    both_up = lambda sf, sr: sf and sr
    if both_up(s10f, s10r):
        print("  -> forensic SEMANTIC CONTEXT ALONE induces a Fake bias (raises both")
        print("     TPR and FPR): class-independent shift, not detection improvement.")
    if both_up(s20f, s20r):
        print("  -> AUXILIARY VISUALIZATION ALONE induces a Fake bias, with no")
        print("     forensic labelling attached.")
    if s32f or s32r:
        print("  -> FORENSIC FRAMING adds bias BEYOND the byte-identical visualization.")
    if (not s32f) and (not s32r) and (tpr["C2"] > tpr["C0"]) and (fpr["C2"] > fpr["C0"]):
        print("  -> C2 ~ C3 with both far above C0: the VISUAL auxiliary-map effect")
        print("     dominates semantic forensic framing.")
    if (abs(tpr["C1"] - tpr["C3"]) < 0.05 and tpr["C1"] > tpr["C2"] and
            tpr["C1"] > tpr["C0"]):
        print("  -> C1 ~ C3, both above C0/C2: FORENSIC SEMANTIC CONTEXT dominates.")
    print("\n  NOTE: C2 is a minimally framed auxiliary-visualization condition, not a")
    print("  framing-free one; C3-C2 is the increment of forensic framing over a")
    print("  neutral auxiliary description.")


if __name__ == "__main__":
    main()
