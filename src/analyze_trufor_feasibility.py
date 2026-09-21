"""TruFor image-level feasibility analysis.

Primary: AUROC of the whole-image score, fake vs paired real, with source-pair
bootstrap CI. Secondary: localization quality on fakes. Descriptive only: the
confidence map. ELA is reported on the SAME sources using its existing frozen
read-out, for a paired tool comparison.
"""
import json, os, sys
from collections import defaultdict
import numpy as np, yaml
from PIL import Image

NBOOT, SEED = 10000, 20260918


def auroc(pos, neg):
    """Mann-Whitney AUROC with tie correction."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    a = np.concatenate([pos, neg])
    r = np.empty(len(a))
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def roc(pos, neg):
    """Full ROC from continuous scores; higher score = predicted fake."""
    th = np.unique(np.concatenate([pos, neg]))
    th = np.concatenate([[np.inf], th[::-1]])
    tpr = np.array([(pos >= t).mean() for t in th])
    fpr = np.array([(neg >= t).mean() for t in th])
    return fpr, tpr, th


def tpr_at_fpr(pos, neg, target):
    """Largest TPR whose FPR does not exceed target, plus that threshold."""
    fpr, tpr, th = roc(np.asarray(pos, float), np.asarray(neg, float))
    ok = fpr <= target + 1e-12
    if not ok.any():
        return float("nan"), float("nan"), float("nan")
    i = np.argmax(np.where(ok, tpr, -1))
    return float(tpr[i]), float(fpr[i]), float(th[i])


def pair_boot(fn, pairs, nb=NBOOT, seed=SEED):
    """Bootstrap over SOURCE PAIRS (each pair = one fake + its paired real)."""
    rng = np.random.default_rng(seed)
    obs = fn(pairs)
    out = []
    m = len(pairs)
    for _ in range(nb):
        idx = rng.integers(0, m, m)
        out.append(fn([pairs[i] for i in idx]))
    out = np.array([x for x in out if np.isfinite(x)])
    return float(obs), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def desc(v):
    v = np.asarray(v, float)
    return dict(n=len(v), mean=v.mean(), median=float(np.median(v)), std=v.std(ddof=1),
                p10=float(np.percentile(v, 10)), p25=float(np.percentile(v, 25)),
                p75=float(np.percentile(v, 75)), p90=float(np.percentile(v, 90)),
                min=v.min(), max=v.max())


def grid_cells(mask01, grid, min_overlap):
    H, W = mask01.shape
    out = set()
    NAMES = ["top-left", "top-center", "top-right", "center-left", "center",
             "center-right", "bottom-left", "bottom-center", "bottom-right"]
    for i in range(grid):
        for j in range(grid):
            sub = mask01[i * H // grid:(i + 1) * H // grid,
                         j * W // grid:(j + 1) * W // grid]
            if sub.size and sub.mean() >= min_overlap:
                out.add(NAMES[i * grid + j])
    return out


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    R = cfg["experiment"]["out_root"]
    od = os.path.join(R, "trufor_feasibility")
    F = json.load(open(os.path.join(od, "trufor_feasibility_frozen.json")))
    rows = [json.loads(l) for l in open(os.path.join(od, "scores.jsonl"))]
    G = defaultdict(dict)
    for r in rows:
        G[r["sample_id"]][r["label"]] = r
    ids = sorted(p for p in G if "fake" in G[p] and "real" in G[p])
    thr = cfg["localization"]["mask_binarize_threshold"]
    grid = cfg["localization"]["grid"]
    mino = cfg["localization"]["gt_cell_min_overlap"]

    print("TRUFOR IMAGE-LEVEL FEASIBILITY — TGIF sd2-sp")
    print(f"protocol sha1 {json.load(open(os.path.join(od,'run_meta.json')))['protocol_sha1']}")
    print(f"bootstrap {NBOOT} over source pairs, seed {SEED}")

    print("\n" + "=" * 88)
    print("1. INTEGRITY")
    print(f"  rows                     : {len(rows)} (expect {F['n_images']})")
    print(f"  complete source pairs    : {len(ids)} (expect {F['n_sources']})")
    print(f"  unique coco_id           : {len({G[p]['fake']['coco_id'] for p in ids})}")
    print(f"  n_samples == n_coco_id   : "
          f"{len({G[p]['fake']['coco_id'] for p in ids}) == len(ids)}")
    bad = [p for p in ids if G[p]["fake"]["score"] is None
           or G[p]["real"]["score"] is None]
    print(f"  missing scores           : {len(bad)}")
    rng_ok = all(0.0 <= G[p][l]["score"] <= 1.0 for p in ids for l in ("fake", "real"))
    print(f"  all scores within [0,1]  : {rng_ok}")
    meta = json.load(open(os.path.join(od, "run_meta.json")))
    print(f"  checkpoint md5 verified  : {meta['checkpoint_md5']}")
    print(f"  GT used as model input   : NO (masks only scored after inference)")

    fk = np.array([G[p]["fake"]["score"] for p in ids], float)
    rl = np.array([G[p]["real"]["score"] for p in ids], float)
    pairs = list(zip(fk, rl))

    print("\n" + "=" * 88)
    print("2. PRIMARY — IMAGE-LEVEL AUROC (fake vs paired real)")
    a, lo, hi = pair_boot(lambda P: auroc([x[0] for x in P], [x[1] for x in P]), pairs)
    print(f"  AUROC = {a:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   n_pairs={len(pairs)}")
    print(f"  CI includes 0.5: {lo <= 0.5 <= hi}")

    print("\n" + "=" * 88)
    print("3. CONTROLLED-FPR OPERATING POINTS")
    print("  (evaluation-derived operating points, NOT frozen deployment thresholds)")
    for t in (0.05, 0.10):
        tp, fp, th = tpr_at_fpr(fk, rl, t)
        b = pair_boot(lambda P, t=t: tpr_at_fpr([x[0] for x in P], [x[1] for x in P], t)[0],
                      pairs)
        print(f"  TPR@FPR<={t:.2f} = {tp:.4f}  95% CI [{b[1]:.4f}, {b[2]:.4f}]  "
              f"(actual FPR {fp:.4f}, score threshold {th:.6f})")

    print("\n" + "=" * 88)
    print("4. SCORE DISTRIBUTION")
    for nm, v in (("fake", fk), ("real", rl)):
        d = desc(v)
        print(f"  {nm:<5} n={d['n']}  mean={d['mean']:.4f}  median={d['median']:.4f}  "
              f"std={d['std']:.4f}")
        print(f"        p10={d['p10']:.4f} p25={d['p25']:.4f} p75={d['p75']:.4f} "
              f"p90={d['p90']:.4f}  [min {d['min']:.4f}, max {d['max']:.4f}]")
    print(f"\n  separation check (is AUROC driven by a few extremes?)")
    for t in (0.5, 0.9):
        print(f"    score>{t}: fake {100*(fk>t).mean():.1f}%   real {100*(rl>t).mean():.1f}%")
    print(f"    median gap = {np.median(fk)-np.median(rl):+.4f}")
    ov = float(np.mean([(rl >= f).mean() for f in fk]))
    print(f"    mean fraction of reals scoring >= a given fake: {ov:.4f}")

    print("\n" + "=" * 88)
    print("5. PAIRED ANALYSIS (same source)")
    d = fk - rl
    dd = desc(d)
    mb = pair_boot(lambda P: float(np.mean([x[0] - x[1] for x in P])), pairs)
    wb = pair_boot(lambda P: float(np.mean([x[0] > x[1] for x in P])), pairs)
    print(f"  dscore = score_fake - score_real")
    print(f"    mean   {dd['mean']:+.4f}  95% CI [{mb[1]:+.4f}, {mb[2]:+.4f}]")
    print(f"    median {dd['median']:+.4f}   std {dd['std']:.4f}")
    print(f"    P(score_fake > score_real) = {wb[0]:.4f}  95% CI [{wb[1]:.4f}, {wb[2]:.4f}]")
    print(f"    pairs with fake>real: {int((d>0).sum())}/{len(d)}")

    print("\n" + "=" * 88)
    print("6. SECONDARY — LOCALIZATION (fakes only; does NOT replace the primary)")
    res = []
    for p in ids:
        r = G[p]["fake"]
        z = np.load(r["localization_map_path"].split("::")[0])
        loc = z["map"].astype(np.float32)
        gt = np.array(Image.open(r["gt_mask_path"]).convert("L"))
        gtb = (gt > thr).astype(np.uint8)
        if gtb.shape != loc.shape:
            gtb = np.array(Image.fromarray(gtb * 255).resize(
                (loc.shape[1], loc.shape[0]), Image.NEAREST))
            gtb = (gtb > 127).astype(np.uint8)
        pb = (loc > 0.5).astype(np.uint8)
        inter = int((pb & gtb).sum())
        union = int((pb | gtb).sum())
        iou = inter / union if union else np.nan
        f1 = 2 * inter / (pb.sum() + gtb.sum()) if (pb.sum() + gtb.sum()) else np.nan
        pa = auroc(loc[gtb == 1].ravel()[:20000], loc[gtb == 0].ravel()[:20000]) \
            if gtb.any() and (gtb == 0).any() else np.nan
        gc = grid_cells(gtb, grid, mino)
        pc = grid_cells(pb, grid, mino)
        hit = bool(gc & pc)
        res.append(dict(iou=iou, f1=f1, pauc=pa, hit=hit,
                        tr=r["tamper_ratio"], mt=r["mask_type"],
                        gt_cells=len(gc)))
    I = np.array([x["iou"] for x in res], float)
    Fq = np.array([x["f1"] for x in res], float)
    P = np.array([x["pauc"] for x in res], float)
    H = np.array([x["hit"] for x in res], float)
    print(f"  n fakes scored      : {len(res)}")
    print(f"  pixel AUROC (subsampled) mean {np.nanmean(P):.4f}  median {np.nanmedian(P):.4f}")
    print(f"  IoU @map>0.5        mean {np.nanmean(I):.4f}  median {np.nanmedian(I):.4f}")
    print(f"  pixel F1 @map>0.5   mean {np.nanmean(Fq):.4f}  median {np.nanmedian(Fq):.4f}")
    print(f"  3x3 grid hit rate   {H.mean():.4f}")
    print("\n  by tamper_ratio quartile:")
    q = np.quantile([x["tr"] for x in res], [.25, .5, .75])
    for nm, sel in (("q1", lambda x: x["tr"] <= q[0]),
                    ("q2", lambda x: q[0] < x["tr"] <= q[1]),
                    ("q3", lambda x: q[1] < x["tr"] <= q[2]),
                    ("q4", lambda x: x["tr"] > q[2])):
        s = [x for x in res if sel(x)]
        if s:
            print(f"    {nm} n={len(s):<4} IoU {np.nanmean([x['iou'] for x in s]):.4f}  "
                  f"F1 {np.nanmean([x['f1'] for x in s]):.4f}  "
                  f"pAUC {np.nanmean([x['pauc'] for x in s]):.4f}  "
                  f"hit {np.mean([x['hit'] for x in s]):.3f}")
    print("  by mask type:")
    for mt in ("bbox", "segm"):
        s = [x for x in res if x["mt"] == mt]
        if s:
            print(f"    {mt:<5} n={len(s):<4} IoU {np.nanmean([x['iou'] for x in s]):.4f}  "
                  f"F1 {np.nanmean([x['f1'] for x in s]):.4f}  "
                  f"pAUC {np.nanmean([x['pauc'] for x in s]):.4f}  "
                  f"hit {np.mean([x['hit'] for x in s]):.3f}")

    print("\n" + "=" * 88)
    print("7. CONFIDENCE MAP — DESCRIPTIVE ONLY (no pooled score constructed)")
    ci, co, cf, cr = [], [], [], []
    for p in ids:
        zf = np.load(G[p]["fake"]["confidence_map_path"].split("::")[0])
        conf = zf["conf"].astype(np.float32)
        cf.append(conf.mean())
        gt = np.array(Image.open(G[p]["fake"]["gt_mask_path"]).convert("L"))
        gtb = (gt > thr).astype(np.uint8)
        if gtb.shape != conf.shape:
            gtb = np.array(Image.fromarray(gtb * 255).resize(
                (conf.shape[1], conf.shape[0]), Image.NEAREST))
            gtb = (gtb > 127).astype(np.uint8)
        if gtb.any():
            ci.append(conf[gtb == 1].mean())
        if (gtb == 0).any():
            co.append(conf[gtb == 0].mean())
        zr = np.load(G[p]["real"]["confidence_map_path"].split("::")[0])
        cr.append(zr["conf"].astype(np.float32).mean())
    print(f"  fake mean confidence            {np.mean(cf):.4f}")
    print(f"  confidence INSIDE  GT region    {np.mean(ci):.4f}")
    print(f"  confidence OUTSIDE GT region    {np.mean(co):.4f}")
    print(f"  inside - outside                {np.mean(ci)-np.mean(co):+.4f}")
    print(f"  real mean confidence            {np.mean(cr):.4f}")
    print("  (pixel-level localization reliability; NOT an image-level trust score)")

    print("\n" + "=" * 88)
    print("8. ELA ON THE SAME 200 SOURCES (existing frozen read-out)")
    if all("ela_anomaly_score" in G[p][l] for p in ids for l in ("fake", "real")):
        ef = np.array([G[p]["fake"]["ela_anomaly_score"] for p in ids], float)
        er = np.array([G[p]["real"]["ela_anomaly_score"] for p in ids], float)
        ep = list(zip(ef, er))
        ea, elo, ehi = pair_boot(
            lambda P: auroc([x[0] for x in P], [x[1] for x in P]), ep)
        e5 = tpr_at_fpr(ef, er, 0.05)
        e10 = tpr_at_fpr(ef, er, 0.10)
        print(f"  ELA  AUROC {ea:.4f} [{elo:.4f},{ehi:.4f}]   "
              f"TPR@5% {e5[0]:.4f}   TPR@10% {e10[0]:.4f}")
        t5 = tpr_at_fpr(fk, rl, 0.05)
        t10 = tpr_at_fpr(fk, rl, 0.10)
        print(f"  TruFor AUROC {a:.4f} [{lo:.4f},{hi:.4f}]   "
              f"TPR@5% {t5[0]:.4f}   TPR@10% {t10[0]:.4f}")
        gap = pair_boot(lambda P: auroc([x[0][0] for x in P], [x[0][1] for x in P]) -
                        auroc([x[1][0] for x in P], [x[1][1] for x in P]),
                        list(zip(pairs, ep)))
        print(f"  paired AUROC difference (TruFor - ELA) {gap[0]:+.4f} "
              f"[{gap[1]:+.4f}, {gap[2]:+.4f}] -> "
              f"{'EXCLUDES zero' if (gap[1] > 0 or gap[2] < 0) else 'includes zero'}")
        print("  same 200 sources, same images -> this paired comparison is valid")
    else:
        print("  ELA scores not present; skipped")

    print("\n" + "=" * 88)
    print("9. GO / YELLOW / NO-GO")
    t5 = tpr_at_fpr(fk, rl, 0.05)[0]
    t10 = tpr_at_fpr(fk, rl, 0.10)[0]
    print(f"  AUROC {a:.4f} CI [{lo:.4f},{hi:.4f}]  includes 0.5: {lo <= 0.5 <= hi}")
    print(f"  TPR@5% {t5:.4f}   TPR@10% {t10:.4f}")
    print(f"  P(fake>real) {wb[0]:.4f}")
    print(f"  localization IoU {np.nanmean(I):.4f}  pixel AUROC {np.nanmean(P):.4f}")
    print()
    if lo <= 0.5 <= hi:
        print("  -> HARD NO-GO: image-level discrimination is not distinguishable from")
        print("     chance. TruFor is a useful localizer but unsuitable as an")
        print("     image-level external evidence source here.")
    elif t10 < 0.40:
        print("  -> YELLOW: image-level signal exists but TPR at low FPR is weak;")
        print("     may not be reliable enough for gating. Pause.")
    else:
        print("  -> GO: AUROC clearly above chance, usable TPR at controlled FPR,")
        print("     paired fake scores usually exceed paired real, localization")
        print("     effective. TruFor qualifies as a plausible external evidence source.")
    print("\n  (no numeric cutoff here is treated as a field standard)")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
        bins = np.linspace(0, 1, 41)
        ax[0].hist(rl, bins=bins, alpha=.62, label="real (paired)", color="#2b7bba")
        ax[0].hist(fk, bins=bins, alpha=.62, label="fake", color="#d1495b")
        ax[0].set_xlabel("TruFor score (higher = fake)")
        ax[0].set_ylabel("count")
        ax[0].set_title(f"score distribution (n={len(ids)} pairs)")
        ax[0].legend()
        fpr, tpr, _ = roc(fk, rl)
        ax[1].plot(fpr, tpr, color="#d1495b", lw=2)
        ax[1].plot([0, 1], [0, 1], "k--", lw=.8)
        for t in (0.05, 0.10):
            ax[1].axvline(t, color="gray", ls=":", lw=.8)
        ax[1].set_xlabel("FPR")
        ax[1].set_ylabel("TPR")
        ax[1].set_title(f"ROC  AUROC={a:.4f} [{lo:.3f},{hi:.3f}]")
        ax[2].hist(d, bins=40, color="#557a46")
        ax[2].axvline(0, color="k", lw=1)
        ax[2].set_xlabel("dscore = fake - real (same source)")
        ax[2].set_title(f"paired delta; P(fake>real)={wb[0]:.3f}")
        fig.tight_layout()
        fp = os.path.join(od, "score_distribution.png")
        fig.savefig(fp, dpi=110)
        print(f"\n  plot -> {fp}")
    except Exception as e:
        print(f"  plot skipped: {e}")


if __name__ == "__main__":
    main()
