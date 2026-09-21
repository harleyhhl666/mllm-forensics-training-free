"""Stage 2 of Gate-1 read-out research: grouped 5-fold CV scorer selection.

Reads the precomputed blind window table and evaluates the SIX frozen scorers.
No anomaly map is recomputed and no new scorer may be added here.

THE SIX FROZEN SCORERS
  A            : max over scales of the within-image standardized window peak
  B_otsu       : inside/outside contrast of an Otsu-thresholded candidate region
  B_p99        : same, 99th-percentile region
  B_p995       : same, 99.5th-percentile region
  C            : A + real-null: per scale, standardize the window peak against the
                 REAL images' distribution of that same statistic, then take the
                 max over scales.  (the A+C combination)
  C_then_B     : C for the score, but the candidate region comes from B_p99, i.e.
                 null-calibrated detection with a tighter region for localization

GROUPED CV: folds are built over coco_id, so every bbox/segm variant and the
paired real image of one source photograph live in exactly one fold. Real-null
statistics for family C are estimated on the TRAINING folds' real images only and
applied to the held-out fold, so no image contributes to its own normalization.

Selection uses ONLY these calibration folds. The test split is never opened.
"""
import argparse, json, os, sys
from collections import defaultdict, Counter
import numpy as np, yaml

SCALES = (0.02, 0.05, 0.10, 0.20)
REGION_RULES = ("otsu", "p99", "p995")
SCORERS = ("A", "B_otsu", "B_p99", "B_p995", "C", "C_then_B")


def auroc(neg, pos):
    neg = np.asarray([x for x in neg if np.isfinite(x)], float)
    pos = np.asarray([x for x in pos if np.isfinite(x)], float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    a = np.concatenate([neg, pos])
    r = a.argsort().argsort() + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def load(path, tool):
    rows = []
    for line in open(path):
        d = json.loads(line)
        if d["tool"] == tool and d.get("valid") and "blind" in d:
            rows.append(d)
    return rows


def make_folds(rows, k, seed):
    """Group by coco_id, stratify folds by category, deterministic given seed."""
    cid_cat, cids_by_cat = {}, defaultdict(list)
    for r in rows:
        cid_cat[r["coco_id"]] = r["category"]
    for cid, cat in cid_cat.items():
        cids_by_cat[cat].append(cid)
    rng = np.random.default_rng(seed)
    assign = {}
    for cat in sorted(cids_by_cat):
        ids = sorted(cids_by_cat[cat])
        rng.shuffle(ids)
        for i, cid in enumerate(ids):          # round-robin keeps categories even
            assign[cid] = i % k
    folds = [[] for _ in range(k)]
    for r in rows:
        folds[assign[r["coco_id"]]].append(r)
    return folds, assign


# ------------------------------------------------------------------ scorers

def score_A(r):
    z = [w["within_image_z"] for w in r["blind"]["scan"]]
    i = int(np.argmax(z))
    return float(z[i]), r["blind"]["scan"][i]["bbox"], SCALES[i], ("scan", i)


def score_B(r, rule):
    i = REGION_RULES.index(rule)
    g = r["blind"]["region"][i]
    if g["degenerate"]:
        return float("-inf"), None, None, ("region", i)
    return float(g["contrast"]), g["bbox"], g["area_frac"], ("region", i)


def fit_null(train_rows):
    """Real-null statistics per scale: median and MAD of the within-image z of
    REAL images. Estimated on training folds only."""
    null = {}
    for i, s in enumerate(SCALES):
        v = np.array([r["blind"]["scan"][i]["within_image_z"]
                      for r in train_rows if r["label"] == "real"], float)
        v = v[np.isfinite(v)]
        if len(v) < 10:
            null[i] = (0.0, 1.0)
            continue
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826 + 1e-9
        null[i] = (med, mad)
    return null


def score_C(r, null):
    best, bi = -np.inf, 0
    for i in range(len(SCALES)):
        med, mad = null[i]
        z = (r["blind"]["scan"][i]["within_image_z"] - med) / mad
        if z > best:
            best, bi = float(z), i
    return best, r["blind"]["scan"][bi]["bbox"], SCALES[bi], ("scan", bi)


def score_C_then_B(r, null):
    s, _, sc, _ = score_C(r, null)
    i = REGION_RULES.index("p99")
    g = r["blind"]["region"][i]
    bbox = None if g["degenerate"] else g["bbox"]
    return s, bbox, sc, ("region", i)


def apply_scorer(name, r, null):
    if name == "A":
        return score_A(r)
    if name.startswith("B_"):
        return score_B(r, name[2:])
    if name == "C":
        return score_C(r, null)
    if name == "C_then_B":
        return score_C_then_B(r, null)
    raise ValueError(name)


def ev_for(r, src):
    kind, i = src
    ev = r.get("eval")
    if not ev:
        return None
    return ev[kind][i]


def tpr_at_fpr(real_scores, fake_scores, fpr):
    rs = np.asarray([x for x in real_scores if np.isfinite(x)], float)
    fs = np.asarray([x for x in fake_scores if np.isfinite(x)], float)
    if len(rs) < 5 or len(fs) < 5:
        return float("nan"), float("nan")
    thr = float(np.percentile(rs, 100 * (1 - fpr)))
    return float(np.mean(fs > thr)), thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--tool", default="ela")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_root = cfg["experiment"]["out_root"]
    seed = cfg["experiment"]["seed"]
    path = os.path.join(out_root, "readout_tgif", "windows_calibration.jsonl")
    rows = load(path, args.tool)
    reals = [r for r in rows if r["label"] == "real"]
    fakes = [r for r in rows if r["label"] == "fake"]
    print(f"tool={args.tool}  valid rows: real {len(reals)}, fake {len(fakes)}")
    print(f"coco_ids: {len({r['coco_id'] for r in rows})}   "
          f"categories: {len({r['category'] for r in rows})}")
    print(f"FROZEN scorers: {SCORERS}")
    print(f"grouped {args.folds}-fold CV over coco_id, category round-robin, seed={seed}\n")

    folds, assign = make_folds(rows, args.folds, seed)
    for i, f in enumerate(folds):
        cids = {r["coco_id"] for r in f}
        print(f"  fold {i}: {len(cids)} coco_ids, {len(f)} images "
              f"({sum(1 for r in f if r['label']=='fake')} fake)")

    # ---- out-of-fold predictions for every scorer
    oof = {s: [] for s in SCORERS}
    for k in range(args.folds):
        train = [r for i, f in enumerate(folds) if i != k for r in f]
        null = fit_null(train)
        for r in folds[k]:
            for s in SCORERS:
                sc, bbox, scale, src = apply_scorer(s, r, null)
                oof[s].append(dict(
                    pair_id=r["pair_id"], coco_id=r["coco_id"], category=r["category"],
                    label=r["label"], tamper_ratio=r["tamper_ratio"], fold=k,
                    score=sc, bbox=bbox, scale=scale, eval=ev_for(r, src)))

    report = {}
    print("\n" + "=" * 96)
    print("GROUPED-CV RESULTS (out-of-fold, calibration only)")
    print(f"{'scorer':<11}{'AUROC':>7}{'±fold':>7}{'TPR@1%':>8}{'TPR@5%':>8}{'TPR@10%':>9}"
          f"{'hit':>7}{'ctr_in':>8}{'IoU':>7}{'prec':>7}")
    for s in SCORERS:
        d = oof[s]
        rv = [x["score"] for x in d if x["label"] == "real"]
        fv = [x["score"] for x in d if x["label"] == "fake"]
        a = auroc(rv, fv)
        # per-fold AUROC spread
        pf = []
        for k in range(args.folds):
            rk = [x["score"] for x in d if x["label"] == "real" and x["fold"] == k]
            fk = [x["score"] for x in d if x["label"] == "fake" and x["fold"] == k]
            if len(rk) >= 5 and len(fk) >= 5:
                pf.append(auroc(rk, fk))
        t1, _ = tpr_at_fpr(rv, fv, 0.01)
        t5, thr5 = tpr_at_fpr(rv, fv, 0.05)
        t10, _ = tpr_at_fpr(rv, fv, 0.10)
        ev = [x["eval"] for x in d if x["label"] == "fake" and x["eval"]]
        hit = float(np.mean([e["hit"] for e in ev])) if ev else float("nan")
        ctr = float(np.mean([e["center_in_mask"] for e in ev])) if ev else float("nan")
        iou = float(np.median([e["iou"] for e in ev])) if ev else float("nan")
        prec = float(np.median([e["precision"] for e in ev])) if ev else float("nan")
        print(f"{s:<11}{a:>7.3f}{(np.std(pf) if pf else float('nan')):>7.3f}"
              f"{t1:>8.3f}{t5:>8.3f}{t10:>9.3f}{hit:>7.3f}{ctr:>8.3f}{iou:>7.3f}{prec:>7.3f}")
        report[s] = dict(auroc=a, auroc_fold_std=float(np.std(pf)) if pf else None,
                         auroc_per_fold=[float(x) for x in pf],
                         tpr_at_1pct=t1, tpr_at_5pct=t5, tpr_at_10pct=t10,
                         threshold_at_5pct=thr5,
                         blind_region_hit_rate=hit, center_in_mask_rate=ctr,
                         median_iou=iou, median_precision=prec,
                         real_median=float(np.median(rv)), fake_median=float(np.median(fv)),
                         real_p95=float(np.percentile(rv, 95)))

    # ---- detail for the best scorer by CV AUROC
    best = max(SCORERS, key=lambda s: (report[s]["auroc"] if np.isfinite(report[s]["auroc"]) else -1))
    print(f"\nbest by grouped-CV AUROC: {best}")
    d = oof[best]
    rv = [x["score"] for x in d if x["label"] == "real"]
    _, thr5 = tpr_at_fpr(rv, [x["score"] for x in d if x["label"] == "fake"], 0.05)

    print(f"\nscore distribution ({best}):")
    for lab in ("real", "fake"):
        v = np.array([x["score"] for x in d if x["label"] == lab], float)
        v = v[np.isfinite(v)]
        print(f"  {lab:<5} n={len(v):>4} p5={np.percentile(v,5):7.2f} med={np.median(v):7.2f} "
              f"p95={np.percentile(v,95):7.2f} max={v.max():7.2f}")

    print(f"\nby tamper_ratio ({best}, thr@5%FPR={thr5:.3f}):")
    for lo, hi in [(0, .01), (.01, .03), (.03, .10), (.10, .60)]:
        sub = [x for x in d if x["label"] == "fake" and lo <= x["tamper_ratio"] < hi]
        if not sub:
            continue
        t = float(np.mean([x["score"] > thr5 for x in sub]))
        ev = [x["eval"] for x in sub if x["eval"]]
        h = float(np.mean([e["hit"] for e in ev])) if ev else float("nan")
        io = float(np.median([e["iou"] for e in ev])) if ev else float("nan")
        print(f"  [{lo:.2f},{hi:.2f}) n={len(sub):>4} TPR={t:.3f} hit={h:.3f} IoU={io:.3f}")

    print(f"\nby category ({best}, TPR@5%FPR / blind hit):")
    for cat in sorted({x["category"] for x in d}):
        sub = [x for x in d if x["label"] == "fake" and x["category"] == cat]
        if len(sub) < 3:
            continue
        t = float(np.mean([x["score"] > thr5 for x in sub]))
        ev = [x["eval"] for x in sub if x["eval"]]
        h = float(np.mean([e["hit"] for e in ev])) if ev else float("nan")
        print(f"  {cat:<15} n={len(sub):>4} TPR={t:.3f} hit={h:.3f}")

    od = os.path.join(out_root, "readout_tgif")
    json.dump(dict(tool=args.tool, folds=args.folds, seed=seed,
                   frozen_scales=list(SCALES), frozen_region_rules=list(REGION_RULES),
                   frozen_scorers=list(SCORERS), best_by_cv_auroc=best,
                   results=report),
              open(os.path.join(od, f"cv_selection_{args.tool}.json"), "w"), indent=1)
    json.dump({s: oof[s] for s in SCORERS},
              open(os.path.join(od, f"oof_{args.tool}.json"), "w"))
    print(f"\nwrote {od}/cv_selection_{args.tool}.json")
    print("NOTHING FROZEN YET.")


if __name__ == "__main__":
    main()
