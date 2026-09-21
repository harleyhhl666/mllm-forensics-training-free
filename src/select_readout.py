"""Gate-1 read-out selection by GROUPED 5-fold cross-validation.

Discipline enforced here:
  * Folds are grouped by coco_id, so all bbox/segm variants and the paired real
    image of one source photograph always sit in the same fold. Category
    proportions are kept close across folds.
  * Family C's real-null statistics are built ONLY from the training folds' REAL
    images and then applied to the held-out fold. Building the null on all data
    would leak the test fold's own statistics into its score.
  * Scorer comparison uses held-out folds only. The frozen search space is the
    one declared in readout.py; nothing is added after seeing results.
  * The TGIF test split is never opened by this script.
  * GT masks are used only after a read-out has produced its candidate region.
"""
import json, os, sys, argparse
from collections import defaultdict, Counter
import numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readout import (SCALES, REGION_RULES, multi_scale_scan, region_contrast,
                     build_real_null, real_null_multiscale, bbox_hit,
                     expected_random_hit)


def auroc(neg, pos):
    neg = np.asarray(neg, float); pos = np.asarray(pos, float)
    neg = neg[np.isfinite(neg)]; pos = pos[np.isfinite(pos)]
    if not len(neg) or not len(pos):
        return float("nan")
    a = np.concatenate([neg, pos]); r = a.argsort().argsort() + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def tpr_at_fpr(rv, fv, fpr):
    if not len(rv) or not len(fv):
        return float("nan"), float("nan")
    thr = float(np.percentile(rv, 100 * (1 - fpr)))
    return float(np.mean(np.asarray(fv) > thr)), thr


def make_folds(meta, k, seed):
    """Grouped folds by coco_id, balanced per category."""
    cid_cat, per_cid = {}, defaultdict(list)
    for m in meta:
        cid_cat[m["coco_id"]] = m["category"]
        per_cid[m["coco_id"]].append(m["pair_id"])
    by_cat = defaultdict(list)
    for cid, cat in cid_cat.items():
        by_cat[cat].append(cid)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for cat in sorted(by_cat):
        ids = sorted(by_cat[cat])
        rng.shuffle(ids)
        for i, cid in enumerate(ids):          # round-robin keeps categories even
            folds[i % k].append(cid)
    return folds, cid_cat, per_cid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    seed = cfg["experiment"]["seed"]
    cache = os.path.join(cfg["experiment"]["out_root"], "ela_cache")
    out_dir = os.path.join(cfg["experiment"]["out_root"], "readout_cv")
    os.makedirs(out_dir, exist_ok=True)
    meta = [m for m in json.load(open(os.path.join(cache, "cache_meta.json")))
            if m["real_valid"] and m["fake_valid"]]
    print(f"usable calibration pairs (ELA valid both sides): {len(meta)}")

    folds, cid_cat, per_cid = make_folds(meta, args.folds, seed)
    print(f"grouped {args.folds}-fold by coco_id:")
    for i, f in enumerate(folds):
        cc = Counter(cid_cat[c] for c in f)
        print(f"  fold {i}: {len(f)} coco_ids, {sum(len(per_cid[c]) for c in f)} pairs, "
              f"{len(cc)} categories")

    by_pid = {m["pair_id"]: m for m in meta}

    def load(pid):
        z = np.load(os.path.join(cache, pid + ".npz"))
        return (np.asarray(z["real"], np.float32), np.asarray(z["fake"], np.float32),
                np.asarray(z["mask_eval_only"]).astype(bool))

    # ---- every read-out in the FROZEN space
    readouts = {"A_multi_scale": ("A", None)}
    for r in REGION_RULES:
        readouts[f"B_region_{r}"] = ("B", r)
    readouts["C_real_null_multiscale"] = ("C", None)

    results = defaultdict(lambda: defaultdict(list))   # name -> field -> values
    rows = []

    for fi, test_cids in enumerate(folds):
        train_cids = [c for j, f in enumerate(folds) if j != fi for c in f]
        test_pids = [p for c in test_cids for p in per_cid[c]]
        train_pids = [p for c in train_cids for p in per_cid[c]]

        # family C: real-null from TRAINING folds' real images only
        train_real = []
        for p in train_pids:
            z = np.load(os.path.join(cache, p + ".npz"))
            train_real.append(np.asarray(z["real"], np.float32))
        null = build_real_null(train_real, SCALES)
        del train_real

        for p in test_pids:
            rm, fm, mk = load(p)
            m = by_pid[p]
            for name, (fam, param) in readouts.items():
                for lab, mp in (("real", rm), ("fake", fm)):
                    if fam == "A":
                        o = multi_scale_scan(mp)
                    elif fam == "B":
                        o = region_contrast(mp, param)
                    else:
                        o = real_null_multiscale(mp, null)
                    results[name][f"{lab}_score"].append(o["score"])
                    if lab == "fake":
                        h = bbox_hit(o["candidate_bbox"], mk)
                        rnd = expected_random_hit(o["candidate_bbox"], mk)
                        for kk, vv in h.items():
                            results[name][f"fake_{kk}"].append(vv)
                        results[name]["fake_random_hit"].append(rnd)
                        results[name]["fake_tamper"].append(m["tamper_ratio"])
                        results[name]["fake_cat"].append(m["category"])
                        results[name]["fake_scale"].append(o["candidate_scale"])
                        rows.append(dict(fold=fi, pair_id=p, readout=name,
                                         score=o["score"], scale=o["candidate_scale"],
                                         bbox=o["candidate_bbox"],
                                         tamper_ratio=m["tamper_ratio"],
                                         category=m["category"], **h,
                                         random_hit=rnd))
        print(f"  fold {fi} done ({len(test_pids)} pairs)", flush=True)

    json.dump(rows, open(os.path.join(out_dir, "cv_rows.json"), "w"), indent=1)

    # -------------------------------------------------- report
    print("\n" + "=" * 92)
    print("GROUPED 5-FOLD CV, held-out folds pooled  (ELA map, blind read-out)")
    print(f"{'read-out':<26}{'AUROC':>7}{'TPR@1%':>8}{'TPR@5%':>8}{'TPR@10%':>9}"
          f"{'hit':>7}{'rand':>7}{'cIn':>6}{'IoU':>7}")
    summary = {}
    for name in readouts:
        r = results[name]
        rv, fv = r["real_score"], r["fake_score"]
        a = auroc(rv, fv)
        t1, th1 = tpr_at_fpr(rv, fv, 0.01)
        t5, th5 = tpr_at_fpr(rv, fv, 0.05)
        t10, th10 = tpr_at_fpr(rv, fv, 0.10)
        hit = float(np.mean(r["fake_hit"]))
        rnd = float(np.nanmean(r["fake_random_hit"]))
        cin = float(np.mean(r["fake_center_in_mask"]))
        iou = float(np.mean(r["fake_iou"]))
        print(f"{name:<26}{a:>7.3f}{t1:>8.3f}{t5:>8.3f}{t10:>9.3f}"
              f"{hit:>7.3f}{rnd:>7.3f}{cin:>6.3f}{iou:>7.3f}")
        summary[name] = dict(auroc=a, tpr_at_1pct=t1, tpr_at_5pct=t5, tpr_at_10pct=t10,
                             thr_1pct=th1, thr_5pct=th5, thr_10pct=th10,
                             hit_rate=hit, random_hit=rnd, center_in_mask=cin,
                             mean_iou=iou,
                             real_median=float(np.median(rv)),
                             fake_median=float(np.median(fv)),
                             n_real=len(rv), n_fake=len(fv))

    # blind localization vs its own random baseline
    print("\nBLIND CANDIDATE-REGION LOCALIZATION (hit vs size-matched random placement)")
    for name in readouts:
        r = results[name]
        hit = np.asarray(r["fake_hit"], float)
        rnd = np.asarray(r["fake_random_hit"], float)
        ok = np.isfinite(rnd)
        d = float(hit[ok].mean() - rnd[ok].mean())
        z = d / (np.sqrt(max(rnd[ok].mean() * (1 - rnd[ok].mean()) / ok.sum(), 1e-12)))
        print(f"  {name:<26} hit={hit.mean():.3f}  random={rnd[ok].mean():.3f}  "
              f"diff={d:+.3f}  z={z:+.1f}  center_in_mask={np.mean(r['fake_center_in_mask']):.3f}")
        summary[name]["hit_minus_random"] = d
        summary[name]["hit_z"] = float(z)

    # stratification for the strongest read-out by AUROC
    best = max(summary, key=lambda k: (summary[k]["auroc"]
                                       if np.isfinite(summary[k]["auroc"]) else -1))
    print(f"\nSTRATIFIED VIEW for best-AUROC read-out: {best}")
    r = results[best]
    thr5 = summary[best]["thr_5pct"]
    fs = np.asarray(r["fake_score"], float)
    tr = np.asarray(r["fake_tamper"], float)
    hitv = np.asarray(r["fake_hit"], float)
    print(f"  by tamper_ratio (threshold @5% real FPR = {thr5:.3f}):")
    strat = {}
    for lo, hi in [(0, .01), (.01, .03), (.03, .10), (.10, .60)]:
        s = (tr >= lo) & (tr < hi)
        if s.sum() == 0:
            continue
        print(f"    [{lo:.2f},{hi:.2f}) n={int(s.sum()):>4}  TPR={np.mean(fs[s] > thr5):.3f}  "
              f"hit={hitv[s].mean():.3f}")
        strat[f"{lo}-{hi}"] = dict(n=int(s.sum()), tpr=float(np.mean(fs[s] > thr5)),
                                   hit=float(hitv[s].mean()))
    cats = np.asarray(r["fake_cat"])
    print("  by category:")
    catstab = {}
    for c in sorted(set(cats.tolist())):
        s = cats == c
        v = dict(n=int(s.sum()), tpr=float(np.mean(fs[s] > thr5)), hit=float(hitv[s].mean()))
        catstab[c] = v
        print(f"    {c:<16} n={v['n']:>4}  TPR={v['tpr']:.3f}  hit={v['hit']:.3f}")
    sc = Counter(r["fake_scale"])
    print(f"  chosen candidate scale distribution: {dict(sorted(sc.items(), key=lambda x: str(x[0])))}")

    summary[best]["tamper_strata"] = strat
    summary[best]["category_stability"] = catstab
    summary["_meta"] = dict(folds=args.folds, seed=seed, n_pairs=len(meta),
                            frozen_scales=list(SCALES), frozen_region_rules=list(REGION_RULES),
                            best_by_auroc=best)
    json.dump(summary, open(os.path.join(out_dir, "cv_summary.json"), "w"), indent=1)
    print(f"\nwrote {out_dir}/cv_summary.json and cv_rows.json")
    print("NOTHING FROZEN YET.")


if __name__ == "__main__":
    main()
