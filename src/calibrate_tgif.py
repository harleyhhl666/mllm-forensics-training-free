"""TGIF Gate-1 calibration: TOOL LAYER ONLY. No MLLM is loaded or run here.

Two levels of analysis, as required:

  A. Image-level discrimination -- blind anomaly score on paired real vs fake.
     Every scorer reads ONLY the tool's anomaly map. No GT mask, no mask-guided
     pooling, no per-image normalization by GT. This is what Gate 1 will use.

  B. Spatial localization quality -- anomaly map vs GT mask, on calibration only.
     GT is allowed here because the purpose is to decide whether the tool carries
     spatially relevant evidence at all. Nothing computed in (B) feeds a threshold.

Candidate scorers include boundary-sensitive ones: TGIF-sp is a pixel-exact
splice, so the discontinuity lives at the mask BOUNDARY, and a scorer that only
looks for a hot blob may miss it. Adding these candidates is a pre-registered
hypothesis about the physics, not post-hoc fishing: all candidates are scored on
calibration only and the choice is recorded.
"""
import json, os, sys, argparse
from collections import defaultdict
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools, CELL_NAMES


# ---------------------------------------------------------------- blind scorers
# signature: (map01) -> (score, peak_cell_index) ; grid fixed at 3

def _cells(m, grid=3):
    h, w = m.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    return np.array([m[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean()
                     for i in range(grid) for j in range(grid)], np.float32)


def _patch(m, frac=0.06):
    h, w = m.shape
    k = max(8, int(round(min(h, w) * frac)) | 1)
    return cv2.blur(m, (k, k))


def _peak_cell(m, grid=3):
    return int(_cells(_patch(m), grid).argmax())


def sc_cell_peak_z(m):
    c = _cells(m)
    return float((c.max() - c.mean()) / (c.std() + 1e-6)), int(c.argmax())


def sc_patch_peak_z(m):
    p = _patch(m)
    return float((p.max() - p.mean()) / (p.std() + 1e-6)), _peak_cell(m)


def sc_patch_top1pct_z(m):
    p = _patch(m).ravel()
    t = p[p >= np.percentile(p, 99.0)]
    return float((t.mean() - p.mean()) / (p.std() + 1e-6)), _peak_cell(m)


def sc_global_mean(m):
    return float(m.mean()), _peak_cell(m)


def sc_kurtosis(m):
    """Heavy-tailed response distribution: a confined anomaly makes the map's
    value distribution more peaked/skewed than a uniformly textured image."""
    x = _patch(m).ravel().astype(np.float64)
    s = x.std() + 1e-9
    return float((((x - x.mean()) / s) ** 4).mean() - 3.0), _peak_cell(m)


def sc_grad_max(m):
    """BOUNDARY-sensitive: a pixel-exact splice creates a step in the residual
    statistics along the mask outline. Measure the strongest smoothed gradient."""
    p = _patch(m, frac=0.03)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.magnitude(gx, gy)
    g = cv2.blur(g, (9, 9))
    return float((g.max() - g.mean()) / (g.std() + 1e-6)), _peak_cell(g)


def sc_grad_top1pct(m):
    p = _patch(m, frac=0.03)
    gx = cv2.Sobel(p, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(p, cv2.CV_32F, 0, 1, ksize=3)
    g = cv2.blur(cv2.magnitude(gx, gy), (9, 9))
    f = g.ravel()
    t = f[f >= np.percentile(f, 99.0)]
    return float((t.mean() - f.mean()) / (f.std() + 1e-6)), _peak_cell(g)


def sc_local_std_contrast(m):
    """Local heterogeneity contrast: ratio of the most-variable window's local
    std to the image's median local std. Catches 'this patch's noise structure
    differs from the rest' without assuming the direction of the difference."""
    h, w = m.shape
    k = max(8, int(round(min(h, w) * 0.06)) | 1)
    mu = cv2.blur(m, (k, k))
    mu2 = cv2.blur(m * m, (k, k))
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0))
    med = float(np.median(sd)) + 1e-6
    return float(np.percentile(sd, 99.5) / med), _peak_cell(sd)


def sc_twosided_mad(m):
    c = _cells(_patch(m))
    med = float(np.median(c)); mad = float(np.median(np.abs(c - med))) + 1e-6
    dev = (c - med) / mad
    k = int(np.abs(dev).argmax())
    return float(abs(dev[k])), k


SCORERS = {
    "cell_peak_z": sc_cell_peak_z,
    "patch_peak_z": sc_patch_peak_z,
    "patch_top1pct_z": sc_patch_top1pct_z,
    "global_mean": sc_global_mean,
    "kurtosis": sc_kurtosis,
    "grad_max": sc_grad_max,
    "grad_top1pct": sc_grad_top1pct,
    "local_std_contrast": sc_local_std_contrast,
    "twosided_mad": sc_twosided_mad,
}


# ---------------------------------------------------------------- evaluation

def auroc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    neg = neg[np.isfinite(neg)]; pos = pos[np.isfinite(pos)]
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    a = np.concatenate([neg, pos])
    r = a.argsort().argsort() + 1.0
    return float((r[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def pixel_auroc(m, mask, cap=120000, rs=None):
    rs = rs or np.random.default_rng(0)
    if m.shape != mask.shape:
        m = cv2.resize(m, (mask.shape[1], mask.shape[0]))
    pos, neg = m[mask], m[~mask]
    if pos.size < 20 or neg.size < 20:
        return float("nan")
    if pos.size > cap: pos = rs.choice(pos, cap, replace=False)
    if neg.size > cap: neg = rs.choice(neg, cap, replace=False)
    return auroc(neg, pos)


def gt_cells(maskf, grid, mo):
    h, w = maskf.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    out = [i * grid + j for i in range(grid) for j in range(grid)
           if maskf[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean() >= mo]
    if not out:
        cov = [maskf[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean()
               for i in range(grid) for j in range(grid)]
        out = [int(np.argmax(cov))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--limit", type=int, default=0, help="cap calibration pairs (0=all)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = cfg["datasets"]["tgif_sp"]
    root = ds["root"]
    grid = cfg["localization"]["grid"]
    mo = cfg["localization"]["gt_cell_min_overlap"]
    mbt = cfg["localization"]["mask_binarize_threshold"]
    tool_cfg = cfg["forensic_tools"]
    out_dir = os.path.join(cfg["experiment"]["out_root"], "calibration_tgif")
    os.makedirs(out_dir, exist_ok=True)

    split = json.load(open(os.path.join(cfg["experiment"]["out_root"],
                                        "splits_tgif", "split_frozen.json")))
    cal = set(split["calibration_pair_ids"])
    recs = [r for r in json.load(open(ds["index"])) if r["pair_id"] in cal]
    if args.limit:
        recs = recs[:args.limit]
    print(f"calibration pairs: {len(recs)}  (from {split['n_calibration_ids']} coco_ids)")
    print(f"mask binarization: >{mbt}  (TGIF masks are feathered RGB)")

    rows = []
    jl = open(os.path.join(out_dir, "calib_raw.jsonl"), "w", buffering=1)
    for n, rec in enumerate(recs):
        mk = np.array(Image.open(os.path.join(root, rec["mask"])).convert("L")) > mbt
        gc = gt_cells(mk.astype(np.float32), grid, mo)
        for lab, key in (("real", "real"), ("fake", "fake")):
            tools, _ = run_tools(os.path.join(root, rec[key]), tool_cfg, grid=grid)
            for tname, r in tools.items():
                row = dict(pair_id=rec["pair_id"], coco_id=rec["coco_id"],
                           category=rec["category"], mask_type=rec["mask_type"],
                           label=lab, tool=tname, valid=bool(r["valid"]),
                           invalid_reasons=r["invalid_reasons"],
                           tamper_ratio=rec["tamper_ratio"] if lab == "fake" else 0.0,
                           n_gt_cells=len(gc))
                if r["valid"]:
                    for sn, fn in SCORERS.items():
                        s, pc = fn(r["map01"])
                        row[f"s__{sn}"] = s
                        if lab == "fake":
                            row[f"hit__{sn}"] = int(pc in gc)
                    if lab == "fake":
                        row["pixel_auroc"] = pixel_auroc(r["map01"], mk)
                        row["pixel_auroc_inv"] = pixel_auroc(-r["map01"], mk)
                rows.append(row)
                jl.write(json.dumps(row) + "\n")
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(recs)} pairs", flush=True)
    jl.close()

    # ------------------------------------------------- report
    tools_seen = sorted({r["tool"] for r in rows})
    report = {}
    print("\n" + "=" * 78)
    print("GATE 0: TOOL VALIDITY ON TGIF (all PNG)")
    for t in tools_seen:
        sub = [r for r in rows if r["tool"] == t]
        v = sum(1 for r in sub if r["valid"])
        print(f"  {t:<18} valid {v}/{len(sub)} = {v/len(sub):.3f}")
        rs = defaultdict(int)
        for r in sub:
            for x in r["invalid_reasons"]: rs[x] += 1
        for k, c in sorted(rs.items(), key=lambda x: -x[1])[:3]:
            print(f"      {k}: {c}")

    for t in tools_seen:
        sub = [r for r in rows if r["tool"] == t and r["valid"]]
        fk = [r for r in sub if r["label"] == "fake"]
        rl = [r for r in sub if r["label"] == "real"]
        print("\n" + "=" * 78)
        print(f"TOOL: {t}   (valid: real {len(rl)}, fake {len(fk)})")
        if len(fk) < 20 or len(rl) < 20:
            print("  too few valid images -> EXCLUDED from Gate 1")
            report[t] = dict(usable=False, n_real=len(rl), n_fake=len(fk))
            continue

        pa = np.array([r.get("pixel_auroc", np.nan) for r in fk], float)
        pai = np.array([r.get("pixel_auroc_inv", np.nan) for r in fk], float)
        print(f"  B. SPATIAL: pixel AUROC med={np.nanmedian(pa):.3f}  "
              f"frac>0.6={np.nanmean(pa > 0.6):.3f}  frac>0.7={np.nanmean(pa > 0.7):.3f}")
        print(f"     inverted-map pixel AUROC med={np.nanmedian(pai):.3f}")
        rb = float(np.mean([r["n_gt_cells"] / (grid * grid) for r in fk]))
        print(f"     (random 3x3 cell-hit baseline for these samples: {rb:.3f})")

        report[t] = dict(usable=True, n_real=len(rl), n_fake=len(fk),
                         pixel_auroc_median=float(np.nanmedian(pa)),
                         pixel_auroc_frac_gt60=float(np.nanmean(pa > 0.6)),
                         pixel_auroc_inv_median=float(np.nanmedian(pai)),
                         random_cell_hit_baseline=rb, scorers={})

        print(f"  A. IMAGE-LEVEL (blind) + cell-hit vs random:")
        print(f"     {'scorer':<20}{'AUROC':>8}{'thr@5%FPR':>12}{'TPR@thr':>9}"
              f"{'cellhit':>9}{'z_hit':>7}")
        for sn in SCORERS:
            rv = [r[f"s__{sn}"] for r in rl if f"s__{sn}" in r]
            fv = [r[f"s__{sn}"] for r in fk if f"s__{sn}" in r]
            if len(rv) < 20 or len(fv) < 20:
                continue
            a = auroc(rv, fv)
            thr = float(np.percentile(rv, 95.0))          # 5% FPR on paired reals
            tpr = float(np.mean(np.asarray(fv) > thr))
            hits = [r[f"hit__{sn}"] for r in fk if f"hit__{sn}" in r]
            ch = float(np.mean(hits)) if hits else float("nan")
            z = (ch - rb) / np.sqrt(max(rb * (1 - rb) / max(len(hits), 1), 1e-12))
            print(f"     {sn:<20}{a:>8.3f}{thr:>12.3f}{tpr:>9.3f}{ch:>9.3f}{z:>7.2f}")
            report[t]["scorers"][sn] = dict(
                auroc=a, threshold_at_5pct_fpr=thr, tpr_at_threshold=tpr,
                real_median=float(np.median(rv)), fake_median=float(np.median(fv)),
                cell_hit_rate=ch, cell_hit_z=float(z))

        # tamper_ratio stratification for the best scorer by AUROC
        best = max(report[t]["scorers"], key=lambda k: report[t]["scorers"][k]["auroc"])
        thr = report[t]["scorers"][best]["threshold_at_5pct_fpr"]
        print(f"  C. BY TAMPER_RATIO (scorer={best}, thr={thr:.3f} @5% FPR):")
        strat = {}
        for lo, hi in [(0, .01), (.01, .03), (.03, .10), (.10, .60)]:
            s = [r for r in fk if lo <= r["tamper_ratio"] < hi]
            if not s:
                continue
            det = float(np.mean([r[f"s__{best}"] > thr for r in s]))
            px = float(np.nanmedian([r.get("pixel_auroc", np.nan) for r in s]))
            print(f"     [{lo:.2f},{hi:.2f}) n={len(s):>4}  TPR={det:.3f}  pixelAUROC={px:.3f}")
            strat[f"{lo}-{hi}"] = dict(n=len(s), tpr=det, pixel_auroc_median=px)
        report[t]["best_scorer_by_auroc"] = best
        report[t]["tamper_ratio_strata"] = strat

    json.dump(report, open(os.path.join(out_dir, "calibration_report.json"), "w"), indent=1)
    print(f"\nwrote {out_dir}/calibration_report.json and calib_raw.jsonl")
    print("\nNOTHING IS FROZEN YET. Inspect, then write frozen_gate1.json explicitly.")


if __name__ == "__main__":
    main()
