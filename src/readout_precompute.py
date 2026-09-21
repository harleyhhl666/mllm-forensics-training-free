"""Stage 1 of Gate-1 read-out research: PRECOMPUTE blind window statistics.

Nothing here is a scorer decision and nothing is thresholded. This pass computes,
once per image, the raw material every candidate read-out needs, so that the
grouped-CV selection afterwards is pure arithmetic on a table (fast, and no map is
recomputed per fold).

FROZEN hyperparameter space, declared before any result is seen:
  scales (area fractions)      : 0.02, 0.05, 0.10, 0.20        [4, fixed]
  family A aggregation         : max over scales                [fixed]
  family B region rules        : otsu, p99, p995                [3, fixed]
  family C null scaling        : median / MAD                   [fixed]
  => 6 scorers total: A, B_otsu, B_p99, B_p995, C(=A+C), C_then_B
No scale, rule, or scorer may be added after seeing results.

BLINDNESS: every quantity written under the 'blind' key is computed from the
anomaly map alone. GT-derived fields live under 'eval' and are computed AFTER the
blind candidate bbox is already fixed, purely to score it later. No 'eval' field
may ever be read by a scorer or a threshold.
"""
import argparse, json, os, sys, time
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from forensic_tools import run_tools

SCALES = (0.02, 0.05, 0.10, 0.20)          # FROZEN
REGION_RULES = ("otsu", "p99", "p995")      # FROZEN


# --------------------------------------------------------------- window scan (A)

def window_stats(map01, area_frac):
    """Slide a square window of the given AREA FRACTION over the map and return
    the within-image standardized peak plus that window's bbox.

    Standardizing inside the image is what makes this blind: no cross-image
    constant, no GT, no dataset-level statistic enters.
    """
    h, w = map01.shape
    side = int(round(np.sqrt(area_frac * h * w)))
    side = max(16, min(side, min(h, w)))
    stride = max(4, side // 4)
    integ = cv2.integral(map01.astype(np.float64))      # (h+1, w+1)

    ys = list(range(0, h - side + 1, stride)) or [0]
    xs = list(range(0, w - side + 1, stride)) or [0]
    if ys[-1] != h - side:
        ys.append(max(0, h - side))
    if xs[-1] != w - side:
        xs.append(max(0, w - side))

    Y = np.array(ys)[:, None]
    X = np.array(xs)[None, :]
    A = integ[Y, X]
    B = integ[Y, X + side]
    C = integ[Y + side, X]
    D = integ[Y + side, X + side]
    means = (D - B - C + A) / float(side * side)

    flat = means.ravel()
    mu, sd = float(flat.mean()), float(flat.std())
    k = int(flat.argmax())
    iy, ix = divmod(k, means.shape[1])
    y0, x0 = int(ys[iy]), int(xs[ix])
    z = (float(flat.max()) - mu) / (sd + 1e-9)
    return dict(scale=area_frac, side=side, n_windows=int(flat.size),
                peak_mean=float(flat.max()), win_mu=mu, win_sd=sd,
                within_image_z=float(z),
                bbox=[x0, y0, x0 + side, y0 + side])


# ------------------------------------------------------- region contrast (B)

def region_contrast(map01, rule):
    """Blind candidate region from the map itself, then inside/outside contrast."""
    u8 = np.clip(map01 * 255, 0, 255).astype(np.uint8)
    if rule == "otsu":
        t, _ = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = float(t) / 255.0
    elif rule == "p99":
        thr = float(np.percentile(map01, 99.0))
    elif rule == "p995":
        thr = float(np.percentile(map01, 99.5))
    else:
        raise ValueError(rule)

    reg = (map01 > thr).astype(np.uint8)
    if reg.sum() < 16 or reg.sum() > 0.9 * reg.size:
        return dict(rule=rule, contrast=0.0, area_frac=float(reg.mean()),
                    bbox=None, degenerate=True)

    # keep the largest connected component as THE candidate region
    n, lab, stats, _ = cv2.connectedComponentsWithStats(reg, connectivity=8)
    if n <= 1:
        return dict(rule=rule, contrast=0.0, area_frac=float(reg.mean()),
                    bbox=None, degenerate=True)
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    m = lab == big
    x, y, ww, hh = (int(stats[big, cv2.CC_STAT_LEFT]), int(stats[big, cv2.CC_STAT_TOP]),
                    int(stats[big, cv2.CC_STAT_WIDTH]), int(stats[big, cv2.CC_STAT_HEIGHT]))
    inside = map01[m]
    outside = map01[~m]
    if inside.size < 16 or outside.size < 16:
        return dict(rule=rule, contrast=0.0, area_frac=float(m.mean()),
                    bbox=None, degenerate=True)
    contrast = float((inside.mean() - outside.mean()) / (map01.std() + 1e-9))
    return dict(rule=rule, contrast=contrast, area_frac=float(m.mean()),
                bbox=[x, y, x + ww, y + hh], degenerate=False)


# --------------------------------------------------------------- GT evaluation
# Computed only AFTER the blind bbox exists; never read by a scorer.

def bbox_eval(bbox, mask):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    H, W = mask.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    sub = mask[y1:y2, x1:x2]
    inter = int(sub.sum())
    barea = (x2 - x1) * (y2 - y1)
    marea = int(mask.sum())
    cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
    return dict(
        hit=bool(inter > 0),
        precision=float(inter / max(barea, 1)),          # bbox purity
        recall=float(inter / max(marea, 1)),             # mask coverage
        iou=float(inter / max(barea + marea - inter, 1)),
        center_in_mask=bool(mask[min(cy, H - 1), min(cx, W - 1)]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--split", default="calibration")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tools", default="ela,noise_residual")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds = cfg["datasets"]["tgif_sp"]
    root = ds["root"]
    mbt = cfg["localization"]["mask_binarize_threshold"]
    tool_cfg = cfg["forensic_tools"]
    want = set(args.tools.split(","))

    split = json.load(open(os.path.join(cfg["experiment"]["out_root"],
                                        "splits_tgif", "split_frozen.json")))
    keep = set(split[f"{args.split}_pair_ids"])
    recs = [r for r in json.load(open(ds["index"])) if r["pair_id"] in keep]
    if args.limit:
        recs = recs[:args.limit]

    out_dir = args.out or os.path.join(cfg["experiment"]["out_root"], "readout_tgif")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"windows_{args.split}.jsonl")
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                d = json.loads(line); done.add((d["pair_id"], d["label"], d["tool"]))
            except Exception:
                pass
    fh = open(path, "a", buffering=1)
    print(f"{args.split}: {len(recs)} pairs, tools={sorted(want)}, resume={len(done)} rows")
    print(f"FROZEN scales={SCALES}  region_rules={REGION_RULES}")

    t0 = time.time()
    for n, rec in enumerate(recs):
        mask = np.array(Image.open(os.path.join(root, rec["mask"])).convert("L")) > mbt
        for lab, key in (("real", "real"), ("fake", "fake")):
            need = [t for t in want if (rec["pair_id"], lab, t) not in done]
            if not need:
                continue
            tools, _ = run_tools(os.path.join(root, rec[key]), tool_cfg,
                                grid=cfg["localization"]["grid"])
            for tname in need:
                r = tools.get(tname)
                if r is None:
                    continue
                row = dict(pair_id=rec["pair_id"], coco_id=rec["coco_id"],
                           category=rec["category"], mask_type=rec["mask_type"],
                           variant=rec["variant"], label=lab, tool=tname,
                           valid=bool(r["valid"]), invalid_reasons=r["invalid_reasons"],
                           tamper_ratio=rec["tamper_ratio"] if lab == "fake" else 0.0,
                           height=int(r["map01"].shape[0]), width=int(r["map01"].shape[1]))
                if r["valid"]:
                    m = r["map01"]
                    blind = dict(scan=[window_stats(m, s) for s in SCALES],
                                 region=[region_contrast(m, q) for q in REGION_RULES])
                    row["blind"] = blind
                    if lab == "fake":
                        row["eval"] = dict(
                            scan=[bbox_eval(w["bbox"], mask) for w in blind["scan"]],
                            region=[bbox_eval(g["bbox"], mask) for g in blind["region"]],
                        )
                fh.write(json.dumps(row) + "\n")
        if (n + 1) % 25 == 0:
            el = time.time() - t0
            print(f"  {n+1}/{len(recs)} pairs  {el/60:.1f} min "
                  f"({el/(n+1):.2f} s/pair)", flush=True)
    fh.close()
    print(f"done -> {path}")


if __name__ == "__main__":
    main()
