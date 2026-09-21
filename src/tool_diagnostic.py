"""POST-HOC tool-quality diagnostic (calibration split only).

GT masks are read here for EVALUATION ONLY. Nothing computed in this file feeds
back into any Gate threshold or any blind score. Purpose: distinguish two very
different situations that the image-level AUC table cannot separate:

  (a) the tools produce no usable signal at all  -> Gate 1 can never pass
      honestly, and Phase 1 cannot run on this dataset;
  (b) the tools DO mark the tampered region inside a fake image, but the
      image-level score is not discriminative across images -> the tool is
      usable as localized evidence, and the Gate-1 criterion needs to be a
      within-image localized-anomaly test, declared before touching test data.

Metrics:
  pixel_auc   : anomaly map vs GT mask, per fake image (chance = 0.5)
  peak_hit    : tool's peak 3x3 cell falls in a GT cell (chance = |GT|/9)
  random_base : mean |GT cells|/9 over the same images
"""
import json, os, sys, random
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from forensic_tools import run_tools, CELL_NAMES
from calibrate import SCORERS, _cellmeans


def gt_cells(mask, grid, min_overlap):
    h, w = mask.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    out = []
    for i in range(grid):
        for j in range(grid):
            blk = mask[ys[i]:ys[i+1], xs[j]:xs[j+1]]
            if blk.mean() >= min_overlap:
                out.append(i * grid + j)
    if not out:  # degenerate: fall back to the single most-covered cell
        cov = [[mask[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean() for j in range(grid)]
               for i in range(grid)]
        out = [int(np.argmax(np.array(cov).ravel()))]
    return out


def pixel_auc(map01, mask_bin, max_px=200000):
    pos = map01[mask_bin]; neg = map01[~mask_bin]
    if pos.size < 10 or neg.size < 10:
        return float("nan")
    rs = np.random.default_rng(0)
    if pos.size > max_px: pos = rs.choice(pos, max_px, replace=False)
    if neg.size > max_px: neg = rs.choice(neg, max_px, replace=False)
    allv = np.concatenate([neg, pos]); rank = allv.argsort().argsort() + 1.0
    return float((rank[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    root = cfg["datasets"]["cocoglide"]["root"]
    grid = cfg["localization"]["grid"]
    mo = cfg["localization"]["gt_cell_min_overlap"]
    tool_cfg = cfg["forensic_tools"]
    out_dir = os.path.join(cfg["experiment"]["out_root"], "calibration")
    cal_ids = set(json.load(open(os.path.join(out_dir, "splits.json")))["calibration"])
    idx = [r for r in json.load(open("/mnt/disk3/borui/fevi/runs/phase0/cocoglide_index.json"))
           if r["pair_id"] in cal_ids]
    print(f"diagnostic on {len(idx)} calibration FAKE images (GT used for evaluation only)\n")

    rows = []
    for k, rec in enumerate(idx):
        mask = np.array(Image.open(os.path.join(root, rec["mask"])).convert("L")) > 127
        gcells = gt_cells(mask.astype(np.float32), grid, mo)
        tools, _ = run_tools(os.path.join(root, rec["fake"]), tool_cfg, grid=grid)
        for tname, r in tools.items():
            if not r["valid"]:
                continue
            m = r["map01"]
            if m.shape != mask.shape:
                m = cv2.resize(m, (mask.shape[1], mask.shape[0]))
            row = dict(pair_id=rec["pair_id"], tool=tname,
                       tamper_ratio=rec["tamper_ratio"],
                       n_gt_cells=len(gcells),
                       pixel_auc=pixel_auc(m, mask))
            for sname, fn in SCORERS.items():
                _, pc = fn(m, grid)
                row[f"peakhit__{sname}"] = int(pc in gcells)
            rows.append(row)
        if (k + 1) % 30 == 0:
            print(f"  {k+1}/{len(idx)}", flush=True)
    json.dump(rows, open(os.path.join(out_dir, "tool_diagnostic.json"), "w"), indent=1)

    print(f"\n{'tool':<16}{'n':>5}{'pixel_auc_med':>15}{'pAUC>0.6':>10}{'rand_base':>11}")
    for tname in sorted({r["tool"] for r in rows}):
        sub = [r for r in rows if r["tool"] == tname]
        pa = np.array([r["pixel_auc"] for r in sub], float)
        rb = np.mean([r["n_gt_cells"] / (grid * grid) for r in sub])
        print(f"{tname:<16}{len(sub):>5}{np.nanmedian(pa):>15.3f}{np.nanmean(pa>0.6):>10.3f}{rb:>11.3f}")
        for sname in SCORERS:
            hit = np.mean([r[f"peakhit__{sname}"] for r in sub])
            # binomial z vs the per-image random baseline
            p0 = rb; n = len(sub)
            z = (hit - p0) / np.sqrt(p0 * (1 - p0) / n)
            print(f"    peak_hit[{sname:<20}] = {hit:.3f}   random={p0:.3f}   z={z:+.2f}")

    print("\nstratified by tamper_ratio (pixel_auc median):")
    bins = [(0, .05), (.05, .15), (.15, .35), (.35, 1.01)]
    for tname in sorted({r["tool"] for r in rows}):
        line = [f"  {tname:<16}"]
        for lo, hi in bins:
            sub = [r["pixel_auc"] for r in rows
                   if r["tool"] == tname and lo <= r["tamper_ratio"] < hi]
            line.append(f"[{lo:.2f},{hi:.2f}) n={len(sub):>3} med="
                        f"{(np.nanmedian(sub) if sub else float('nan')):.3f}")
        print("  ".join(line))


if __name__ == "__main__":
    main()
