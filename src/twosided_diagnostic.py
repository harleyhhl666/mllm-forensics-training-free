"""Two-sided anomaly diagnostic (calibration split only, GT for evaluation only).

Motivation from the first diagnostic: pixel-AUC sat BELOW 0.5 for large-area
edits and ABOVE 0.5 for small ones, which means the GLIDE-inpainted region is
often noise-DEFICIENT rather than noise-excessive. A one-sided "find the hottest
patch" detector is structurally blind to that. Here we test whether a two-sided
(deviation in either direction) blind criterion recovers real signal.

Still blind: every score is computed from the anomaly map alone. GT enters only
in pixel_auc / peak_hit evaluation.
"""
import json, os, sys
import numpy as np, yaml, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from forensic_tools import run_tools, CELL_NAMES
from tool_diagnostic import gt_cells, pixel_auc


def patch_map(map01, frac=0.06):
    h, w = map01.shape
    k = max(8, int(round(min(h, w) * frac)) | 1)
    return cv2.blur(map01, (k, k))


def two_sided(map01, grid):
    """Blind two-sided score: largest |cell mean - median cell| in units of MAD.
    Returns (score, peak_cell, direction)."""
    pm = patch_map(map01)
    h, w = pm.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    c = np.array([pm[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean()
                  for i in range(grid) for j in range(grid)], np.float32)
    med = float(np.median(c))
    mad = float(np.median(np.abs(c - med))) + 1e-6
    dev = (c - med) / mad
    k = int(np.abs(dev).argmax())
    return float(abs(dev[k])), k, ("high" if dev[k] > 0 else "low")


def low_only(map01, grid):
    """Blind one-sided LOW score: the most noise-deficient cell."""
    pm = patch_map(map01)
    h, w = pm.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    c = np.array([pm[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean()
                  for i in range(grid) for j in range(grid)], np.float32)
    med = float(np.median(c)); mad = float(np.median(np.abs(c - med))) + 1e-6
    dev = (c - med) / mad
    k = int(dev.argmin())
    return float(-dev[k]), k, "low"


SC2 = {"two_sided_mad": two_sided, "low_only_mad": low_only}


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    root = cfg["datasets"]["cocoglide"]["root"]
    grid = cfg["localization"]["grid"]; mo = cfg["localization"]["gt_cell_min_overlap"]
    tool_cfg = cfg["forensic_tools"]
    out_dir = os.path.join(cfg["experiment"]["out_root"], "calibration")
    cal = set(json.load(open(os.path.join(out_dir, "splits.json")))["calibration"])
    idx = [r for r in json.load(open("/mnt/disk3/borui/fevi/runs/phase0/cocoglide_index.json"))
           if r["pair_id"] in cal]

    rows = []
    for k, rec in enumerate(idx):
        mask = np.array(Image.open(os.path.join(root, rec["mask"])).convert("L")) > 127
        gcells = gt_cells(mask.astype(np.float32), grid, mo)
        for lab, key in (("real", "real"), ("fake", "fake")):
            tools, _ = run_tools(os.path.join(root, rec[key]), tool_cfg, grid=grid)
            for tname, r in tools.items():
                if not r["valid"]:
                    continue
                m = r["map01"]
                row = dict(pair_id=rec["pair_id"], label=lab, tool=tname,
                           tamper_ratio=rec["tamper_ratio"] if lab == "fake" else 0.0,
                           n_gt_cells=len(gcells))
                mm = m if m.shape == mask.shape else cv2.resize(m, (mask.shape[1], mask.shape[0]))
                if lab == "fake":
                    row["pixel_auc_inverted"] = pixel_auc(-mm, mask)
                for sname, fn in SC2.items():
                    s, pc, d = fn(m, grid)
                    row[f"score__{sname}"] = s
                    row[f"dir__{sname}"] = d
                    row[f"peakhit__{sname}"] = int(pc in gcells) if lab == "fake" else None
                rows.append(row)
        if (k + 1) % 40 == 0:
            print(f"  {k+1}/{len(idx)}", flush=True)
    json.dump(rows, open(os.path.join(out_dir, "twosided_diagnostic.json"), "w"), indent=1)

    def auc(neg, pos):
        neg, pos = np.asarray(neg, float), np.asarray(pos, float)
        allv = np.concatenate([neg, pos]); rk = allv.argsort().argsort() + 1.0
        return float((rk[len(neg):].sum() - len(pos)*(len(pos)+1)/2) / (len(neg)*len(pos)))

    print("\n=== two-sided blind criteria, calibration split ===")
    for tname in sorted({r["tool"] for r in rows}):
        sub = [r for r in rows if r["tool"] == tname]
        fake = [r for r in sub if r["label"] == "fake"]
        real = [r for r in sub if r["label"] == "real"]
        inv = np.array([r["pixel_auc_inverted"] for r in fake], float)
        print(f"\n{tname}  (valid: real {len(real)}, fake {len(fake)})")
        print(f"  pixel_auc with INVERTED map: med={np.nanmedian(inv):.3f}  frac>0.6={np.nanmean(inv>0.6):.3f}")
        rb = np.mean([r["n_gt_cells"]/(grid*grid) for r in fake])
        for sname in SC2:
            a = auc([r[f"score__{sname}"] for r in real], [r[f"score__{sname}"] for r in fake])
            hit = np.mean([r[f"peakhit__{sname}"] for r in fake])
            z = (hit - rb)/np.sqrt(rb*(1-rb)/len(fake))
            dirs = {}
            for r in fake: dirs[r[f"dir__{sname}"]] = dirs.get(r[f"dir__{sname}"], 0) + 1
            print(f"  {sname:<16} image_AUC={a:.3f}  peak_hit={hit:.3f} (random {rb:.3f}, z={z:+.2f})  dirs={dirs}")
        print("  pixel_auc_inverted by tamper_ratio:")
        for lo, hi in [(0,.05),(.05,.15),(.15,.35),(.35,1.01)]:
            v = [r["pixel_auc_inverted"] for r in fake if lo <= r["tamper_ratio"] < hi]
            print(f"    [{lo:.2f},{hi:.2f}) n={len(v):>3} med={(np.nanmedian(v) if v else float('nan')):.3f}")


if __name__ == "__main__":
    main()
