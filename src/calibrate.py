"""Gate-1 blind scorer selection + threshold calibration.

Protocol compliance:
  * Runs ONLY on the calibration split. The test split is untouched here.
  * Candidate scorers are all BLIND (no GT mask ever read).
  * The winner and its threshold are written to a frozen JSON. main_run.py
    refuses to run without that file and never re-derives a threshold.
  * Selecting among pre-declared candidates using calibration-split labels is
    legitimate; doing it on the test split would not be.
"""
import json, os, sys, random
import numpy as np, yaml, cv2
sys.path.insert(0, os.path.dirname(__file__))
from forensic_tools import run_tools, CELL_NAMES


# --------------------------------------------- candidate blind image-level scorers
# Each takes the normalized anomaly map (blind) and returns (score, peak_cell_idx).

def _cellmeans(map01, grid):
    h, w = map01.shape
    ys = np.linspace(0, h, grid + 1).astype(int)
    xs = np.linspace(0, w, grid + 1).astype(int)
    return np.array([[map01[ys[i]:ys[i+1], xs[j]:xs[j+1]].mean()
                      for j in range(grid)] for i in range(grid)], np.float32)


def _peak_cell_from_patchmap(pm, grid):
    """Localize by which coarse cell holds the strongest fine-grained patch."""
    cm = _cellmeans(pm, grid)
    return int(cm.ravel().argmax())


def sc_cell3_peak_z(map01, grid):
    c = _cellmeans(map01, grid).ravel()
    return float((c.max() - c.mean()) / (c.std() + 1e-6)), int(c.argmax())


def _patch_mean_map(map01, frac=0.06):
    """Box-mean map at a scale proportional to image size (~6% of short side)."""
    h, w = map01.shape
    k = max(8, int(round(min(h, w) * frac)) | 1)
    return cv2.blur(map01, (k, k)), k


def sc_patch_peak_z(map01, grid):
    pm, _ = _patch_mean_map(map01)
    z = (pm.max() - pm.mean()) / (pm.std() + 1e-6)
    return float(z), _peak_cell_from_patchmap(pm, grid)


def sc_patch_p999_over_med(map01, grid):
    pm, _ = _patch_mean_map(map01)
    med = float(np.median(pm))
    return float(np.percentile(pm, 99.9) / (med + 1e-6)), _peak_cell_from_patchmap(pm, grid)


def sc_patch_top1pct_z(map01, grid):
    """Mean of the top-1% patch values, standardized by the image's own spread.
    Robust to a single hot pixel, still sensitive to a small confined region."""
    pm, _ = _patch_mean_map(map01)
    f = pm.ravel()
    thr = np.percentile(f, 99.0)
    top = f[f >= thr]
    return float((top.mean() - f.mean()) / (f.std() + 1e-6)), _peak_cell_from_patchmap(pm, grid)


def sc_global_mean(map01, grid):
    """Non-localizing control: pure global anomaly energy."""
    c = _cellmeans(map01, grid).ravel()
    return float(map01.mean()), int(c.argmax())


SCORERS = {
    "cell3_peak_z": sc_cell3_peak_z,
    "patch_peak_z": sc_patch_peak_z,
    "patch_p999_over_med": sc_patch_p999_over_med,
    "patch_top1pct_z": sc_patch_top1pct_z,
    "global_mean": sc_global_mean,
}


def auc(neg, pos):
    neg, pos = np.asarray(neg, float), np.asarray(pos, float)
    if len(neg) == 0 or len(pos) == 0:
        return float("nan")
    allv = np.concatenate([neg, pos])
    rank = allv.argsort().argsort() + 1.0
    return float((rank[len(neg):].sum() - len(pos) * (len(pos) + 1) / 2) / (len(neg) * len(pos)))


def main():
    cfg = yaml.safe_load(open(sys.argv[1]))
    seed = cfg["experiment"]["seed"]
    out_dir = os.path.join(cfg["experiment"]["out_root"], "calibration")
    os.makedirs(out_dir, exist_ok=True)
    root = cfg["datasets"]["cocoglide"]["root"]
    grid = cfg["localization"]["grid"]
    tool_cfg = cfg["forensic_tools"]

    idx = json.load(open("/mnt/disk3/borui/fevi/runs/phase0/cocoglide_index.json"))
    rng = random.Random(seed)
    order = list(range(len(idx)))
    rng.shuffle(order)
    n_cal = cfg["sampling"]["calibration_n"]
    cal_ids = sorted(order[:n_cal])
    test_ids = sorted(order[n_cal:])
    json.dump({"calibration": [idx[i]["pair_id"] for i in cal_ids],
               "test": [idx[i]["pair_id"] for i in test_ids],
               "seed": seed},
              open(os.path.join(out_dir, "splits.json"), "w"), indent=1)
    print(f"split: calibration={len(cal_ids)} test={len(test_ids)} (disjoint, seed={seed})")

    # per-image raw records on the calibration split only
    recs = []
    for k, i in enumerate(cal_ids):
        rec = idx[i]
        for lab, key in (("real", "real"), ("fake", "fake")):
            tools, _ = run_tools(os.path.join(root, rec[key]), tool_cfg, grid=grid)
            for tname, r in tools.items():
                row = dict(pair_id=rec["pair_id"], label=lab, tool=tname,
                           valid=bool(r["valid"]), reasons=r["invalid_reasons"],
                           tamper_ratio=rec["tamper_ratio"] if lab == "fake" else 0.0)
                for sname, fn in SCORERS.items():
                    s, pc = fn(r["map01"], grid)
                    row[f"score__{sname}"] = s
                    row[f"peak__{sname}"] = CELL_NAMES[pc]
                recs.append(row)
        if (k + 1) % 20 == 0:
            print(f"  {k+1}/{len(cal_ids)} pairs processed", flush=True)
    json.dump(recs, open(os.path.join(out_dir, "calibration_raw.json"), "w"), indent=1)

    # --------------------------------------------------- evaluate candidates
    print("\nBlind Gate-1 scorer comparison on calibration split")
    print("(AUC = separability of fake vs real using the blind score; VALID images only)")
    report = {}
    for tname in ("ela", "noise_residual", "dct_jpeg"):
        sub = [r for r in recs if r["tool"] == tname and r["valid"]]
        nreal = sum(1 for r in sub if r["label"] == "real")
        nfake = sum(1 for r in sub if r["label"] == "fake")
        tot = sum(1 for r in recs if r["tool"] == tname)
        print(f"\n{tname}: valid {len(sub)}/{tot}  (real {nreal}, fake {nfake})")
        if nreal < 10 or nfake < 10:
            print("  -> too few valid images; tool excluded from Gate 1")
            report[tname] = {"usable": False, "n_valid_real": nreal, "n_valid_fake": nfake}
            continue
        report[tname] = {"usable": True, "n_valid_real": nreal, "n_valid_fake": nfake,
                         "scorers": {}}
        for sname in SCORERS:
            r_ = [x[f"score__{sname}"] for x in sub if x["label"] == "real"]
            f_ = [x[f"score__{sname}"] for x in sub if x["label"] == "fake"]
            a = auc(r_, f_)
            thr = float(np.percentile(r_, 100 * (1 - cfg["gate1"]["calibration_fpr"])))
            tpr = float(np.mean(np.asarray(f_) > thr))
            # localization sanity of the scorer's own peak cell vs GT cells
            print(f"  {sname:<22} AUC={a:6.3f}  thr@FPR{cfg['gate1']['calibration_fpr']:.2f}={thr:8.3f}"
                  f"  gate1_pass_rate_fake={tpr:6.3f}")
            report[tname]["scorers"][sname] = dict(auc=a, threshold=thr, fake_pass_rate=tpr,
                                                  real_median=float(np.median(r_)),
                                                  fake_median=float(np.median(f_)))
    json.dump(report, open(os.path.join(out_dir, "scorer_comparison.json"), "w"), indent=1)
    print(f"\nwrote {out_dir}/scorer_comparison.json, calibration_raw.json, splits.json")
    print("\nNOTE: no scorer/threshold is frozen automatically. Inspect the table, then")
    print("write frozen_gate1.json explicitly so the choice is on the record.")


if __name__ == "__main__":
    main()
